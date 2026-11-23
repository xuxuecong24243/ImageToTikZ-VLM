"""Validate a narrow scene grammar using only stdlib.

Math uses an allowlist, not arbitrary TeX. This is not a complete mathematical
parser or a TeX sandbox. Public compilation still needs OS-level isolation.
"""
from __future__ import annotations

import math
import re

from .schema import (CARDINAL_ANCHORS, COLORS, LABEL_ANCHORS, MATH_COMMANDS,
                     MAX_RICH_SEGMENTS, NODE_TYPES, STYLE_DEFAULTS, TEXT_MODES,
                     OBJECT_CONTRACTS, STYLE_LIMITS, ALIGNMENTS, ROUTINGS, ARROWS, AXES)
from .geometry import edge_route_points, diamond_path_error


class SceneValidationError(ValueError):
    """The scene violates the explicit supported contract."""


def _object(value, allowed, required, location):
    if not isinstance(value, dict):
        raise SceneValidationError(location + " must be an object.")
    unknown, missing = set(value) - set(allowed), set(required) - set(value)
    if unknown or missing:
        raise SceneValidationError("{}: unknown fields {}; missing fields {}.".format(
            location, sorted(str(k) for k in unknown), sorted(missing)))


def _contract_object(value, name, location):
    contract = OBJECT_CONTRACTS[name]
    _object(value, contract['properties'], contract['required'], location)


def _number(value, location, minimum=None, maximum=None, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SceneValidationError(location + " must be a finite number.")
    if ((positive and value <= 0) or (minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)):
        raise SceneValidationError(location + " is outside the supported range.")
    return value


def _choice(value, choices, location):
    if not isinstance(value, str) or value not in choices:
        raise SceneValidationError(location + " has an unsupported value.")


def _text(value, location):
    if not isinstance(value, str) or len(value) > 4000:
        raise SceneValidationError(location + " must be a string of at most 4000 characters.")
    if any(ord(c) < 32 and c != "\n" for c in value) or "\x7f" in value:
        raise SceneValidationError(location + " contains unsupported control characters.")


def escape_text(text):
    """Escape characters once, without reinterpreting literal command strings."""
    _text(text, "text")
    replacements = {
        "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&",
        "#": r"\#", "%": r"\%", "_": r"\_", "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}", "\n": r"\\",
    }
    return "".join(replacements.get(c, c) for c in text)


def validate_math(text):
    """Validate without rewriting the accepted LaTeX expression."""
    _text(text, "math text")
    if not text.strip() or any(c in text for c in "$%#&\n") or "^^" in text:
        raise SceneValidationError("Math needs content without delimiters, comments or control escapes.")
    depth, index = 0, 0
    while index < len(text):
        character = text[index]
        if character == "\\":
            match = re.match(r"\\([A-Za-z]+|.)", text[index:])
            if not match:
                raise SceneValidationError("A math command is incomplete.")
            command = match.group(1)
            if command not in MATH_COMMANDS and command not in {" ", ",", ";", ":", "!", "{", "}", "|"}:
                raise SceneValidationError("Unsupported math command: \\" + command)
            index += len(match.group(0))
            continue
        if character == "{":
            depth += 1
            if depth > 32:
                raise SceneValidationError("Math grouping exceeds 32 levels.")
        elif character == "}":
            depth -= 1
            if depth < 0:
                raise SceneValidationError("Unbalanced math braces.")
        index += 1
    if depth:
        raise SceneValidationError("Unbalanced math braces.")


def _style(style, location):
    _contract_object(style, 'style', location)
    for field in ("fill", "stroke", "text_color"):
        if field in style:
            _choice(style[field], set(COLORS) | ({"none"} if field in {"fill", "stroke"} else set()), location + "." + field)
    for field in ('fill_color','stroke_color','text_color_rgb'):
        if field in style and (not isinstance(style[field],str) or re.fullmatch(r'#[0-9A-Fa-f]{6}',style[field]) is None):
            raise SceneValidationError(location + '.' + field + ' requires #RRGGBB.')
    for field, limits in STYLE_LIMITS.items():
        if field in style:
            _number(style[field], location + "." + field, *limits)
    if "bold" in style and not isinstance(style["bold"], bool):
        raise SceneValidationError(location + ".bold must be Boolean.")
    if 'font_family' in style:
        _choice(style['font_family'], {'serif','sans','mono'}, location+'.font_family')
    if 'preserve_spaces' in style and not isinstance(style['preserve_spaces'],bool):
        raise SceneValidationError(location+'.preserve_spaces must be Boolean.')


def resolve_style(scene, item):
    """Return an independent default-filled style, leaving the scene untouched."""
    raw = item.get("style", {})
    if isinstance(raw, str):
        if raw not in scene.get("styles", {}):
            raise SceneValidationError("Unknown style name: " + raw)
        raw = scene["styles"][raw]
    _style(raw, "style")
    result = dict(STYLE_DEFAULTS)
    result.update(raw)
    if "font_size" in item:
        _number(item["font_size"], "font_size", 6, 80)
        result["font_size"] = item["font_size"]
    return result


def _point(value, width, height, location):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SceneValidationError(location + " must be [x,y].")
    _number(value[0], location + ".x", 0, width)
    _number(value[1], location + ".y", 0, height)


def validate_rich_segments(segments):
    """Validate ordered spans before emitting any raw math into a TikZ node."""
    if not isinstance(segments, list) or not 1 <= len(segments) <= MAX_RICH_SEGMENTS:
        raise SceneValidationError("rich segments needs between 1 and 128 entries.")
    total = 0
    for index, segment in enumerate(segments):
        location = "segments[{}]".format(index)
        _contract_object(segment, 'segment', location)
        _choice(segment["type"], {"text", "math"}, location + ".type")
        _text(segment["content"], location + ".content")
        total += len(segment["content"])
        if segment["type"] == "math":
            validate_math(segment["content"])
    if not 1 <= total <= 4000:
        raise SceneValidationError("rich content must contain 1 to 4000 characters in total.")


def _label_fields(item, location, *, required_text=False):
    mode = item.get("mode", "text")
    _choice(mode, TEXT_MODES, location + ".mode")
    _choice(item.get("align", "center"), ALIGNMENTS, location + ".align")
    if "font_size" in item:
        _number(item["font_size"], location + ".font_size", 6, 80)
    if mode == "rich":
        validate_rich_segments(item.get("segments"))
        if "text" in item:
            _text(item["text"], location + ".text")  # provenance, never rendered twice
        return
    if "segments" in item:
        raise SceneValidationError(location + ": segments requires mode=rich.")
    if required_text or "text" in item:
        _text(item.get("text"), location + ".text")
        if item.get("mode", "text") == "math":
            validate_math(item["text"])


def validate_scene(scene):
    """Validate structure and references before generating any TikZ."""
    _contract_object(scene, 'scene', 'scene')
    canvas = scene["canvas"]
    _contract_object(canvas, 'canvas', 'canvas')
    if 'layered' in canvas and not isinstance(canvas['layered'], bool):
        raise SceneValidationError('canvas.layered must be Boolean.')
    if 'measure_text' in canvas and not isinstance(canvas['measure_text'],bool):
        raise SceneValidationError('canvas.measure_text must be Boolean.')
    width = _number(canvas["width"], "canvas.width", 1, 20000)
    height = _number(canvas["height"], "canvas.height", 1, 20000)
    _choice(canvas["coordinate_system"], {"top-left"}, "canvas.coordinate_system")
    _number(canvas.get("scale", .4), "canvas.scale", .01, 2)
    if not isinstance(scene["styles"], dict) or len(scene["styles"]) > 100:
        raise SceneValidationError("styles must contain at most 100 entries.")
    for name, value in scene["styles"].items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name):
            raise SceneValidationError("Style names must be simple identifiers.")
        _style(value, "styles." + name)
    identifiers, node_ids = set(), set()
    for kind in ("nodes", "edges", "labels"):
        if not isinstance(scene[kind], list) or len(scene[kind]) > 500:
            raise SceneValidationError(kind + " must be an array of at most 500 objects.")
        for index, item in enumerate(scene[kind]):
            name = item.get("id") if isinstance(item, dict) else None
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name):
                raise SceneValidationError("{}.{} needs a simple unique id.".format(kind, index))
            if name in identifiers:
                raise SceneValidationError("Duplicate ID: " + name)
            identifiers.add(name)
            if 'z_index' in item and (type(item['z_index']) is not int or not 0 <= item['z_index'] <= 10000):
                raise SceneValidationError(kind + '.' + str(index) + '.z_index must be an integer in [0,10000].')
            if kind == "nodes":
                node_ids.add(name)
    for node in scene["nodes"]:
        location = "node " + node["id"]
        _contract_object(node, 'node', location)
        _choice(node["type"], NODE_TYPES, location + ".type")
        if "wrap" in node and (node['type'] != 'text' or not isinstance(node['wrap'], bool)):
            raise SceneValidationError(location + '.wrap must be Boolean and is only supported for text nodes.')
        box = node["bbox"]
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise SceneValidationError(location + ".bbox must be [x,y,width,height].")
        x, y, w, h = box
        _point([x, y], width, height, location + ".bbox")
        _number(w, location + ".width", positive=True)
        _number(h, location + ".height", positive=True)
        _point([x + w, y + h], width, height, location + ".bbox end")
        if "background_shape" in node:
            _choice(node["background_shape"], {"circle"}, location + ".background_shape")
            if node["type"] != "text" or w != h:
                raise SceneValidationError(location + ".background_shape requires a text node with a square bbox.")
        if node["type"] in {"matrix", "tensor3d"}:
            grid = node.get("grid")
            _contract_object(grid, 'grid', location + '.grid')
            for field in ("rows", "cols"):
                value = grid[field]
                if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 50:
                    raise SceneValidationError(location + ".grid." + field + " must be an integer in [1,50].")
        elif "grid" in node:
            raise SceneValidationError(location + " does not support a grid.")
        if node["type"] == "tensor3d":
            depth = node.get("depth")
            _contract_object(depth, 'depth', location + '.depth')
            dx = _number(depth["dx"], location + ".depth.dx", positive=True)
            dy = _number(depth["dy"], location + ".depth.dy")
            if dy >= 0:
                raise SceneValidationError(location + ".depth.dy must be negative for top-left coordinates.")
            _point([x + dx, y + dy], width, height, location + ".back top-left")
            _point([x + w + dx, y + h + dy], width, height, location + ".back bottom-right")
        elif "depth" in node:
            raise SceneValidationError(location + " does not support depth.")
        _label_fields(node, location, required_text=node["type"] == "text")
        resolve_style(scene, node)
    nodes_by_id = {node['id']: node for node in scene['nodes']}
    for edge in scene["edges"]:
        location = "edge " + edge["id"]
        _contract_object(edge, 'edge', location)
        if "points" in edge:
            if any(key in edge for key in ("from", "to", "from_anchor", "to_anchor")):
                raise SceneValidationError(location + " cannot mix absolute points and node references.")
            points = edge["points"]
            if not isinstance(points, list) or not 2 <= len(points) <= 100:
                raise SceneValidationError(location + ".points needs between 2 and 100 points.")
            for index, point in enumerate(points):
                _point(point, width, height, location + ".points." + str(index))
            if any(a == b for a, b in zip(points, points[1:])):
                raise SceneValidationError(location + " contains a zero-length segment.")
        else:
            for field in ("from", "to"):
                if not isinstance(edge.get(field), str) or edge[field] not in node_ids:
                    raise SceneValidationError(location + "." + field + " references an unknown node.")
            for field, default in (("from_anchor", "east"), ("to_anchor", "west")):
                _choice(edge.get(field, default), CARDINAL_ANCHORS, location + "." + field)
        _choice(edge.get("routing", "straight"), ROUTINGS, location + ".routing")
        _choice(edge.get("first_axis", "x"), AXES, location + ".first_axis")
        _choice(edge.get("arrow", "none"), ARROWS, location + ".arrow")
        if "dashed" in edge and not isinstance(edge["dashed"], bool):
            raise SceneValidationError(location + ".dashed must be Boolean.")
        resolve_style(scene, edge)
        metadata = edge.get('metadata', {})
        metadata = metadata if isinstance(metadata, dict) else {}
        for role, key, meta_key in (('source', 'from', 'source_id'), ('target', 'to', 'target_id')):
            identifier = edge.get(key, metadata.get(meta_key))
            node = nodes_by_id.get(identifier) if isinstance(identifier, str) else None
            if node is not None and node['type'] == 'diamond':
                problem = diamond_path_error(edge_route_points(edge, nodes_by_id), node, endpoint=role)
                if problem:
                    raise SceneValidationError(location + '.' + key + ': ' + problem)
    for label in scene["labels"]:
        location = "label " + label["id"]
        _contract_object(label, 'label', location)
        _point([label["x"], label["y"]], width, height, location)
        _choice(label.get("anchor", "center"), LABEL_ANCHORS, location + ".anchor")
        _label_fields(label, location, required_text=True)
        resolve_style(scene, label)
