"""Limited scene contract; dimensions, fonts and line widths use source pixels.

canvas.scale is points per pixel (default 0.4), shared by both axes.
bbox=[left,top,width,height]; tensor bbox describes its front face.
depth={dx>0,dy<0} projects towards the source image's upper right.
Styles select only the keys/colors below; raw TikZ options are not supported.
Fill and stroke may be "none" to disable painting; text_color requires a color.
Text-bearing objects accept mode=text|math|rich. Rich uses ordered segments
[{type: text|math, content: str}]; optional text retains the unmodified source.
Segments are only legal in rich mode; math content has no outer delimiters.
Text nodes may set wrap=False to omit bbox-derived paragraph width, preserving
their anchor bbox and font size. The legacy default is wrap=True.
Text nodes may explicitly set background_shape="circle" with a square bbox.
The bbox defines the center and minimum diameter; the native TikZ node contains
both background and centered text, without paragraph width. Long text can grow
the circle beyond that minimum; no automatic text fitting is performed.
Edges either reference two nodes and cardinal anchors, or contain an absolute
points polyline. Metadata is never emitted and can record manual review notes.
"""

COLORS = {
    "light_blue": "D8EDF8", "dark_blue": "26445C", "blue": "2477AA",
    "black": "111111", "white": "FFFFFF", "gray": "777777",
}
NODE_TYPES = frozenset({"rectangle", "rounded_rectangle", "diamond", "matrix", "tensor3d", "text"})
TEXT_MODES = frozenset({"text", "math", "rich"})
MAX_RICH_SEGMENTS = 128
CARDINAL_ANCHORS = frozenset({"north", "south", "east", "west"})
LABEL_ANCHORS = CARDINAL_ANCHORS | {
    "center", "north east", "north west", "south east", "south west",
}
STYLE_DEFAULTS = {
    "fill": "none", "stroke": "dark_blue", "text_color": "black",
    "line_width": 1.2, "rounded_radius": 8, "font_size": 22, "bold": False,
}
MATH_COMMANDS = frozenset({
    "mathbb", "mathrm", "mathbf", "mathit", "operatorname", "frac", "sqrt",
    "in", "times", "cdots", "ldots", "vdots", "ddots", "ell",
    "alpha", "beta", "gamma", "delta", "theta", "sigma", "sum", "prod",
    "left", "right", "le", "ge", "ne", "pm", "cdot", "otimes", "oplus",
    "underbrace", "overbrace", "text", "quad", "qquad",
})

# Shared structural contract: the stdlib validator consumes these object fields;
# model prompts/exporters consume the same definitions. Relational geometry,
# literal text/TeX safety and cross-object references still require validate_scene.
ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_]{0,63}(?![\s\S])"
STYLE_LIMITS = {"line_width": (.1, 10), "rounded_radius": (0, 50), "font_size": (6, 80), "opacity": (0, 1),
                "font_scale": (.5,3), "line_spacing": (.8,2.2)}
ALIGNMENTS = frozenset({"left", "center", "right"})
ROUTINGS = frozenset({"straight", "orthogonal"})
ARROWS = frozenset({"none", "start", "end", "both"})
AXES = frozenset({"x", "y"})
CANVAS_DEFAULT_SCALE = .4


def _ref(name):
    return {"$ref": "#/$defs/" + name}


def _enum(values):
    return {"type": "string", "enum": sorted(values)}


def _number_rule(minimum=None, maximum=None, *, positive=False):
    rule = {"type": "number"}
    if minimum is not None:
        rule["minimum"] = minimum
    if maximum is not None:
        rule["maximum"] = maximum
    if positive:
        rule["exclusiveMinimum"] = 0
    return rule


def _object_rule(properties, required=()):
    return {"type": "object", "properties": properties,
            "required": list(required), "additionalProperties": False}


_ID_RULE = {"type": "string", "pattern": ID_PATTERN}
_TEXT_RULE = {"type": "string", "maxLength": 4000}
_TEXT_PROPERTIES = {"text": _TEXT_RULE, "mode": _enum(TEXT_MODES),
                    "segments": {"type": "array", "items": _ref("segment"),
                                 "minItems": 1, "maxItems": MAX_RICH_SEGMENTS},
                    "font_size": _number_rule(6, 80), "align": _enum(ALIGNMENTS)}
_STYLE_REFERENCE = {"oneOf": [_ID_RULE, _ref("style")]}
_Z_INDEX_RULE = {"type":"integer", "minimum":0, "maximum":10000}

OBJECT_CONTRACTS = {
    "scene": _object_rule({
        "canvas": _ref("canvas"),
        "styles": {"type": "object", "maxProperties": 100,
                   "propertyNames": _ID_RULE, "additionalProperties": _ref("style")},
        **{key: {"type": "array", "items": _ref(kind), "maxItems": 500}
           for key, kind in (("nodes", "node"), ("edges", "edge"), ("labels", "label"))},
        "metadata": {},
    }, ("canvas", "styles", "nodes", "edges", "labels")),
    "canvas": _object_rule({
        "width": _number_rule(1, 20000), "height": _number_rule(1, 20000),
        "coordinate_system": {"const": "top-left"}, "scale": _number_rule(.01, 2),
        "layered": {"type":"boolean"},
        "measure_text": {"type":"boolean"},
    }, ("width", "height", "coordinate_system")),
    "style": _object_rule({
        **{key: _enum(set(COLORS) | ({"none"} if key != "text_color" else set()))
           for key in ("fill", "stroke", "text_color")},
        **{key: {"type":"string", "pattern":r'^#[0-9A-Fa-f]{6}(?![\s\S])'}
           for key in ("fill_color", "stroke_color", "text_color_rgb")},
        **{key: _number_rule(*limits) for key, limits in STYLE_LIMITS.items()},
        "bold": {"type": "boolean"},
        "font_family": _enum({'serif','sans','mono'}),
        "preserve_spaces": {"type":"boolean"},
    }),
    "grid": _object_rule({key: {"type": "integer", "minimum": 1, "maximum": 50}
                          for key in ("rows", "cols")}, ("rows", "cols")),
    "depth": _object_rule({"dx": _number_rule(positive=True),
                           "dy": {"type": "number", "exclusiveMaximum": 0}}, ("dx", "dy")),
    "segment": _object_rule({"type": _enum({"text", "math"}), "content": _TEXT_RULE},
                            ("type", "content")),
    "point": {"type": "array", "minItems": 2, "maxItems": 2,
              "items": _number_rule(0)},
    "node": _object_rule({
        "id": _ID_RULE, "type": _enum(NODE_TYPES),
        "bbox": {"type": "array", "minItems": 4, "maxItems": 4,
                 "prefixItems": [_number_rule(0), _number_rule(0),
                                 _number_rule(positive=True), _number_rule(positive=True)],
                 "items": False},
        "style": _STYLE_REFERENCE, "grid": _ref("grid"), "depth": _ref("depth"),
        **_TEXT_PROPERTIES, "wrap": {"type": "boolean"},
        "background_shape": {"const": "circle"}, "metadata": {}, "z_index": _Z_INDEX_RULE,
    }, ("id", "type", "bbox")),
    "label": _object_rule({"id": _ID_RULE, "x": _number_rule(0), "y": _number_rule(0),
        "anchor": _enum(LABEL_ANCHORS), "style": _STYLE_REFERENCE,
        **_TEXT_PROPERTIES, "metadata": {}, "z_index": _Z_INDEX_RULE,
    }, ("id", "x", "y")),
    "edge": _object_rule({
        "id": _ID_RULE, "from": _ID_RULE, "to": _ID_RULE,
        "from_anchor": _enum(CARDINAL_ANCHORS), "to_anchor": _enum(CARDINAL_ANCHORS),
        "points": {"type": "array", "items": _ref("point"), "minItems": 2, "maxItems": 100},
        "routing": _enum(ROUTINGS), "first_axis": _enum(AXES), "arrow": _enum(ARROWS),
        "dashed": {"type": "boolean"}, "style": _STYLE_REFERENCE, "metadata": {}, "z_index": _Z_INDEX_RULE,
    }, ("id",)),
}


def _has_value(field, value):
    return {"required": [field], "properties": {field: {"const": value}}}


_RICH_CONDITION = {"if": _has_value("mode", "rich"), "then": {"required": ["segments"]},
                   "else": {"not": {"required": ["segments"]}}}
OBJECT_CONTRACTS["node"]["allOf"] = [
    _RICH_CONDITION,
    {"if": {"properties": {"type": {"enum": ["matrix", "tensor3d"]}}, "required": ["type"]},
     "then": {"required": ["grid"]}, "else": {"not": {"required": ["grid"]}}},
    {"if": _has_value("type", "tensor3d"), "then": {"required": ["depth"]},
     "else": {"not": {"required": ["depth"]}}},
    {"if": _has_value("type", "text"), "then": {
        "if": _has_value("mode", "rich"), "else": {"required": ["text"]}},
     "else": {"not": {"anyOf": [{"required": ["wrap"]}, {"required": ["background_shape"]}]}}},
]
OBJECT_CONTRACTS["label"]["allOf"] = [
    _RICH_CONDITION,
    {"if": _has_value("mode", "rich"), "else": {"required": ["text"]}},
]
OBJECT_CONTRACTS["edge"]["oneOf"] = [
    {"required": ["points"], "not": {"anyOf": [{"required": [key]} for key in
                                               ("from", "to", "from_anchor", "to_anchor")]}},
    {"required": ["from", "to"], "not": {"required": ["points"]}},
]

RUNTIME_CONSTRAINTS = [
    "Optional font_family selects existing TeX serif/sans/mono families; font_scale multiplies only glyph sizes, never geometry. line_spacing changes baseline spacing. preserve_spaces emits each literal prose space explicitly. Missing fields keep legacy bytes.",
    "Optional canvas.measure_text emits actual compiled text-node dimensions to the XeLaTeX log; it does not change geometry or content.",
    "Existing fill/stroke/text_color retain their named palette. Optional fill_color/stroke_color/text_color_rgb are strict #RRGGBB overrides. Optional opacity is [0,1], with no implicit new default.",
    "Optional canvas.layered=true paints z_index-sorted fills, then borders, then edges, then text. Callers derive z_index from containment/occlusion evidence; missing layered retains legacy ordering.",
    "All numbers must be finite; booleans are not numbers. All nodes/edges/labels share a unique ID namespace.",
    "Canvas is source pixels, origin top-left, x right, y down; scale is points per pixel, default 0.4.",
    "All points, full bboxes and tensor projected corners must lie inside the canvas.",
    "tensor3d bbox is the FRONT FACE; grid rows/cols are drawn cell counts, not mathematical dimensions.",
    "grid rows/cols must parse as Python int, not bool or float (even 1.0); JSON Schema integer alone does not express this legacy restriction.",
    "A text node circle background requires an exactly square bbox. No generic polygon primitive exists.",
    "diamond bbox is its exact outer box; vertices are the north/east/south/west midpoints. Sharp corners; rounded_radius does not apply.",
    "Diamond text is centered and measured at compile time: text_width/bbox_width + text_height/bbox_height <= 0.9. Overflow is an explicit error; no resizing or text rewriting.",
    "Diamond edge endpoints must use cardinal vertices and their routes may not enter that diamond's interior. Absolute paths with endpoint metadata receive the same check.",
    "Named style references must resolve in styles; inline/omitted styles merge with STYLE_DEFAULTS.",
    "Edge from/to must reference existing nodes. Absolute paths may not repeat consecutive points.",
    "Absolute edge topology can be retained in metadata.source_id/target_id/from_anchor/to_anchor; metadata does not draw.",
    "Text is literal Unicode (max 4000 chars), no controls except newline. Escape JSON, not literal TeX prose.",
    "Math forbids dollars, comments, controls, newlines, macros and environments; braces balanced to depth 32, commands allowlisted.",
    "Rich segments are ordered text/math spans, max 4000 characters combined; optional text is provenance, not rendered twice.",
    "metadata is not rendered and remains backward compatible; model proposals must use object metadata and unverified provenance.",
    "Schema legality never establishes completeness, OCR correctness, visual fidelity or connector topology accuracy.",
]


def scene_contract(image_size=None):
    """Reviewable JSON Schema plus explicit runtime checks, not a new Scene dialect."""
    from copy import deepcopy
    definitions = deepcopy(OBJECT_CONTRACTS)
    if image_size is not None:
        if (not isinstance(image_size, (list, tuple)) or len(image_size) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 20000
                       for v in image_size)):
            raise ValueError("Image dimensions must be integers in [1,20000].")
        for key, value in zip(("width", "height"), image_size):
            definitions["canvas"]["properties"][key]["const"] = value
    return {
        "contract_version": "1.0", "authoritative_validator": "structured_tikz.validator.validate_scene",
        "json_schema": {"$schema": "https://json-schema.org/draft/2020-12/schema",
                        "$ref": "#/$defs/scene", "$defs": definitions},
        "runtime_constraints": list(RUNTIME_CONSTRAINTS),
        "style_defaults": dict(STYLE_DEFAULTS), "colors": dict(COLORS),
        "math_commands": sorted(MATH_COMMANDS),
        "unsupported_native_primitives": ["polygon", "ellipse", "embedded_image"],
    }
