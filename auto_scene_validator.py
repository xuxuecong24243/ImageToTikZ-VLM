"""Bounded Scene diagnostics. Geometry checks are not a visual/semantic oracle."""
from __future__ import annotations

import math

from structured_tikz.schema import NODE_TYPES
from structured_tikz.validator import validate_math, validate_scene
from semantic_edge_router import _inside_length


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def box_valid(box):
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(finite(v) for v in box) and box[2] > 0 and box[3] > 0)


def records(scene, key):
    value = scene.get(key, []) if isinstance(scene, dict) else []
    return [x for x in value[:500] if isinstance(x, dict)] if isinstance(value, list) else []


def metadata(item):
    value = item.get('metadata', {})
    return value if isinstance(value, dict) else {}


def whole_box(node):
    box = node.get('bbox')
    if not box_valid(box):
        return None
    x, y, w, h = box
    depth = node.get('depth')
    if node.get('type') == 'tensor3d':
        if not isinstance(depth, dict) or not all(finite(depth.get(k)) for k in ('dx', 'dy')):
            return None
        dx, dy = depth['dx'], depth['dy']
        return [min(x, x+dx), min(y, y+dy), w+abs(dx), h+abs(dy)]
    return list(box)


def overlap(a, b):
    return max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(
        0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))


def within(box, width, height):
    return box[0] >= 0 and box[1] >= 0 and box[0]+box[2] <= width and box[1]+box[3] <= height


def label_box(label):
    """Return (bbox, evidence); estimate is never a measured text boundary."""
    box = metadata(label).get('bbox')
    if box_valid(box):
        return list(box), 'supplied_bbox_not_font_verified'
    if not all(finite(label.get(k)) for k in ('x', 'y')) or not isinstance(label.get('text'), str):
        return None, 'unavailable'
    size = label.get('font_size', 22)
    if not finite(size) or size <= 0:
        return None, 'unavailable'
    text = label['text']
    w = max(size, sum(size if ord(c) > 255 else size*.55 for c in text))
    h = size*1.25
    x, y = label['x'], label['y']
    anchor = label.get('anchor', 'center')
    anchor = anchor if isinstance(anchor, str) else 'center'
    left = x if 'west' in anchor else x-w if 'east' in anchor else x-w/2
    top = y if 'north' in anchor else y-h if 'south' in anchor else y-h/2
    return [left, top, w, h], 'heuristic_extent_not_font_metrics'


def anchor(node, name):
    if not isinstance(name, str) or not box_valid(node.get('bbox')):
        return None
    x, y, w, h = node['bbox']
    return {'north': [x+w/2, y], 'south': [x+w/2, y+h],
            'east': [x+w, y+h/2], 'west': [x, y+h/2]}.get(name)


def edge_points(edge, nodes):
    points = edge.get('points')
    if points is not None:
        if (isinstance(points, list) and 2 <= len(points) <= 100
                and all(isinstance(p, (list, tuple)) and len(p) == 2 and all(finite(v) for v in p) for p in points)):
            return points
        return None
    if not all(isinstance(edge.get(k), str) for k in ('from', 'to')):
        return None
    source, target = nodes.get(edge.get('from')), nodes.get(edge.get('to'))
    if not source or not target:
        return None
    a = anchor(source, edge.get('from_anchor', 'east'))
    b = anchor(target, edge.get('to_anchor', 'west'))
    if a is None or b is None:
        return None
    if edge.get('routing') == 'orthogonal':
        middle = [b[0], a[1]] if edge.get('first_axis', 'x') == 'x' else [a[0], b[1]]
        return [a, middle, b]
    return [a, b]


def diagnose_scene(scene, image_size=None):
    """Collect deterministic errors, risk warnings, and explicitly unknown semantics."""
    report = {'schema_valid': False, 'errors': [], 'risks': [], 'unknowns': []}

    def add(kind, code, identifier, message, **evidence):
        report[kind].append(dict(code=code, id=identifier, message=message, **evidence))

    try:
        validate_scene(scene)
        report['schema_valid'] = True
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        add('errors', 'schema_invalid', None, str(error))
    if not isinstance(scene, dict):
        return report
    canvas = scene.get('canvas')
    canvas = canvas if isinstance(canvas, dict) else {}
    width, height = canvas.get('width'), canvas.get('height')
    usable_canvas = finite(width) and finite(height) and width > 0 and height > 0
    if image_size is not None and (width, height) != tuple(image_size):
        add('errors', 'canvas_image_mismatch', None, 'Scene canvas must match original image pixels.')
    nodes = {n.get('id'): n for n in records(scene, 'nodes') if isinstance(n.get('id'), str)}
    boxes = {}
    for node in records(scene, 'nodes'):
        identifier, kind = node.get('id'), node.get('type')
        if not isinstance(kind, str) or kind not in NODE_TYPES:
            add('errors', 'unknown_node_type', identifier, 'Unsupported drawing primitive.')
        if not box_valid(node.get('bbox')):
            add('errors', 'invalid_bbox', identifier, 'bbox requires finite x,y and positive width,height.')
        if kind in ('matrix', 'tensor3d'):
            grid = node.get('grid')
            if not isinstance(grid, dict) or any(not isinstance(grid.get(k), int) or isinstance(grid.get(k), bool)
                                                or not 1 <= grid[k] <= 50 for k in ('rows', 'cols')):
                add('errors', 'missing_or_invalid_grid', identifier, 'Drawing grid rows/cols must be integers in 1..50.')
        if kind == 'tensor3d':
            depth = node.get('depth')
            if (not isinstance(depth, dict) or not all(finite(depth.get(k)) for k in ('dx', 'dy'))
                    or depth['dx'] <= 0 or depth['dy'] >= 0):
                add('errors', 'missing_or_invalid_depth', identifier, 'Tensor needs front bbox and dx>0,dy<0; no depth is guessed.')
        box = whole_box(node)
        if box and isinstance(identifier, str):
            boxes[identifier] = box
            if usable_canvas and not within(box, width, height):
                add('errors', 'node_outside_canvas', identifier, 'Object including tensor projection exceeds canvas.', bbox=box)
    for i, (identifier, box) in enumerate(boxes.items()):
        for other, other_box in list(boxes.items())[i+1:]:
            ratio = overlap(box, other_box)/min(box[2]*box[3], other_box[2]*other_box[3])
            if ratio > .5:
                add('risks', 'large_node_overlap', identifier, 'Large bounding-box overlap; containment may be intentional.',
                    other_id=other, overlap_fraction=round(ratio, 4))
    labels = []
    for label in records(scene, 'labels'):
        identifier = label.get('id')
        if not all(finite(label.get(k)) for k in ('x', 'y')):
            add('errors', 'label_position_missing', identifier, 'Label requires finite x,y coordinates.')
        if usable_canvas and all(finite(label.get(k)) for k in ('x', 'y')):
            if not 0 <= label['x'] <= width or not 0 <= label['y'] <= height:
                add('errors', 'label_anchor_outside_canvas', identifier, 'Label anchor is outside canvas.')
        if not label.get('text'):
            add('errors', 'label_content_missing', identifier, 'Empty text is not a recognized label.')
        box, evidence = label_box(label)
        if box:
            labels.append((identifier, box, evidence))
            if usable_canvas and not within(box, width, height):
                add('risks', 'possible_text_clipping', identifier, 'Text extent may exceed canvas.', evidence=evidence)
            for node_id, node_box in boxes.items():
                if metadata(label).get('node_id') == node_id:
                    continue  # An explicitly owned label may intentionally be inside its node.
                if overlap(box, node_box) > .2*box[2]*box[3]:
                    add('risks', 'possible_label_node_overlap', identifier, 'Text box overlaps an unrelated object.',
                        other_id=node_id, evidence=evidence)
    for i, (identifier, box, evidence) in enumerate(labels):
        for other, other_box, other_evidence in labels[i+1:]:
            if overlap(box, other_box) > .5*min(box[2]*box[3], other_box[2]*other_box[3]):
                add('risks', 'possible_text_overlap', identifier, 'Large text-box overlap; font layout remains unverified.',
                    other_id=other, evidence=[evidence, other_evidence])
    for item in records(scene, 'nodes')+records(scene, 'labels'):
        if item.get('mode') == 'math':
            try:
                validate_math(item.get('text'))
            except (ValueError, TypeError) as error:
                add('errors', 'invalid_math', item.get('id'), str(error))
    for edge in records(scene, 'edges'):
        identifier = edge.get('id')
        if 'points' not in edge and any(not isinstance(edge.get(key), str) or edge[key] not in nodes for key in ('from', 'to')):
            add('errors', 'unknown_edge_node', identifier, 'Edge endpoint references a nonexistent node.')
        points = edge_points(edge, nodes)
        if points is None:
            add('errors', 'invalid_edge_path', identifier, 'No usable path or endpoints.')
            continue
        endpoints = {value for value in (edge.get('from'), edge.get('to'), metadata(edge).get('source_id'),
                                        metadata(edge).get('target_id')) if isinstance(value, str)}
        for node_id, box in boxes.items():
            if node_id not in endpoints and any(_inside_length(a, b, box) > 2 for a, b in zip(points, points[1:])):
                add('risks', 'edge_crosses_unrelated_node', identifier, 'Path enters an unrelated object bounding box.', other_id=node_id)
        for label_id, box, evidence in labels:
            if any(_inside_length(a, b, box) > 2 for a, b in zip(points, points[1:])):
                add('risks', 'possible_edge_text_crossing', identifier, 'Path intersects a text extent.', other_id=label_id, evidence=evidence)
    meta = metadata(scene)
    for key in ('pending_nodes', 'pending_labels', 'pending_parameters'):
        if meta.get(key):
            add('errors', 'unresolved_'+key, None, 'Incomplete objects/parameters need an authorized completion or explicit failure.', count=len(meta[key]) if isinstance(meta[key], list) else None)
    if meta.get('pending_edges'):
        add('risks', 'unrouted_semantic_edges', None, 'Some semantic relationships have no accepted drawable path.',
            count=len(meta['pending_edges']) if isinstance(meta['pending_edges'], list) else None)
    add('unknowns', 'visual_semantics_unverified', None, 'Chinese transcription, math meaning, omitted objects, topology and visual fidelity cannot be proven by schema/geometry checks.')
    report['quality_verified'] = False
    return report
