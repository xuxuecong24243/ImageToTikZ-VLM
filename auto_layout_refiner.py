"""Finite local repairs and orthogonal route proposals, never visual recognition."""
from __future__ import annotations

from copy import deepcopy
import math

from auto_scene_validator import (anchor, box_valid, edge_points, finite, label_box,
                                  metadata, overlap, records, whole_box, within)
from semantic_edge_router import _inside_length
from structured_tikz.validator import validate_scene


def _protected(item):
    return metadata(item).get('review_status') in ('verified', 'rejected')


def _compact(points):
    result = []
    for point in points:
        if not result or point != result[-1]:
            result.append(list(point))
    return result


def _safe(points, obstacles, width, height):
    return (len(points) >= 2 and all(0 <= p[0] <= width and 0 <= p[1] <= height for p in points)
            and not any(_inside_length(a, b, box) > .1 for a, b in zip(points, points[1:]) for box in obstacles))


def _route(edge, scene, nodes):
    if (any(not isinstance(edge.get(k), str) or edge[k] not in nodes for k in ('from', 'to'))
            or edge['from'] == edge['to']):
        return None
    if edge.get('arrow') not in ('none', 'start', 'end', 'both'):
        return None
    start = anchor(nodes[edge['from']], edge.get('from_anchor'))
    end = anchor(nodes[edge['to']], edge.get('to_anchor'))
    if start is None or end is None:
        return None
    normals = {'north': [0, -1], 'south': [0, 1], 'west': [-1, 0], 'east': [1, 0]}
    a = [start[i]+normals[edge['from_anchor']][i]*8 for i in (0, 1)]
    b = [end[i]+normals[edge['to_anchor']][i]*8 for i in (0, 1)]
    obstacles = []
    for node in nodes.values():
        box = whole_box(node)
        if box:
            inset = min(.5, box[2]/4, box[3]/4)
            obstacles.append([box[0]+inset, box[1]+inset, box[2]-2*inset, box[3]-2*inset])
    for label in records(scene, 'labels'):
        box, _ = label_box(label)
        if box:
            obstacles.append(box)
    middle = [(a[0]+b[0])/2, (a[1]+b[1])/2]
    nearest = sorted(obstacles, key=lambda box: abs(box[0]+box[2]/2-middle[0])+abs(box[1]+box[3]/2-middle[1]))[:16]
    paths = [[a, [b[0], a[1]], b], [a, [a[0], b[1]], b]]
    for box in nearest:
        for x in (box[0]-8, box[0]+box[2]+8):
            paths.append([a, [x, a[1]], [x, b[1]], b])
        for y in (box[1]-8, box[1]+box[3]+8):
            paths.append([a, [a[0], y], [b[0], y], b])
    width, height = scene['canvas']['width'], scene['canvas']['height']
    candidates = [_compact([start]+path+[end]) for path in paths]
    safe = [points for points in candidates if _safe(points, obstacles, width, height)]
    if not safe:
        return None
    points = min(safe, key=lambda p: (sum(math.dist(x, y) for x, y in zip(p, p[1:])), len(p), p))
    result = {key: deepcopy(edge[key]) for key in ('id', 'arrow', 'style', 'dashed') if key in edge}
    result.update(points=points, metadata=deepcopy(metadata(edge)))
    result['metadata'].update(source='deterministic_route_not_measured', review_status='unverified',
                              source_id=edge['from'], target_id=edge['to'],
                              from_anchor=edge['from_anchor'], to_anchor=edge['to_anchor'],
                              topology_source=metadata(edge).get('source', 'model_proposal'),
                              visual_path_verified=False)
    try:
        validate_scene(dict(canvas=scene['canvas'], styles=scene.get('styles', {}),
                            nodes=list(nodes.values()), labels=[], edges=[result]))
    except (ValueError, TypeError, KeyError):
        return None
    return result


def refine_scene(scene):
    """Return a copy and audit changes; caller limits iterations and validates final Scene."""
    result, changes = deepcopy(scene), []
    if not isinstance(result, dict) or _protected(result):
        return result, changes
    canvas = result.get('canvas', {})
    if not isinstance(canvas, dict) or not all(finite(canvas.get(k)) and canvas[k] > 0 for k in ('width', 'height')):
        return result, changes
    width, height = canvas['width'], canvas['height']

    def record(item, before, reason):
        item.setdefault('metadata', {}).update(review_status='unverified')
        changes.append({'id': item.get('id'), 'before': before, 'after': deepcopy(item),
                        'reason': reason, 'source': 'deterministic_local_rule_not_measurement'})

    # Only tiny out-of-canvas errors for non-tensor shapes; no rescaling or global moves.
    for node in records(result, 'nodes'):
        if _protected(node) or node.get('type') == 'tensor3d' or not box_valid(node.get('bbox')):
            continue
        x, y, w, h = node['bbox']
        if w > width or h > height:
            continue
        nx, ny = min(max(x, 0), width-w), min(max(y, 0), height-h)
        if (x, y) != (nx, ny) and max(abs(nx-x), abs(ny-y)) <= 5:
            before = deepcopy(node)
            node['bbox'] = [nx, ny, w, h]
            record(node, before, 'clamp_minor_canvas_overflow_at_most_5px')
    nodes = {n['id']: n for n in records(result, 'nodes') if isinstance(n.get('id'), str)}
    for label in records(result, 'labels'):
        if _protected(label) or not box_valid(metadata(label).get('bbox')) or not all(finite(label.get(k)) for k in ('x', 'y')):
            continue
        box = list(metadata(label)['bbox'])
        owner = metadata(label).get('node_id')
        blockers = [whole_box(node) for node_id, node in nodes.items() if node_id != owner and whole_box(node)]
        blockers += [metadata(other)['bbox'] for other in records(result, 'labels')
                     if other.get('id') != label.get('id') and box_valid(metadata(other).get('bbox'))]
        collisions = [b for b in blockers if overlap(box, b) > 0]
        # A fully contained label with no ownership may be intentional: do not move it.
        if not collisions or any(overlap(box, b) >= box[2]*box[3] for b in collisions):
            continue
        candidates = []
        for b in collisions:
            candidates += [(b[0]-box[0]-box[2]-3, 0), (b[0]+b[2]-box[0]+3, 0),
                           (0, b[1]-box[1]-box[3]-3), (0, b[1]+b[3]-box[1]+3)]
        for dx, dy in sorted(candidates, key=lambda shift: abs(shift[0])+abs(shift[1])):
            shifted = [box[0]+dx, box[1]+dy, box[2], box[3]]
            if (max(abs(dx), abs(dy)) <= 20 and within(shifted, width, height)
                    and not any(overlap(shifted, b) > 0 for b in blockers)):
                before = deepcopy(label)
                label.update(x=label['x']+dx, y=label['y']+dy)
                label['metadata']['bbox'] = shifted
                record(label, before, 'small_label_shift_from_supplied_bbox_conflict')
                break
    # Existing explicit paths are preserved: a visible arrow may deliberately stop short.
    for index, edge in enumerate(records(result, 'edges')):
        if _protected(edge) or 'points' in edge:
            continue
        points = edge_points(edge, nodes)
        obstacle_boxes = [whole_box(node) for node_id, node in nodes.items()
                          if node_id not in (edge.get('from'), edge.get('to')) and whole_box(node)]
        obstacle_boxes += [box for box, _ in (label_box(label) for label in records(result, 'labels')) if box]
        conflict = points and any(_inside_length(a, b, box) > 1 for a, b in zip(points, points[1:]) for box in obstacle_boxes)
        if conflict:
            candidate = dict(edge, from_anchor=edge.get('from_anchor', 'east'), to_anchor=edge.get('to_anchor', 'west'), arrow=edge.get('arrow', 'none'))
            routed = _route(candidate, result, nodes)
            if routed:
                before = deepcopy(edge)
                edge.clear(); edge.update(routed)
                record(edge, before, 'reroute_reference_edge_around_known_boxes')
    pending = metadata(result).get('pending_edges', [])
    if isinstance(pending, list):
        remaining = []
        identifiers = {item.get('id') for key in ('nodes', 'labels', 'edges') for item in records(result, key) if isinstance(item.get('id'), str)}
        for edge in pending[:500]:
            if (not isinstance(edge, dict) or _protected(edge) or not isinstance(edge.get('id'), str)
                    or edge['id'] in identifiers):
                remaining.append(edge); continue
            routed = _route(edge, result, nodes)
            if routed:
                result.setdefault('edges', []).append(routed)
                identifiers.add(routed['id'])
                record(routed, deepcopy(edge), 'route_pending_semantic_relationship')
            else:
                remaining.append(edge)
        remaining += pending[500:]
        if pending:
            result['metadata']['pending_edges'] = remaining
    return result, changes
