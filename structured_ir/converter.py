"""Deterministic IR -> existing Scene. Never infer missing content or topology.

Regions are segmentation/ownership metadata, not implicit painted containers.
Only explicit drawable elements become nodes. The returned report distinguishes
complete representability from partial conversion and from visual fidelity.
"""
from copy import deepcopy
import hashlib
import math

from structured_tikz.validator import validate_scene
from structured_tikz.geometry import node_anchor, edge_route_points, diamond_path_error
from .validator import validate_ir


TEXT_TYPES = {'text', 'multiline_text', 'code', 'math'}


def convert_ir(ir, style_map=None, config=None):
    """Return (Scene or None, diagnostics). Input is never mutated.

style_map maps neutral IR style_role names to existing Scene inline styles.
config currently accepts only scale (points per input pixel). No fitting,
reflow, text correction, shape substitution or synthetic endpoints occurs.
"""
    validate_ir(ir)
    if style_map is not None and not isinstance(style_map, dict):
        raise ValueError('style_map must map roles to Scene style objects.')
    if config is not None and not isinstance(config, dict):
        raise ValueError('config must be an object.')
    style_map = deepcopy({} if style_map is None else style_map)
    config = dict({} if config is None else config)
    if set(config) - {'scale'}:
        raise ValueError('Unknown conversion configuration: ' + ', '.join(sorted(set(config)-{'scale'})))
    scene = {'canvas': {k: ir['canvas'][k] for k in ('width', 'height', 'coordinate_system')},
             'styles': {}, 'nodes': [], 'edges': [], 'labels': [],
             'metadata': {'origin': 'deterministic_ir_conversion', 'ir_version': ir['version'],
                          'canvas_id': ir['canvas']['id'], 'regions': deepcopy(ir['regions']),
                          'ir_connections': deepcopy(ir['connections']),
                          'ir_provenance': deepcopy(ir.get('provenance', {})),
                          'quality': 'unverified'}}
    if 'scale' in config:
        scene['canvas']['scale'] = config['scale']
    if 'layered' in ir:
        scene['canvas']['layered'] = ir['layered']
    if 'measure_text' in ir:
        scene['canvas']['measure_text'] = ir['measure_text']
    report = {'status': 'converting', 'scene_valid': False, 'complete': False,
              'converted_elements': [], 'converted_connections': [], 'diagnostics': [],
              'unsupported_elements': [], 'unrendered_text': [], 'api_calls': 0,
              'hierarchy': 'preserved_in_scene_metadata; regions_do_not_imply_painted_shapes',
              'visual_fidelity': 'not_evaluated'}

    def issue(code, item, message, *, unsupported=False, text=False, blocking=True):
        diagnostic = {'code': code, 'element_id': item.get('id'), 'message': message,
                      'blocking_complete_conversion': blocking}
        report['diagnostics'].append(diagnostic)
        if unsupported:
            report['unsupported_elements'].append({'id': item['id'], 'type': item.get('type'), 'code': code})
        if text and item.get('raw_text'):
            report['unrendered_text'].append({'id': item['id'], 'raw_text': item['raw_text'], 'reason': code})

    try:
        validate_scene(scene)
    except ValueError as exc:
        issue('scene_canvas_incompatible', ir['canvas'], str(exc))
        report['status'] = 'failed'
        return None, report

    def style(item, output):
        role = item.get('style_role')
        if role in style_map:
            output['style'] = deepcopy(style_map[role])
        elif role:
            issue('style_role_unmapped', item, 'Using existing default style; appearance requires a style mapping.', blocking=False)
        if 'visual_style' in item:
            output['style'] = {**output.get('style', {}), **deepcopy(item['visual_style'])}
        if 'z_index' in item:
            output['z_index'] = item['z_index']

    def add_checked(collection, value, original):
        scene[collection].append(value)
        try:
            validate_scene(scene)
        except ValueError as exc:
            scene[collection].pop()
            issue('scene_representation_rejected', original, str(exc), unsupported=True, text=True)
            return False
        return True

    for item in ir['elements']:
        kind = item['type']
        meta = {'ir_id': item['id'], 'ir_type': kind, 'region_id': item['region_id'],
                'raw_text': item.get('raw_text'), 'reading_order': item.get('reading_order')}
        if kind in {'connector', 'arrow'}:
            edge = {'id': item['id'], 'points': deepcopy(item['points']),
                    'arrow': 'end' if kind == 'arrow' else 'none', 'metadata': meta}
            style(item, edge)
            if add_checked('edges', edge, item):
                report['converted_elements'].append(item['id'])
            if item.get('raw_text'):
                issue('unplaced_graphic_text', item, 'Geometric line text has no independent placement; retained in IR only.', text=True)
            continue
        node = {'id': item['id'], 'bbox': deepcopy(item['bbox']), 'metadata': meta}
        if kind in TEXT_TYPES:
            node.update(type='text', text=item['raw_text'], mode='math' if kind == 'math' else 'text')
        elif kind == 'circle':
            if item['bbox'][2] != item['bbox'][3]:
                issue('unsupported_shape', item, 'Scene circle requires a square bbox; no geometry distortion applied.', unsupported=True, text=True)
                continue
            # This empty string paints a genuine source circle, not a placeholder
            # endpoint. It is the existing Scene circle primitive's representation.
            node.update(type='text', background_shape='circle', text=item.get('raw_text', ''), wrap=False)
        else:
            node['type'] = 'rectangle' if kind == 'title_band' else kind
        if 'raw_text' in item:
            node['text'] = item['raw_text']
            node['mode'] = 'math' if item.get('text_type') == 'math' else 'text'
            if item.get('text_type') == 'code' and 'font_family' not in item.get('visual_style',{}):
                issue('code_typography_default', item, 'Literal code preserved; Scene has no per-node font-family selector.', blocking=False)
        if 'render_segments' in item:
            node.update(mode='rich',segments=deepcopy(item['render_segments']))
        style(item, node)
        if add_checked('nodes', node, item):
            report['converted_elements'].append(item['id'])

    nodes = {node['id']: node for node in scene['nodes']}
    used = {obj['id'] for collection in ('nodes', 'edges', 'labels') for obj in scene[collection]}
    used.update(c['id'] for c in ir['connections'])

    def anchor(node_id, name):
        return node_anchor(nodes[node_id], name)

    for item in ir['connections']:
        if item['source_id'] not in nodes or item['target_id'] not in nodes:
            issue('blocked_connection', item, 'Endpoint was not mapped to a Scene node; semantic connection remains in IR.')
            continue
        edge = {'id': item['id'], 'arrow': {'forward': 'end', 'reverse': 'start', 'both': 'both', 'none': 'none'}[item['direction']],
                'metadata': deepcopy(item)}
        path_mismatch = False
        if item.get('points'):
            edge['points'] = deepcopy(item['points'])
            endpoints = (anchor(item['source_id'], item['source_anchor']), anchor(item['target_id'], item['target_anchor']))
            path_mismatch = any(not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6)
                   for point, expected_point in zip((edge['points'][0], edge['points'][-1]), endpoints)
                   for actual, expected in zip(point, expected_point))
            if path_mismatch:
                issue('connection_path_anchor_mismatch', item,
                      'Explicit path endpoints disagree with semantic anchors; coordinates retained without rerouting.')
        elif 'center' in (item['source_anchor'], item['target_anchor']):
            edge['points'] = [anchor(item['source_id'], item['source_anchor']), anchor(item['target_id'], item['target_anchor'])]
        else:
            edge.update({'from': item['source_id'], 'to': item['target_id'],
                         'from_anchor': item['source_anchor'], 'to_anchor': item['target_anchor']})
        diamond_endpoints = [(role, nodes[item[key]]) for role, key in
                             (('source', 'source_id'), ('target', 'target_id'))
                             if nodes[item[key]]['type'] == 'diamond']
        if diamond_endpoints:
            if path_mismatch:
                issue('diamond_connection_anchor_mismatch', item,
                      'Explicit path disagrees with declared anchors; diamond connection was not drawn or rerouted.')
                continue
            problems = [problem for role, node in diamond_endpoints
                        for problem in [diamond_path_error(edge_route_points(edge, nodes), node, endpoint=role)] if problem]
            if problems:
                issue('diamond_connection_geometry', item, '; '.join(problems))
                continue
        style(item, edge)
        if not add_checked('edges', edge, item):
            continue
        report['converted_connections'].append(item['id'])
        if 'branch_label' in item:
            if 'label_bbox' not in item:
                issue('unplaced_branch_label', item, 'Branch label retained semantically; no guessed label position.')
                continue
            endpoint_regions = {nodes[item[key]]['metadata']['region_id'] for key in ('source_id', 'target_id')}
            matches = [node for node in nodes.values()
                       if node['metadata']['ir_type'] in {'text', 'multiline_text'}
                       and node.get('mode', 'text') == 'text'
                       and node.get('text') == item['branch_label'] and node['bbox'] == item['label_bbox']
                       and node['metadata']['region_id'] in endpoint_regions]
            if len(matches) == 1:
                edge['metadata']['branch_label_node_id'] = matches[0]['id']
                issue('branch_label_reused', item,
                      'Existing text node matches literal label, exact bbox and endpoint region: ' + matches[0]['id'], blocking=False)
                continue
            if len(matches) > 1:
                issue('ambiguous_branch_label', item, 'Multiple text nodes match label placement; no arbitrary association or duplicate label added.')
                continue
            ident = 'label_' + hashlib.sha256(item['id'].encode()).hexdigest()[:16]
            while ident in used:
                ident += 'x'
            used.add(ident)
            x, y, w, h = item['label_bbox']
            label = {'id': ident, 'x': x+w/2, 'y': y+h/2, 'text': item['branch_label'],
                     'anchor': 'center', 'metadata': {'ir_connection': item['id'], 'bbox': deepcopy(item['label_bbox'])}}
            add_checked('labels', label, item)
    validate_scene(scene)
    if not scene['nodes'] and not scene['edges'] and not scene['labels']:
        issue('no_drawable_content', ir['canvas'],
              'No source elements could be drawn; an empty canvas is not a completed conversion.')
    report.update(scene_valid=True,
                  complete=not any(d['blocking_complete_conversion'] for d in report['diagnostics']))
    report['status'] = 'complete' if report['complete'] else 'partial'
    scene['metadata']['conversion_status'] = report['status']
    return scene, report
