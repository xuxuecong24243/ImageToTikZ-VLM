"""Read-only comparisons; call only after the independent enriched draft is saved.

Reference measurements are evaluation targets, never enrichment inputs. Matching
is heuristic and does not prove semantic recognition or equivalent connectivity.
"""
import math
import unicodedata


MIN_BOX_IOU = .25
AMBIGUOUS_IOU_MARGIN = .05
PATH_TOLERANCE_PX = 16.0


def _whole_box(node):
    x, y, w, h = node['bbox']
    depth = node.get('depth', {}) if node.get('type') == 'tensor3d' else {}
    dx, dy = depth.get('dx', 0), depth.get('dy', 0)
    return [min(x, x+dx), min(y, y+dy), w+abs(dx), h+abs(dy)]


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax+aw, bx+bw)-max(ax, bx))*max(0, min(ay+ah, by+bh)-max(ay, by))
    return intersection/(aw*ah+bw*bh-intersection)


def _distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def _texts(scene):
    result = []
    for item in scene.get('labels', [])+scene.get('nodes', []):
        if not item.get('text'):
            continue
        normalized = ' '.join(unicodedata.normalize('NFKC', item['text']).split())
        center = ([item['x'], item['y']] if 'x' in item else
                  [item['bbox'][0]+item['bbox'][2]/2, item['bbox'][1]+item['bbox'][3]/2])
        result.append({'id': item['id'], 'text': item['text'], 'normalized_text': normalized,
                       'mode': item.get('mode', 'text'), 'center_px': center})
    return result


def _point_to_segment(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    length_squared = dx*dx+dy*dy
    t = max(0, min(1, ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/length_squared)) if length_squared else 0
    return _distance(p, [a[0]+t*dx, a[1]+t*dy])


def _path_distance(a, b):
    def one_way(source, target):
        samples = [[p[0]+(q[0]-p[0])*t/4, p[1]+(q[1]-p[1])*t/4]
                   for p, q in zip(source, source[1:]) for t in range(5)]
        return max(min(_point_to_segment(s, p, q) for p, q in zip(target, target[1:])) for s in samples)
    return max(one_way(a, b), one_way(b, a))


def evaluate_enrichment(before, after, semantic_result, reports, reference):
    """Return JSON-serializable evidence without changing any argument."""
    if any(before['canvas'][k] != reference['canvas'][k] or after['canvas'][k] != reference['canvas'][k]
           for k in ('width', 'height', 'coordinate_system')):
        return {'status': 'needs_review', 'reason': 'incompatible_canvas_no_comparison_performed'}
    after_nodes = {n['id']: n for n in after['nodes']}
    ref_nodes = [n for n in reference['nodes'] if n['type'] != 'text']
    used, node_results = set(), []
    for old in before['nodes']:
        if old['type'] == 'text':
            continue
        measured = old.get('metadata', {}).get('geometry_enrichment', {}).get('original_bbox', old['bbox'])
        candidates = sorted((( _iou(measured, _whole_box(r)), r) for r in ref_nodes),
                            key=lambda pair: pair[0], reverse=True)
        entry = {'node_id': old['id'], 'status': 'not_matched', 'original_bbox': list(measured)}
        node_results.append(entry)
        if not candidates or candidates[0][0] < MIN_BOX_IOU:
            continue
        score, ref = candidates[0]
        entry.update(reference_id=ref['id'], whole_box_iou=score)
        if ref['id'] in used or (len(candidates) > 1 and score-candidates[1][0] < AMBIGUOUS_IOU_MARGIN):
            entry.update(status='needs_review', reason='ambiguous_or_already_assigned_reference')
            continue
        used.add(ref['id'])
        node = after_nodes.get(old['id'])
        if node is None:
            entry.update(status='needs_review', reason='original_node_id_missing_after_enrichment')
            continue
        entry.update(status='matched_needs_review', type_before=old['type'], type_after=node['type'],
                     type_reference=ref['type'], type_equal=node['type'] == ref['type'],
                     grid_after=node.get('grid'), grid_reference=ref.get('grid'),
                     grid_equal=node.get('grid') == ref.get('grid') if ref.get('grid') else None)
        if node['type'] == ref['type']:
            entry['drawing_bbox_delta_px'] = [a-b for a, b in zip(node['bbox'], ref['bbox'])]
            entry['drawing_bbox_iou'] = _iou(node['bbox'], ref['bbox'])
            if ref['type'] == 'tensor3d':
                entry['depth_delta_px'] = {k: node['depth'][k]-ref['depth'][k] for k in ('dx', 'dy')}
        else:
            entry['drawing_bbox_comparison'] = 'needs_review_incompatible_front_face_semantics'
    old_texts, new_texts, reference_texts = _texts(before), _texts(after), _texts(reference)
    text_results = []
    for target in reference_texts:
        entry = {'reference_id': target['id'], 'text': target['text']}
        for name, items in (('before', old_texts), ('after', new_texts)):
            matches = [i for i in items if i['normalized_text'] == target['normalized_text'] and i['mode'] == target['mode']]
            entry[name] = {'status': 'exact_content_match' if len(matches) == 1 else
                          'not_matched' if not matches else 'needs_review_duplicate_content',
                          'candidate_ids': [i['id'] for i in matches]}
            if len(matches) == 1:
                entry[name]['center_distance_px'] = _distance(matches[0]['center_px'], target['center_px'])
        text_results.append(entry)
    old_edges = {e['id'] for e in before['edges']}
    ref_paths = [e for e in reference['edges'] if len(e.get('points', [])) >= 2]
    edge_results = []
    for edge in after['edges']:
        if edge['id'] in old_edges:
            continue
        entry = {'edge_id': edge['id'], 'status': 'not_matched'}
        edge_results.append(entry)
        if len(edge.get('points', [])) < 2:
            entry.update(status='needs_review', reason='no_explicit_polyline')
            continue
        distances = sorted((_path_distance(edge['points'], r['points']), r['id']) for r in ref_paths)
        if distances:
            distance, identifier = distances[0]
            entry.update(reference_id=identifier, sampled_symmetric_distance_px=distance,
                         status='path_near_reference_needs_review' if distance <= PATH_TOLERANCE_PX else 'not_matched')
    return {'status': 'evaluation_only_needs_review', 'nodes': node_results, 'texts': text_results,
            'new_edges': edge_results, 'unmatched_reference_node_ids': [n['id'] for n in ref_nodes if n['id'] not in used],
            'source_semantic_object_count': len(semantic_result.get('objects', [])),
            'report_names': sorted(reports), 'thresholds': {'minimum_bbox_iou': MIN_BOX_IOU,
            'ambiguous_iou_margin': AMBIGUOUS_IOU_MARGIN, 'path_tolerance_px': PATH_TOLERANCE_PX},
            'limitations': ['Reference is used only for evaluation, never to fill the draft.',
            'Text comparison is exact after Unicode/whitespace normalization, not OCR or mathematical equivalence.',
            'Text centers may be label anchors; reference text boxes are unavailable. Font sizes/overlap are not scored.',
            'Path distance samples segments and ignores direction; nearby paths do not prove correct endpoint topology.',
            'Explicit reference polylines without node IDs cannot establish recovered semantic connectivity.',
            'Grid counts describe visible cells, not mathematical dimensions; all matches require review.']}
