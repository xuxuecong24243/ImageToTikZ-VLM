"""Conservative semantic suggestions into an unverified copy of a Scene.

No measured geometry, approved object, existing edge or existing label is edited.
No mathematical dimensions, tensor depth, grid count or connection path is guessed.
"""
from __future__ import annotations

import copy

from semantic_schema import SemanticValidationError, scene_text_regions, validate_semantic_result
from structured_tikz.validator import validate_scene


def _status(item):
    metadata = item.get('metadata', {})
    if not isinstance(metadata, dict):
        raise SemanticValidationError('Review metadata must be an object.')
    status = metadata.get('review_status', 'unverified')
    if status not in {'unverified', 'verified', 'rejected'}:
        raise SemanticValidationError('Unknown review status in source Scene.')
    return status


def _can_change_type(scene, node, proposed):
    candidate = copy.deepcopy(node)
    candidate['type'] = proposed
    # Do not invent/remove type-specific fields to force the proposal to pass.
    miniature = {'canvas': scene['canvas'], 'styles': scene['styles'],
                 'nodes': [candidate], 'edges': [], 'labels': []}
    try:
        validate_scene(miniature)
    except ValueError:
        return False
    return True


def merge_semantics(scene_draft: dict, semantic_result: dict) -> dict:
    validate_semantic_result(semantic_result, scene_draft)
    if _status(scene_draft) != 'unverified':
        raise SemanticValidationError('Merge requires an unverified Scene; approved/rejected scenes are not modified.')
    for kind in ('nodes', 'edges', 'labels'):
        for item in scene_draft[kind]:
            _status(item)
    result = copy.deepcopy(scene_draft)
    metadata = result.setdefault('metadata', {})
    review = {
        'source': 'vlm_semantic_proposal', 'review_status': 'unverified',
        'confidence_meaning': 'Model self-report, not calibrated accuracy.',
        'result': copy.deepcopy(semantic_result),
        'relations': [dict(copy.deepcopy(item), review_status='unverified', source='vlm_semantic_proposal')
                      for item in semantic_result['relations']],
        'protected_objects': [], 'protected_texts': [], 'deferred_types': [],
        'pending_texts': [], 'created_label_ids': [], 'applied_type_changes': [],
        'unmatched_objects': copy.deepcopy(semantic_result['unmatched_objects']),
        'unmatched_texts': copy.deepcopy(semantic_result['unmatched_texts']),
        'warnings': copy.deepcopy(semantic_result['warnings']),
    }
    # Prior proposals remain auditable instead of silently being overwritten.
    if 'semantic_review' in metadata:
        review['previous_review'] = copy.deepcopy(metadata['semantic_review'])
    metadata.update(semantic_review=review, review_status='unverified', human_review_required=True)
    nodes = {node['id']: node for node in result['nodes']}
    for proposal in semantic_result['objects']:
        node = nodes[proposal['candidate_id']]
        if _status(node) in {'verified', 'rejected'}:
            review['protected_objects'].append(copy.deepcopy(proposal))
            continue
        node_meta = node.setdefault('metadata', {})
        node_meta['semantic_candidate'] = copy.deepcopy(proposal)
        changed_fields = ['metadata.semantic_candidate']
        proposed = proposal['semantic_type']
        if proposed != 'unknown' and proposed != node['type']:
            if _can_change_type(result, node, proposed):
                old_type = node['type']
                node['type'] = proposed
                changed_fields.append('type')
                review['applied_type_changes'].append({'id': node['id'], 'before': old_type, 'after': proposed})
            else:
                review['deferred_types'].append({
                    'candidate_id': node['id'], 'proposed_type': proposed,
                    'reason': 'Existing fields do not satisfy the proposed type; geometry and drawing parameters were not invented.',
                })
        node_meta['semantic_update'] = {'source': 'vlm_semantic_proposal', 'fields': changed_fields,
                                        'confidence_meaning': 'self_reported_not_probability'}
        node_meta['review_status'] = 'unverified'
    regions = scene_text_regions(scene_draft)
    identifiers = {item['id'] for kind in ('nodes', 'edges', 'labels') for item in result[kind]}
    protected_labels = [item for item in result['labels'] if _status(item) in {'verified', 'rejected'}]
    for text in semantic_result['texts']:
        if text['candidate_id'] is not None and _status(nodes[text['candidate_id']]) in {'verified', 'rejected'}:
            review['protected_texts'].append({'text': copy.deepcopy(text), 'reason': 'associated_node_is_protected'})
            continue
        if text['text_region_id'] is not None:
            bbox = copy.deepcopy(regions[text['text_region_id']]['bbox'])
            position_source = 'existing_text_region_measurement'
        else:
            bbox = copy.deepcopy(text['bbox'])
            position_source = 'vlm_estimate'
        if bbox is None:
            review['pending_texts'].append({'text': copy.deepcopy(text), 'reason': 'position_unknown'})
            continue
        x, y, width, height = bbox
        identifier = 'sem_' + text['id']
        conflict = identifier in identifiers or any(
            (label['text'] == text['text'] and label.get('mode', 'text') == text['mode'])
            or (x <= label['x'] <= x+width and y <= label['y'] <= y+height)
            for label in protected_labels)
        if conflict:
            review['protected_texts'].append({'text': copy.deepcopy(text), 'reason': 'existing_id_or_protected_label_overlap'})
            continue
        label = {
            'id': identifier, 'text': text['text'], 'mode': text['mode'],
            'x': x+width/2, 'y': y+height/2, 'anchor': 'center', 'align': 'center', 'font_size': 22,
            'metadata': {
                'source': 'vlm_semantic_proposal', 'review_status': 'unverified',
                'source_text_id': text['id'], 'candidate_id': text['candidate_id'],
                'text_region_id': text['text_region_id'], 'raw_text': text['raw_text'],
                'bbox': bbox, 'model_proposed_bbox': copy.deepcopy(text['bbox']), 'position_source': position_source,
                'confidence': text['confidence'], 'confidence_meaning': 'self_reported_not_probability',
                'font_size_source': 'fixed_review_default_not_measured',
            },
        }
        result['labels'].append(label)
        identifiers.add(identifier)
        review['created_label_ids'].append(identifier)
    # Relations and unmatched detections stay in metadata, never final edges/nodes.
    validate_scene(result)
    return result


def enrich_semantic_scene(scene_draft: dict, semantic_result: dict, context: dict):
    """Optional offline pass; never consult a reference Scene or create verified data.

    Existing merge_semantics remains unchanged. Incomplete proposals live in
    review metadata/reports; only schema-valid objects enter drawable arrays.
    """
    from semantic_text_aligner import align_texts
    from semantic_shape_enricher import enrich_shapes
    from semantic_edge_router import route_edges

    validate_semantic_result(semantic_result, scene_draft)
    if _status(scene_draft) != 'unverified':
        raise SemanticValidationError('Enrichment requires an unverified Scene.')
    for kind in ('nodes', 'edges', 'labels'):
        for item in scene_draft[kind]:
            _status(item)
    _validate_enrichment_context(scene_draft, context)
    result = copy.deepcopy(scene_draft)
    labels, text_report = align_texts(result, semantic_result, context)
    by_label = {item['id']: index for index, item in enumerate(result['labels'])}
    used_ids = {item['id'] for kind in ('nodes', 'edges', 'labels') for item in result[kind]}
    protected = [item for item in result['labels'] if _status(item) != 'unverified']
    merge_conflicts = []
    for label in labels:
        index = by_label.get(label['id'])
        old = result['labels'][index] if index is not None else None
        if old and _status(old) != 'unverified':
            continue
        bbox = label.get('metadata', {}).get('bbox')
        overlaps_protected = bbox and any(
            bbox[0] <= other['x'] <= bbox[0]+bbox[2] and
            bbox[1] <= other['y'] <= bbox[1]+bbox[3] for other in protected)
        unrelated_id = (old and old.get('metadata', {}).get('source_text_id') !=
                        label.get('metadata', {}).get('source_text_id'))
        if overlaps_protected or unrelated_id or (old is None and label['id'] in used_ids):
            merge_conflicts.append({'id': label['id'], 'reason': 'existing_object_or_protected_label'})
            continue
        if old:
            label['metadata'].setdefault('alignment_previous_position', {
                'x': old['x'], 'y': old['y'], 'bbox': copy.deepcopy(old.get('metadata', {}).get('bbox')),
                'position_source': old.get('metadata', {}).get('position_source')})
            result['labels'][index] = label
        else:
            by_label[label['id']] = len(result['labels'])
            used_ids.add(label['id'])
            result['labels'].append(label)
    nodes, shape_report = enrich_shapes(result, semantic_result)
    by_node = {node['id']: i for i, node in enumerate(result['nodes'])}
    for node in nodes:
        index = by_node[node['id']]
        if _status(result['nodes'][index]) == 'unverified':
            result['nodes'][index] = node
    text_boxes = [item['bbox'] for item in context.get('text_regions', [])]
    text_boxes += [label['metadata']['bbox'] for label in result['labels']
                   if label.get('metadata', {}).get('bbox')]
    edges, edge_report = route_edges(result, semantic_result, text_boxes=text_boxes)
    result['edges'].extend(edges)
    reports = {'text': text_report, 'shapes': shape_report, 'edges': edge_report}
    summary = {
        'before': {key: len(scene_draft[key]) for key in ('nodes', 'labels', 'edges')},
        'after': {key: len(result[key]) for key in ('nodes', 'labels', 'edges')},
        'new_labels_applied': len(result['labels'])-len(scene_draft['labels']),
        'updated_labels_applied': sum(
            (label['x'], label['y']) != (scene_draft['labels'][i]['x'], scene_draft['labels'][i]['y'])
            for i, label in enumerate(result['labels'][:len(scene_draft['labels'])])),
        'shape_nodes_applied': len(nodes), 'new_edges_applied': len(edges),
        'merge_conflicts': merge_conflicts,
    }
    reports['summary'] = summary
    metadata = result.setdefault('metadata', {})
    review = {'source': 'offline_rules_on_existing_semantics_and_geometry',
              'review_status': 'unverified', 'human_review_required': True,
              'reports': copy.deepcopy(reports),
              'limitations': ['No OCR, new CV analysis, API call or reference data used.',
                              'Grid counts describe drawn cells, not tensor dimensions.',
                              'Arrow direction is semantic advice, not measured arrowheads.']}
    if 'offline_enrichment' in metadata:
        review['previous_review'] = copy.deepcopy(metadata['offline_enrichment'])
    metadata.update(offline_enrichment=review, human_review_required=True, review_status='unverified')
    validate_scene(result)
    # Strong postcondition: approved/rejected objects and original semantics survive exactly.
    for kind in ('nodes', 'labels', 'edges'):
        final = {item['id']: item for item in result[kind]}
        for original in scene_draft[kind]:
            if _status(original) != 'unverified' and final[original['id']] != original:
                raise SemanticValidationError('Protected object changed during enrichment.')
    return result, reports


def _validate_enrichment_context(scene, context):
    """Fail on mismatched measured inputs instead of combining separate runs."""
    image = context.get('image', {})
    if any(image.get(key) != scene['canvas'][key] for key in ('width', 'height', 'coordinate_system')):
        raise SemanticValidationError('Context image coordinates do not match Scene canvas.')
    nodes = {node['id']: node for node in scene['nodes']}
    measured = scene_text_regions(scene)
    for collection, key, known in (('candidates', 'candidate_id', nodes),
                                    ('text_regions', 'text_region_id', measured)):
        items = context.get(collection, [])
        if not isinstance(items, list) or len(items) > 500:
            raise SemanticValidationError('Context candidates must be bounded arrays.')
        seen = set()
        for item in items:
            identifier = item.get(key)
            if identifier not in known or identifier in seen:
                raise SemanticValidationError('Context has unknown/duplicate measurement IDs.')
            seen.add(identifier)
            original = known[identifier]
            expected = original.get('metadata', {}).get('geometry_enrichment', {}).get('original_bbox', original['bbox'])
            if item.get('bbox') != expected:
                raise SemanticValidationError('Context bbox differs from original measured Scene bbox.')
