"""Conservative, offline alignment of semantic text to measured text boxes.

There is no OCR here: a measured box establishes geometry, not its contents.
Inferred associations remain unverified, even when the VLM says "high".
"""
from copy import deepcopy
import math
import re


def _bbox(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) for v in value):
        return None
    return list(value) if value[2] > 0 and value[3] > 0 else None


def _union(boxes):
    left = min(b[0] for b in boxes)
    top = min(b[1] for b in boxes)
    return [left, top, max(b[0] + b[2] for b in boxes) - left,
            max(b[1] + b[3] for b in boxes) - top]


def merge_text_regions(regions, owners=None, gap_ratio=0.7,
                       vertical_overlap=0.65, height_ratio=0.65):
    """Join adjacent row fragments, never two differently owned text seeds.

    owners maps region ID to semantic text ID. Each returned component contains
    bbox and region_ids. Spatial adjacency alone is explicitly not an OCR match.
    """
    owners = owners or {}
    usable = []
    for region in regions:
        box = _bbox(region.get('bbox'))
        rid = region.get('text_region_id', region.get('id'))
        if box and rid:
            usable.append({'bbox': box, 'region_ids': [rid],
                           '_owners': {owners[rid]} if rid in owners else set()})
    usable.sort(key=lambda item: (item['bbox'][1], item['bbox'][0], item['region_ids'][0]))
    # Pairwise fragment adjacency, with ownership checked again at component level.
    pairs = []
    for i, first in enumerate(usable):
        a = first['bbox']
        for j in range(i + 1, len(usable)):
            b = usable[j]['bbox']
            overlap = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
            gap = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])
            if (min(a[3], b[3]) / max(a[3], b[3]) >= height_ratio
                    and overlap / max(a[3], b[3]) >= vertical_overlap
                    and 0 <= gap <= gap_ratio * max(a[3], b[3])):
                pairs.append((gap, i, j))
    parent = list(range(len(usable)))

    def find(index):
        while parent[index] != index:
            index = parent[index]
        return index

    for _, i, j in sorted(pairs):
        a, b = find(i), find(j)
        combined = usable[a]['_owners'] | usable[b]['_owners']
        if a != b and len(combined) <= 1:
            parent[b] = a
            usable[a]['_owners'] = combined
    groups = {}
    for i, item in enumerate(usable):
        groups.setdefault(find(i), []).append(item)
    result = []
    for members in groups.values():
        result.append({'bbox': _union([m['bbox'] for m in members]),
                       'region_ids': sorted(r for m in members for r in m['region_ids'])})
    return sorted(result, key=lambda item: (item['bbox'][1], item['bbox'][0]))


def _variable(text):
    value = text.strip().strip('$')
    # Normalize common TeX subscript spellings without interpreting expressions.
    value = re.sub(r'\\(?:mathrm|text|operatorname)\{([^{}]*)\}', r'\1', value)
    value = value.replace('{', '').replace('}', '')
    match = re.match(r'([A-Za-z](?:_[A-Za-z]+)?)\b', value)
    return match.group(1) if match else None


def _association(text, objects):
    explicit = text.get('candidate_id')
    if explicit:
        return [explicit], 'explicit_candidate_id'
    if text.get('mode') != 'math':
        return [], 'no_semantic_anchor'
    variable = _variable(text.get('text') or text.get('raw_text', ''))
    if not variable:
        return [], 'no_leading_variable'
    matches = []
    for obj in objects:
        # Case is meaningful: p and P must not silently become interchangeable.
        label = (obj.get('label') or '').replace('{', '').replace('}', '')
        if re.search(r'(?<![A-Za-z0-9_])' + re.escape(variable) + r'(?![A-Za-z0-9_])', label):
            if obj.get('candidate_id'):
                matches.append(obj['candidate_id'])
    return sorted(set(matches)), 'leading_variable:' + variable


def _protected(item):
    return item.get('metadata', {}).get('review_status') in ('verified', 'rejected')


def _nearby_clusters(node_box, clusters, used_regions, median_height):
    """Only offer nearby above/inside/left boxes; do not invent a text extent."""
    x, y, width, height = node_box
    proposals = []
    for cluster in clusters:
        if used_regions.intersection(cluster['region_ids']):
            continue
        bx, by, bw, bh = cluster['bbox']
        if bh < median_height * 0.65:
            continue  # dots and tiny marks are especially unreliable text anchors
        cx, cy = bx + bw / 2, by + bh / 2
        above_gap = y - (by + bh)
        relation = None
        if (0 <= above_gap <= max(2.5 * bh, height * 0.6)
                and x - width * 0.3 <= cx <= x + width * 1.3):
            relation = 'above'
        elif x <= cx <= x + width and y <= cy <= y + height:
            relation = 'inside'
        elif 0 <= x - (bx + bw) <= max(bh * 2, width * 0.5) and y <= cy <= y + height:
            relation = 'left'
        if relation:
            proposals.append({'bbox': deepcopy(cluster['bbox']),
                              'region_ids': list(cluster['region_ids']),
                              'relation': relation,
                              'position_source': 'geometry_inference'})
    return proposals


def align_texts(scene, semantic_result, context):
    """Return proposed labels plus an explicit report, without input mutation.

    Ambiguous/unresolved texts create no label. Existing protected labels are
    returned unchanged. Other proposed labels retain all existing metadata.
    """
    texts = semantic_result.get('texts', [])
    objects = semantic_result.get('objects', [])
    regions = context.get('text_regions', [])
    owners = {t['text_region_id']: t['id'] for t in texts if t.get('text_region_id')}
    clusters = merge_text_regions(regions, owners=owners)
    cluster_by_region = {rid: c for c in clusters for rid in c['region_ids']}
    labels_by_id = {label['id']: label for label in scene.get('labels', [])}
    nodes_by_id = {node['id']: node for node in scene.get('nodes', [])}
    boxes = {item['candidate_id']: _bbox(item.get('bbox'))
             for item in context.get('candidates', []) if item.get('candidate_id')}
    for nid, node in nodes_by_id.items():
        boxes.setdefault(nid, _bbox(node.get('bbox')))
    heights = sorted(c['bbox'][3] for c in clusters)
    median_height = heights[len(heights) // 2] if heights else 1
    used_regions = {rid for t in texts for rid in
                    cluster_by_region.get(t.get('text_region_id'), {}).get('region_ids', [])}
    associations = {t['id']: _association(t, objects) for t in texts}
    claimants = {}
    for t in texts:
        ids, _ = associations[t['id']]
        if not t.get('text_region_id') and len(ids) == 1:
            claimants.setdefault(ids[0], []).append(t['id'])
    entries = []
    new_labels = []
    counts = dict(original_with_bbox=sum(_bbox(t.get('bbox')) is not None for t in texts),
                  original_positioned=0, newly_positioned=0, remaining_unpositioned=0,
                  updated_existing_positions=0, conflicts=0)
    for item in texts:
        tid = item['id']
        label_id = 'sem_' + tid
        old = labels_by_id.get(label_id)
        positioned = bool(old and isinstance(old.get('x'), (int, float))
                          and isinstance(old.get('y'), (int, float)))
        counts['original_positioned'] += int(positioned)
        entry = {'source_text_id': tid, 'status': 'unresolved', 'bbox': None,
                 'position_source': 'unresolved', 'region_ids': [], 'candidates': [],
                 'evidence': [], 'review_status': 'unverified'}
        ids, association_evidence = associations[tid]
        protected = bool(old and _protected(old)) or any(
            _protected(nodes_by_id[cid]) for cid in ids if cid in nodes_by_id)
        chosen = None
        if protected:
            entry['status'] = 'protected'
            entry['evidence'].append('existing_verified_or_rejected_record_preserved')
            if old:
                new_labels.append(deepcopy(old))
                entry['bbox'] = _bbox(old.get('metadata', {}).get('bbox'))
        else:
            seed = cluster_by_region.get(item.get('text_region_id'))
            if seed:
                chosen = deepcopy(seed)
                entry['position_source'] = ('geometry_inference' if len(seed['region_ids']) > 1 else 'measured')
                entry['evidence'].append('measured_seed_with_adjacent_row_fragments' if len(seed['region_ids']) > 1
                                         else 'explicit_measured_text_region')
            elif _bbox(item.get('bbox')):
                chosen = {'bbox': _bbox(item['bbox']), 'region_ids': []}
                entry['position_source'] = 'vlm_estimate'
                entry['evidence'].append('model_bbox_only_not_measured')
            elif ids:
                entry['evidence'].append(association_evidence)
                for cid in ids:
                    if boxes.get(cid):
                        candidates = _nearby_clusters(boxes[cid], clusters, used_regions, median_height)
                        for proposal in candidates:
                            proposal['candidate_id'] = cid
                        entry['candidates'].extend(candidates)
                    else:
                        entry['evidence'].append('associated_object_has_no_measured_bbox:' + cid)
                contested = any(len(claimants.get(cid, [])) > 1 for cid in ids)
                if len(ids) == 1 and len(entry['candidates']) == 1 and not contested:
                    chosen = deepcopy(entry['candidates'][0])
                    entry['position_source'] = 'geometry_inference'
                    entry['evidence'].append('unique_nearby_measured_cluster_not_ocr_verified')
                elif len(ids) > 1 or len(entry['candidates']) > 1 or contested:
                    entry['status'] = 'ambiguous'
                    if contested:
                        entry['evidence'].append('multiple_semantic_texts_claim_same_object')
                    counts['conflicts'] += 1
            else:
                entry['evidence'].append(association_evidence)
            if chosen:
                entry.update(status='aligned', bbox=chosen['bbox'], region_ids=chosen['region_ids'])
                label = deepcopy(old) if old else {
                    'id': label_id, 'text': item.get('text', item.get('raw_text', '')),
                    'mode': item.get('mode', 'text'), 'anchor': 'center',
                    'align': 'center', 'font_size': 22, 'metadata': {}}
                box = chosen['bbox']
                label.update(x=box[0] + box[2] / 2, y=box[1] + box[3] / 2)
                metadata = label.setdefault('metadata', {})
                metadata.setdefault('source', 'vlm_semantic_proposal')
                metadata.setdefault('raw_text', item.get('raw_text', item.get('text', '')))
                metadata.setdefault('source_semantic_text', deepcopy(item))
                metadata.setdefault('confidence', item.get('confidence'))
                metadata.setdefault('confidence_meaning', 'self_reported_not_probability')
                metadata.setdefault('font_size_source', 'fixed_review_default_not_measured')
                metadata.update(source_text_id=tid, review_status='unverified',
                                bbox=deepcopy(box), position_source=entry['position_source'],
                                alignment_region_ids=list(entry['region_ids']),
                                alignment_evidence=list(entry['evidence']))
                new_labels.append(label)
                if positioned:
                    counts['updated_existing_positions'] += int(
                        old['x'] != label['x'] or old['y'] != label['y'])
                else:
                    counts['newly_positioned'] += 1
        if not positioned and chosen is None:
            counts['remaining_unpositioned'] += 1
        entries.append(entry)
    return new_labels, {'entries': entries, 'counts': counts,
                        'method': 'measured_adjacency_and_conservative_symbol_association',
                        'limitations': ['No OCR content matching; positions are unverified.',
                                        'Measured text boxes may omit symbols or combine unrelated marks.',
                                        'Unanchored natural-language text is not positioned by guesswork.']}
