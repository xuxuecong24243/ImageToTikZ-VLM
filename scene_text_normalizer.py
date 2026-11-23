"""Conservative Scene text normalization; no drawing, model calls or file IO.

Only explicitly delimited, allowlisted math changes mode. Mixed prose/math is
opt-in via enable_rich_text; otherwise it remains literal and pending.
Duplicate comparison never rewrites the retained content.
An edge/owner-referenced node is retained; no implicit reference migration occurs.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re
import unicodedata

from structured_tikz.schema import MATH_COMMANDS
from structured_tikz.validator import resolve_style, validate_math, validate_scene
from rich_text_parser import parse_rich_text, RichTextParseError


def _meta(item):
    value = item.get('metadata', {})
    return value if isinstance(value, dict) else {}


def _protected(item):
    return _meta(item).get('review_status') in ('verified', 'rejected')


def _dollars(text):
    result = []
    for index, char in enumerate(text):
        if char != '$':
            continue
        cursor = index-1
        while cursor >= 0 and text[cursor] == '\\':
            cursor -= 1
        if (index-1-cursor) % 2 == 0:
            result.append(index)
    return result


def _math_parts(text):
    """Return exact inner text only for ONE complete supported outer delimiter."""
    stripped = text.strip()
    dollars = _dollars(stripped)
    if len(stripped) >= 2 and dollars == [0, len(stripped)-1]:
        return stripped[1:-1], '$...$'
    if (stripped.startswith(r'\(') and stripped.endswith(r'\)')
            and stripped.count(r'\(') == stripped.count(r'\)') == 1 and not dollars):
        return stripped[2:-2], r'\(...\)'
    return None, None


def _source_text(item):
    return item.get('text', ''.join(span['content'] for span in item.get('segments', [])))


def _classify(item):
    text = _source_text(item)
    if item.get('mode') == 'rich':
        return 'existing_rich', None
    if item.get('mode', 'text') == 'math':
        return 'existing_math', None
    inner, delimiter = _math_parts(text)
    if inner is not None:
        try:
            validate_math(inner)
        except ValueError as error:
            return 'invalid_math', str(error)
        return 'pure_delimited_math', {'inner': inner, 'delimiter': delimiter}
    if _dollars(text) or r'\(' in text or r'\)' in text:
        return 'mixed_text_math', 'Literal mixed content needs opt-in validated rich-text conversion.'
    commands = set(re.findall(r'\\([A-Za-z]+)', text)) & MATH_COMMANDS
    if commands:
        return 'math_commands_in_text', 'Math-like commands without complete outer delimiters; meaning is uncertain.'
    return 'plain_text', None


def _plain_key(text):
    # Comparison only: retained Unicode, punctuation and spaces are not edited.
    # Normalize punctuation only: NFKC on all text would turn e.g. x² into x2.
    punctuation = ''.join(unicodedata.normalize('NFKC', char)
                          if unicodedata.category(char).startswith('P') else char for char in text)
    value = ' '.join(punctuation.split())
    value = re.sub(r'\s*([(),:;!?])\s*', r'\1', value)
    return re.sub(r'(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])', '', value)


def _math_key(text):
    # Only a SINGLE TeX token may lose optional script braces. _{tok} != _tok.
    value = re.sub(r'(?<!\\)([_^])\{([A-Za-z0-9])\}', r'\1\2', text)
    if re.search(r'\\(?:text|operatorname)\b', value):
        return value  # Preserve meaningful spaces within text-bearing commands.
    return re.sub(r'\s+', '', value)


def _content_key(item):
    text, mode = _source_text(item), item.get('mode', 'text')
    if mode == 'rich':
        return ('rich', tuple((span['type'], _math_key(span['content']) if span['type'] == 'math'
                              else _plain_key(span['content'])) for span in item['segments']))
    if mode == 'math':
        return ('math', _math_key(text))
    inner, _ = _math_parts(text)
    if inner is not None:
        try:
            validate_math(inner)
            return ('math', _math_key(inner))
        except ValueError:
            return ('invalid_math_literal', text)
    # For mixed text, compare closed spans separately, without converting them.
    spans, end = [], 0
    for match in re.finditer(r'(?<!\\)\$([^$\n]*)\$|\\\((.*?)\\\)', text, re.DOTALL):
        spans.append(('text', _plain_key(text[end:match.start()])))
        math_text = match.group(1) if match.group(1) is not None else match.group(2)
        try:
            validate_math(math_text)
            spans.append(('math', _math_key(math_text)))
        except ValueError:
            spans.append(('literal', match.group(0)))
        end = match.end()
    spans.append(('text', _plain_key(text[end:])))
    return ('text', tuple(spans))


def _box(value):
    return (list(value) if isinstance(value, (list, tuple)) and len(value) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)
            and value[2] > 0 and value[3] > 0 else None)


def _position(item, collection):
    if collection == 'nodes':
        x, y, w, h = item['bbox']
        return {'point': [x+w/2, y+h/2], 'bbox': list(item['bbox']), 'anchor': 'center',
                'source': 'existing_node_bbox'}
    return {'point': [item['x'], item['y']], 'bbox': _box(_meta(item).get('bbox')),
            'anchor': item.get('anchor', 'center'), 'source': 'existing_label_anchor_and_optional_bbox'}


def _overlap(a, b):
    area = max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))*max(
        0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    return area/(a[2]*a[3]+b[2]*b[3]-area)


def _spatial(a, b):
    first, second = a['position'], b['position']
    distance = math.dist(first['point'], second['point'])
    boxes = first['bbox'], second['bbox']
    iou = _overlap(*boxes) if all(boxes) else None
    same_anchor = first['anchor'] == second['anchor']
    tolerance = min(20.0, max(3.0, min(boxes[0][3], boxes[1][3])*.5)) if all(boxes) else 2.0
    close = same_anchor and distance <= tolerance and (iou is None or iou >= .75)
    return {'clear_spatial_duplicate': close, 'anchor_distance_px': distance,
            'bbox_iou': iou, 'distance_tolerance_px': tolerance,
            'nearby_but_uncertain': not close and (distance <= 30 or iou is not None and iou > .2)}


def _references(scene):
    result = {}
    keys = {'from', 'to', 'node_id', 'source_id', 'target_id', 'candidate_id'}

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in keys and isinstance(child, str):
                    result.setdefault(child, []).append(path+'.'+key)
                elif isinstance(child, (dict, list)):
                    visit(child, path+'.'+key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path+'['+str(index)+']')

    for kind in ('nodes', 'edges', 'labels'):
        for item in scene[kind]:
            visit(item, kind+'.'+item['id'])
    visit(scene.get('metadata', {}).get('pending_edges', []), 'metadata.pending_edges')
    return result


def _entries(scene):
    entries = []
    for kind in ('labels', 'nodes'):  # Prefer an existing independent label.
        for item in scene[kind]:
            if not item.get('text') and item.get('mode') != 'rich':
                continue
            entries.append({'collection': kind, 'id': item['id'], 'item': item,
                            'key': _content_key(item), 'position': _position(item, kind)})
    return entries


def _pair_reason(scene, a, b, references):
    owners = [_meta(e['item']).get('node_id') for e in (a, b)]
    if all(owners) and owners[0] != owners[1]:
        return 'conflicting_owner_ids'
    for entry in (a, b):
        item = entry['item']
        if _protected(item):
            return 'protected_object'
        if references.get(entry['id']):
            return 'referenced_object_no_safe_migration'
        if entry['collection'] == 'nodes':
            if item['type'] != 'text' or resolve_style(scene, item)['fill'] != 'none':
                return 'node_has_or_may_intend_geometry'
    styles = [resolve_style(scene, e['item']) for e in (a, b)]
    if any(styles[0][key] != styles[1][key] for key in ('font_size', 'text_color', 'bold')):
        return 'different_text_styles'
    if a['item'].get('align', 'center') != b['item'].get('align', 'center'):
        return 'different_text_alignment'
    return None


def normalize_scene_text(scene: dict, *, enable_rich_text: bool = False) -> tuple[dict, dict]:
    """Return schema-valid copy + audit. No coordinate/route edits or API calls.

    Incomplete/invalid input Scene is rejected. Unsafe delimited math in literal
    text is preserved and reported; it is NEVER passed to the math renderer.
    Re-running returns the same Scene content (reports describe each call).
    Rich conversion is opt-in and preserves original text as provenance.
    """
    if not isinstance(enable_rich_text, bool):
        raise ValueError('enable_rich_text must be Boolean.')
    validate_scene(scene)
    result = deepcopy(scene)
    original_entries = _entries(scene)
    original_by_id = {entry['id']: entry for entry in original_entries}
    report = {'schema_input_valid': True, 'changes': [], 'pending_review': [],
              'diagnosis': {'inventory': [], 'duplicate_candidates': [], 'spatial_repetitions': []},
              'rules': {'bbox_iou_minimum': .75, 'maximum_anchor_gap_px': 20,
                        'coordinates_changed': False, 'references_migrated': False,
                        'mixed_content_supported_by_current_renderer': True,
                        'rich_text_enabled': enable_rich_text}}
    references = _references(scene)
    for entry in original_entries:
        item = entry['item']
        source_text = _source_text(item)
        classification, detail = _classify(item)
        report['diagnosis']['inventory'].append({
            'id': entry['id'], 'collection': entry['collection'], 'text': source_text,
            'mode': item.get('mode', 'text'), 'position': entry['position'],
            'classification': classification, 'references': references.get(entry['id'], []),
            'contains_chinese': bool(re.search(r'[\u3400-\u9fff]', source_text)),
            'contains_latin': bool(re.search(r'[A-Za-z]', source_text)),
            'math_commands': sorted(set(re.findall(r'\\([A-Za-z]+)', source_text)) & MATH_COMMANDS)})
    for kind in ('nodes', 'labels'):
        for item in result[kind]:
            if not item.get('text'):
                continue
            classification, detail = _classify(item)
            if classification == 'pure_delimited_math':
                if kind == 'nodes' and item['type'] != 'text':
                    report['pending_review'].append({'id': item['id'], 'reason': 'geometric_node_text_requires_review'})
                    continue
                if _protected(scene) or _protected(item):
                    report['pending_review'].append({'id': item['id'], 'reason': 'protected_math_object'})
                    continue
                before = deepcopy(item)
                item.update(text=detail['inner'], mode='math')
                report['changes'].append({'operation': 'normalize_pure_math', 'id': item['id'],
                                          'collection': kind, 'before': before, 'after': deepcopy(item),
                                          'reason': 'complete_math_delimiters_and_existing_allowlist',
                                          'delimiter': detail['delimiter'], 'source': 'syntax_only_no_semantic_rewrite'})
            elif classification == 'mixed_text_math' and enable_rich_text:
                if kind == 'nodes' and item['type'] != 'text':
                    report['pending_review'].append({'id': item['id'], 'reason': 'geometric_node_text_requires_review'})
                    continue
                if _protected(scene) or _protected(item):
                    report['pending_review'].append({'id': item['id'], 'reason': 'protected_math_object'})
                    continue
                try:
                    segments = parse_rich_text(item['text'])
                except RichTextParseError as error:
                    report['pending_review'].append({'id': item['id'], 'collection': kind,
                        'reason': 'rich_text_parse_failed', 'detail': str(error), 'text': item['text']})
                    continue
                if not any(span['type'] == 'math' for span in segments):
                    continue
                before = deepcopy(item)
                item.update(mode='rich', segments=segments)
                report['changes'].append({'operation': 'normalize_rich_text', 'id': item['id'],
                    'collection': kind, 'original_text': before['text'],
                    'before': before, 'after': deepcopy(item),
                    'reason': 'explicit_inline_math_delimiters_and_existing_allowlist',
                    'source': 'syntax_only_no_semantic_rewrite'})

    entries = _entries(result)
    removed = set()
    for index, entry in enumerate(entries):
        if entry['id'] in removed:
            continue
        for other in entries[index+1:]:
            if other['id'] in removed or entry['key'] != other['key']:
                continue
            evidence = _spatial(entry, other)
            if not evidence['clear_spatial_duplicate']:
                category = 'duplicate_candidates' if evidence['nearby_but_uncertain'] else 'spatial_repetitions'
                report['diagnosis'][category].append({'ids': [entry['id'], other['id']],
                    'status': 'uncertain' if evidence['nearby_but_uncertain'] else 'retained_different_positions',
                    **evidence})
                if evidence['nearby_but_uncertain']:
                    report['pending_review'].append({'ids': [entry['id'], other['id']], 'reason': 'uncertain_duplicate_position'})
                continue
            reason = 'protected_scene' if _protected(scene) else _pair_reason(result, entry, other, references)
            report['diagnosis']['duplicate_candidates'].append({'ids': [entry['id'], other['id']],
                'status': 'pending_review' if reason else 'confirmed_duplicate', 'reason': reason,
                'content_comparison': 'whitespace_punctuation_and_single_token_scripts_only', **evidence})
            if reason:
                report['pending_review'].append({'ids': [entry['id'], other['id']], 'reason': reason})
                continue
            # The retained object is never moved, rewritten or restyled.
            removed.add(other['id'])
            report['changes'].append({'operation': 'remove_duplicate', 'id': other['id'],
                'collection': other['collection'], 'retained_id': entry['id'],
                'retained_collection': entry['collection'], 'original_text': _source_text(original_by_id[other['id']]['item']),
                'original_position': original_by_id[other['id']]['position'],
                'original_object': deepcopy(original_by_id[other['id']]['item']),
                'before': deepcopy(other['item']), 'after': None,
                'retained_before': deepcopy(entry['item']), 'retained_after': deepcopy(entry['item']),
                'reason': 'equivalent_content_same_style_overlapping_position_no_live_references',
                'source': 'conservative_data_deduplication', 'evidence': evidence})
    for kind in ('nodes', 'labels'):
        result[kind] = [item for item in result[kind] if item['id'] not in removed]
    # Pending mixed/invalid text is reported for survivors, with original text intact.
    remaining_entries = _entries(result)
    for entry in remaining_entries:
        classification, detail = _classify(entry['item'])
        if classification in ('mixed_text_math', 'invalid_math', 'math_commands_in_text'):
            report['pending_review'].append({'id': entry['id'], 'collection': entry['collection'],
                'reason': classification, 'detail': detail, 'text': _source_text(entry['item']),
                'position': entry['position'], 'review_status': 'unverified'})
    validate_scene(result)
    report['schema_output_valid'] = True
    report['counts'] = {
        'text_nodes_before': sum(n['type'] == 'text' for n in scene['nodes']),
        'text_nodes_after': sum(n['type'] == 'text' for n in result['nodes']),
        'labels_before': len(scene['labels']), 'labels_after': len(result['labels']),
        'duplicate_objects_removed': len(removed),
        'confirmed_duplicate_pairs': sum(e['status'] == 'confirmed_duplicate' for e in report['diagnosis']['duplicate_candidates']),
        'pure_math_converted': sum(e['operation'] == 'normalize_pure_math' for e in report['changes']),
        'rich_text_converted': sum(e['operation'] == 'normalize_rich_text' for e in report['changes']),
        'rich_text_parse_failures': sum(e['reason'] == 'rich_text_parse_failed' for e in report['pending_review']),
        'mixed_text_math_before': sum(e['classification'] == 'mixed_text_math' for e in report['diagnosis']['inventory']),
        'mixed_text_math_remaining': sum(e['reason'] == 'mixed_text_math' for e in report['pending_review']),
        'invalid_math_remaining': sum(e['reason'] == 'invalid_math' for e in report['pending_review'])}
    report['invariants'] = {
        'edges_unchanged': result['edges'] == scene['edges'],
        'geometric_nodes_unchanged': [n for n in result['nodes'] if n['type'] != 'text'] ==
                                     [n for n in scene['nodes'] if n['type'] != 'text'],
        'retained_positions_unchanged': all(
            next(e['position'] for e in original_entries if e['id'] == entry['id']) == entry['position']
            for entry in remaining_entries)}
    if not all(report['invariants'].values()):
        raise ValueError('Text normalization must preserve geometry, routes and retained positions.')
    return result, report
