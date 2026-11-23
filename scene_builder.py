"""OpenCV candidates -> unverified, schema-compatible scene draft. No recognition.

The manual reference is optional REPORT input only; it never supplies scene data.
All source fields are retained in object metadata or source_geometry_metadata.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re

from structured_tikz.validator import validate_scene


class SceneBuildError(ValueError):
    """Geometry cannot be converted without ambiguity or data loss."""


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SceneBuildError('Duplicate JSON field: ' + key)
        result[key] = value
    return result


def read_geometry(path):
    source = Path(path)
    if source.stat().st_size > 5_000_000:
        raise SceneBuildError('Geometry JSON exceeds the 5 MB experiment limit.')
    try:
        result = json.loads(source.read_text(encoding='utf-8-sig'), object_pairs_hook=_unique_keys,
                            parse_constant=lambda value: _bad_constant(value))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SceneBuildError('Invalid UTF-8 geometry JSON: ' + str(exc)) from exc
    if not isinstance(result, dict):
        raise SceneBuildError('Geometry root must be an object.')
    return result


def _bad_constant(value):
    raise SceneBuildError('Non-finite JSON constant: ' + value)


def _number(value, location, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SceneBuildError(location + ' must be finite numeric data.')
    if value < 0 or maximum is not None and value > maximum:
        raise SceneBuildError(location + ' is outside the image coordinate range.')
    return value


def _bbox(item, width, height):
    raw = item.get('bbox_px')
    normalized = item.get('bbox')
    if isinstance(raw, dict) and set(raw) == {'x', 'y', 'width', 'height'}:
        box = [raw[k] for k in ('x', 'y', 'width', 'height')]
        source = 'bbox_px_dict_to_bbox_array'
    elif isinstance(raw, list) and len(raw) == 4:
        box, source = list(raw), 'bbox_px_array'
    elif raw is None and isinstance(normalized, dict):
        box = [_number(normalized.get(k), 'normalized bbox.' + k, 1) * factor
               for k, factor in zip(('x', 'y', 'width', 'height'), (width, height, width, height))]
        source = 'normalized_bbox_scaled_to_pixels'
    else:
        raise SceneBuildError('Candidate requires bbox_px or a normalized bbox: ' + item['id'])
    x, y, w, h = [_number(v, 'bbox') for v in box]
    if w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise SceneBuildError('Invalid bbox size/bounds: ' + item['id'])
    if normalized is not None:
        if not isinstance(normalized, dict) or set(normalized) != {'x', 'y', 'width', 'height'}:
            raise SceneBuildError('Invalid normalized bbox: ' + item['id'])
        calculated = [_number(normalized[k], 'normalized bbox.' + k, 1) * factor
                      for k, factor in zip(('x', 'y', 'width', 'height'), (width, height, width, height))]
        if any(abs(a - b) > .02 for a, b in zip(box, calculated)):
            raise SceneBuildError('Pixel and normalized bbox disagree: ' + item['id'])
    return box, source


def _line_points(item, width, height):
    keys = ('x1', 'y1', 'x2', 'y2')
    normalized = all(key in item for key in keys)
    if 'pixel' in item:
        pixel = item['pixel']
        if not isinstance(pixel, dict) or set(pixel) != set(keys):
            raise SceneBuildError('Invalid line pixel coordinates: ' + item['id'])
        coords = [_number(pixel[k], 'line.pixel.' + k, limit)
                  for k, limit in zip(keys, (width, height, width, height))]
        mapping = 'pixel_endpoints'
    elif normalized:
        coords = [_number(item[k], 'normalized line.' + k, 1) * limit
                  for k, limit in zip(keys, (width, height, width, height))]
        mapping = 'normalized_endpoints_scaled_to_pixels'
    else:
        raise SceneBuildError('Line endpoints are missing: ' + item['id'])
    if normalized:
        calculated = [_number(item[k], 'normalized line.' + k, 1) * limit
                      for k, limit in zip(keys, (width, height, width, height))]
        if any(abs(a - b) > .02 for a, b in zip(coords, calculated)):
            raise SceneBuildError('Pixel and normalized line endpoints disagree: ' + item['id'])
    if coords[:2] == coords[2:]:
        raise SceneBuildError('Zero-length line: ' + item['id'])
    return [coords[:2], coords[2:]], mapping


def build_scene_draft(geometry_data: dict, image_width: int, image_height: int) -> dict:
    """Preserve rectangle bboxes; never infer text, tensor types, or topology."""
    if not isinstance(geometry_data, dict) or geometry_data.get('schema_version') != '1.0':
        raise SceneBuildError('Expected geometry schema_version 1.0.')
    if geometry_data.get('coordinate_system') != 'top-left-normalized':
        raise SceneBuildError('Expected top-left-normalized source coordinates.')
    if geometry_data.get('axes', {'x': 'right', 'y': 'down'}) != {'x': 'right', 'y': 'down'}:
        raise SceneBuildError('Geometry axes must be x:right, y:down.')
    image = geometry_data.get('image')
    if not isinstance(image, dict):
        raise SceneBuildError('Geometry image metadata is missing.')
    for name, value in (('width', image_width), ('height', image_height)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 20000:
            raise SceneBuildError('Image dimensions must be positive integers <= 20000.')
        if image.get(name) != value:
            raise SceneBuildError('Supplied image dimensions disagree with geometry.image.')
    # Validate JSON-only metadata and reject NaN even in retained source evidence.
    try:
        json.dumps(geometry_data, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SceneBuildError('Geometry must contain finite JSON-compatible data.') from exc
    collections = {}
    identifiers = set()
    for kind in ('rectangles', 'lines', 'text_regions', 'regions'):
        values = geometry_data.get(kind, [])
        if not isinstance(values, list) or len(values) > 500:
            raise SceneBuildError(kind + ' must be an array of at most 500 candidates.')
        for item in values:
            identifier = item.get('id') if isinstance(item, dict) else None
            if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', identifier):
                raise SceneBuildError('Candidate needs a simple ID in ' + kind)
            if identifier in identifiers:
                raise SceneBuildError('Duplicate candidate ID: ' + identifier)
            identifiers.add(identifier)
        collections[kind] = values
    scene = {'canvas': {'width': image_width, 'height': image_height, 'coordinate_system': 'top-left'},
             'styles': {}, 'nodes': [], 'edges': [], 'labels': [],
             'metadata': {'source': 'opencv_geometry_scene_draft', 'review_status': 'unverified',
                          'source_image': copy.deepcopy(image),
                          'source_geometry_metadata': copy.deepcopy({k: v for k, v in geometry_data.items() if k not in collections}),
                          'review_candidates': {'lines': [], 'excluded_lines': [], 'text_regions': [], 'regions': []},
                          'conversion_warnings': [],
                          'score_meaning': 'Source heuristic ranks are not calibrated recognition probabilities.',
                          'style_origin': 'Fixed neutral outline for review, not a measured source color.',
                          'coordinate_conversion': 'Prefer original pixels; normalized-only values multiplied by image dimensions.',
                          'human_review_required': True}}
    candidates = scene['metadata']['review_candidates']
    for item in collections['rectangles']:
        box, mapping = _bbox(item, image_width, image_height)
        scene['nodes'].append({'id': item['id'], 'type': 'rectangle', 'bbox': box,
                               'style': {'fill': 'none', 'stroke': 'dark_blue'},
                               'metadata': {'source': 'opencv', 'source_id': item['id'],
                                            'review_status': 'unverified', 'classification': 'rectangle_candidate',
                                            'coordinate_mapping': mapping, 'source_candidate': copy.deepcopy(item)}})
    for item in collections['lines']:
        points, mapping = _line_points(item, image_width, image_height)
        classification = item.get('classification', 'uncertain')
        if classification not in {'geometry', 'text_stroke_candidate', 'uncertain'}:
            raise SceneBuildError('Unsupported line classification: ' + item['id'])
        if 'classification' not in item:
            scene['metadata']['conversion_warnings'].append('Missing line classification treated as uncertain: ' + item['id'])
        candidate = {'id': item['id'], 'source_id': item['id'], 'source': 'opencv',
                     'classification': classification, 'points': points, 'arrow': 'unknown',
                     'review_status': 'unverified', 'coordinate_mapping': mapping,
                     'source_candidate': copy.deepcopy(item)}
        if classification == 'text_stroke_candidate':
            candidate['exclusion_reason'] = 'Classified as text stroke; excluded from edge selection, not human-rejected.'
            candidates['excluded_lines'].append(candidate)
        else:
            candidates['lines'].append(candidate)
    for kind in ('text_regions', 'regions'):
        for item in collections[kind]:
            box, mapping = _bbox(item, image_width, image_height)
            record = {'id': item['id'], 'bbox': box, 'review_status': 'unverified',
                      'coordinate_mapping': mapping, 'source_candidate': copy.deepcopy(item)}
            if kind == 'text_regions':
                record.update(text_region_id=item['id'], recognized_text=None,
                              source=item.get('source', 'unspecified_text_region_detector'))
            candidates[kind].append(record)
    validate_scene(scene)
    return scene


def edge_from_candidate(scene, candidate_id, edge_id):
    """Explicit human selection only; produces an unverified, directionless edge."""
    validate_scene(scene)
    candidate = next((item for item in scene.get('metadata', {}).get('review_candidates', {}).get('lines', [])
                      if item['id'] == candidate_id), None)
    if candidate is None:
        raise SceneBuildError('Unknown or excluded line candidate: ' + candidate_id)
    edge = {'id': edge_id, 'points': copy.deepcopy(candidate['points']), 'arrow': 'none',
            'metadata': {'source': 'human_selected_opencv_line', 'source_id': candidate_id,
                         'review_status': 'unverified', 'classification': candidate['classification']}}
    checked = copy.deepcopy(scene)
    checked['edges'].append(edge)
    validate_scene(checked)
    return edge


def _iou(a, b):
    overlap = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(
        0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return overlap / (a[2] * a[3] + b[2] * b[3] - overlap)


def compare_reference(scene, reference):
    """One-to-one greedy IoU matches, REPORT ONLY, never change the draft."""
    validate_scene(reference)
    if any(reference['canvas'][k] != scene['canvas'][k] for k in ('width', 'height', 'coordinate_system')):
        raise SceneBuildError('Reference canvas differs; no implicit rescaling is performed.')
    pairs = []
    for node in scene['nodes']:
        for target in reference['nodes']:
            x, y, w, h = target['bbox']
            if target['type'] == 'tensor3d':
                dx, dy = target['depth']['dx'], target['depth']['dy']
                extent = [x, y + dy, w + dx, h - dy]
            else:
                extent = target['bbox']
            pairs.append((_iou(node['bbox'], extent), node, target, extent))
    used_nodes, used_targets, matches = set(), set(), []
    for overlap, node, target, extent in sorted(pairs, key=lambda p: (-p[0], p[1]['id'], p[2]['id'])):
        if overlap < .5 or node['id'] in used_nodes or target['id'] in used_targets:
            continue
        used_nodes.add(node['id']); used_targets.add(target['id'])
        box = node['bbox']
        matches.append({'node_id': node['id'], 'reference_id': target['id'],
                        'reference_type_for_comparison_only': target['type'],
                        'iou_reference_extent': round(overlap, 6),
                        'iou_reference_front_bbox': round(_iou(box, target['bbox']), 6),
                        'extent_center_distance_px': round(math.hypot(box[0]+box[2]/2-extent[0]-extent[2]/2,
                                                                    box[1]+box[3]/2-extent[1]-extent[3]/2), 3),
                        'meaning': 'spatial_candidate_match_not_semantic_recognition'})
    return {'method': 'one_to_one_greedy_IoU_at_least_0.5; tensor_reference_uses_projected_extent',
            'matches': matches, 'matched_nodes': len(matches),
            'unmatched_draft_ids': [n['id'] for n in scene['nodes'] if n['id'] not in used_nodes],
            'unmatched_reference_ids': [n['id'] for n in reference['nodes'] if n['id'] not in used_targets],
            'reference_counts': {k: len(reference[k]) for k in ('nodes', 'edges', 'labels')},
            'reference_tensor_count': sum(n['type'] == 'tensor3d' for n in reference['nodes']),
            'automatically_recognized_tensor_count': 0,
            'limitations': ['No text, formula, grid, depth, node type or edge topology is recovered by bbox matching.',
                            'All reference edges and labels need human review; Hough segments do not prove complete connections.']}


def build_report(scene, reference=None):
    review = scene['metadata']['review_candidates']
    report = {'kind': 'geometry_to_scene_draft', 'api_called': False, 'review_status': 'unverified',
              'input_rectangles': len(scene['nodes']), 'converted_nodes': len(scene['nodes']),
              'excluded_rectangles': 0, 'nodes_requiring_review': len(scene['nodes']),
              'input_lines': len(review['lines']) + len(review['excluded_lines']),
              'retained_line_candidates': len(review['lines']),
              'excluded_text_stroke_candidates': len(review['excluded_lines']),
              'uncertain_line_candidates': sum(c['classification'] == 'uncertain' for c in review['lines']),
              'automatic_edges': 0, 'automatic_labels': 0,
              'unrecognized_text_regions': len(review['text_regions']),
              'geometric_groups_retained': len(review['regions']),
              'coordinate_mappings': {n['id']: n['metadata']['coordinate_mapping'] for n in scene['nodes']},
              'warnings': scene['metadata']['conversion_warnings']}
    if reference is not None:
        report['reference_comparison'] = compare_reference(scene, reference)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New scene_draft.json; never overwritten')
    parser.add_argument('--reference', type=Path, help='Manual scene used only for geometric comparison report')
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        report_path = output.parent / 'scene_build_report.json'
        if output.suffix.lower() != '.json' or output == report_path or output.exists() or report_path.exists():
            raise SceneBuildError('Use a new .json output and a directory without an existing build report.')
        geometry = read_geometry(args.geometry)
        size = geometry.get('image', {})
        scene = build_scene_draft(geometry, size.get('width'), size.get('height'))
        reference = None
        if args.reference:
            from structured_tikz.generator import load_scene
            reference, _ = load_scene(args.reference)
        report = build_report(scene, reference)
        report['geometry_file_sha256'] = hashlib.sha256(args.geometry.read_bytes()).hexdigest()
        report['geometry_path'] = str(args.geometry.resolve())
        report['scene_path'] = str(output)
        scene_text = json.dumps(scene, ensure_ascii=False, indent=2, allow_nan=False)
        if len(scene_text.encode('utf-8')) > 1_000_000:
            raise SceneBuildError('Draft exceeds existing scene loader 1 MB limit; nothing was written.')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as handle:
            handle.write(scene_text)
        with report_path.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, 'Scene build error: ' + str(exc) + '\n')
    print('Draft:', output)
    print('Report:', report_path)
    print('Unverified nodes:', len(scene['nodes']), '| Retained line candidates:', report['retained_line_candidates'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
