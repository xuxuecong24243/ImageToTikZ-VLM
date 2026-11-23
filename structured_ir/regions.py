"""Geometry candidates and explicit crop-coordinate recovery; never recognition.

This adapter reuses the existing source-bound crop implementation. A geometric
candidate is deliberately not an IR semantic region until a recognizer says so.
Transforms use source pixels per local pixel, followed by a source-pixel offset.
"""
from copy import deepcopy
import math

from region_segmenter import (_image_info, _intersection, _records,
                              segment_regions, save_region_crops, MAX_REGIONS)


def _finite(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _mapped(values):
    if not all(_finite(value) for value in values):
        raise ValueError('coordinate transform produced non-finite coordinates')
    return values


def _box(value):
    if isinstance(value, dict):
        value = [value.get(key) for key in ('x', 'y', 'width', 'height')]
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or not all(_finite(v) for v in value) or min(value[2:]) <= 0):
        raise ValueError('bbox must contain finite [x,y,width,height] with positive size')
    return list(value)


def _clip_box(value, width, height):
    x, y, w, h = _box(value)
    left, top = max(0, math.floor(x)), max(0, math.floor(y))
    right, bottom = min(width, math.ceil(x + w)), min(height, math.ceil(y + h))
    if right <= left or bottom <= top:
        raise ValueError('candidate is entirely outside the image')
    return [left, top, right - left, bottom - top]


def _transform(value):
    if not isinstance(value, dict) or set(value) != {'offset', 'scale'}:
        raise ValueError('transform requires exactly offset and scale')
    offset, scale = value['offset'], value['scale']
    if (any(not isinstance(v, (list, tuple)) or len(v) != 2 for v in (offset, scale))
            or not all(_finite(v) for v in list(offset) + list(scale))
            or min(scale) <= 0):
        raise ValueError('transform needs finite offset and positive scale pairs')
    return offset, scale


def local_to_global_bbox(bbox, transform):
    """Map a box without rounding or rescaling its meaning."""
    x, y, width, height = _box(bbox)
    (dx, dy), (sx, sy) = _transform(transform)
    return _mapped([dx + x * sx, dy + y * sy, width * sx, height * sy])


def local_to_global_points(points, transform):
    """Map explicit path points; semantic endpoints are not coordinates."""
    (dx, dy), (sx, sy) = _transform(transform)
    if (not isinstance(points, list)
            or any(not isinstance(p, (list, tuple)) or len(p) != 2
                   or not all(_finite(v) for v in p) for p in points)):
        raise ValueError('points must be an array of finite [x,y] pairs')
    return [_mapped([dx + x * sx, dy + y * sy]) for x, y in points]


def plan_regions(image_path, geometry, *, overlap_px=12, duplicate_iou=.97):
    """Plan bounded crops from verified cached geometric evidence.

    Existing automatic column segmentation is retained as independent evidence.
    Rectangle/group candidates also survive ambiguous layouts, with deduplication,
    clipping and overlap diagnostics. No text-stroke count is treated as content.
    """
    if not _finite(duplicate_iou) or not 0 < duplicate_iou <= 1:
        raise ValueError('duplicate_iou must be in (0,1]')
    if geometry is None:
        raise ValueError('plan_regions requires existing geometry; no implicit CV run')
    # Reuse identity validation and current automatic segmentation, not a new grid.
    segmentation = segment_regions(image_path, geometry=geometry, overlap_px=overlap_px)
    info = _image_info(image_path)
    width, height = info['width'], info['height']
    candidates, diagnostics = [], []
    for collection in ('rectangles', 'regions'):
        records = geometry.get(collection, [])
        if not isinstance(records, list) or len(records) > 500:
            raise ValueError('geometry ' + collection + ' must be a bounded array')
        for index, record in enumerate(records):
            path = '/' + collection + '/' + str(index)
            try:
                if not isinstance(record, dict):
                    raise ValueError('candidate must be an object')
                original = _box(record.get('bbox_px'))
                bbox = _clip_box(original, width, height)
            except ValueError as error:
                diagnostics.append({'code': 'invalid_candidate', 'path': path, 'message': str(error)})
                continue
            source = {'collection': collection, 'index': index, 'source_id': record.get('id'),
                      'original_bbox': original}
            if original != bbox:
                diagnostics.append({'code': 'candidate_clipped_or_pixel_enclosed',
                                    'path': path + '/bbox_px', 'before': original, 'after': bbox})
            duplicate = None
            for previous in candidates:
                intersection = _intersection(bbox, previous['bbox'])
                union = bbox[2] * bbox[3] + previous['bbox'][2] * previous['bbox'][3] - intersection
                if intersection / union >= duplicate_iou:
                    duplicate = previous
                    break
            if duplicate is not None:
                duplicate['sources'].append(source)
                diagnostics.append({'code': 'duplicate_candidate', 'path': path,
                                    'retained_id': duplicate['id'], 'threshold_iou': duplicate_iou})
                continue
            candidates.append({'id': 'candidate_%03d' % (len(candidates) + 1),
                               'type': 'geometric_candidate', 'bbox': bbox, 'sources': [source],
                               'confidence': None,
                               'uncertainty': 'Geometric evidence only; semantic role and hierarchy unrecognized.'})
    overlaps = []
    for i, first in enumerate(candidates):
        for second in candidates[i + 1:]:
            intersection = _intersection(first['bbox'], second['bbox'])
            if not intersection:
                continue
            first_area, second_area = (obj['bbox'][2] * obj['bbox'][3] for obj in (first, second))
            relationship = ('first_contains_second' if intersection == second_area else
                            'second_contains_first' if intersection == first_area else 'partial_overlap')
            overlaps.append({'first_id': first['id'], 'second_id': second['id'],
                             'relationship': relationship, 'intersection_area_px': intersection,
                             'handling': 'retained; geometric containment does not assign semantic parents'})
    if candidates:
        selected = sorted(candidates, key=lambda c: (-c['bbox'][2] * c['bbox'][3], c['id']))[:MAX_REGIONS]
        crop_regions = _records([c['bbox'] for c in selected], info, 'geometry_candidates',
                               {'method': 'deduplicated_clipped_existing_geometry',
                                'confidence_basis': 'unrecognized geometric candidates'}, overlap_px)
        for record, candidate in zip(crop_regions, selected):
            record['candidate_id'] = candidate['id']
            candidate['crop_region_id'] = record['region_id']
            candidate['crop_bbox'] = record['crop_bbox']
            candidate['local_bbox'] = [record['bbox'][0] - record['offset'][0],
                                       record['bbox'][1] - record['offset'][1],
                                       record['bbox'][2], record['bbox'][3]]
            candidate['transform'] = {'offset': list(record['offset']), 'scale': [1, 1]}
        if len(candidates) > MAX_REGIONS:
            diagnostics.append({'code': 'crop_limit', 'limit': MAX_REGIONS,
                                'message': 'All candidates retained; only largest candidates scheduled for crops.'})
    else:
        crop_regions = segmentation
    diagnostics.append({'code': 'semantic_segmentation_unverified',
                        'message': 'No OCR/VLM was called; candidates are not recognized IR regions.'})
    return {'canvas': {'width': width, 'height': height, 'coordinate_system': 'top-left'},
            'image': info, 'geometry_reused': True, 'api_calls': 0,
            'status': 'geometric_candidates_only', 'candidates': candidates,
            'candidate_count': len(candidates), 'overlaps': overlaps, 'diagnostics': diagnostics,
            'existing_segmentation': segmentation, 'crop_regions': crop_regions,
            'counts': {'rectangles': len(geometry.get('rectangles', [])),
                       'lines': len(geometry.get('lines', [])),
                       'geometric_groups': len(geometry.get('regions', []))},
            'limitations': ['Semantic roles, text, arrows and branching require recognition.',
                            'Overlaps and nested candidates are preserved, not inferred as semantic hierarchy.',
                            'Text-like components are evidence, not a count of real text nodes.']}


def save_crops(image_path, plan, output_dir):
    """Reuse source identity, pixel-copy and no-overwrite checks from crop helper."""
    return save_region_crops(image_path, plan['crop_regions'], output_dir)


def remap_ir(ir, transform, canvas, *, region_transforms=None, connection_transforms=None):
    """Return a copy with declared boxes/path points in global coordinates.

    IDs, hierarchy, endpoint references, direction, branch labels and anchors are
    never rewritten. With multiple crop transforms, each geometric connection
    needs an external transform keyed by connection ID: path frames cannot be
    guessed from endpoint regions. This keeps crop bookkeeping outside the IR.
    This
    function translates known geometry, not arbitrary metadata; validate output
    with the independent IR validator before conversion.
    """
    _transform(transform)
    if not isinstance(ir, dict) or not isinstance(canvas, dict):
        raise ValueError('IR and destination canvas must be objects')
    transforms = {} if region_transforms is None else region_transforms
    if not isinstance(transforms, dict):
        raise ValueError('region_transforms must be an object')
    for value in transforms.values():
        _transform(value)
    edge_transforms = {} if connection_transforms is None else connection_transforms
    if not isinstance(edge_transforms, dict):
        raise ValueError('connection_transforms must be an object')
    for value in edge_transforms.values():
        _transform(value)
    result = deepcopy(ir)
    region_records = result.get('regions', [])
    if not isinstance(region_records, list) or any(not isinstance(r, dict) for r in region_records):
        raise ValueError('regions must be an array of objects')
    region_parents = {r.get('id'): r.get('parent_id') for r in region_records}

    def region_transform(identifier):
        # A crop transform applies to its descendants unless explicitly replaced.
        seen = set()
        while identifier is not None:
            if identifier in seen:
                raise ValueError('Cyclic region ancestry in coordinate mapping')
            seen.add(identifier)
            if identifier in transforms:
                return transforms[identifier]
            identifier = region_parents.get(identifier)
        return transform

    for collection in ('regions', 'elements', 'connections'):
        records = result.get(collection, [])
        if not isinstance(records, list):
            raise ValueError(collection + ' must be an array')
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(collection + ' entries must be objects')
            region_id = record.get('region_id')
            if collection == 'regions':
                region_id = record.get('id')
            if collection == 'connections':
                if (transforms and ('points' in record or 'label_bbox' in record)
                        and record.get('id') not in edge_transforms):
                    raise ValueError('Geometric connection needs an explicit connection transform for multiple crops')
                selected = edge_transforms.get(record.get('id'), transform)
            else:
                selected = region_transform(region_id)
            if 'bbox' in record:
                record['bbox'] = local_to_global_bbox(record['bbox'], selected)
            if 'label_bbox' in record:
                record['label_bbox'] = local_to_global_bbox(record['label_bbox'], selected)
            if 'points' in record:
                record['points'] = local_to_global_points(record['points'], selected)
    result['canvas'] = deepcopy(canvas)
    return result
