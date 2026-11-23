"""Evidence-based column crops using existing geometry, never VLM recognition.

The first supported automatic layout is separated, tall panel columns. Other
layouts are retained as one explicitly unsegmented image instead of guessing.
All boxes are source-pixel [x, y, width, height]; crops are never rescaled.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re


MAX_REGIONS = 16
MAX_PIXELS = 25_000_000


def _image_info(image_path):
    from PIL import Image
    path = Path(image_path).expanduser().resolve()
    with Image.open(path) as image:
        width, height = image.size
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Region crops require a single-frame image.')
        if image.getexif().get(274, 1) != 1:
            raise ValueError('Normalize EXIF rotation before source-pixel cropping.')
        if width < 1 or height < 1 or width * height > MAX_PIXELS:
            raise ValueError('Source image exceeds the supported pixel limit.')
    return {'path': str(path), 'width': width, 'height': height,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _integer_box(value, width, height):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(isinstance(v, bool) or not isinstance(v, int) for v in value)):
        raise ValueError('Pixel bbox must contain four integers [x,y,width,height].')
    x, y, w, h = value
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
        raise ValueError('Pixel bbox exceeds the source image or has nonpositive size.')
    return list(value)


def _geometry_box(value, width, height):
    if not isinstance(value, dict):
        raise ValueError('Geometry bbox_px must be an object.')
    box = [value.get(k) for k in ('x', 'y', 'width', 'height')]
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in box):
        raise ValueError('Geometry bbox_px must contain finite pixel values.')
    x, y, w, h = box
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
        raise ValueError('Geometry bbox_px is outside the image.')
    left, top = math.floor(x), math.floor(y)
    return [left, top, math.ceil(x + w) - left, math.ceil(y + h) - top]


def _intersection(a, b):
    return (max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
            * max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])))


def _union(boxes):
    x, y = min(b[0] for b in boxes), min(b[1] for b in boxes)
    return [x, y, max(b[0] + b[2] for b in boxes) - x,
            max(b[1] + b[3] for b in boxes) - y]


def _vertical_coverage(boxes):
    merged = []
    for low, high in sorted((b[1], b[1] + b[3]) for b in boxes):
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return sum(high - low for low, high in merged)


def _automatic_columns(geometry, info):
    width, height = info['width'], info['height']
    rectangles = geometry.get('rectangles', [])
    if not isinstance(rectangles, list) or len(rectangles) > 500:
        raise ValueError('Geometry rectangles must be a bounded array.')
    panels, ignored = [], []
    for index, record in enumerate(rectangles):
        if not isinstance(record, dict):
            ignored.append({'index': index, 'reason': 'not_an_object'})
            continue
        try:
            box = _geometry_box(record.get('bbox_px'), width, height)
        except ValueError as error:
            ignored.append({'index': index, 'reason': str(error)})
            continue
        if box[2] >= .15 * width and box[3] >= .20 * height and box[2] * box[3] >= .04 * width * height:
            panels.append({'id': str(record.get('id', 'candidate_' + str(index))), 'bbox': box})
    # Large outer panels have priority over enclosed detail boxes.
    panels.sort(key=lambda p: (-p['bbox'][2] * p['bbox'][3], p['bbox'][0], p['bbox'][1], p['id']))
    outer, suppressed = [], []
    for panel in panels:
        box = panel['bbox']
        if any(_intersection(box, other['bbox']) >= .98 * box[2] * box[3] for other in outer):
            suppressed.append(panel['id'])
        else:
            outer.append(panel)
    groups = []
    ambiguous = False
    for panel in sorted(outer, key=lambda p: (p['bbox'][0], p['bbox'][1], p['id'])):
        x, _, w, _ = panel['bbox']
        matches = []
        for index, group in enumerate(groups):
            gx, _, gw, _ = _union([p['bbox'] for p in group])
            overlap = max(0, min(x + w, gx + gw) - max(x, gx))
            if overlap >= .8 * min(w, gw) and abs((x + w / 2) - (gx + gw / 2)) <= .25 * max(w, gw):
                matches.append(index)
        if len(matches) > 1:
            ambiguous = True
            break
        if matches:
            groups[matches[0]].append(panel)
        else:
            groups.append([panel])
    groups.sort(key=lambda group: _union([p['bbox'] for p in group])[0])
    boxes = [_union([p['bbox'] for p in group]) for group in groups]
    coverage = sum(box[2] for box in boxes) / width
    vertical = [_vertical_coverage([p['bbox'] for p in group]) / height for group in groups]
    disjoint = all(a[0] + a[2] <= b[0] for a, b in zip(boxes, boxes[1:]))
    evidence = {
        'method': 'large_outer_panels_with_aligned_vertical_groups',
        'confidence_basis': 'heuristic geometric evidence, not a calibrated probability or semantic recognition',
        'panel_candidates': panels, 'suppressed_nested_candidate_ids': suppressed,
        'ignored_candidates': ignored,
        'column_groups': [{'member_ids': [p['id'] for p in group], 'bbox': box}
                          for group, box in zip(groups, boxes)],
        'horizontal_coverage_fraction': coverage, 'vertical_coverage_fractions': vertical,
        'thresholds': {'minimum_panel_width_fraction': .15, 'minimum_panel_height_fraction': .20,
                       'minimum_panel_area_fraction': .04, 'nested_containment_fraction': .98,
                       'aligned_x_overlap_fraction': .8, 'minimum_total_width_fraction': .60,
                       'minimum_column_vertical_coverage_fraction': .45},
        'limitation': 'Only strongly supported separated columns are split automatically; stacked or ambiguous layouts remain whole-image.',
    }
    supported = (not ambiguous and 2 <= len(groups) <= MAX_REGIONS and disjoint
                 and coverage >= .60 and all(value >= .45 for value in vertical))
    if not supported:
        evidence['fallback_reason'] = 'insufficient_or_ambiguous_separated_column_evidence'
        return [[0, 0, width, height]], 'whole_image_fallback', evidence
    boundaries = [0] + [(a[0] + a[2] + b[0]) // 2 for a, b in zip(boxes, boxes[1:])] + [width]
    cores = [[left, 0, right - left, height] for left, right in zip(boundaries, boundaries[1:])]
    evidence['boundary_rule'] = 'midpoints of measured inter-column gaps; full-height strips retain image margins without scaling'
    return cores, 'automatic_geometry_columns', evidence


def _records(boxes, info, source, evidence, overlap_px):
    width, height = info['width'], info['height']
    regions = []
    for index, box in enumerate(boxes, 1):
        x, y, w, h = _integer_box(box, width, height)
        left, top = max(0, x - overlap_px), max(0, y - overlap_px)
        right, bottom = min(width, x + w + overlap_px), min(height, y + h + overlap_px)
        basis = deepcopy(evidence)
        basis.update(image_sha256=info['sha256'], image_size=[width, height])
        regions.append({'region_id': 'region_%03d' % index, 'bbox': list(box),
                        'crop_bbox': [left, top, right - left, bottom - top],
                        'offset': [left, top], 'type': 'panel' if source != 'whole_image_fallback' else 'whole_image',
                        'coordinate_system': 'local', 'has_overlap': False,
                        'overlap_px_requested': overlap_px, 'source': source, 'evidence': basis})
    for index, region in enumerate(regions):
        region['has_overlap'] = any(_intersection(region['crop_bbox'], other['crop_bbox']) > 0
                                    for j, other in enumerate(regions) if j != index)
    return regions


def segment_regions(image_path, *, geometry=None, region_config=None, overlap_px=12):
    """Return source-bound local crop records without writing files.

    Provided geometry must match both the file SHA-256 and decoded dimensions.
    Without geometry, reuse the existing deterministic geometry analyzer once.
    A list of integer boxes (or a JSON file containing that list under regions)
    is an explicit, clearly labelled manual fallback; it is never automatic.
    """
    info = _image_info(image_path)
    if isinstance(overlap_px, bool) or not isinstance(overlap_px, int) or not 0 <= overlap_px <= 256:
        raise ValueError('Overlap padding must be an integer in [0,256] pixels.')
    if region_config is not None:
        config = region_config
        if isinstance(config, (str, Path)):
            config = json.loads(Path(config).read_text(encoding='utf-8-sig'))
        if isinstance(config, dict):
            config = config.get('regions')
        if not isinstance(config, list) or not 1 <= len(config) <= MAX_REGIONS:
            raise ValueError('Manual configuration needs 1 to 16 region boxes.')
        boxes = [_integer_box(item.get('bbox') if isinstance(item, dict) else item,
                              info['width'], info['height']) for item in config]
        if len({tuple(box) for box in boxes}) != len(boxes):
            raise ValueError('Manual configuration repeats a region box.')
        return _records(boxes, info, 'manual_configuration',
                        {'method': 'explicit_user_region_configuration',
                         'confidence_basis': 'human supplied, not automatic image analysis'}, overlap_px)
    supplied_geometry = geometry is not None
    if geometry is None:
        from geometry_analyzer import analyze_image
        geometry = analyze_image(info['path'])
    if not isinstance(geometry, dict) or not isinstance(geometry.get('image'), dict):
        raise ValueError('Geometry must include its source image metadata.')
    origin = geometry['image']
    if (origin.get('sha256') != info['sha256'] or origin.get('width') != info['width']
            or origin.get('height') != info['height']):
        raise ValueError('Geometry image hash/dimensions do not match the supplied image.')
    boxes, source, evidence = _automatic_columns(geometry, info)
    evidence['geometry_source'] = 'provided_hash_matched_geometry' if supplied_geometry else 'existing_geometry_analyzer'
    return _records(boxes, info, source, evidence, overlap_px)


def save_region_crops(image_path, regions, output_dir):
    """Write exact crops, preview and manifest into a fresh artifact namespace.

    Existing files are never overwritten. This consumes segment_regions output;
    the recorded image identity must still match when crops are materialized.
    """
    from PIL import Image, ImageDraw, ImageFont
    info = _image_info(image_path)
    if not isinstance(regions, list) or not 1 <= len(regions) <= MAX_REGIONS:
        raise ValueError('Expected 1 to 16 region records.')
    checked, seen = [], set()
    for original in regions:
        if not isinstance(original, dict):
            raise ValueError('A region must be an object.')
        record = deepcopy(original)
        name = record.get('region_id')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', name) or name in seen:
            raise ValueError('Region IDs must be unique simple identifiers.')
        seen.add(name)
        core = _integer_box(record.get('bbox'), info['width'], info['height'])
        crop = _integer_box(record.get('crop_bbox'), info['width'], info['height'])
        if (_intersection(core, crop) != core[2] * core[3] or record.get('offset') != crop[:2]
                or record.get('coordinate_system') != 'local'):
            raise ValueError('Crop must contain its core with a matching local offset.')
        basis = record.get('evidence', {})
        if (not isinstance(basis, dict) or basis.get('image_sha256') != info['sha256']
                or basis.get('image_size') != [info['width'], info['height']]):
            raise ValueError('Region records belong to a different source image.')
        record['crop_path'] = 'regions/' + name + '.png'
        checked.append(record)
    destination = Path(output_dir).expanduser().resolve()
    planned = [destination / 'region_manifest.json', destination / 'region_preview.png']
    planned += [destination / item['crop_path'] for item in checked]
    if any(path.exists() for path in planned):
        raise FileExistsError('Region output already exists; use a fresh directory.')
    (destination / 'regions').mkdir(parents=True, exist_ok=True)
    with Image.open(info['path']) as image:
        image.load()
        preview = image.convert('RGB')
        for record in checked:
            x, y, w, h = record['crop_bbox']
            crop = image.crop((x, y, x + w, y + h))
            with (destination / record['crop_path']).open('xb') as handle:
                crop.save(handle, format='PNG')
    draw = ImageDraw.Draw(preview)
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()
    colors = ('#cc3311', '#0077bb', '#228833', '#aa3377', '#ee7733', '#009988')
    for index, record in enumerate(checked):
        x, y, w, h = record['crop_bbox']
        color = colors[index % len(colors)]
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=color, width=2)
        x, y, w, h = record['bbox']
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=color, width=4)
        draw.text((x + 5, y + 5), record['region_id'], fill=color, font=font,
                  stroke_width=2, stroke_fill='white')
    with (destination / 'region_preview.png').open('xb') as handle:
        preview.save(handle, format='PNG')
    unchanged = hashlib.sha256(Path(info['path']).read_bytes()).hexdigest() == info['sha256']
    if not unchanged:
        raise ValueError('Source image changed while saving region crops.')
    sources = {record.get('source') for record in checked}
    automatic = sources == {'automatic_geometry_columns'}
    status = ('automatic_columns' if automatic else 'manual_configuration'
              if sources == {'manual_configuration'} else 'insufficient_evidence_whole_image')
    manifest = {'image': info, 'regions': checked, 'automatic': automatic,
                'segmentation_status': status, 'manual_configuration_used': 'manual_configuration' in sources,
                'original_image_unchanged': unchanged, 'crop_scaling': 'none_original_pixels',
                'preview': 'region_preview.png', 'vlm_called': False,
                'interpretation': 'Heuristic layout regions, not recognized semantic contents. Local coordinates start at crop_bbox top-left.'}
    with (destination / 'region_manifest.json').open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    return manifest
