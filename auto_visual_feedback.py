"""Bounded image-space evidence, never a semantic correctness certificate.

The rendered page is letterboxed into the original canvas with one uniform
scale. We deliberately do not deform/crop content to improve a similarity score.
Page margins and registration error can therefore lower these coarse metrics.
"""
from pathlib import Path
import math

from image_compare import _read_rgb


MAX_SIDE = 1600
EDGE_TOLERANCE_PX = 2


def _dependencies():
    import cv2
    import numpy as np
    return cv2, np


def _fit(image, size):
    from PIL import Image, ImageOps
    fitted = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    offset = ((size[0]-fitted.width)//2, (size[1]-fitted.height)//2)
    canvas = Image.new('RGB', size, 'white')
    canvas.paste(fitted, offset)
    return canvas, {'uniform_scale': min(size[0]/image.width, size[1]/image.height),
                    'offset_px': list(offset), 'fitted_size': list(fitted.size)}


def _foreground(array, np):
    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]))
    background = np.median(border, axis=0)
    mask = np.max(np.abs(array.astype(float)-background), axis=2) > 18
    ys, xs = np.where(mask)
    bounds = ([int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)]
              if len(xs) else None)
    return mask, {'background_rgb': [round(float(v), 2) for v in background],
                  'foreground_mean_rgb': [round(float(v), 2) for v in array[mask].mean(axis=0)] if mask.any() else None,
                  'foreground_fraction': round(float(mask.mean()), 6), 'foreground_bbox_px': bounds}


def _coverage(edges, other_near, np):
    count = int(np.count_nonzero(edges))
    return round(float(np.count_nonzero(edges & other_near))/count, 6) if count else None


def _region_boxes(scene):
    for node in scene.get('nodes', []):
        if 'bbox' not in node:
            continue
        x, y, w, h = node['bbox']
        if node.get('type') == 'tensor3d' and node.get('depth'):
            dx, dy = node['depth']['dx'], node['depth']['dy']
            x, y, w, h = min(x, x+dx), min(y, y+dy), w+abs(dx), h+abs(dy)
        yield 'node', node['id'], [x, y, w, h]
    for label in scene.get('labels', []):
        box = label.get('metadata', {}).get('bbox')
        if box:
            yield 'label', label['id'], box
    for edge in scene.get('edges', []):
        points = edge.get('points')
        if points:
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            yield 'edge', edge['id'], [min(xs)-3, min(ys)-3, max(xs)-min(xs)+6, max(ys)-min(ys)+6]


def compare_visual(original_path, rendered_path, scene, output_dir):
    """Return diagnostics and save visual_diff.png; never change Scene coordinates.

    Region results sample supplied Scene locations in both images. They are not
    independent object/text detections. All outputs remain unverified.
    """
    limitations = [
        'No OCR: Chinese content and mathematical equivalence are not verified.',
        'Edges and foreground statistics do not verify topology, tensor depth or grid counts.',
        'Region IDs come from the proposed Scene, not independent image recognition.',
        'Only aspect-preserving page alignment is used; margins/residual offsets can affect all metrics.',
        'No automatic Scene edits are justified by these pixel metrics alone.',
    ]
    try:
        cv2, np = _dependencies()
    except ImportError:
        return {'status': 'unavailable', 'reason': 'OpenCV/numpy not installed; install existing requirements-geometry.txt if needed.',
                'needs_review': True, 'limitations': limitations}
    original, rendered = _read_rgb(original_path), _read_rgb(rendered_path)
    canvas = scene.get('canvas', {})
    if (canvas.get('width'), canvas.get('height')) != original.size:
        raise ValueError('Visual comparison requires Scene coordinates matching the original image dimensions.')
    scale = min(1.0, MAX_SIDE/max(original.size))
    size = (max(1, round(original.width*scale)), max(1, round(original.height*scale)))
    first, original_transform = _fit(original, size)
    second, rendered_transform = _fit(rendered, size)
    a, b = np.array(first), np.array(second)
    edges_a = cv2.Canny(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), 80, 180) > 0
    edges_b = cv2.Canny(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), 80, 180) > 0
    kernel = np.ones((2*EDGE_TOLERANCE_PX+1,)*2, dtype=np.uint8)
    near_a = cv2.dilate(edges_a.astype(np.uint8), kernel) > 0
    near_b = cv2.dilate(edges_b.astype(np.uint8), kernel) > 0
    foreground_a, appearance_a = _foreground(a, np)
    foreground_b, appearance_b = _foreground(b, np)
    region_results = []
    for kind, identifier, box in _region_boxes(scene):
        if len(box) != 4 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
            continue
        x, y, w, h = box
        x1, y1 = max(0, math.floor(x*scale)), max(0, math.floor(y*scale))
        x2, y2 = min(size[0], math.ceil((x+w)*scale)), min(size[1], math.ceil((y+h)*scale))
        if x2 <= x1 or y2 <= y1:
            continue
        roi = (slice(y1, y2), slice(x1, x2))
        coverage = _coverage(edges_a[roi], near_b[roi], np)
        reverse = _coverage(edges_b[roi], near_a[roi], np)
        region_results.append({'kind': kind, 'id': identifier, 'bbox_original_px': box,
                               'original_edge_coverage': coverage, 'rendered_edge_coverage': reverse,
                               'original_foreground_fraction': round(float(foreground_a[roi].mean()), 6),
                               'rendered_foreground_fraction': round(float(foreground_b[roi].mean()), 6),
                               'risk': 'insufficient_edges' if coverage is None or reverse is None else
                               'possible_local_difference' if min(coverage, reverse) < .65 else 'not_semantically_verified'})
    # Color-coded geometric differences, no claim to recognize the missing item.
    diff = (a.astype(float)*.3+255*.7).astype(np.uint8)
    diff[edges_a & near_b] = [25, 150, 60]
    diff[edges_a & ~near_b] = [230, 50, 40]
    diff[edges_b & ~near_a] = [30, 130, 230]
    from PIL import Image
    destination = Path(output_dir)/'visual_diff.png'
    if destination.resolve() in (Path(original_path).resolve(), Path(rendered_path).resolve()):
        raise ValueError('Visual output must not replace an input image.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        Image.fromarray(diff).save(stream, format='PNG')
    return {'status': 'compared_unverified', 'needs_review': True,
            'registration': {'method': 'uniform_page_letterbox_no_content_warp', 'size_px': list(size),
                             'original': original_transform, 'rendered': rendered_transform,
                             'residual_alignment_error_px': None,
                             'residual_status': 'not_estimated', 'foreground_bbox_registration': False},
            'geometry': {'original_edge_coverage': _coverage(edges_a, near_b, np),
                         'rendered_edge_coverage': _coverage(edges_b, near_a, np),
                         'original_edge_pixels': int(edges_a.sum()), 'rendered_edge_pixels': int(edges_b.sum()),
                         'tolerance_in_comparison_pixels': EDGE_TOLERANCE_PX},
            'appearance': {'original': appearance_a, 'rendered': appearance_b},
            'regions': region_results,
            'unassessed_label_ids': [label['id'] for label in scene.get('labels', []) if not label.get('metadata', {}).get('bbox')],
            'visual_diff': str(destination),
            'diff_legend': {'green': 'nearby edges', 'red': 'original edge without nearby rendered edge',
                            'blue': 'rendered edge without nearby original edge'},
            'semantic_checks': {name: 'unverified' for name in
                                ('chinese_text', 'math_content', 'connection_topology', 'tensor_projection', 'matrix_grid')},
            'limitations': limitations}
