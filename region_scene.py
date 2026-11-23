"""Local Scene envelopes; drawing objects always use the existing Scene schema.

Coordinates are unscaled crop pixels. A namespaced reference is a pair of IDs
in this intermediate envelope, never a slash-containing formal Scene ID.
Unknown cross-region endpoints stay outside the drawable edges array.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re

from structured_tikz.validator import validate_scene

_ID = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,63}')


def _identifier(value, location):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(location + ' must be a Scene-compatible identifier.')


def _integer(value, location, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0):
        raise ValueError(location + ' must be a nonnegative integer pixel coordinate.')


def validate_manifest(manifest):
    """Check the crop-to-original contract without reading or modifying files."""
    if not isinstance(manifest, dict) or not isinstance(manifest.get('image'), dict):
        raise ValueError('Region manifest needs image metadata.')
    image = manifest['image']
    width, height = image.get('width'), image.get('height')
    _integer(width, 'image.width', positive=True)
    _integer(height, 'image.height', positive=True)
    if max(width, height) > 20000 or width * height > 25_000_000:
        raise ValueError('Image exceeds existing Scene/pixel limits.')
    if 'sha256' in image and (not isinstance(image['sha256'], str)
                             or not re.fullmatch(r'[0-9a-f]{64}', image['sha256'])):
        raise ValueError('Source SHA-256 must be 64 lowercase hexadecimal characters.')
    regions = manifest.get('regions')
    if not isinstance(regions, list) or not 1 <= len(regions) <= 32:
        raise ValueError('Manifest needs 1..32 regions.')
    seen = set()
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError('Each region must be an object.')
        identifier = region.get('region_id')
        _identifier(identifier, 'region_id')
        if identifier in seen:
            raise ValueError('Duplicate region ID: ' + identifier)
        seen.add(identifier)
        box = region.get('crop_bbox')
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError('crop_bbox must be [x,y,width,height].')
        for index, value in enumerate(box):
            _integer(value, 'crop_bbox', positive=index >= 2)
        x, y, w, h = box
        if x + w > width or y + h > height:
            raise ValueError('Region crop extends beyond the original image.')
        if region.get('offset') != [x, y]:
            raise ValueError('Region offset must equal the actual crop origin.')
        core = region.get('bbox', box)
        if not isinstance(core, list) or len(core) != 4:
            raise ValueError('Region bbox must be [x,y,width,height].')
        for index, value in enumerate(core):
            _integer(value, 'bbox', positive=index >= 2)
        if not (x <= core[0] and y <= core[1] and core[0] + core[2] <= x + w
                and core[1] + core[3] <= y + h):
            raise ValueError('Core region must be contained in its crop.')
    return manifest


def validate_region_scene(bundle, region):
    """Validate local drawing plus unresolved relations; no inferred endpoints."""
    if not isinstance(bundle, dict) or set(bundle) - {'region_id', 'scene', 'pending_connections', 'metadata'}:
        raise ValueError('Unsupported local Scene envelope fields.')
    if bundle.get('region_id') != region.get('region_id'):
        raise ValueError('Local Scene region ID differs from its crop manifest.')
    _identifier(bundle['region_id'], 'region_id')
    scene = bundle.get('scene')
    validate_scene(scene)
    if [scene['canvas']['width'], scene['canvas']['height']] != region['crop_bbox'][2:]:
        raise ValueError('Local Scene canvas must match unscaled crop dimensions.')
    if not isinstance(bundle.get('metadata', {}), dict):
        raise ValueError('Local envelope metadata must be an object.')
    pending = bundle.get('pending_connections', [])
    if not isinstance(pending, list) or len(pending) > 500:
        raise ValueError('pending_connections must contain at most 500 entries.')
    seen = {item['id'] for kind in ('nodes', 'edges', 'labels') for item in scene[kind]}
    for relation in pending:
        if not isinstance(relation, dict):
            raise ValueError('Pending connection must be an object.')
        _identifier(relation.get('id'), 'pending connection id')
        if relation['id'] in seen:
            raise ValueError('Pending connection ID collides with another local object.')
        seen.add(relation['id'])
        for field in ('source', 'target'):
            endpoint = relation.get(field)
            if endpoint is None:
                continue
            if not isinstance(endpoint, dict) or set(endpoint) != {'region_id', 'id'}:
                raise ValueError('Cross-region references need {region_id,id} or null.')
            _identifier(endpoint['region_id'], field + '.region_id')
            _identifier(endpoint['id'], field + '.id')
        if not isinstance(relation.get('evidence', {}), dict):
            raise ValueError('Pending evidence must be an object.')
    json.dumps(bundle, allow_nan=False)
    return bundle


def make_region_scene(region, scene, *, pending_connections=None, metadata=None):
    """Wrap an existing local Scene without mutating the caller's values."""
    bundle = {'region_id': region['region_id'], 'scene': deepcopy(scene),
              'pending_connections': deepcopy(pending_connections or []),
              'metadata': deepcopy(metadata or {})}
    validate_region_scene(bundle, region)
    return bundle
