"""Evidence-gated local geometry repair; never infer text or connection semantics.

Observations are reviewed measurements, not model confidence or geometry-only
semantic guesses. The caller verifies the image hash against the source bytes.
No filesystem, network, renderer or configuration access occurs here.
"""
from copy import deepcopy
import hashlib
import json
import re

from .regions import _box
from .validator import inspect_ir, _contains


def ir_fingerprint(ir):
    return hashlib.sha256(json.dumps(ir, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _intersection(a, b):
    return max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))


def _union(a, b):
    x, y = min(a[0], b[0]), min(a[1], b[1])
    return [x, y, max(a[0]+a[2], b[0]+b[2])-x, max(a[1]+a[3], b[1]+b[3])-y]


def _error_keys(validation):
    return {(e['path'], e['code']) for e in validation['errors']}


def _geometry_diff(before, after):
    changes = []
    for collection, fields in [('elements', ('bbox', 'region_id')), ('regions', ('bbox', 'children'))]:
        for index, (old, new) in enumerate(zip(before[collection], after[collection])):
            for field in fields:
                if old.get(field) != new.get(field):
                    changes.append({'path': f'/{collection}/{index}/{field}', 'id': old['id'],
                                    'before': deepcopy(old.get(field)), 'after': deepcopy(new.get(field))})
    return changes


def repair_geometry(ir, evidence=None, *, image_sha256):
    """Return an independent IR and audit report. Uncertain proposals stay unresolved.

Evidence binds image_sha256, ir_sha256, coordinate_system, canvas_size and an
observations array. Each observation requires element_id, expected_bbox,
expected_region_id, confirmed=True and a nonempty basis. Supported actions:

* reassign: optional measured_bbox, ownership='unique_region' or 'unassigned'.
  Unique means exactly one containing region. Unassigned additionally requires
  page_level=True and zero overlap with every existing region.
* measure_bbox: replace only the box, preserving its current owner.
* expand_region: verified_owner=True and region_support_bbox bound the measured
  support. Use only the minimal union of the existing owner and the element;
  do not introduce additional overlap with unrelated regions.

Every observation commits atomically only if it introduces no new validation
errors. Input text, IDs, ordering, types and the full connection graph are fixed.
This is not an automatic image recognizer or a general JSON patch interface.
"""
    original = deepcopy(ir)
    result = deepcopy(ir)
    before = inspect_ir(ir)
    report = {'status': 'unresolved', 'before_validation': before, 'after_validation': before,
              'changes': [], 'unresolved': [], 'api_calls': 0, 'semantic_content_preserved': True}

    def unresolved(code, message, index=None):
        report['unresolved'].append({'code': code, 'message': message, 'observation_index': index})

    # Only element geometry/ownership failures are repairable by this adapter.
    if any(not re.fullmatch(r'/elements/\d+/(bbox|region_id)', e['path']) or
           e['code'] not in {'outside_region', 'out_of_bounds', 'invalid_reference'} for e in before['errors']):
        unresolved('unsupported_input_error', 'Non-local or structural errors require separate review.')
        return result, report
    if evidence is None:
        if before['valid']:
            report['status'] = 'unchanged'
        else:
            unresolved('insufficient_evidence', 'No reviewed source measurements supplied.')
        return result, report
    try:
        if not isinstance(evidence, dict):
            raise ValueError('Evidence must be an object.')
        if not isinstance(image_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', image_sha256):
            raise ValueError('Expected a source image SHA-256.')
        if evidence.get('image_sha256') != image_sha256 or evidence.get('ir_sha256') != ir_fingerprint(ir):
            raise ValueError('Source image or IR fingerprint mismatch.')
        canvas = ir['canvas']
        if evidence.get('coordinate_system') != canvas['coordinate_system'] or evidence.get('canvas_size') != [canvas['width'], canvas['height']]:
            raise ValueError('Evidence must use the original canvas coordinates.')
        observations = evidence.get('observations')
        if not isinstance(observations, list) or len(observations) > 16:
            raise ValueError('Evidence requires at most 16 local observations.')
    except (ValueError, KeyError, TypeError) as exc:
        unresolved('invalid_evidence', str(exc))
        return result, report
    extent = [0, 0, canvas['width'], canvas['height']]
    for index, observation in enumerate(observations):
        try:
            if not isinstance(observation, dict) or observation.get('confirmed') is not True or not isinstance(observation.get('basis'), str) or not observation['basis'].strip():
                raise ValueError('A confirmed observation and review basis are required.')
            trial = deepcopy(result)
            element = next((e for e in trial['elements'] if e['id'] == observation.get('element_id')), None)
            if element is None:
                raise ValueError('Observation references an unknown element.')
            if ('expected_bbox' not in observation or 'expected_region_id' not in observation or
                    element['bbox'] != observation['expected_bbox'] or element['region_id'] != observation['expected_region_id']):
                raise ValueError('Original geometry preconditions do not match.')
            action = observation.get('action')
            regions = {r['id']: r for r in trial['regions']}
            if action in {'reassign', 'measure_bbox'}:
                box = _box(observation.get('measured_bbox', element['bbox']))
                if not _contains(extent, box):
                    raise ValueError('Measured box is outside the image; no clipping or guessing.')
                element['bbox'] = box
                if action == 'reassign':
                    containing = [r['id'] for r in trial['regions'] if _contains(r['bbox'], box)]
                    if observation.get('ownership') == 'unique_region':
                        if len(containing) != 1:
                            raise ValueError('Region assignment is ambiguous or has no containing region.')
                        owner = containing[0]
                    elif observation.get('ownership') == 'unassigned':
                        if observation.get('page_level') is not True or any(_intersection(r['bbox'], box) for r in trial['regions']):
                            raise ValueError('Page-level ownership needs explicit review and no region overlap.')
                        owner = None
                    else:
                        raise ValueError('Unknown ownership evidence.')
                    previous_owner = element['region_id']
                    element['region_id'] = owner
                    if previous_owner != owner:
                        if previous_owner in regions and 'children' in regions[previous_owner]:
                            regions[previous_owner]['children'] = [c for c in regions[previous_owner]['children'] if c != element['id']]
                        if owner in regions and 'children' in regions[owner]:
                            if element['id'] not in regions[owner]['children']:
                                regions[owner]['children'].append(element['id'])
            elif action == 'expand_region':
                if observation.get('verified_owner') is not True or element['region_id'] not in regions:
                    raise ValueError('Expansion requires independently verified existing ownership.')
                region = regions[element['region_id']]
                expanded = _union(region['bbox'], element['bbox'])
                support = _box(observation.get('region_support_bbox'))
                if not _contains(extent, support) or not _contains(support, expanded):
                    raise ValueError('Expansion exceeds verified region support or image bounds.')
                def ancestor(first, second):
                    seen = set()
                    while second in regions and second not in seen:
                        if first == second:
                            return True
                        seen.add(second)
                        second = regions[second]['parent_id']
                    return False
                for other in trial['regions']:
                    if ancestor(region['id'], other['id']) or ancestor(other['id'], region['id']):
                        continue
                    if _intersection(expanded, other['bbox']) > _intersection(region['bbox'], other['bbox']):
                        raise ValueError('Minimal expansion would create or increase overlap with another region.')
                region['bbox'] = expanded
            else:
                raise ValueError('Unsupported repair action.')
            after = inspect_ir(trial)
            if _error_keys(after) - _error_keys(inspect_ir(result)):
                raise ValueError('Proposal introduces new validation errors; atomic change rejected.')
            changes = _geometry_diff(result, trial)
            for change in changes:
                change.update(observation_index=index, basis=observation['basis'])
            report['changes'].extend(changes)
            result = trial
        except (ValueError, KeyError, TypeError) as exc:
            unresolved('proposal_rejected', str(exc), index)
    after = inspect_ir(result)
    report['after_validation'] = after
    report['remaining_errors'] = deepcopy(after['errors'])
    # Exhaustively mask allowed geometry fields and compare every other value.
    fixed = deepcopy(result)
    for collection, fields in [('elements', ('bbox', 'region_id')), ('regions', ('bbox', 'children'))]:
        for old, new in zip(original[collection], fixed[collection]):
            for field in fields:
                if field in old:
                    new[field] = deepcopy(old[field])
                else:
                    new.pop(field, None)
    if fixed != original:
        raise AssertionError('Geometry repair changed semantic content.')
    report['status'] = ('repaired' if report['changes'] else 'unchanged') if after['valid'] and not report['unresolved'] else 'unresolved'
    return result, report
