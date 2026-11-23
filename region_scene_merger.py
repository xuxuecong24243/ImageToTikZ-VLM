"""Deterministic crop-local Scene merge; no model, OCR or compilation calls.

Only explicit, evidenced cross-region references become drawable edges. Original
metadata remains auditable in local-coordinate provenance, never guessed global.
"""
from copy import deepcopy
import hashlib
import json
import math
import re

from region_scene import validate_manifest, validate_region_scene
from structured_tikz.validator import validate_scene, resolve_style


BOX_KEYS = {'bbox', 'expected_bbox', 'model_proposed_bbox', 'bbox_scene'}
POSITION_KEYS = {'position', 'original_position'}
REF_KEYS = {'node_id', 'candidate_id', 'source_id', 'target_id', 'owner_id', 'object_id'}
SCOPED_KEYS = {'region_local_metadata', 'region_provenance'}


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _identifier(kind, region, local):
    stem = re.sub('[^A-Za-z0-9_]', '_', region + '_' + local)[:42]
    digest = hashlib.sha256(_encoded([kind, region, local]).encode('utf-8')).hexdigest()[:12]
    return kind + '_' + stem + '_' + digest


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _box(value):
    return (isinstance(value, list) and len(value) == 4 and all(_finite(v) for v in value)
            and value[2] > 0 and value[3] > 0)


def _area_intersection(a, b):
    return max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(
        0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))


def _move_box(value, dx, dy):
    if not _box(value):
        raise ValueError('Declared metadata bbox must be finite [x,y,width,height].')
    return [value[0]+dx, value[1]+dy, value[2], value[3]]


def _move_points(value, dx, dy):
    if not isinstance(value, list) or any(not isinstance(p, list) or len(p) != 2
            or not all(_finite(v) for v in p) for p in value):
        raise ValueError('Declared metadata points must be finite [x,y] pairs.')
    return [[p[0]+dx, p[1]+dy] for p in value]


def _move_metadata(metadata, dx, dy, region, identifier, report):
    """Translate declared absolute fields; scope ambiguous fields as local."""
    if not isinstance(metadata, dict):
        raise ValueError('Object metadata must be an object.')
    result, deferred = {}, {}
    for key, value in metadata.items():
        if key in SCOPED_KEYS:
            deferred[key] = deepcopy(value)
        elif key in BOX_KEYS and value is not None:
            result[key] = _move_box(value, dx, dy)
        elif key == 'line_boxes_scene':
            if not isinstance(value, list):
                raise ValueError('line_boxes_scene must be an array.')
            result[key] = [_move_box(box, dx, dy) for box in value]
        elif key in POSITION_KEYS and value is not None:
            if not isinstance(value, dict) or not all(_finite(value.get(k)) for k in ('x', 'y')):
                raise ValueError('Declared metadata position needs finite x/y.')
            result[key] = dict(deepcopy(value), x=value['x']+dx, y=value['y']+dy)
        elif key == 'points':
            result[key] = _move_points(value, dx, dy)
        elif key in {'pending_edges', 'pending_connections'}:
            deferred[key] = deepcopy(value)
        elif re.search(r'bbox|bounds|position|coordinate|centroid|(^|_)center|(^|_)points|^x$|^y$', key):
            deferred[key] = deepcopy(value)
        elif isinstance(value, dict):
            result[key] = _move_metadata(value, dx, dy, region, identifier, report)
        elif isinstance(value, list) and any(isinstance(v, dict) for v in value):
            # Unknown arrays may mix coordinate/reference conventions.
            deferred[key] = deepcopy(value)
        else:
            result[key] = deepcopy(value)
    if deferred:
        result['region_local_metadata'] = {'region_id': region, 'coordinate_system': 'local', 'values': deferred}
        report['review_required'].append({'region_id': region, 'local_id': identifier,
            'reason': 'uninterpreted_metadata_preserved_in_local_scope', 'fields': sorted(deferred)})
    return result


def _map_metadata(metadata, region, id_map, report, identifier):
    result, unresolved = {}, {}
    for key, value in metadata.items():
        if key in SCOPED_KEYS:
            result[key] = deepcopy(value)
        elif key in REF_KEYS:
            if value is None:
                result[key] = None
            elif isinstance(value, str) and value in id_map[region]:
                result[key] = id_map[region][value]
            else:
                unresolved[key] = deepcopy(value)
        elif key.endswith('_id') or key.endswith('_ids'):
            # No knowledge of an arbitrary reference domain; retain but scope it.
            unresolved[key] = deepcopy(value)
        elif isinstance(value, dict):
            result[key] = _map_metadata(value, region, id_map, report, identifier)
        else:
            result[key] = deepcopy(value)
    if unresolved:
        local = result.setdefault('region_local_metadata', {'region_id': region, 'coordinate_system': 'local', 'values': {}})
        local['values'].update(unresolved)
        report['review_required'].append({'region_id': region, 'local_id': identifier,
            'reason': 'unresolved_metadata_references', 'references': unresolved})
    return result


def _reference_values(metadata):
    result = {}
    for key, value in metadata.items():
        if key in SCOPED_KEYS:
            continue
        if key in REF_KEYS:
            result[key] = value
        elif isinstance(value, dict):
            nested = _reference_values(value)
            if nested:
                result[key] = nested
    return result


def _has_local_metadata(value):
    if isinstance(value, dict):
        return 'region_local_metadata' in value or any(_has_local_metadata(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_local_metadata(v) for v in value)
    return False


def _spatially_close(a, b, collection):
    """Possible overlap only, never sufficient evidence to remove an object."""
    if collection == 'nodes':
        first, second = a['bbox'], b['bbox']
        if any(abs(x-y) > 1 for x,y in zip(first,second)):
            return False
        intersection = _area_intersection(first,second)
        return intersection / (first[2]*first[3]+second[2]*second[3]-intersection) >= .95
    if collection == 'labels':
        return math.dist([a['x'],a['y']],[b['x'],b['y']]) <= 1
    if 'points' in a and 'points' in b:
        return len(a['points']) == len(b['points']) and all(
            math.dist(x,y) <= 1 for x,y in zip(a['points'],b['points']))
    if 'points' not in a and 'points' not in b:
        return (a['from'],a['to'],a.get('from_anchor','east'),a.get('to_anchor','west')) == (
            b['from'],b['to'],b.get('from_anchor','east'),b.get('to_anchor','west'))
    return False


def _equivalent(a, b, collection, scene):
    """Conservative equality: no text normalization or semantic guessing."""
    if _has_local_metadata(a.get('metadata',{})) or _has_local_metadata(b.get('metadata',{})):
        return False
    if collection == 'nodes':
        first, second = a['bbox'], b['bbox']
        if any(abs(x-y) > 1 for x, y in zip(first, second)):
            return False
        intersection = _area_intersection(first, second)
        union = first[2]*first[3] + second[2]*second[3] - intersection
        if intersection / union < .95:
            return False
        ignored = {'id', 'metadata', 'style', 'bbox'}
    elif collection == 'labels':
        if math.dist([a['x'], a['y']], [b['x'], b['y']]) > 1:
            return False
        first, second = a.get('metadata', {}).get('bbox'), b.get('metadata', {}).get('bbox')
        if (first is None) != (second is None):
            return False
        if first is not None and any(abs(x-y) > 1 for x, y in zip(first, second)):
            return False
        ignored = {'id', 'metadata', 'style', 'x', 'y'}
    else:
        ignored = {'id', 'metadata', 'style'}
    return ({k:v for k,v in a.items() if k not in ignored} == {k:v for k,v in b.items() if k not in ignored}
        and resolve_style(scene, a) == resolve_style(scene, b)
        and _reference_values(a.get('metadata', {})) == _reference_values(b.get('metadata', {})))


def _anchor(node, name):
    x, y, w, h = node['bbox']
    return {'north':[x+w/2,y], 'south':[x+w/2,y+h], 'east':[x+w,y+h/2], 'west':[x,y+h/2]}.get(name)


def merge_region_scenes(manifest, bundles):
    """Return a new existing-schema Scene and a full ID/provenance audit."""
    validate_manifest(manifest)
    _encoded(bundles)
    regions = {r['region_id']:r for r in manifest['regions']}
    if not isinstance(bundles, list) or len(bundles) != len(regions):
        raise ValueError('Supply exactly one local Scene for every manifest region.')
    supplied = {}
    for bundle in bundles:
        rid = bundle.get('region_id')
        if rid not in regions or rid in supplied:
            raise ValueError('Unknown or repeated local Scene region.')
        validate_region_scene(bundle, regions[rid])
        supplied[rid] = bundle
    if set(supplied) != set(regions):
        raise ValueError('Local Scene coverage is incomplete.')
    scales = {b['scene']['canvas'].get('scale', .4) for b in bundles}
    if len(scales) != 1:
        raise ValueError('Local canvas scales differ; merge never silently rescales text or geometry.')
    if sum(len(b['scene'][k]) for b in bundles for k in ('nodes','labels','edges')) > 2000:
        raise ValueError('Local inputs exceed the 2000-object merge resource bound.')
    ordered = sorted(supplied)
    scene = {'canvas':{'width':manifest['image']['width'],'height':manifest['image']['height'],
             'coordinate_system':'top-left','scale':next(iter(scales))},
             'styles':{}, 'nodes':[], 'labels':[], 'edges':[], 'metadata':{
                 'source':'offline_region_scene_merge','real_vlm_recognition_verified':False,
                 'coordinate_system':'global','region_sources':[]}}
    report = {'id_map':{r:{} for r in ordered},'style_map':{r:{} for r in ordered},
              'deduplicated':[],'review_required':[],'pending_connections':[], 'resolved_connections':[],
              'api_called':False,'recognition_quality':'not_tested_by_merge'}
    id_map, style_map, used, style_content = report['id_map'], report['style_map'], set(), {}
    def allocate(kind, rid, local):
        name = _identifier(kind, rid, local)
        if name in used:
            raise ValueError('Generated identifier collision; refusing overwrite.')
        used.add(name)
        return name
    for rid in ordered:
        bundle, local = supplied[rid], supplied[rid]['scene']
        scene['metadata']['region_sources'].append({'region_id':rid,'coordinate_system':'local',
            'scene_metadata':deepcopy(local.get('metadata',{})), 'bundle_metadata':deepcopy(bundle.get('metadata',{}))})
        for field in ('pending_edges','pending_connections'):
            if local.get('metadata',{}).get(field):
                report['pending_connections'].append({'region_id':rid,'reason':'legacy_local_metadata_requires_explicit_resolution',
                    'field':field,'coordinate_system':'local','value':deepcopy(local['metadata'][field])})
        for name, value in sorted(local['styles'].items()):
            key = _encoded(value)
            if key not in style_content:
                global_name = allocate('s', rid, name)
                style_content[key] = global_name
                scene['styles'][global_name] = deepcopy(value)
            style_map[rid][name] = style_content[key]
        for collection in ('nodes','labels','edges'):
            for item in local[collection]:
                id_map[rid][item['id']] = allocate('o', rid, item['id'])
    origins = {}
    # Nodes must be canonical before label ownership and edge endpoints are mapped.
    for collection in ('nodes','labels','edges'):
        staged = []
        for rid in ordered:
            dx, dy = regions[rid]['offset']
            for raw in supplied[rid]['scene'][collection]:
                item = deepcopy(raw); item['id'] = id_map[rid][raw['id']]
                if isinstance(item.get('style'),str):
                    item['style'] = style_map[rid][item['style']]
                if collection == 'nodes':
                    item['bbox'] = _move_box(item['bbox'], dx, dy)
                elif collection == 'labels':
                    item['x'] += dx; item['y'] += dy
                else:
                    if 'points' in item:
                        item['points'] = _move_points(item['points'], dx, dy)
                    else:
                        item['from'] = id_map[rid][raw['from']]
                        item['to'] = id_map[rid][raw['to']]
                meta = _move_metadata(item.get('metadata',{}), dx, dy, rid, raw['id'], report)
                item['metadata'] = _map_metadata(meta, rid, id_map, report, raw['id'])
                staged.append((rid, raw['id'], item))
        for rid, local_id, item in staged:
            ambiguous_local = any(r == rid and other is not item and _equivalent(item,other,collection,scene)
                                  for r,_,other in staged)
            overlapping_prior = [prior for prior in scene[collection]
                if rid not in origins[prior['id']]
                and any(_area_intersection(regions[rid]['crop_bbox'],regions[r]['crop_bbox']) > 0
                        for r in origins[prior['id']])]
            matches = [prior for prior in overlapping_prior if _equivalent(item,prior,collection,scene)]
            conflicts = [prior['id'] for prior in overlapping_prior
                         if _spatially_close(item,prior,collection) and not _equivalent(item,prior,collection,scene)]
            source = {'region_id':rid,'local_id':local_id,'coordinate_system':'local',
                      'source':supplied[rid].get('metadata',{}).get('source','unverified_local_scene')}
            if conflicts:
                report['review_required'].append(dict(source,collection=collection,
                    reason='possible_duplicate_conflicting_content_or_style',candidate_global_ids=conflicts,
                    interpretation='Spatial proximity only; content, style, references or unscoped evidence differ.'))
            if len(matches) == 1 and not ambiguous_local:
                canonical = matches[0]
                id_map[rid][local_id] = canonical['id']
                origins[canonical['id']].add(rid)
                canonical['metadata']['region_provenance'].append(source)
                report['deduplicated'].append(dict(source, collection=collection,global_id=canonical['id'],
                    reason='overlapping_crops_matching_geometry_content_style_and_references'))
            else:
                if matches or ambiguous_local:
                    report['review_required'].append(dict(source,reason='ambiguous_duplicate_preserved'))
                item['metadata']['region_provenance'] = [source]
                scene[collection].append(item); origins[item['id']] = {rid}
    # References in earlier node metadata may point to a node canonicalized later.
    aliases = {_identifier('o',rid,local):global_id for rid,values in id_map.items() for local,global_id in values.items()}
    def canonical_refs(value):
        if isinstance(value,dict):
            for key,item in list(value.items()):
                if key in SCOPED_KEYS: continue
                if key in REF_KEYS and isinstance(item,str): value[key] = aliases.get(item,item)
                elif isinstance(item,(dict,list)): canonical_refs(item)
        elif isinstance(value,list):
            for item in value: canonical_refs(item)
    for collection in ('nodes','labels','edges'):
        for item in scene[collection]: canonical_refs(item.get('metadata',{}))
    nodes = {n['id']:n for n in scene['nodes']}
    for rid in ordered:
        for pending in supplied[rid].get('pending_connections',[]):
            record = {'region_id':rid,'connection':deepcopy(pending)}
            def endpoint(value):
                if not isinstance(value,dict): return None
                return id_map.get(value.get('region_id'),{}).get(value.get('id'))
            source, target = endpoint(pending.get('source')), endpoint(pending.get('target'))
            reason = None
            evidence = pending.get('evidence',{})
            if source not in nodes or target not in nodes or source == target:
                reason = 'missing_or_ambiguous_endpoint'
            elif not isinstance(evidence,dict) or evidence.get('confirmed') is not True:
                reason = 'connection_evidence_not_confirmed'
            from_anchor, to_anchor = pending.get('from_anchor','east'),pending.get('to_anchor','west')
            if not reason and (_anchor(nodes[source],from_anchor) is None or _anchor(nodes[target],to_anchor) is None):
                reason = 'unsupported_anchor'
            edge = {'id':_identifier('c',rid,pending['id']),'from':source,'to':target,
                    'from_anchor':from_anchor,'to_anchor':to_anchor,'arrow':pending.get('arrow','end')}
            if not reason and evidence.get('kind') == 'mock_confirmed':
                if supplied[rid].get('metadata',{}).get('source') != 'mock_not_model_recognition':
                    reason = 'mock_confirmation_requires_mock_origin'
            elif not reason and evidence.get('kind') == 'geometric_path':
                points = evidence.get('points')
                valid_points = (isinstance(points,list) and 2 <= len(points) <= 100
                    and all(isinstance(p,list) and len(p)==2 and all(_finite(v) for v in p) for p in points))
                if (not valid_points or evidence.get('coordinate_system') != 'global'
                        or not manifest['image'].get('sha256')
                        or evidence.get('source_image_sha256') != manifest['image']['sha256']
                        or math.dist(points[0],_anchor(nodes[source],from_anchor)) > 2
                        or math.dist(points[-1],_anchor(nodes[target],to_anchor)) > 2):
                    reason = 'geometric_path_not_global_or_not_bound_to_image_and_endpoints'
                else:
                    edge = {'id':edge['id'],'points':deepcopy(points),'arrow':edge['arrow']}
            elif not reason:
                reason = 'unsupported_connection_evidence'
            if not reason and pending['id'] in id_map[rid]: reason = 'connection_id_conflicts_with_local_object'
            if not reason:
                edge['metadata'] = {'source_id':source,'target_id':target,'from_anchor':from_anchor,
                    'to_anchor':to_anchor,'connection_evidence':deepcopy(evidence),
                    'mock_only':evidence['kind']=='mock_confirmed','region_provenance':[{'region_id':rid,'local_id':pending['id']}]}
                if 'style' in pending:
                    edge['style'] = style_map[rid].get(pending['style'],pending['style']) if isinstance(pending['style'],str) else deepcopy(pending['style'])
                trial = dict(scene,edges=scene['edges']+[edge])
                try: validate_scene(trial)
                except (ValueError,TypeError,KeyError) as exc: reason = 'invalid_confirmed_edge: '+str(exc)
            if reason:
                report['pending_connections'].append(dict(record,reason=reason))
            else:
                if edge['id'] in used: raise ValueError('Cross-connection identifier collision.')
                used.add(edge['id']); scene['edges'].append(edge);id_map[rid][pending['id']] = edge['id']
                report['resolved_connections'].append(dict(record,global_id=edge['id'],mock_only=edge['metadata']['mock_only']))
    validate_scene(scene)
    report['schema_valid'] = True
    report['counts'] = {'regions':len(regions),'input':{k:sum(len(b['scene'][k]) for b in bundles) for k in ('nodes','labels','edges')},
        'merged':{k:len(scene[k]) for k in ('nodes','labels','edges')},'deduplicated':len(report['deduplicated']),
        'resolved_cross_region':len(report['resolved_connections']),'pending_cross_region':len(report['pending_connections'])}
    scene['metadata']['pending_cross_region_count'] = len(report['pending_connections'])
    return scene, report
