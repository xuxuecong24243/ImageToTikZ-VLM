"""One-pass, non-semantic recovery of model Scene structure. No image recognition/API.

Original data is never mutated. Unknown fields, content, geometry and references
are not guessed away. A successful structural recovery says nothing about coverage.
"""
from copy import deepcopy
import json

from structured_tikz.schema import OBJECT_CONTRACTS, scene_contract
from structured_tikz.validator import validate_scene, SceneValidationError


def inspect_structure(scene, image_size=None):
    errors = []
    def error(code, message, path):
        errors.append({'category':'structural_format', 'code':code, 'path':path, 'message':message})
    try:
        validate_scene(scene)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        error('schema_invalid', str(exc), '/')
    if isinstance(scene, dict):
        spec = OBJECT_CONTRACTS['scene']
        for key in sorted(set(spec['required'])-set(scene)):
            error('missing_root_field', 'Required Scene field is absent.', '/'+key)
        for key in sorted(set(scene)-set(spec['properties']), key=str):
            error('unknown_root_field', 'Unsupported root field is preserved for review.', '/'+str(key))
        for key in ('nodes','edges','labels'):
            if key in scene and not isinstance(scene[key],list):
                error('invalid_collection_type', 'Drawing collection must be an array; no replacement is guessed.', '/'+key)
        if 'canvas' in scene and 'styles' in scene:
            try:
                # A validation probe only, never a replacement Scene or output.
                validate_scene({'canvas':scene['canvas'],'styles':scene['styles'],
                                'nodes':[],'edges':[],'labels':[]})
            except (ValueError,TypeError,KeyError,OverflowError) as exc:
                error('root_configuration_invalid', str(exc), '/canvas_or_styles')
        if image_size is not None:
            canvas = scene.get('canvas')
            if isinstance(canvas, dict) and (canvas.get('width'),canvas.get('height')) != tuple(image_size):
                error('canvas_image_mismatch', 'Existing canvas dimensions do not match the input image.', '/canvas')
    return {'schema_valid':not any(e['code']=='schema_invalid' for e in errors),
            'structure_accepted':not errors, 'errors':errors,
            'content_coverage':'not_evaluated', 'visual_fidelity':'not_evaluated'}


def recover_scene_structure(scene, image_size, *, coordinate_system=None,
                            response_format=None, default_styles=None):
    """Apply at most one pass of three explicitly justified root repairs.

    image_size comes from the decoded image, coordinate_system from the prompt
    contract, response_format from trusted request diagnostics (never model data).
    A format echo may be removed only when it exactly matches that transport mode.
    Missing named styles require explicit caller configuration; no names are invented.
    Valid old Scenes remain identical. Missing drawing collections are never created.
    """
    scene_contract(image_size)  # Strict validation of authoritative dimensions.
    before = inspect_structure(scene, image_size)
    report = {'policy':'deterministic_root_recovery_v1', 'passes':1, 'max_passes':1,
              'before':before, 'changes':[], 'unresolved':[],
              'content_restored':False, 'semantic_objects_added':0, 'api_calls':0}
    try:
        encoded = json.dumps(scene, ensure_ascii=False, allow_nan=False)
        if len(encoded) > 500000:
            raise ValueError('oversize')
    except (ValueError, TypeError, RecursionError, OverflowError):
        report.update(status='unrecoverable', after=before,
                      unresolved=[{'code':'unsafe_or_oversized_input','path':'/'}])
        return deepcopy(scene), report
    result = deepcopy(scene)
    meta = scene.get('metadata', {}) if isinstance(scene, dict) else {}
    status = meta.get('review_status') if isinstance(meta,dict) else None
    if not isinstance(scene, dict) or (isinstance(status,str) and status in {'verified','rejected'}):
        report.update(status='not_modified', after=before,
                      unresolved=[{'code':'protected_or_nonobject_input','path':'/'}])
        return result, report

    def change(path, old, new, reason, evidence, *, operation='add'):
        report['changes'].append({'op':operation, 'path':path, 'before':deepcopy(old),
                                  'after':deepcopy(new), 'reason':reason, 'evidence':evidence})
    # This is a transport format echo, not a recognized legacy Scene dialect.
    if ('type' in result and result['type'] == 'json_object'
            and response_format == {'type':'json_object'}):
        old = result.pop('type')
        change('/type', old, None, 'remove_confirmed_transport_format_echo',
               {'trusted_request_response_format':response_format,
                'not_a_supported_drawing_field':True}, operation='remove')
    elif 'type' in result:
        report['unresolved'].append({'code':'unknown_type_semantics','path':'/type',
                                    'reason':'No exact trusted transport-format match; field retained.'})
    if coordinate_system == 'top-left':
        values = dict(width=image_size[0], height=image_size[1], coordinate_system=coordinate_system)
        if 'canvas' not in result:
            result['canvas'] = values
            change('/canvas', None, values, 'canvas_from_input_dimensions_and_prompt_coordinates',
                   {'image_size':list(image_size),'prompt_coordinates':coordinate_system})
        elif isinstance(result['canvas'],dict):
            for key,value in values.items():
                if key not in result['canvas']:
                    result['canvas'][key] = value
                    change('/canvas/'+key,None,value,'missing_canvas_field_from_authoritative_input',
                           {'image_size':list(image_size),'prompt_coordinates':coordinate_system})
    elif 'canvas' not in result:
        report['unresolved'].append({'code':'coordinate_system_not_confirmed','path':'/canvas'})
    if 'styles' not in result:
        named = set()
        collections_known = all(isinstance(result.get(k),list) for k in ('nodes','edges','labels'))
        if collections_known:
            for kind in ('nodes','edges','labels'):
                for item in result[kind]:
                    if isinstance(item,dict) and isinstance(item.get('style'),str):
                        named.add(item['style'])
        configured = default_styles if isinstance(default_styles,dict) else {}
        if collections_known and named <= set(configured):
            proposed = {name:deepcopy(configured[name]) for name in sorted(named)}
            try:
                validate_scene({'canvas':{'width':1,'height':1,'coordinate_system':'top-left'},
                                'styles':proposed,'nodes':[],'edges':[],'labels':[]})
            except (ValueError,TypeError,KeyError):
                report['unresolved'].append({'code':'invalid_configured_styles','path':'/styles'})
            else:
                result['styles'] = proposed
                change('/styles', None, proposed, 'use_existing_style_defaults_or_configured_names',
                       {'named_references':sorted(named),'default_source':'structured_tikz.schema.STYLE_DEFAULTS',
                        'configured_names_only':True})
        else:
            report['unresolved'].append({'code':'missing_style_definitions','path':'/styles',
                                        'unconfigured_names':sorted(named-set(configured)),
                                        'collections_known':collections_known})
    after = inspect_structure(result, image_size)
    report['after'] = after
    report['unresolved'].extend(after['errors'])
    report['status'] = ('recovered_structure_only' if report['changes'] else 'not_needed') if after['structure_accepted'] else (
        'partially_recovered_invalid' if report['changes'] else 'unrecoverable')
    return result, report
