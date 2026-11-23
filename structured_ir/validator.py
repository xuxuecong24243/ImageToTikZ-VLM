"""Schema-driven structural checks plus IR references/geometry, with JSON pointers.

This stdlib walker implements only the keywords used in our exported contract.
It is not advertised as a general JSON Schema implementation. No Scene, font,
model client, configuration loader or network dependency is needed.
"""
import math
import re

from .schema import ir_schema


class IRValidationError(ValueError):
    def __init__(self, report):
        self.report = report
        first = report['errors'][0]
        super().__init__('{}: {} ({})'.format(first['path'] or '/', first['message'], first['code']))


def _path(parent, key):
    return parent + '/' + str(key).replace('~', '~0').replace('/', '~1')


def _issue(errors, path, code, message):
    errors.append({'path': path, 'code': code, 'message': message})


def _number(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _matches_type(value, name):
    return {'object': isinstance(value, dict), 'array': isinstance(value, list),
            'string': isinstance(value, str), 'number': _number(value),
            'integer': type(value) is int, 'boolean': type(value) is bool,
            'null': value is None}[name]


def _walk(value, rule, path, definitions, errors):
    if rule is False:
        _issue(errors, path, 'forbidden_field', 'This field or array item is not allowed here.')
        return
    if '$ref' in rule:
        _walk(value, definitions[rule['$ref'].split('/')[-1]], path, definitions, errors)
        return
    kinds = rule.get('type')
    if kinds is not None and not any(_matches_type(value, k) for k in
                                     (kinds if isinstance(kinds, list) else [kinds])):
        _issue(errors, path, 'field_type', 'Expected {}.'.format(kinds))
        return
    if 'const' in rule and (type(value) is not type(rule['const']) or value != rule['const']):
        _issue(errors, path, 'constant', 'Expected {!r}.'.format(rule['const']))
    if 'enum' in rule and value not in rule['enum']:
        _issue(errors, path, 'unknown_type' if path.endswith('/type') else 'enum',
               'Expected one of {}.'.format(rule['enum']))
    if isinstance(value, dict):
        for key in rule.get('required', []):
            if key not in value:
                _issue(errors, _path(path, key), 'missing_field', 'Required field is absent.')
        properties = rule.get('properties', {})
        for key, item in value.items():
            if key in properties:
                _walk(item, properties[key], _path(path, key), definitions, errors)
            elif rule.get('additionalProperties') is False:
                _issue(errors, _path(path, key), 'unknown_field', 'Unrecognized field is preserved and rejected.')
    if isinstance(value, list):
        for limit, comparator in [('minItems', lambda x, y: x < y), ('maxItems', lambda x, y: x > y)]:
            if limit in rule and comparator(len(value), rule[limit]):
                _issue(errors, path, 'array_length', '{} is {}.'.format(limit, rule[limit]))
        prefix = rule.get('prefixItems', [])
        for index, item in enumerate(value):
            child = prefix[index] if index < len(prefix) else rule.get('items', {})
            _walk(item, child, _path(path, index), definitions, errors)
    if isinstance(value, str):
        if 'pattern' in rule and re.fullmatch(rule['pattern'], value) is None:
            _issue(errors, path, 'identifier', 'Identifier must start with a letter and contain <=64 ASCII identifier characters.')
        for limit, comparator in [('minLength', lambda x, y: x < y), ('maxLength', lambda x, y: x > y)]:
            if limit in rule and comparator(len(value), rule[limit]):
                _issue(errors, path, 'text_length', '{} is {}.'.format(limit, rule[limit]))
    if _number(value):
        for key, fails in [('minimum', lambda x, y: x < y), ('maximum', lambda x, y: x > y),
                           ('exclusiveMinimum', lambda x, y: x <= y)]:
            if key in rule and fails(value, rule[key]):
                _issue(errors, path, 'number_range', '{} is {}.'.format(key, rule[key]))
    for child in rule.get('allOf', []):
        _walk(value, child, path, definitions, errors)
    if 'if' in rule:
        condition_errors = []
        _walk(value, rule['if'], path, definitions, condition_errors)
        chosen = rule.get('then' if not condition_errors else 'else', {})
        _walk(value, chosen, path, definitions, errors)


def _json_safety(value, path, errors, ancestors=(), depth=0):
    if depth > 64:
        _issue(errors, path, 'depth_limit', 'JSON nesting exceeds 64 levels.')
        return
    if type(value) is float and not math.isfinite(value):
        _issue(errors, path, 'non_finite', 'JSON numbers must be finite.')
    elif type(value) in (dict, list):
        if id(value) in ancestors:
            _issue(errors, path, 'cyclic_data', 'Python object cycles are not JSON.')
            return
        ancestors = ancestors + (id(value),)
        for key, item in (value.items() if isinstance(value, dict) else enumerate(value)):
            if isinstance(value, dict) and not isinstance(key, str):
                _issue(errors, path, 'json_key', 'Object keys must be strings.')
            else:
                _json_safety(item, _path(path, key), errors, ancestors, depth + 1)
    elif value is not None and type(value) not in (str, int, float, bool):
        _issue(errors, path, 'json_type', 'Value is not a JSON data type.')


def _contains(outer, inner):
    return (inner[0] >= outer[0] and inner[1] >= outer[1] and
            inner[0] + inner[2] <= outer[0] + outer[2] and
            inner[1] + inner[3] <= outer[1] + outer[3])


def _relations(ir, errors, warnings):
    canvas = ir['canvas']
    extent = [0, 0, canvas['width'], canvas['height']]
    namespace = {canvas['id']: '/canvas/id'}
    regions = {r['id']: r for r in ir['regions']}
    elements = {e['id']: e for e in ir['elements']}
    entries = {}
    for kind in ('regions', 'elements', 'connections'):
        for index, item in enumerate(ir[kind]):
            path = '/{}/{}'.format(kind, index)
            identifier = item['id']
            if identifier in namespace:
                _issue(errors, path + '/id', 'duplicate_id', 'Already declared at {}.'.format(namespace[identifier]))
            else:
                namespace[identifier] = path + '/id'
            entries[identifier] = path
            for field in ('bbox', 'label_bbox'):
                if field in item and not _contains(extent, item[field]):
                    _issue(errors, path + '/' + field, 'out_of_bounds', 'Full box must lie inside the original image.')
            for n, point in enumerate(item.get('points', [])):
                if point[0] > extent[2] or point[1] > extent[3]:
                    _issue(errors, path + '/points/' + str(n), 'out_of_bounds', 'Path point lies outside the original image.')
                if n and point == item['points'][n-1]:
                    _issue(errors, path + '/points/' + str(n), 'duplicate_point', 'Consecutive path points must differ.')
                if kind == 'elements' and not _contains(item['bbox'], [point[0], point[1], 0, 0]):
                    _issue(errors, path + '/points/' + str(n), 'outside_element', 'Point lies outside its element bbox.')
    for kind, reference in [('regions', 'parent_id'), ('elements', 'region_id')]:
        for index, item in enumerate(ir[kind]):
            path = '/{}/{}'.format(kind, index)
            owner = item[reference]
            if owner is not None:
                if owner not in regions:
                    _issue(errors, path + '/' + reference, 'invalid_reference', 'Expected an existing region ID.')
                elif not _contains(regions[owner]['bbox'], item['bbox']):
                    _issue(errors, path + '/bbox', 'outside_region', 'Box lies outside its declared parent/owner region.')
    for index, region in enumerate(ir['regions']):
        path = '/regions/' + str(index)
        seen, current = set(), region['id']
        while current in regions:
            if current in seen:
                _issue(errors, path + '/parent_id', 'parent_cycle', 'Region parent hierarchy contains a cycle.')
                break
            seen.add(current)
            current = regions[current]['parent_id']
        if region['type'] == 'unknown':
            _issue(warnings, path + '/type', 'uncertain_region', 'Unknown region semantics require review.')
        if 'children' in region:
            expected = {r['id'] for r in ir['regions'] if r['parent_id'] == region['id']}
            expected.update(e['id'] for e in ir['elements'] if e['region_id'] == region['id'])
            listed = region['children']
            for n, child in enumerate(listed):
                pointer = path + '/children/' + str(n)
                if child not in regions and child not in elements:
                    _issue(errors, pointer, 'invalid_reference', 'Expected a child region or element ID.')
                elif child not in expected:
                    _issue(errors, pointer, 'ownership_mismatch', 'Child ownership does not match this region.')
                if child in listed[:n]:
                    _issue(errors, pointer, 'duplicate_reference', 'Child ID appears more than once.')
            if expected - set(listed):
                _issue(errors, path + '/children', 'missing_children', 'Declared children omit: {}.'.format(sorted(expected - set(listed))))
    branches = set()
    for index, connection in enumerate(ir['connections']):
        path = '/connections/' + str(index)
        for field in ('source_id', 'target_id'):
            if connection[field] not in elements:
                _issue(errors, path + '/' + field, 'invalid_reference', 'Expected an existing element ID.')
        if connection['kind'] == 'branch':
            label = connection['branch_label']
            if not label.strip():
                _issue(errors, path + '/branch_label', 'empty_branch_label', 'Branch label must contain visible text.')
            branch = (connection['source_id'], label)
            if branch in branches:
                _issue(errors, path + '/branch_label', 'ambiguous_branch_label', 'Same source has repeated branch label.')
            branches.add(branch)
        if connection['kind'] == 'branch' and connection['direction'] == 'none':
            _issue(warnings, path + '/direction', 'undirected_branch', 'Branch has no direction; review topology.')
    for index, element in enumerate(ir['elements']):
        expected_type = {'text': 'plain', 'multiline_text': 'plain', 'code': 'code', 'math': 'math'}.get(element['type'])
        if expected_type and element['text_type'] != expected_type:
            _issue(errors, '/elements/{}/text_type'.format(index), 'text_type_mismatch',
                   'This element type requires text_type={}.'.format(expected_type))


def inspect_ir(ir):
    """Return diagnostics without mutating input; structural errors gate references."""
    errors, warnings = [], []
    _json_safety(ir, '', errors)
    if not errors:
        schema = ir_schema()
        _walk(ir, schema, '', schema['$defs'], errors)
    if not errors:
        _relations(ir, errors, warnings)
    return {'valid': not errors, 'errors': errors, 'warnings': warnings}


def validate_ir(ir):
    """Validate IR independently of rendering; return the unchanged object."""
    report = inspect_ir(ir)
    if not report['valid']:
        raise IRValidationError(report)
    return ir
