"""Read-only IR coverage evaluation against optional independent expectations.

Expectations: region_types/texts/element_types are string lists; connections is a
list of dictionaries selecting source_id, target_id, direction, kind and/or
branch_label. minimum_counts optionally maps regions/elements/connections/texts
to nonnegative integers. Text matching is literal containment, not OCR scoring.
No image-specific inventory is embedded in this module.
"""
from copy import deepcopy
from .validator import inspect_ir


def evaluate_ir_coverage(ir, expectations=None, geometry=None, conversion_report=None):
    validation = inspect_ir(ir)
    data = ir if isinstance(ir, dict) else {}
    collections = {key: [v for v in data.get(key, []) if isinstance(v, dict)]
                   if isinstance(data.get(key), list) else [] for key in ('regions', 'elements', 'connections')}
    regions, elements, connections = (collections[k] for k in ('regions', 'elements', 'connections'))
    texts = [e for e in elements if isinstance(e.get('raw_text'), str)]
    counts = {key: len(value) for key, value in collections.items()}
    counts['texts'] = len(texts)
    report = {'format_validity': validation, 'counts': counts, 'recognized': [], 'missing': [],
              'unsupported': [], 'unknown': [], 'warnings': [],
              'content': {'status': 'not_evaluated_without_expectations'},
              'relationships': {'status': 'not_evaluated_without_expectations'},
              'geometry': {'status': 'not_verified', 'reason': 'Valid bounds do not prove image alignment.'},
              'conversion': deepcopy(conversion_report),
              'observed': {'region_ids': [r.get('id') for r in regions],
                           'elements': [{'id': e.get('id'), 'type': e.get('type')} for e in elements],
                           'texts': [{'id': e.get('id'), 'raw_text': e['raw_text']} for e in texts],
                           'connection_ids': [c.get('id') for c in connections]}}
    if not validation['valid']:
        report['warnings'].append('IR format is invalid; coverage is diagnostic only.')
    if not connections:
        report['warnings'].append('No semantic connections observed; absence is not proof of missing arrows without expectations.')
    if expectations is None:
        report['unknown'].append('Image content coverage and semantic relationship accuracy require independent expectations or review.')
    else:
        allowed = {'region_types', 'texts', 'element_types', 'connections', 'minimum_counts'}
        if not isinstance(expectations, dict) or set(expectations) - allowed:
            raise ValueError('Unknown or invalid coverage expectations.')
        for field, key, records in [('region_types', 'type', regions), ('texts', 'raw_text', texts),
                                    ('element_types', 'type', elements)]:
            expected = expectations.get(field, [])
            if not isinstance(expected, list) or any(not isinstance(s, str) or not s for s in expected):
                raise ValueError(field + ' must be a list of nonempty strings.')
            for value in expected:
                found = [record.get('id') for record in records
                         if (value in record.get(key, '') if field == 'texts' else record.get(key) == value)]
                item = {'dimension': 'content', 'kind': field, 'expected': value, 'matched_ids': found}
                report['recognized' if found else 'missing'].append(item)
        relation_fields = {'id', 'source_id', 'target_id', 'direction', 'kind', 'branch_label'}
        wanted_connections = expectations.get('connections', [])
        if not isinstance(wanted_connections, list):
            raise ValueError('connections expectations must be a list.')
        for expected in wanted_connections:
            if (not isinstance(expected, dict) or not expected or set(expected) - relation_fields
                    or any(not isinstance(v, str) or not v for v in expected.values())):
                raise ValueError('Invalid connection expectation.')
            found = [c.get('id') for c in connections if all(c.get(k) == v for k, v in expected.items())]
            item = {'dimension': 'relationships', 'kind': 'connection', 'expected': deepcopy(expected), 'matched_ids': found}
            report['recognized' if found else 'missing'].append(item)
        minima = expectations.get('minimum_counts', {})
        if (not isinstance(minima, dict) or set(minima) - set(counts)
                or any(type(v) is not int or v < 0 for v in minima.values())):
            raise ValueError('Invalid minimum_counts expectation.')
        for key, number in minima.items():
            item = {'dimension': 'relationships' if key == 'connections' else 'content',
                    'kind': 'minimum_count', 'collection': key, 'expected': number, 'observed': counts[key]}
            report['recognized' if counts[key] >= number else 'missing'].append(item)
        for dimension in ('content', 'relationships'):
            evaluated = any(i['dimension'] == dimension for i in report['missing'] + report['recognized'])
            report[dimension]['status'] = ('missing_expected_items' if any(i['dimension'] == dimension for i in report['missing'])
                                           else 'expectations_present' if evaluated else 'not_evaluated')
        report['content']['method'] = 'Literal expected snippets/types/counts; cannot verify transcription accuracy or full image coverage.'
        report['relationships']['method'] = 'Exact declared endpoint, direction, kind and label fields; no image topology inference.'
        report['unknown'].append('Unspecified image content and actual visual/topology correctness remain unverified.')
    if isinstance(geometry, dict):
        report['geometry']['candidate_counts'] = {key: len(geometry.get(key, []))
                                                for key in ('rectangles', 'lines', 'regions') if isinstance(geometry.get(key), list)}
        report['geometry']['warning'] = 'Candidates are not semantic truth; no completeness ratio is computed from text-like components.'
    for item in regions + elements + connections:
        confidence = item.get('confidence', 1)
        low = type(confidence) in (int, float) and confidence < .5
        if item.get('uncertainty') or item.get('type') == 'unknown' or low:
            report['unknown'].append({'id': item.get('id'), 'uncertainty': item.get('uncertainty', 'Low or unknown recognition confidence.')})
    if conversion_report is not None:
        report['unsupported'] = deepcopy(conversion_report.get('unsupported_elements', []))
        report['unrendered_text'] = deepcopy(conversion_report.get('unrendered_text', []))
        if not conversion_report.get('complete'):
            report['warnings'].append('Downstream conversion is partial or failed; valid IR is not fully rendered.')
    else:
        report['unknown'].append('Downstream representability not evaluated.')
    report['quality_warning'] = bool(report['missing'] or report['unknown'] or report['unsupported'] or report['warnings'])
    return report
