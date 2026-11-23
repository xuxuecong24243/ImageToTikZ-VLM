"""Strict, bounded semantic-response contract. No model or HTTP implementation.

Confidence is a model's self-report, never a calibrated recognition probability.
The math allowlist is shared with the existing Scene validator, not a TeX sandbox.
"""
from __future__ import annotations

import json
import math
import re

from structured_tikz.validator import validate_math, validate_scene


SEMANTIC_TYPES = frozenset({'rectangle', 'rounded_rectangle', 'matrix', 'tensor3d', 'text', 'unknown'})
CONFIDENCES = frozenset({'high', 'medium', 'low', 'unknown'})
RESULT_FIELDS = frozenset({'objects', 'relations', 'texts', 'unmatched_objects', 'unmatched_texts', 'warnings'})
MAX_RESPONSE_CHARACTERS = 250_000

CONTRACT = """Return one strict JSON object with exactly these six array fields:
objects, relations, texts, unmatched_objects, unmatched_texts, warnings.
No Markdown fences, explanations, comments, TikZ, NaN, or extra fields.
All item fields below are REQUIRED; use explicit null where permitted.
objects items: {candidate_id, semantic_type, label, confidence, review_required}.
candidate_id must be a supplied node ID, used at most once. semantic_type is one
of rectangle, rounded_rectangle, matrix, tensor3d, text, unknown. label is a
nonempty plain string or null. Never return object bbox/grid/depth modifications.
relations items: {source_id, target_id, relation, direction, confidence, review_required}.
Endpoints must be supplied node IDs, or null only for an unknown relation.
relation is direct_line, arrow, proximity, or unknown. direction is forward,
reverse, bidirectional, none, or unknown. direct_line/proximity permit none or
unknown; an unknown relation requires unknown direction. Arrow direction may be
unknown. Do not connect a node to itself or repeat the same ordered endpoint pair.
texts items: {id, candidate_id, text_region_id, raw_text, text, mode, bbox,
confidence, review_required}. IDs use 1-60 ASCII letters/digits/underscores and
start with a letter; they must be unique across texts and unmatched_texts.
candidate_id is a supplied node ID or null; text_region_id is a supplied text
region ID or null. raw_text preserves the visible transcription; text contains
plain text for mode=text or a LaTeX math expression without dollar delimiters for
mode=math. Both strings must be nonempty. Preserve original math transcription
and converted LaTeX separately. bbox is [x,y,width,height] in ORIGINAL image
pixels, top-left origin, or null when position is uncertain. Do not invent exact
positions. Reuse a measured text_region_id when it reliably covers the text.
unmatched_texts items have the same fields as texts, but candidate_id and
text_region_id must both be null; do not invent a matched ID.
unmatched_objects items: {description, semantic_type, bbox, confidence,
review_required}; description is nonempty plain text, bbox has the same meaning
as text bbox or is null. No candidate_id is allowed for unmatched objects.
confidence is always high, medium, low, or unknown and is self-reported uncertainty,
not measured accuracy. review_required must always be true. warnings contains
plain strings (at most 50, each at most 500 characters). Other arrays contain at
most 500 items, strings at most 4000 characters. Empty arrays are valid.
"""


class SemanticValidationError(ValueError):
    """Semantic data failed the explicit response contract."""


def _fields(value, expected, location):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise SemanticValidationError(location + ' has missing or unexpected fields.')


def _string(value, location, maximum=4000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise SemanticValidationError(location + ' must be nonempty bounded text.')
    if any(ord(c) < 32 and c != '\n' for c in value) or '\x7f' in value:
        raise SemanticValidationError(location + ' contains unsupported control characters.')


def _choice(value, allowed, location):
    if not isinstance(value, str) or value not in allowed:
        raise SemanticValidationError(location + ' has an unsupported value.')


def _review(item, location):
    _choice(item['confidence'], CONFIDENCES, location + '.confidence')
    if item['review_required'] is not True:
        raise SemanticValidationError(location + '.review_required must be true.')


def _bbox(value, scene, location):
    if value is None:
        return
    if not isinstance(value, list) or len(value) != 4:
        raise SemanticValidationError(location + ' must be [x,y,width,height] or null.')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise SemanticValidationError(location + ' must contain finite coordinates.')
    x, y, w, h = value
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > scene['canvas']['width'] or y+h > scene['canvas']['height']:
        raise SemanticValidationError(location + ' is outside the original image.')


def scene_text_regions(scene):
    """Return measured auxiliary text boxes; metadata is not recognized text."""
    metadata = scene.get('metadata', {})
    if not isinstance(metadata, dict):
        raise SemanticValidationError('Scene metadata must be an object.')
    review = metadata.get('review_candidates', {})
    if not isinstance(review, dict):
        raise SemanticValidationError('Scene review_candidates must be an object.')
    regions = review.get('text_regions', [])
    if not isinstance(regions, list) or len(regions) > 500:
        raise SemanticValidationError('Scene text_regions must be a bounded array.')
    result = {}
    for item in regions:
        name = item.get('id') if isinstance(item, dict) else None
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', name) or name in result:
            raise SemanticValidationError('Scene text region IDs must be simple and unique.')
        _bbox(item.get('bbox'), scene, 'Scene text region bbox')
        if item.get('bbox') is None:
            raise SemanticValidationError('Scene text region needs its measured bbox.')
        result[name] = item
    return result


def _candidate(value, node_ids, location, *, nullable=False):
    if value is None and nullable:
        return
    if not isinstance(value, str) or value not in node_ids:
        raise SemanticValidationError(location + ' does not reference a supplied candidate ID.')


def validate_semantic_result(result, scene):
    try:
        validate_scene(scene)
    except ValueError as exc:
        raise SemanticValidationError('Input Scene failed the existing schema validation.') from exc
    _fields(result, RESULT_FIELDS, 'Semantic result')
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SemanticValidationError('Semantic result must contain finite JSON-compatible values.') from exc
    for key in RESULT_FIELDS:
        maximum = 50 if key == 'warnings' else 500
        if not isinstance(result[key], list) or len(result[key]) > maximum:
            raise SemanticValidationError(key + ' must be a bounded array.')
    node_ids = {node['id'] for node in scene['nodes']}
    regions = scene_text_regions(scene)
    seen_objects = set()
    for item in result['objects']:
        _fields(item, {'candidate_id', 'semantic_type', 'label', 'confidence', 'review_required'}, 'Object')
        _candidate(item['candidate_id'], node_ids, 'Object.candidate_id')
        if item['candidate_id'] in seen_objects:
            raise SemanticValidationError('Duplicate semantic object candidate_id.')
        seen_objects.add(item['candidate_id'])
        _choice(item['semantic_type'], SEMANTIC_TYPES, 'Object.semantic_type')
        if item['label'] is not None:
            _string(item['label'], 'Object.label')
        _review(item, 'Object')
    seen_relations = set()
    for item in result['relations']:
        _fields(item, {'source_id', 'target_id', 'relation', 'direction', 'confidence', 'review_required'}, 'Relation')
        _choice(item['relation'], {'direct_line', 'arrow', 'proximity', 'unknown'}, 'Relation.relation')
        _choice(item['direction'], {'forward', 'reverse', 'bidirectional', 'none', 'unknown'}, 'Relation.direction')
        for key in ('source_id', 'target_id'):
            _candidate(item[key], node_ids, 'Relation.' + key, nullable=item['relation'] == 'unknown')
        pair = (item['source_id'], item['target_id'])
        if pair in seen_relations or pair[0] is not None and pair[0] == pair[1]:
            raise SemanticValidationError('Duplicate relation endpoints or a self-relation.')
        seen_relations.add(pair)
        if item['relation'] in {'direct_line', 'proximity'} and item['direction'] not in {'none', 'unknown'}:
            raise SemanticValidationError('A non-arrow relation cannot assert arrow direction.')
        if item['relation'] == 'unknown' and item['direction'] != 'unknown':
            raise SemanticValidationError('Unknown relations need unknown direction.')
        _review(item, 'Relation')
    seen_texts = set()
    for collection in ('texts', 'unmatched_texts'):
        for item in result[collection]:
            _fields(item, {'id', 'candidate_id', 'text_region_id', 'raw_text', 'text', 'mode', 'bbox',
                           'confidence', 'review_required'}, 'Text')
            if not isinstance(item['id'], str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,59}', item['id']):
                raise SemanticValidationError('Text IDs must be simple identifiers of at most 60 characters.')
            if item['id'] in seen_texts:
                raise SemanticValidationError('Duplicate text ID.')
            seen_texts.add(item['id'])
            _candidate(item['candidate_id'], node_ids, 'Text.candidate_id', nullable=True)
            _candidate(item['text_region_id'], regions, 'Text.text_region_id', nullable=True)
            if collection == 'unmatched_texts' and (item['candidate_id'] is not None or item['text_region_id'] is not None):
                raise SemanticValidationError('Unmatched text cannot claim a matched candidate ID.')
            _string(item['raw_text'], 'Text.raw_text')
            _string(item['text'], 'Text.text')
            _choice(item['mode'], {'text', 'math'}, 'Text.mode')
            _bbox(item['bbox'], scene, 'Text.bbox')
            if item['mode'] == 'math':
                try:
                    validate_math(item['text'])
                except ValueError as exc:
                    raise SemanticValidationError('Math text failed the existing command/group safety check.') from exc
            _review(item, 'Text')
    for item in result['unmatched_objects']:
        _fields(item, {'description', 'semantic_type', 'bbox', 'confidence', 'review_required'}, 'Unmatched object')
        _string(item['description'], 'Unmatched object.description')
        _choice(item['semantic_type'], SEMANTIC_TYPES, 'Unmatched object.semantic_type')
        _bbox(item['bbox'], scene, 'Unmatched object.bbox')
        _review(item, 'Unmatched object')
    for warning in result['warnings']:
        _string(warning, 'Warning', 500)


def parse_semantic_response(text, scene):
    """Accept JSON only; do not strip fences or repair malformed model output."""
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_RESPONSE_CHARACTERS:
        raise SemanticValidationError('Semantic response must be nonempty text within 250000 characters.')

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise SemanticValidationError('Duplicate JSON object key.')
            value[key] = item
        return value

    def nonfinite(_):
        raise SemanticValidationError('Non-finite JSON constants are not allowed.')

    try:
        result = json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise SemanticValidationError('Model output is not one strict JSON document.') from exc
    validate_semantic_result(result, scene)
    return result
