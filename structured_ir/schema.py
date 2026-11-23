"""Independent image-understanding IR, expressed once for prompts and validation.

Coordinates are original-image pixels, top-left origin. This contract deliberately
does not import Scene/TikZ: a valid diamond may still be unrenderable downstream.
"""
from copy import deepcopy

IR_VERSION = '1.0'
ID_PATTERN = r'^[A-Za-z][A-Za-z0-9_]{0,63}(?![\s\S])'
REGION_TYPES = frozenset({'title', 'panel', 'flow', 'code', 'text', 'unknown'})
TEXT_TYPES = frozenset({'text', 'multiline_text', 'code', 'math'})
ELEMENT_TYPES = TEXT_TYPES | frozenset({'rectangle', 'rounded_rectangle', 'circle',
                                      'diamond', 'title_band', 'connector', 'arrow'})
ANCHORS = frozenset({'north', 'south', 'east', 'west', 'center'})


def _ref(name):
    return {'$ref': '#/$defs/' + name}


def _object(properties, required):
    return {'type': 'object', 'properties': properties, 'required': list(required),
            'additionalProperties': False}


def _array(items, maximum=1000, minimum=0):
    return {'type': 'array', 'items': items, 'minItems': minimum, 'maxItems': maximum}


def _enum(values):
    return {'type': 'string', 'enum': sorted(values)}


_ID = {'type': 'string', 'pattern': ID_PATTERN}
_NULL_ID = {'type': ['string', 'null'], 'pattern': ID_PATTERN}
_TEXT = {'type': 'string', 'minLength': 1, 'maxLength': 4000}
_NUMBER = {'type': 'number', 'minimum': 0}
_POSITIVE = {'type': 'number', 'exclusiveMinimum': 0}
_EVIDENCE = {'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
             'uncertainty': _TEXT}
_STYLE = {'style_role': {'type': 'string', 'minLength': 1, 'maxLength': 128},
          'visual_style': _ref('visual_style'),
          'z_index': {'type': 'integer', 'minimum': 0, 'maximum': 10000}}


def _color_rule(allow_none=False):
    return _enum({'light_blue','dark_blue','blue','black','white','gray'} | ({'none'} if allow_none else set()))

DEFINITIONS = {
    'visual_style': _object({
        'fill': _color_rule(True), 'stroke': _color_rule(True), 'text_color': _color_rule(),
        **{key: {'type':'string','pattern':r'^#[0-9A-Fa-f]{6}(?![\s\S])'}
           for key in ('fill_color','stroke_color','text_color_rgb')},
        **{key: {'type':'number', 'minimum':low, 'maximum':high}
           for key, (low, high) in {'line_width':(.1,10), 'rounded_radius':(0,50),
                                   'font_size':(6,80), 'opacity':(0,1),
                                   'font_scale':(.5,3), 'line_spacing':(.8,2.2)}.items()},
        'bold': {'type':'boolean'},
        'font_family': _enum({'serif','sans','mono'}),
        'preserve_spaces': {'type':'boolean'},
    }, ()),
    'render_segment': _object({'type':_enum({'text','math'}),
                              'content':{'type':'string','maxLength':4000}}, ('type','content')),
    'canvas': _object({'id': _ID, 'width': {**_POSITIVE, 'maximum': 20000},
                       'height': {**_POSITIVE, 'maximum': 20000},
                       'coordinate_system': {'const': 'top-left'}, 'source': _TEXT},
                      ('id', 'width', 'height', 'coordinate_system')),
    'bbox': {'type': 'array', 'minItems': 4, 'maxItems': 4,
             'prefixItems': [_NUMBER, _NUMBER, _POSITIVE, _POSITIVE], 'items': False},
    'point': {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': _NUMBER},
    'region': _object({'id': _ID, 'type': _enum(REGION_TYPES), 'bbox': _ref('bbox'),
                       'parent_id': _NULL_ID, 'children': _array(_ID), **_EVIDENCE},
                      ('id', 'type', 'bbox', 'parent_id')),
    'element': _object({'id': _ID, 'type': _enum(ELEMENT_TYPES), 'bbox': _ref('bbox'),
                        'region_id': _NULL_ID, 'raw_text': _TEXT,
                        'text_type': _enum({'plain', 'code', 'math'}),
                        'reading_order': {'type': 'integer', 'minimum': 0},
                        'points': _array(_ref('point'), 100, 2),
                        'render_segments': _array(_ref('render_segment'),128,1), **_EVIDENCE, **_STYLE},
                       ('id', 'type', 'bbox', 'region_id')),
    'connection': _object({'id': _ID, 'source_id': _ID, 'target_id': _ID,
                           'direction': _enum({'forward', 'reverse', 'both', 'none'}),
                           'source_anchor': _enum(ANCHORS), 'target_anchor': _enum(ANCHORS),
                           'kind': _enum({'flow', 'branch', 'merge'}),
                           'points': _array(_ref('point'), 100, 2),
                           'branch_label': _TEXT, 'label_bbox': _ref('bbox'),
                           **_EVIDENCE, **_STYLE},
                          ('id', 'source_id', 'target_id', 'direction', 'source_anchor',
                           'target_anchor', 'kind')),
}
DEFINITIONS['element']['allOf'] = [
    {'if': {'required':['render_segments']},
     'then': {'required':['raw_text'], 'properties':{'type':_enum(TEXT_TYPES)}}},
    {'if': {'properties': {'type': _enum(TEXT_TYPES)}, 'required': ['type']},
     'then': {'required': ['raw_text', 'text_type', 'reading_order']}},
    {'if': {'properties': {'type': _enum({'connector', 'arrow'})}, 'required': ['type']},
     'then': {'required': ['points']}, 'else': {'properties': {'points': False}}},
    {'if': {'required': ['raw_text']}, 'then': {'required': ['text_type', 'reading_order']}},
]
DEFINITIONS['connection']['allOf'] = [
    {'if': {'properties': {'kind': {'const': 'branch'}}, 'required': ['kind']},
     'then': {'required': ['branch_label']}},
    {'if': {'required': ['label_bbox']}, 'then': {'required': ['branch_label']}},
]
DEFINITIONS['ir'] = _object({
    'version': {'const': IR_VERSION}, 'canvas': _ref('canvas'),
    'regions': _array(_ref('region')), 'elements': _array(_ref('element')),
    'connections': _array(_ref('connection')),
    'layered': {'type': 'boolean'}, 'measure_text': {'type':'boolean'},
    'provenance': {'type': 'object'}, 'diagnostics': _array(_TEXT, 100),
}, ('version', 'canvas', 'regions', 'elements', 'connections'))

RUNTIME_CONSTRAINTS = [
    'Optional render_segments are a display representation; raw_text remains unmodified provenance. Math segment safety is checked by Scene conversion. Optional font_family, line_spacing, preserve_spaces and font_scale affect text only.',
    'Optional visual_style uses named fill/stroke/text_color and literal #RRGGBB overrides fill_color/stroke_color/text_color_rgb, pixel dimensions and opacity; no drawing code. Missing fields retain existing defaults.',
    'Optional layered enables separate backgrounds/borders/edges/text. z_index is a caller-supplied compositing order within each phase, not semantic topology.',
    'All IDs, including canvas, share one namespace; references must resolve.',
    'All numbers are finite; booleans are not numbers. All boxes and points lie in the image.',
    'Boxes are [left,top,width,height], with positive width and height.',
    'Regions have an acyclic parent tree; listed children agree with actual ownership.',
    'Child regions/elements lie within their declared parent/owner region.',
    'Points may not repeat consecutively. Branch labels are nonempty and unambiguous per source.',
    'Connection graph cycles are allowed; only parent-region cycles are forbidden.',
    'raw_text is preserved, not corrected. Confidence is a self-report, not calibrated accuracy.',
    'unknown region type requires review. Supported IR types need not be supported by Scene.',
    'Schema validity alone proves neither content coverage nor topology/visual accuracy.',
]


def ir_schema():
    """Fresh reviewable JSON Schema used unchanged by the prompt and validator."""
    return {'$schema': 'https://json-schema.org/draft/2020-12/schema',
            '$id': 'urn:imagetotikz:ir:1.0', 'title': 'Image structure IR 1.0',
            '$ref': '#/$defs/ir', '$defs': deepcopy(DEFINITIONS),
            'x-runtime-constraints': list(RUNTIME_CONSTRAINTS)}
