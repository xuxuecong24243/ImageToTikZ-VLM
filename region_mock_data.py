"""Hand-authored three-column fixtures for the offline region pipeline.

These objects and connections are synthetic test data, NOT VLM recognition or
an asserted transcription of the input image. Coordinates are derived from the
actual crop sizes; no full-image diagram coordinates are embedded here.
"""
from __future__ import annotations

from copy import deepcopy


SOURCE = 'mock_not_model_recognition'


def _bundle(region, title, box_style):
    _, _, width, height = region['crop_bbox']
    if width < 160 or height < 360:
        raise ValueError('The three-column demonstration needs crops at least 160 x 360 pixels.')
    styles = {
        'box': deepcopy(box_style),
        'shared': {'text_color': 'black', 'font_size': 18},
        'heading': {'fill': 'dark_blue', 'stroke': 'dark_blue',
                    'text_color': 'white', 'font_size': 22, 'bold': True},
        'overlap_marker': {'fill': 'white', 'stroke': 'gray',
                           'text_color': 'black', 'line_width': .8, 'font_size': 12},
    }
    scene = {
        'canvas': {'width': width, 'height': height,
                   'coordinate_system': 'top-left', 'scale': .4},
        'styles': styles,
        'nodes': [
            {'id': 'panel', 'type': 'rectangle', 'bbox': [4, 4, width-8, height-8],
             'style': {'fill': 'none', 'stroke': 'gray', 'line_width': .6}},
            {'id': 'heading', 'type': 'rounded_rectangle',
             'bbox': [.06*width, .04*height, .88*width, .065*height],
             'text': title, 'mode': 'text', 'style': 'heading'},
            {'id': 'fixture_note', 'type': 'text',
             'bbox': [.07*width, .92*height, .86*width, .045*height],
             'text': '离线 Mock：非真实模型识别结果', 'mode': 'text',
             'style': {'text_color': 'gray', 'font_size': 14}},
        ],
        'labels': [], 'edges': [],
        'metadata': {'source': SOURCE, 'manual_fixture': True,
                     'recognition_performed': False},
    }
    return {'region_id': region['region_id'], 'scene': scene,
            'pending_connections': [],
            'metadata': {'source': SOURCE, 'manual_fixture': True,
                         'not_evidence_of_visual_recognition': True}}


def _box(scene, identifier, text, x, y, width, height):
    scene['nodes'].append({'id': identifier, 'type': 'rounded_rectangle',
                           'bbox': [x, y, width, height], 'text': text,
                           'mode': 'text', 'style': 'box'})


def _edge(scene, identifier, source, target, **options):
    scene['edges'].append(dict(id=identifier, **{
        'from': source, 'to': target, 'from_anchor': 'south', 'to_anchor': 'north',
        'arrow': 'end', 'style': {'stroke': 'dark_blue', 'line_width': 1.2},
        **options}))


def _math_label(scene, x, y):
    scene['labels'].append({
        'id': 'label', 'text': '768', 'mode': 'math', 'x': x, 'y': y,
        'anchor': 'center', 'style': 'shared',
        'metadata': {'source': SOURCE, 'bbox': [x-20, y-12, 40, 24],
                     'fixture_role': 'equal_text_at_distinct_positions'},
    })


def _left(scene):
    width, height = scene['canvas']['width'], scene['canvas']['height']
    x, box_width, box_height = .15*width, .70*width, .065*height
    for identifier, text, y in [
            ('box', 'Token / 位置嵌入', .18*height),
            ('transformer', 'TransformerBlock', .34*height),
            ('output', '输出层 out_head', .50*height)]:
        _box(scene, identifier, text, x, y, box_width, box_height)
    _edge(scene, 'internal_1', 'box', 'transformer')
    _edge(scene, 'internal_2', 'transformer', 'output')
    scene['nodes'].append({
        'id': 'tensor', 'type': 'tensor3d',
        'bbox': [.23*width, .69*height, .38*width, .09*height],
        'depth': {'dx': min(18, .04*width), 'dy': -min(12, .015*height)},
        'grid': {'rows': 2, 'cols': 3},
        'style': {'fill': 'light_blue', 'stroke': 'blue', 'line_width': 1.2},
        'metadata': {'source': SOURCE, 'fixture_role': 'depth_vector_translation_check'},
    })
    _math_label(scene, .75*width, .73*height)


def _middle(scene):
    width, height = scene['canvas']['width'], scene['canvas']['height']
    x, box_width, box_height = .10*width, .80*width, .065*height
    for identifier, text, y in [
            ('box', '步骤 1：嵌入权重加载', .18*height),
            ('step2', '步骤 2：Q / K / V 拆分', .34*height),
            ('step3', '步骤 3：转置后赋值', .50*height)]:
        _box(scene, identifier, text, x, y, box_width, box_height)
    _edge(scene, 'internal_1', 'box', 'step2')
    scene['edges'].append({
        'id': 'internal_2',
        'points': [[.50*width, .34*height+box_height],
                   [.94*width, .34*height+box_height],
                   [.94*width, .48*height], [.50*width, .48*height],
                   [.50*width, .50*height]],
        'arrow': 'end', 'style': {'stroke': 'blue', 'line_width': 1.2},
        'metadata': {'source_id': 'step2', 'target_id': 'step3',
                     'from_anchor': 'south', 'to_anchor': 'north', 'source': SOURCE},
    })
    scene['nodes'].append({
        'id': 'matrix', 'type': 'matrix',
        'bbox': [.18*width, .68*height, .42*width, .10*height],
        'grid': {'rows': 2, 'cols': 4},
        'style': {'fill': 'light_blue', 'stroke': 'dark_blue', 'line_width': 1.2},
    })
    _math_label(scene, .76*width, .73*height)
    scene['labels'].append({
        'id': 'rich_example', 'mode': 'rich', 'x': .5*width, 'y': .86*height,
        'anchor': 'center', 'style': 'shared',
        'segments': [{'type': 'text', 'content': '示例矩阵：'},
                     {'type': 'math', 'content': r'W\in\mathbb{R}^{4\times8}'}],
        'metadata': {'source': SOURCE, 'node_id': 'matrix',
                     'fixture_role': 'synthetic_rich_text_preservation'},
    })


def _right(scene):
    width, height = scene['canvas']['width'], scene['canvas']['height']
    _box(scene, 'box', 'params', .10*width, .18*height, .33*width, .065*height)
    for identifier, text, y in [
            ('wpe', 'wpe：位置嵌入', .35*height),
            ('wte', 'wte：词嵌入', .48*height),
            ('blocks', 'blocks：Transformer 块', .61*height)]:
        _box(scene, identifier, text, .43*width, y, .50*width, .06*height)
        _edge(scene, 'tree_'+identifier, 'box', identifier,
              to_anchor='west', routing='orthogonal', first_axis='y')
    _math_label(scene, .60*width, .79*height)


def _overlap_duplicate(first, second, first_region, second_region):
    a, b = first_region['crop_bbox'], second_region['crop_bbox']
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    if right-left < 12 or bottom-top < 100:
        raise ValueError('The overlap-duplicate Mock requires an actual shared crop strip.')
    box_width, box_height = min(18, right-left-4), 20
    gx, gy = left+2, top+.82*(bottom-top)
    for bundle, region in [(first, first_region), (second, second_region)]:
        ox, oy = region['offset']
        x, y = gx-ox, gy-oy
        scene = bundle['scene']
        scene['nodes'].append({
            'id': 'shared_rect', 'type': 'rectangle',
            'bbox': [x, y, box_width, box_height], 'style': 'overlap_marker',
            'metadata': {'source': SOURCE, 'fixture_role': 'overlap_duplicate'},
        })
        scene['labels'].append({
            'id': 'shared_label', 'text': 'X', 'mode': 'math',
            'x': x+box_width/2, 'y': y+box_height/2,
            'anchor': 'center', 'style': 'overlap_marker',
            'metadata': {'node_id': 'shared_rect', 'source': SOURCE,
                         'bbox': [x+2, y+3, box_width-4, box_height-6]},
        })


def build_mock_region_scenes(manifest):
    """Return local fixtures for exactly three detected, overlapping columns.

    Repeated ``box``/``label`` IDs intentionally exercise namespace remapping.
    The cross-column connection is explicitly synthetic, not image evidence.
    Unsupported segmentation is reported rather than silently replaced.
    """
    regions = sorted(manifest['regions'], key=lambda r: (r['crop_bbox'][0], r['crop_bbox'][1]))
    if len(regions) != 3:
        raise ValueError('This hand-authored GPT-2 layout Mock requires exactly three detected regions.')
    palettes = [
        {'fill': 'light_blue', 'stroke': 'dark_blue', 'text_color': 'black', 'font_size': 18},
        {'fill': 'white', 'stroke': 'blue', 'text_color': 'black', 'font_size': 18},
        {'fill': 'light_blue', 'stroke': 'black', 'text_color': 'dark_blue', 'font_size': 17},
    ]
    titles = ['Mock：模型结构', 'Mock：权重加载步骤', 'Mock：参数树']
    bundles = [_bundle(region, title, palette)
               for region, title, palette in zip(regions, titles, palettes)]
    for populate, bundle in zip((_left, _middle, _right), bundles):
        populate(bundle['scene'])
    _overlap_duplicate(bundles[0], bundles[1], regions[0], regions[1])
    left_id, middle_id = regions[0]['region_id'], regions[1]['region_id']
    bundles[0]['pending_connections'].extend([
        {'id': 'cross_confirmed', 'source': {'region_id': left_id, 'id': 'box'},
         'target': {'region_id': middle_id, 'id': 'box'},
         'from_anchor': 'east', 'to_anchor': 'west', 'arrow': 'end',
         'evidence': {'kind': 'mock_confirmed', 'confirmed': True},
         'metadata': {'source': SOURCE, 'fixture_role': 'synthetic_cross_region_test'}},
        {'id': 'pending_unknown', 'source': {'region_id': left_id, 'id': 'output'},
         'target': None, 'from_anchor': 'east', 'to_anchor': 'west', 'arrow': 'end',
         'evidence': {'kind': 'unknown', 'confirmed': False},
         'metadata': {'source': SOURCE, 'fixture_role': 'unresolved_connection_test'}},
    ])
    from region_scene import validate_region_scene
    for bundle, region in zip(bundles, regions):
        validate_region_scene(bundle, region)
    return bundles
