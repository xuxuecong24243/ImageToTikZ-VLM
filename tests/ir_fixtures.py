"""Artificial generic fixtures, never OCR/VLM output or a page_01 reconstruction."""
from copy import deepcopy


def example_ir():
    return {'version': '1.0',
            'canvas': {'id': 'canvas', 'width': 640, 'height': 420, 'coordinate_system': 'top-left'},
            'regions': [{'id': 'panel', 'type': 'flow', 'bbox': [10, 10, 620, 400], 'parent_id': None,
                         'children': ['heading', 'start', 'finish']}],
            'elements': [
                {'id': 'heading', 'type': 'text', 'bbox': [40, 25, 540, 45], 'region_id': 'panel',
                 'raw_text': '离线合成示例 / Offline fixture', 'text_type': 'plain', 'reading_order': 0},
                {'id': 'start', 'type': 'rounded_rectangle', 'bbox': [60, 115, 210, 85], 'region_id': 'panel',
                 'raw_text': 'value = data[0]', 'text_type': 'code', 'reading_order': 1},
                {'id': 'finish', 'type': 'rectangle', 'bbox': [370, 115, 210, 85], 'region_id': 'panel',
                 'raw_text': '保存结果', 'text_type': 'plain', 'reading_order': 2}],
            'connections': [{'id': 'flow', 'source_id': 'start', 'target_id': 'finish',
                             'direction': 'forward', 'source_anchor': 'east', 'target_anchor': 'west', 'kind': 'flow'}],
            'provenance': {'kind': 'synthetic_test_fixture', 'recognition_performed': False}}


def branch_ir():
    ir = deepcopy(example_ir())
    ir['regions'][0].pop('children')
    ir['elements'] = [
        {'id': 'decision', 'type': 'diamond', 'bbox': [240, 40, 120, 70], 'region_id': 'panel',
         'raw_text': '有效？', 'text_type': 'plain', 'reading_order': 0},
        {'id': 'left', 'type': 'rectangle', 'bbox': [80, 180, 160, 70], 'region_id': 'panel', 'raw_text': '处理 A', 'text_type': 'plain', 'reading_order': 1},
        {'id': 'right', 'type': 'rectangle', 'bbox': [400, 180, 160, 70], 'region_id': 'panel', 'raw_text': '处理 B', 'text_type': 'plain', 'reading_order': 2},
        {'id': 'merge', 'type': 'rounded_rectangle', 'bbox': [240, 320, 120, 60], 'region_id': 'panel', 'raw_text': '结束', 'text_type': 'plain', 'reading_order': 3}]
    ir['connections'] = []
    for ident, source, target, kind, label in [
            ('yes', 'decision', 'left', 'branch', '是'), ('no', 'decision', 'right', 'branch', '否'),
            ('join_a', 'left', 'merge', 'merge', None), ('join_b', 'right', 'merge', 'merge', None)]:
        connection = {'id': ident, 'source_id': source, 'target_id': target, 'direction': 'forward',
                      'source_anchor': 'south', 'target_anchor': 'north', 'kind': kind}
        if label:
            connection.update(branch_label=label, label_bbox=[140 if label == '是' else 450, 120, 40, 40])
        ir['connections'].append(connection)
    return ir
