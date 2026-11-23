"""Small, deterministic offline repair of the reviewed GPT-2 middle Scene.

No I/O, model calls or layout changes. Refuse ambiguous duplicate removal.
"""
from copy import deepcopy

from structured_tikz.validator import validate_scene, resolve_style

BADGE_IDS = ('step1_badge', 'step2_badge', 'step3_badge')


def _references(value, target, path='$'):
    if isinstance(value, dict):
        return [found for key, child in value.items()
                for found in _references(child, target, path + '.' + str(key))]
    if isinstance(value, list):
        return [found for index, child in enumerate(value)
                for found in _references(child, target, path + '[' + str(index) + ']')]
    return [path] if value == target else []


def repair_middle_scene(scene):
    """Return a repaired copy and changes; repeated calls are a no-op.

    Treat every external occurrence of the removed ID as a potential reference,
    including metadata. Never rewrite references to make deletion succeed.
    """
    validate_scene(scene)
    repaired = deepcopy(scene)
    nodes = {node['id']: node for node in repaired['nodes']}
    title = nodes.get('top_title_label')
    if not title or title['type'] != 'text' or title.get('mode', 'text') != 'text':
        raise ValueError('Expected the reviewed top_title_label text node.')
    changes = []
    duplicate = next((item for item in repaired['labels'] if item['id'] == 'lbl_fragment'), None)
    without_duplicate = deepcopy(repaired)
    without_duplicate['labels'] = [item for item in without_duplicate['labels'] if item['id'] != 'lbl_fragment']
    references = _references(without_duplicate, 'lbl_fragment')
    if references:
        raise ValueError('Refusing removal: lbl_fragment is referenced at ' + ', '.join(references))
    if duplicate is not None:
        owner = duplicate.get('metadata', {})
        x, y, w, h = title['bbox']
        same_region = (duplicate.get('x') == x and y <= duplicate.get('y', -1) <= y + h
                       and duplicate.get('anchor') == 'south west')
        if not (duplicate.get('text') == title.get('text')
                and duplicate.get('mode', 'text') == 'text'
                and owner.get('node_id') == title['id'] and owner.get('bbox') == title['bbox']
                and same_region
                and resolve_style(repaired, duplicate)['font_size'] == resolve_style(repaired, title)['font_size']):
            raise ValueError('Cannot confirm lbl_fragment is the reviewed redundant title.')
        changes.append({'id': duplicate['id'], 'operation': 'remove_label',
                        'reason': 'Same title text, owner, bbox and font size; offset duplicate confirmed in the previous PNG.',
                        'original': deepcopy(duplicate), 'retained_id': title['id'],
                        'retained_original': deepcopy(title), 'external_references': references})
        repaired['labels'] = without_duplicate['labels']
    for digit, node_id in enumerate(BADGE_IDS, 1):
        node = nodes.get(node_id)
        if not node or node['type'] != 'text' or node.get('text') != str(digit) or node.get('mode', 'text') != 'text':
            raise ValueError('Expected the reviewed number text: ' + node_id)
        x, y, w, h = node['bbox']
        style = resolve_style(repaired, node)
        if w != h or style['fill'] != 'blue' or style['text_color'] != 'white':
            raise ValueError('Badge needs its existing square bbox and blue/white style: ' + node_id)
        if node.get('background_shape') != 'circle':
            changes.append({'id': node_id, 'operation': 'add_field', 'field': 'background_shape',
                            'before_present': 'background_shape' in node, 'before': node.get('background_shape'),
                            'after': 'circle', 'original': deepcopy(node), 'center': [x + w / 2, y + h / 2],
                            'diameter_source_pixels': w,
                            'reason': 'Draw existing white number and blue fill in one native circle; preserve bbox and typography.'})
            node['background_shape'] = 'circle'
    validate_scene(repaired)
    return repaired, changes
