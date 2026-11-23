"""Small TikZ primitives. All geometry is computed in the source pixel system."""
from .validator import escape_text, resolve_style, validate_rich_segments
from .geometry import diamond_vertices


def number(value):
    return format(float(value), '.5f').rstrip('0').rstrip('.') or '0'


def point(x, y, canvas):
    scale = canvas.get('scale', .4)
    return '({}pt,{}pt)'.format(number(x * scale), number((canvas['height'] - y) * scale))


def color(name):
    if name.startswith('#'):
        return 'scenergb' + name[1:].upper()
    return 'none' if name == 'none' else 'scene' + name.replace('_', '')


def draw_style(style, canvas, *, fill=False):
    options = ['draw=' + color(style.get('stroke_color', style.get('stroke', 'dark_blue'))),
               'line width=' + number(style.get('line_width', 1.5) * canvas.get('scale', .4)) + 'pt']
    if fill:
        options.append('fill=' + color(style.get('fill_color', style.get('fill', 'none'))))
    if 'opacity' in style:
        options.append('opacity=' + number(style['opacity']))
    return ','.join(options)


def anchor_node(node, canvas, options='draw=none,fill=none'):
    x, y, width, height = node['bbox']
    scale = canvas.get('scale', .4)
    return (r'\node[rectangle,inner sep=0pt,outer sep=0pt,minimum width=%spt,minimum height=%spt,%s] '
            r'(n-%s) at %s {};' % (number(width * scale), number(height * scale), options,
                                    node['id'], point(x + width / 2, y + height / 2, canvas)))


def grid_lines(node, canvas, style, *, tensor=False):
    x, y, w, h = node['bbox']
    rows, cols = node['grid']['rows'], node['grid']['cols']
    depth = node.get('depth', {'dx': 0, 'dy': 0})
    dx, dy = depth['dx'], depth['dy']
    options = draw_style(dict(style, line_width=style.get('line_width', 1.5) * .65), canvas)
    commands = []
    def draw(a, b):
        commands.append(r'\draw[%s] %s -- %s;' % (options, point(*a, canvas), point(*b, canvas)))
    for col in range(1, cols):
        xx = x + w * col / cols
        draw((xx, y), (xx, y + h))
        if tensor:
            draw((xx, y), (xx + dx, y + dy))
    for row in range(1, rows):
        yy = y + h * row / rows
        draw((x, yy), (x + w, yy))
        if tensor:
            draw((x + w, yy), (x + w + dx, yy + dy))
    return commands


def tensor3d(node, canvas, style):
    """Front bbox + (dx,dy) defines all three faces and their editable grids."""
    x, y, w, h = node['bbox']
    dx, dy = node['depth']['dx'], node['depth']['dy']
    faces = [
        ('top', [(x, y), (x + dx, y + dy), (x + w + dx, y + dy), (x + w, y)]),
        ('right', [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)]),
        ('front', [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]),
    ]
    commands = []
    for name, vertices in faces:
        commands.append('% face: ' + name)
        commands.append(r'\path[%s] %s -- cycle;' % (
            draw_style(style, canvas, fill=True), ' -- '.join(point(*p, canvas) for p in vertices)))
    commands.extend(grid_lines(node, canvas, style, tensor=True))
    commands.append(anchor_node(node, canvas))
    return commands


def shape(node, scene):
    canvas, kind = scene['canvas'], node['type']
    style = resolve_style(scene, node)
    if kind == 'diamond':
        # Explicit vertices avoid native TikZ diamond text/minimum-size growth.
        return [r'\path[%s] %s -- cycle;' % (
            draw_style(style, canvas, fill=True),
            ' -- '.join(point(*p, canvas) for p in diamond_vertices(node['bbox']))),
            anchor_node(node, canvas)]
    if kind == 'tensor3d':
        return tensor3d(node, canvas, style)
    if kind == 'text':
        if node.get('background_shape') == 'circle':
            x, y, width, height = node['bbox']
            item = dict(node, x=x + width / 2, y=y + height / 2,
                        anchor='center', align='center')
            return [label(item, scene, name_prefix='n-', circle_diameter=width)]
        return [anchor_node(node, canvas)]
    options = draw_style(style, canvas, fill=True)
    if kind == 'rounded_rectangle':
        options += ',rounded corners=' + number(style.get('rounded_radius', 10) * canvas.get('scale', .4)) + 'pt'
    commands = [anchor_node(node, canvas, options)]
    if kind == 'matrix':
        commands.extend(grid_lines(node, canvas, style))
    return commands


def render_label_text(item, *, preserve_spaces=False):
    """One node's contents: escape prose, emit only validated math spans."""
    def prose(text):
        escaped = escape_text(text)
        return escaped.replace(' ',r'\ ') if preserve_spaces else escaped
    mode = item.get('mode', 'text')
    if mode == 'rich':
        validate_rich_segments(item['segments'])
        return ''.join('$' + span['content'] + '$' if span['type'] == 'math'
                       else prose(span['content']) for span in item['segments'])
    return '$' + item['text'] + '$' if mode == 'math' else prose(item['text'])


def _font(item, canvas, style):
    size = item.get('font_size', style.get('font_size', 22)) * style.get('font_scale',1) * canvas.get('scale', .4)
    font = r'\fontsize{%spt}{%spt}\selectfont' % (number(size), number(size * style.get('line_spacing',1.25)))
    if 'font_family' in style:
        font += {'serif':r'\rmfamily','sans':r'\sffamily','mono':r'\ttfamily'}[style['font_family']]
    if style.get('bold', False):
        font += r'\bfseries'
    return font


def label(item, scene, *, name_prefix='l-', text_width=None, circle_diameter=None, _content=None, _text_opacity=None):
    canvas, style = scene['canvas'], resolve_style(scene, item)
    font = _font(item, canvas, style)
    options = ['inner sep=0pt', 'outer sep=0pt', 'anchor=' + item.get('anchor', 'center'),
               'align=' + item.get('align', 'center'), 'text=' + color(style.get('text_color_rgb', style.get('text_color', 'black'))),
               'font={' + font + '}']
    if text_width is not None:
        options.append('text width=' + number(text_width * canvas.get('scale', .4)) + 'pt')
    if circle_diameter is not None:
        options.extend(['circle', 'minimum size=' + number(circle_diameter * canvas.get('scale', .4)) + 'pt',
                        draw_style(style, canvas, fill=True)])
    elif 'opacity' in style:
        options.append('opacity=' + number(style['opacity']))
    if _text_opacity is not None:
        options.append('text opacity=' + number(_text_opacity))
    text = render_label_text(item,preserve_spaces=style.get('preserve_spaces',False)) if _content is None else _content
    return r'\node[%s] (%s%s) at %s {%s};' % (
        ','.join(options), name_prefix, item['id'], point(item['x'], item['y'], canvas), text)


def diamond_text(node, scene):
    """Measure real glyphs before painting. Overflow stops compilation explicitly.

    A centered text rectangle lies inside a diamond iff Wtext/W + Htext/H <= 1.
    Use .9 for an interior margin; preserve the font, source text and outer bbox.
    """
    canvas, style = scene['canvas'], resolve_style(scene, node)
    x, y, w, h = node['bbox']
    scale = canvas.get('scale', .4)
    item = dict(node, x=x+w/2, y=y+h/2, anchor='center')
    alignment = {'left': 'l', 'center': 'c', 'right': 'r'}[node.get('align', 'center')]
    # Measure inside the ordinary TikZ text node: PGF deliberately replaces
    # selectfont with nullfont outside text nodes inside a picture.
    contents = '\n'.join([
        r'\begingroup',
        r'\setbox0=\hbox{\shortstack[%s]{%s}}' % (alignment, render_label_text(node,preserve_spaces=style.get('preserve_spaces',False))),
        r'\pgfmathparse{(\the\wd0)/(%spt)+((\the\ht0)+(\the\dp0))/(%spt)}' % (number(w*scale), number(h*scale)),
        r'\message{diamond_text_fit %s ratio=\pgfmathresult}' % node['id'],
        r'\ifdim\pgfmathresult pt>0.9pt\relax',
        r'\PackageError{structured-tikz}{diamond_text_overflow: %s}{Text exceeds diamond interior. Supply a larger bbox or an explicit reviewed font size.}' % node['id'],
        r'\fi',
        r'\copy0',
        r'\endgroup',
    ])
    return label(item, scene, name_prefix='t-', _content=contents)


def text_measurement_commands(node_name, ident):
    """Report the actual compiled node box in TeX pt, without painting content.

    Names are restricted because this helper may also be called by a batch
    measurement document. Scratch dimensions are scoped to this group.
    """
    import re
    if (not isinstance(node_name,str) or re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,127}',node_name) is None
            or not isinstance(ident,str) or re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,127}',ident) is None):
        raise ValueError('Measurement names must be safe identifiers.')
    return [r'\begingroup',
        r'\pgfextractx{\dimen0}{\pgfpointdiff{\pgfpointanchor{%s}{west}}{\pgfpointanchor{%s}{east}}}' % (node_name,node_name),
        r'\pgfextracty{\dimen2}{\pgfpointdiff{\pgfpointanchor{%s}{south}}{\pgfpointanchor{%s}{north}}}' % (node_name,node_name),
        r'\typeout{ITTEXT|%s|\the\dimen0|\the\dimen2}' % ident,
        r'\endgroup']


def edge(item, scene):
    canvas = scene['canvas']
    options = [draw_style(resolve_style(scene, item), canvas)]
    arrow = {'none': '-', 'start': '<-', 'end': '->', 'both': '<->'}[item.get('arrow', 'none')]
    options.append(arrow)
    if item.get('dashed', False):
        options.append('dashed')
    if 'points' in item:
        route = ' -- '.join(point(*p, canvas) for p in item['points'])
    else:
        source = '(n-%s.%s)' % (item['from'], item.get('from_anchor', 'east'))
        target = '(n-%s.%s)' % (item['to'], item.get('to_anchor', 'west'))
        join = ' -- '
        if item.get('routing', 'straight') == 'orthogonal':
            join = ' -| ' if item.get('first_axis', 'x') == 'x' else ' |- '
        route = source + join + target
    return r'\draw[%s] %s;' % (','.join(options), route)
