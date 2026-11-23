"""Source-pixel geometry shared by validation and deterministic conversion."""
from copy import deepcopy
import math


def _bbox(bbox):
    if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox)
            or min(bbox[:2]) < 0 or min(bbox[2:]) <= 0):
        raise ValueError('bbox must be finite [x>=0,y>=0,width>0,height>0].')
    return bbox


def diamond_vertices(bbox):
    """Exact outer bbox vertices, clockwise from north; never an inscribed bbox."""
    x, y, w, h = _bbox(bbox)
    return [[x+w/2, y], [x+w, y+h/2], [x+w/2, y+h], [x, y+h/2]]


def diamond_text_box(bbox):
    """One safe central rectangle; its corners have normalized diamond norm .9.

    Rendering uses actual TeX width/height and the same .9 norm, so short wide
    lines can use more width than this illustrative equal-fraction rectangle.
    """
    x, y, w, h = _bbox(bbox)
    return [x+.275*w, y+.275*h, .45*w, .45*h]


def node_anchor(node, name):
    x, y, w, h = _bbox(node['bbox'])
    points = dict(zip(('north', 'east', 'south', 'west'), diamond_vertices(node['bbox'])))
    points['center'] = [x+w/2, y+h/2]
    if name not in points:
        raise ValueError('Unknown node anchor: ' + str(name))
    return points[name]


def edge_route_points(edge, nodes):
    """Resolve the existing straight/one-bend routing without changing its shape."""
    if 'points' in edge:
        return deepcopy(edge['points'])
    start = node_anchor(nodes[edge['from']], edge.get('from_anchor', 'east'))
    end = node_anchor(nodes[edge['to']], edge.get('to_anchor', 'west'))
    points = [start]
    if edge.get('routing', 'straight') == 'orthogonal':
        points.append([end[0], start[1]] if edge.get('first_axis', 'x') == 'x' else [start[0], end[1]])
    points.append(end)
    return [point for index, point in enumerate(points) if index == 0 or point != points[index-1]]


def diamond_path_error(points, node, *, endpoint='source'):
    """Reject wrong endpoints and any segment entering this endpoint diamond.

    The normalized L1 distance is piecewise linear, with breakpoints where x or
    y crosses the center. Evaluating these breakpoints exactly finds its minimum
    on a segment; no sampling gaps or image-specific route rules are involved.
    """
    if endpoint not in {'source', 'target'}:
        raise ValueError('endpoint must be source or target.')
    if len(points) < 2:
        return 'Diamond connection needs a nonzero path.'
    end = points[0 if endpoint == 'source' else -1]
    if not any(all(math.isclose(a, b, rel_tol=0, abs_tol=1e-6) for a, b in zip(end, vertex))
               for vertex in diamond_vertices(node['bbox'])):
        return 'Diamond endpoint must lie at a north/south/east/west vertex; center is not a boundary anchor.'
    x, y, w, h = node['bbox']
    center = [x+w/2, y+h/2]
    for start, finish in zip(points, points[1:]):
        fractions = [0, 1]
        for axis in (0, 1):
            delta = finish[axis]-start[axis]
            if delta:
                t = (center[axis]-start[axis])/delta
                if 0 < t < 1:
                    fractions.append(t)
        for t in fractions:
            px, py = [a+t*(b-a) for a, b in zip(start, finish)]
            if abs(px-center[0])/(w/2)+abs(py-center[1])/(h/2) < 1-1e-9:
                return 'Connection path crosses the diamond interior; supply an outward route.'
    return None
