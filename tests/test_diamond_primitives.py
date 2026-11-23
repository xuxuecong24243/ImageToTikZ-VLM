"""Offline diamond geometry, text safety, routes and legacy rendering."""
from copy import deepcopy
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from structured_tikz.geometry import diamond_vertices, diamond_text_box, node_anchor, edge_route_points, diamond_path_error
from structured_tikz.generator import generate_tikz
from structured_tikz.primitives import shape, edge
from structured_tikz.schema import scene_contract
from structured_tikz.validator import validate_scene


def diamond(**updates):
    return dict({'id': 'd', 'type': 'diamond', 'bbox': [200, 150, 200, 100]}, **updates)


def scene(*nodes):
    return {'canvas': {'width': 800, 'height': 600, 'coordinate_system': 'top-left'},
            'styles': {}, 'nodes': list(nodes), 'edges': [], 'labels': []}


def routed(**updates):
    return dict({'id': 'e', 'from': 'd', 'to': 'target', 'from_anchor': 'east', 'to_anchor': 'west'}, **updates)


def nodes():
    return {'d': diamond(), 'target': {'id': 'target', 'type': 'rectangle', 'bbox': [500, 300, 100, 100]}}


class DiamondPrimitiveTests(OfflineTestCase):
    def test_vertices_exact_outer_bbox(self):
        self.assertEqual(diamond_vertices(diamond()['bbox']), [[300, 150], [400, 200], [300, 250], [200, 200]])

    def test_non_square_and_translated_geometry(self):
        self.assertEqual(diamond_vertices([10, 20, 360, 80]), [[190, 20], [370, 60], [190, 100], [10, 60]])
        before = diamond_vertices([10, 20, 80, 360])
        self.assertEqual(diamond_vertices([77, 53, 80, 360]), [[x+67, y+33] for x,y in before])

    def test_tiny_positive_dimensions(self):
        value = diamond(bbox=[1, 2, .001, .002])
        validate_scene(scene(value))
        p = diamond_vertices(value['bbox'])
        self.assertAlmostEqual(p[1][0]-p[3][0], .001)
        self.assertAlmostEqual(p[2][1]-p[0][1], .002)

    def test_invalid_dimensions_and_bounds(self):
        for box in ([20,20,0,20], [20,20,-1,20], [20,20,20,0], [20,20,True,20],
                    [20,20,float('nan'),20], [20,20,20,float('inf')], [-1,20,100,100], [750,20,100,100]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                validate_scene(scene(diamond(bbox=box)))

    def test_safe_text_box_corners(self):
        for box in ([0,0,100,100], [10,20,400,40], [100,200,40,200]):
            x,y,w,h = box
            a,b,c,d = diamond_text_box(box)
            for px,py in ((a,b),(a+c,b),(a+c,b+d),(a,b+d)):
                self.assertLessEqual(abs(px-x-w/2)/(w/2)+abs(py-y-h/2)/(h/2), .90000001)

    def test_all_cardinal_anchors(self):
        for name, point in zip(('north','east','south','west'), diamond_vertices(diamond()['bbox'])):
            self.assertEqual(node_anchor(diamond(), name), point)
        self.assertEqual(node_anchor(diamond(), 'center'), [300,200])
        with self.assertRaises(ValueError): node_anchor(diamond(), 'invented')

    def test_shared_schema_and_illegal_features(self):
        contract = scene_contract()
        self.assertIn('diamond', contract['json_schema']['$defs']['node']['properties']['type']['enum'])
        self.assertNotIn('diamond', contract['unsupported_native_primitives'])
        for update in ({'grid':{'rows':2,'cols':2}}, {'background_shape':'circle'}, {'wrap':False}):
            with self.subTest(update=update), self.assertRaises(ValueError): validate_scene(scene(diamond(**update)))
        with self.assertRaises(ValueError): validate_scene(scene(diamond(), diamond()))

    def test_exact_tikz_vertices_and_style(self):
        output = generate_tikz(scene(diamond(style={'fill':'light_blue','stroke':'black','line_width':2})))
        for expected in ('(120pt,180pt)','(160pt,160pt)','(120pt,140pt)','(80pt,160pt)',
                         '-- cycle','fill=scenelightblue','draw=sceneblack','line width=0.8pt'):
            self.assertIn(expected, output)

    def test_chinese_english_and_multiline(self):
        for text in ('条件满足吗？', 'ready_x & y?', '第一行\nSecond line'):
            output = generate_tikz(scene(diamond(text=text)))
            self.assertIn('diamond_text_overflow: d', output)
            self.assertIn('shortstack', output)
        self.assertIn(r'ready\_x \& y?', generate_tikz(scene(diamond(text='ready_x & y?'))))
        self.assertIn(r'第一行\\Second line', generate_tikz(scene(diamond(text='第一行\nSecond line'))))

    def test_math_rich_and_no_provenance_duplication(self):
        output = generate_tikz(scene(diamond(text=r'\alpha + x', mode='math')))
        self.assertIn(r'$\alpha + x$', output)
        output = generate_tikz(scene(diamond(text='SOURCE', mode='rich', segments=[
            {'type':'text','content':'满足 '}, {'type':'math','content':r'x \ge 0'}])))
        self.assertIn(r'$x \ge 0$', output)
        self.assertNotIn('SOURCE', output)
        with self.assertRaises(ValueError): generate_tikz(scene(diamond(text=r'\input{file}', mode='math')))

    def test_long_text_has_measured_compile_guard_without_rewriting(self):
        text = 'A long decision statement '*40
        data = scene(diamond(text=text)); original = deepcopy(data)
        output = generate_tikz(data)
        for term in (text, 'PackageError', r'\the\wd0', r'\the\ht0', '0.9pt'):
            self.assertIn(term, output)
        self.assertEqual(data, original)

    def test_straight_and_orthogonal_routes(self):
        self.assertEqual(edge_route_points(routed(), nodes()), [[400,200],[500,350]])
        for axis, elbow in (('x',[500,200]),('y',[400,350])):
            self.assertEqual(edge_route_points(routed(routing='orthogonal', first_axis=axis), nodes()), [[400,200],elbow,[500,350]])

    def test_explicit_waypoints_preserved_and_duplicates_rejected(self):
        points = [[400,200],[450,200],[450,350],[500,350]]
        self.assertEqual(edge_route_points({'points':points}, {}), points)
        data = scene(*nodes().values())
        data['edges'] = [{'id':'e','points':[points[0],points[0],points[-1]]}]
        with self.assertRaises(ValueError): validate_scene(data)
        aligned = nodes(); aligned['target']['bbox'] = [500,150,100,100]
        self.assertEqual(edge_route_points(routed(routing='orthogonal'), aligned), [[400,200],[500,200]])

    def test_each_outward_anchor_path(self):
        for name, far in {'north':[300,100],'east':[500,200],'south':[300,300],'west':[100,200]}.items():
            vertex = node_anchor(diamond(),name)
            self.assertIsNone(diamond_path_error([vertex,far],diamond(),endpoint='source'))
            self.assertIsNone(diamond_path_error([far,vertex],diamond(),endpoint='target'))

    def test_reject_interior_crossing_and_nonvertex(self):
        for points in ([[300,150],[300,300]], [[350,175],[500,100]],
                       [[400,200],[500,200],[500,400],[300,400],[300,100]]):
            self.assertIsInstance(diamond_path_error(points,diamond()),str)
        data = scene(*nodes().values()); data['edges'] = [routed(from_anchor='west')]
        validate_scene(data)  # this original slope follows the southwest boundary
        data['nodes'][1]['bbox'] = [500,150,100,100]  # now crosses the interior horizontally
        with self.assertRaisesRegex(ValueError,'interior'): validate_scene(data)

    def test_absolute_semantic_edge_is_checked(self):
        data = scene(diamond())
        data['edges'] = [{'id':'bad','points':[[300,150],[300,300]],'metadata':{'source_id':'d'}}]
        with self.assertRaisesRegex(ValueError,'interior'): validate_scene(data)
        for metadata in ([], {'source_id':[]}, {'target_id':{}}, None):
            data['edges'][0]['metadata'] = metadata
            validate_scene(data)  # legacy opaque metadata is never a semantic guess

    def test_arrows_and_invalid_references(self):
        data = scene(*nodes().values())
        for arrow, expected in (('none',',-]'),('start',',<-]'),('end',',->]'),('both',',<->]')):
            data['edges'] = [routed(arrow=arrow)]; validate_scene(data)
            self.assertIn(expected, edge(data['edges'][0], data))
        for update in ({'to':'missing'},{'from_anchor':'northwest'},{'to_anchor':'center'}):
            data['edges'] = [routed(**update)]
            with self.assertRaises(ValueError): validate_scene(data)

    def test_legacy_rectangle_and_rounding_exact(self):
        for kind, extra in [('rectangle',''),('rounded_rectangle',',rounded corners=3.2pt')]:
            node = {'id':'old','type':kind,'bbox':[20,30,100,40]}
            expected = r'\node[rectangle,inner sep=0pt,outer sep=0pt,minimum width=40pt,minimum height=16pt,draw=scenedarkblue,line width=0.48pt,fill=none'+extra+r'] (n-old) at (28pt,220pt) {};'
            self.assertEqual(shape(node,scene(node)),[expected])

    def test_legacy_circle_not_changed(self):
        node = {'id':'old','type':'text','bbox':[20,30,100,100],'text':'A','background_shape':'circle','wrap':False}
        output = generate_tikz(scene(node))
        self.assertIn('circle,minimum size=40pt',output)
        self.assertIn('(n-old) at (28pt,208pt) {A};',output)
        self.assertNotIn('diamond_text_overflow',output)

    def test_determinism_and_no_input_mutation(self):
        data = scene(diamond(text='Continue?')); before = deepcopy(data)
        self.assertEqual(generate_tikz(data),generate_tikz(data))
        self.assertEqual(data,before)
