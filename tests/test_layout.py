"""Advisory TikZ checks; no compiler, model or third-party dependency is used."""
import json
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from layout_validator import validate_layout


def codes(report):
    return [item["code"] for item in report["issues"]]


class LayoutValidatorTests(OfflineTestCase):
    def test_normal_static_nodes_and_anchors(self):
        source = r"""\begin{tikzpicture}
\node[draw] (a) at (0,0) {中文 $x_1$};
\coordinate (b) at (2,1);
\draw[->] (a.east) -- (b);
\end{tikzpicture}"""
        report = validate_layout(source)
        self.assertEqual(report["explicit_node_names"], ["a", "b"])
        self.assertEqual(report["issues"], [])
        self.assertTrue(report["advisory_only"])
        self.assertEqual(report["schema_version"], "1.0")

    def test_duplicate_name_warns_and_locates_later_declaration(self):
        report = validate_layout("\\node (same) at (0,0) {A};\n\\coordinate (same) at (2,2);")
        issue = report["issues"][0]
        self.assertEqual(issue["code"], "duplicate_node_name")
        self.assertEqual(issue["severity"], "warning")
        self.assertEqual(issue["line"], 2)

    def test_node_names_after_at_and_path_nodes(self):
        report = validate_layout(r"\node at (0,0) (a) {A}; \draw (0,1) node[above] (b) {B} -- (a);")
        self.assertEqual(report["explicit_node_names"], ["a", "b"])
        self.assertEqual(report["issues"], [])

    def test_unknown_reference_is_uncertain_warning(self):
        report = validate_layout(r"\node (a) at (0,0) {};\draw (a) -- (missing.west);")
        self.assertEqual(codes(report), ["unresolved_node_reference"])
        self.assertEqual(report["issues"][0]["severity"], "warning")
        self.assertIn("macros/scopes", report["issues"][0]["message"])

    def test_comments_and_label_parentheses_do_not_create_names(self):
        report = validate_layout(r"""% \node (a) {fake}; { (missing)
\node (a) at (0,0) {words (missing) and \{escaped\} 50\%};
\draw (a) -- (current bounding box.north);
""")
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["explicit_node_names"], ["a"])

    def test_nonfinite_literal_coordinates_are_errors_and_json_safe(self):
        for value in ("nan", "inf", "-Infinity", "1e999"):
            with self.subTest(value=value):
                report = validate_layout(r"\draw (0,0) -- (" + value + ", 1);")
                self.assertIn("nonfinite_coordinate", codes(report))
                self.assertEqual(report["summary"]["errors"], 1)
                json.dumps(report, allow_nan=False)

    def test_plain_numbers_dimensions_and_arithmetic_not_confused(self):
        source = r"\draw (0.5,-2e1) -- (1cm,2cm) -- (\x+1,2) -- (2*\x,3);"
        report = validate_layout(source)
        self.assertEqual(report["issues"], [])
        self.assertEqual(len(report["literal_coordinates"]), 1)
        self.assertEqual(report["literal_coordinates"][0]["values"], [0.5, -20.0])

    def test_canvas_must_be_explicit_and_uses_tikz_units(self):
        source = r"\draw (0,0) -- (12,4);"
        self.assertNotIn("coordinate_outside_declared_bounds", codes(validate_layout(source)))
        report = validate_layout(source, canvas_bounds=(0, 0, 10, 5))
        self.assertIn("coordinate_outside_declared_bounds", codes(report))
        self.assertEqual(report["summary"]["errors"], 0)

    def test_relative_transformed_and_scoped_positions_are_not_absolute(self):
        source = r"\draw (0,0) -- ++(12,0);"
        report = validate_layout(source, canvas_bounds=(0, 0, 10, 10))
        self.assertNotIn("coordinate_outside_declared_bounds", codes(report))
        self.assertTrue(report["literal_coordinates"][-1]["relative"])
        report = validate_layout(r"\begin{tikzpicture}[scale=0.1]\draw (0,0)--(100,0);\end{tikzpicture}",
                                 canvas_bounds=(0, 0, 10, 10))
        self.assertNotIn("coordinate_outside_declared_bounds", codes(report))
        scoped = r"\begin{scope}[shift={(100,0)}]\node (a) at (0,0) {};\end{scope}\node (a) at (0,0) {};"
        self.assertNotIn("duplicate_node_name", codes(validate_layout(scoped)))

    def test_loop_definitions_are_not_instantiated_as_static_names(self):
        for loop in (r"\foreach \i in {1,2} {\node (a) at (\i,0) {};}",
                     r"\foreach \i in {1,2} \node (a) at (\i,0) {};"):
            report = validate_layout(loop + r"\node (a) at (0,0) {};")
            self.assertEqual(report["explicit_node_names"], ["a"])
            self.assertNotIn("duplicate_node_name", codes(report))

    def test_name_prefixes_are_not_treated_as_global_duplicate_names(self):
        report = validate_layout(r"\node[name prefix=x] (a) {};\node[name prefix=y] (a) {};")
        self.assertEqual(report["explicit_node_names"], [])
        self.assertNotIn("duplicate_node_name", codes(report))
        self.assertTrue(any("prefix" in item for item in report["limitations"]))

    def test_macros_detect_original_parameter_count_failure(self):
        source = r"""\newcommand{\drawmat}[4]{
\draw (0,0) rectangle (#4,#5); \foreach \i in {1,...,#6} {\node{#7};}
}
\newcommand{\drawtensor}[6]{\draw (#7,#8) -- (0,0);}
\begin{tikzpicture}\drawmat{x}{0}{0}{1}{2}{3}{4}\end{tikzpicture}"""
        report = validate_layout(source)
        self.assertTrue(any(item["code"] == "macro_parameter_out_of_range" and item["line"] == 2
                            for item in report["issues"]))
        self.assertEqual(report["macros"][0]["referenced_parameters"], [4, 5, 6, 7])
        corrected = source.replace(r"\drawmat}[4]", r"\drawmat}[7]").replace(r"\drawtensor}[6]", r"\drawtensor}[8]")
        self.assertEqual(validate_layout(corrected)["summary"]["errors"], 0)

    def test_macro_comments_escaped_hashes_optional_args_and_named_body(self):
        source = r"""\newcommand\example[1][default]{\node (a) {\#2 and #1}; % #9 is not code
}
\node (a) at (0,0) {};"""
        report = validate_layout(source)
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["macros"][0]["referenced_parameters"], [1])
        self.assertEqual(report["explicit_node_names"], ["a"])

    def test_nested_macro_parameters_are_explicitly_skipped(self):
        report = validate_layout(r"\newcommand{\outer}[1]{\newcommand{\inner}[1]{##1 #1}}")
        self.assertEqual(report["summary"]["errors"], 0)
        self.assertEqual(report["macros"][0]["parameter_check"], "skipped_dynamic_or_nested_definition")

    def test_invalid_macro_markers_and_more_than_nine_parameters(self):
        for source in (r"\newcommand{\f}[1]{#0}", r"\newcommand{\f}[10]{#1}", r"\newcommand{\f}{#}"):
            with self.subTest(source=source):
                self.assertGreater(validate_layout(source)["summary"]["errors"], 0)

    def test_brace_balance_and_line_numbers(self):
        report = validate_layout("% } ignored\n\\node {text;\n}")
        self.assertEqual(report["summary"]["errors"], 0)
        report = validate_layout("% { ignored\n\\node {text;")
        self.assertEqual(report["issues"][0]["code"], "unbalanced_brace")
        self.assertEqual(report["issues"][0]["line"], 2)
        self.assertIn("unbalanced_brace", codes(validate_layout("}")))

    def test_no_claim_of_overlap_detection_and_source_not_modified(self):
        source = r"\node (a) at (0,0) {A};\node (b) at (0,0) {B};"
        original = source[:]
        report = validate_layout(source)
        self.assertEqual(source, original)
        self.assertEqual(report["issues"], [])
        self.assertIn({"name": "visual_overlap_and_topology", "status": "not_implemented"}, report["checks"])
        json.dumps(report, ensure_ascii=False, allow_nan=False)

    def test_empty_source_and_invalid_arguments(self):
        self.assertEqual(validate_layout("")["summary"], {"errors": 0, "warnings": 0})
        with self.assertRaises(TypeError):
            validate_layout(None)
        for bounds in ((0, 0, 0, 1), (0, 0, float("inf"), 1), (0, 0, 2), (True, 0, 2, 2)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                validate_layout("", canvas_bounds=bounds)


if __name__ == "__main__":
    unittest.main()
