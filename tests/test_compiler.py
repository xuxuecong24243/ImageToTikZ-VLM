"""Offline tests: these mocks do not demonstrate any AI conversion quality."""
from pathlib import Path
import subprocess
import tempfile
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from unittest.mock import Mock, patch

from latex_compiler import build_compile_command, compile_tex, extract_errors
from latex_template import TikzValidationError, extract_tikz, validate_tikz, wrap_document


FRAGMENT = r"\begin{tikzpicture}\node {中文 $x^2$};\end{tikzpicture}"


class TemplateTests(OfflineTestCase):
    def test_extract_code_fence(self):
        self.assertEqual(extract_tikz("Explanation\n```latex\n" + FRAGMENT + "\n```"), FRAGMENT)

    def test_plain_fragment(self):
        self.assertEqual(extract_tikz(FRAGMENT), FRAGMENT)

    def test_two_figures_are_ambiguous(self):
        with self.assertRaises(TikzValidationError):
            extract_tikz("```tex\n" + FRAGMENT + "\n```\n```tex\n" + FRAGMENT + "\n```")

    def test_fixed_chinese_document(self):
        source = wrap_document(FRAGMENT)
        self.assertEqual(source.count(r"\documentclass"), 1)
        self.assertEqual(source.count(r"\begin{document}"), 1)
        self.assertIn("fontset=fandol", source)
        self.assertIn("amsmath,amssymb", source)
        self.assertIn("中文", source)

    def test_reject_full_document_and_package_injection(self):
        for prefix in (r"\documentclass{article}", r"\usepackage{shellesc}", r"\begin{document}"):
            with self.subTest(prefix=prefix), self.assertRaises(TikzValidationError):
                wrap_document(prefix + FRAGMENT)

    def test_reject_file_access_execution_and_obfuscation(self):
        attacks = [r"\input{secret}", r"\includegraphics{original.png}", r"\write18{bad}",
                   r"\openout1=secret", r"\read1 to\foo", r"\catcode65=0", r"\csname input\endcsname",
                   r"\directlua{os.execute('bad')}", r"\special{psfile=secret}", r"\pdfobj file{secret}",
                   r"^^5cinput{secret}", r"\pgfimage{secret}", r"\@@input{secret}",
                   r"\XeTeXpicfile \"original.png\""]
        for attack in attacks:
            with self.subTest(attack=attack), self.assertRaises(TikzValidationError):
                validate_tikz(FRAGMENT.replace(r"\node", attack + r"\node"))

    def test_allow_editable_macros_styles_and_loops(self):
        code = r"""% A reusable editable tensor macro, not a raster image.
\newcommand{\tensorbox}[2][blue]{\fill[#1!20] (0,0) rectangle (#2,1);}
\definecolor{tensorblue}{RGB}{80,150,210}
\tikzset{box/.style={draw,rounded corners}}
\begin{tikzpicture}
\tensorbox[tensorblue]{2}
\foreach \x in {0,1,2} {\draw (\x,0)--(\x,1);}
\matrix[matrix of math nodes] {1 & 2 \\ 3 & 4 \\};
\node[box] {中文 $8\times4\times256$};
\end{tikzpicture}
"""
        validate_tikz(code)

    def test_comments_and_escaped_percent(self):
        validate_tikz("% \\input{ignored}\n" + FRAGMENT.replace("中文", r"百分比 20\%"))

    def test_reject_prose_and_incomplete_environment(self):
        for code in ("Here is code\n" + FRAGMENT, FRAGMENT + "explanation", r"\begin{tikzpicture}"):
            with self.subTest(code=code), self.assertRaises(TikzValidationError):
                validate_tikz(code)


class CompilerTests(OfflineTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tex = self.root / "generation_01.tex"
        self.tex.write_text(wrap_document(FRAGMENT), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_command_is_explicit_and_disables_shell_escape(self):
        command = build_compile_command(self.tex, "C:/TeX tools/xelatex.exe")
        self.assertEqual(command[0], "C:/TeX tools/xelatex.exe")
        self.assertIn("-no-shell-escape", command)
        self.assertIn("-halt-on-error", command)
        self.assertIn("-file-line-error", command)
        self.assertEqual(command[-1], "./generation_01.tex")

    def test_success_requires_new_pdf_and_keeps_full_logs(self):
        process = Mock(returncode=0)
        full_log = "start\n" + "中文 log\n" * 4000
        def communicate(**kwargs):
            self.tex.with_suffix(".pdf").write_bytes(b"%PDF-1.5\nexample mocked bytes")
            self.tex.with_suffix(".log").write_text(full_log, encoding="utf-8")
            return b"complete stdout", b"stderr"
        process.communicate.side_effect = communicate
        with patch("latex_compiler.subprocess.Popen", return_value=process) as popen:
            result = compile_tex(self.tex)
        self.assertTrue(result.success)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.log_path.read_text(encoding="utf-8"), full_log)
        self.assertIn("complete stdout", result.console_path.read_text(encoding="utf-8"))
        self.assertFalse(popen.call_args.kwargs["shell"])
        self.assertEqual(popen.call_args.kwargs["env"]["openin_any"], "p")

    def test_stale_pdf_cannot_fake_success(self):
        self.tex.with_suffix(".pdf").write_bytes(b"%PDF-old")
        process = Mock(returncode=0)
        process.communicate.return_value = (b"no PDF output", b"")
        with patch("latex_compiler.subprocess.Popen", return_value=process):
            result = compile_tex(self.tex)
        self.assertFalse(result.success)
        self.assertIsNone(result.pdf_path)
        self.assertIn("No newly generated PDF", result.errors)

    def test_compile_error_has_location(self):
        process = Mock(returncode=1)
        process.communicate.return_value = (b"./generation_01.tex:12: Undefined control sequence.\nl.12 \\badmacro", b"")
        with patch("latex_compiler.subprocess.Popen", return_value=process):
            result = compile_tex(self.tex)
        self.assertFalse(result.success)
        self.assertIn(":12:", result.errors)
        self.assertTrue(result.log_path.is_file())

    def test_timeout_records_failure_and_stops_process_tree(self):
        process = Mock(returncode=-9)
        process.communicate.side_effect = [subprocess.TimeoutExpired("xelatex", 1, output=b"partial"), (b"partial", b"timed out")]
        with patch("latex_compiler.subprocess.Popen", return_value=process), patch("latex_compiler._terminate_process_tree") as stop:
            result = compile_tex(self.tex, timeout=1)
        stop.assert_called_once_with(process)
        self.assertTrue(result.timed_out)
        self.assertFalse(result.success)
        self.assertIn("partial", result.console_path.read_text(encoding="utf-8"))

    def test_missing_engine_is_reported(self):
        with patch("latex_compiler.subprocess.Popen", side_effect=FileNotFoundError("xelatex missing")):
            result = compile_tex(self.tex)
        self.assertFalse(result.success)
        self.assertIsNone(result.returncode)
        self.assertIn("could not start", result.errors)

    def test_invalid_timeout_and_missing_source(self):
        for timeout in (0, -1, float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                compile_tex(self.tex, timeout=timeout)
        with self.assertRaises(FileNotFoundError):
            compile_tex(self.root / "missing.tex")

    def test_error_summary_limits_context_not_full_log(self):
        summary = extract_errors("header\n! Undefined control sequence.\nl.42 \\broken\n" + "rest\n" * 20)
        self.assertIn("l.42", summary)
        self.assertLess(len(summary), 300)


if __name__ == "__main__":
    unittest.main()
