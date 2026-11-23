"""A fixed Chinese XeLaTeX template and conservative checks for model output.

These checks reject obvious file/command access. They are NOT a TeX sandbox:
public deployment still requires isolation, restricted permissions and quotas.
"""
from __future__ import annotations

import re


class TikzValidationError(ValueError):
    """The model response violates the fragment or safety contract."""


# Match command names, not substrings: e.g. \\inputlineno is not \\input.
_COMMAND = re.compile(r"\\([A-Za-z@]+|.)", re.DOTALL)
_FORBIDDEN_COMMANDS = {
    "documentclass", "documentstyle", "usepackage", "requirepackage",
    "loadclass", "passoptionstopackage", "passoptionstoclass", "includeonly",
    "input", "@input", "@@input", "@input@", "include", "@include",
    "includegraphics", "graphicspath", "includepdf", "@starttoc",
    "pgfimage", "pgfdeclareimage", "pgfuseimage", "pgfplotstableread",
    "openin", "openout", "closein", "closeout", "read", "readline", "write",
    "immediate", "newread", "newwrite", "newstream", "shellescape",
    "shellesc", "writefile", "@writefile", "inputiffileexists", "iffileexists",
    "getfileinfo", "verbatiminput", "lstinputlisting", "inputminted",
    "directlua", "latelua", "luadirect", "luaexec", "special", "pdfobj",
    "pdfxform", "pdfrefxform", "pdfximage", "pdfrefximage", "pdfextension",
    "pdffilesize", "pdffilemoddate", "pdffiledump", "pdfmdfivesum",
    "pdfcatalog", "pdfnames", "pdftrailer", "pdfannot", "pdfstartlink",
    "pdfmapfile", "pdfmapline", "xetexpicfile", "xetexpdffile",
    "catcode", "mathcode", "lccode", "uccode", "escapechar", "endlinechar",
    "csname", "endcsname", "scantokens", "let", "futurelet",
    "def", "gdef", "edef", "xdef", "renewcommand", "renewenvironment",
    "newenvironment", "makeatletter", "makeatother", "explsyntaxon",
    "explsyntaxoff", "everyjob", "everyeof", "output", "shipout", "dump",
    "endinput", "bye", "stop", "@@end", "primitive", "pdfprimitive",
    "usetikzlibrary", "usepgflibrary", "tikzexternalize", "tikzexternalenable",
    "tikzsetnextfilename", "tikzpicturedependsonfile", "pgfplotsset",
    "font", "fontspec", "setmainfont", "setsansfont", "setmonofont",
    "setcjkmainfont", "setcjksansfont", "setcjkmonofont",
    "bibliography", "bibliographystyle", "addbibresource",
    "openany", "pdfoutput", "pdfshellescape", "batchmode",
}
_FORBIDDEN_ENVIRONMENTS = {
    "document", "filecontents", "filecontents*", "luacode", "luacode*",
    "python", "pythoncode", "pycode", "minted",
}
_BEGIN = re.compile(r"\\begin\s*\{\s*tikzpicture\s*\}")
_END = re.compile(r"\\end\s*\{\s*tikzpicture\s*\}")


def _without_comments(text: str) -> str:
    """Strip TeX comments without treating an escaped percent as a comment."""
    cleaned = []
    for line in text.splitlines():
        for index, char in enumerate(line):
            if char == "%":
                backslashes = 0
                cursor = index - 1
                while cursor >= 0 and line[cursor] == "\\":
                    backslashes += 1
                    cursor -= 1
                if backslashes % 2 == 0:
                    line = line[:index]
                    break
        cleaned.append(line)
    return "\n".join(cleaned)


def _skip_space(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _group_end(text: str, position: int, opener: str = "{") -> int:
    position = _skip_space(text, position)
    if position >= len(text) or text[position] != opener:
        raise TikzValidationError("Expected a balanced {} group in drawing definitions.".format(opener))
    closer = "}" if opener == "{" else "]"
    depth, braces = 1, 0
    cursor = position + 1
    while cursor < len(text):
        char = text[cursor]
        if char == "\\":
            cursor += 2
            continue
        if opener == "[":
            if char == "{":
                braces += 1
            elif char == "}":
                braces -= 1
        if braces == 0:
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return cursor + 1
        cursor += 1
    raise TikzValidationError("Unbalanced group in drawing macro/style definitions.")


def _validate_prelude(text: str) -> None:
    """Permit drawing definitions before the single tikzpicture, not prose."""
    position = 0
    while (position := _skip_space(text, position)) < len(text):
        command = _COMMAND.match(text, position)
        if not command:
            raise TikzValidationError("Only drawing macros/styles may precede tikzpicture; remove prose.")
        name = command.group(1)
        position = _skip_space(text, command.end())
        if name in {"newcommand", "providecommand"}:
            if position < len(text) and text[position] == "*":
                position = _skip_space(text, position + 1)
            if position < len(text) and text[position] == "{":
                end = _group_end(text, position)
                macro_name = text[position + 1:end - 1].strip()
                position = end
            else:
                macro = re.match(r"\\[A-Za-z]+", text[position:])
                if not macro:
                    raise TikzValidationError("A drawing macro must have a normal alphabetic command name.")
                macro_name = macro.group(0)
                position += len(macro_name)
            if not re.fullmatch(r"\\[A-Za-z]+", macro_name):
                raise TikzValidationError("Invalid drawing macro name.")
            for _ in range(2):
                position = _skip_space(text, position)
                if position < len(text) and text[position] == "[":
                    position = _group_end(text, position, "[")
            position = _group_end(text, position)
        elif name in {"tikzset", "definecolor", "colorlet", "pgfmathsetmacro", "pgfmathtruncatemacro"}:
            count = {"tikzset": 1, "definecolor": 3, "colorlet": 2,
                     "pgfmathsetmacro": 2, "pgfmathtruncatemacro": 2}[name]
            for _ in range(count):
                position = _group_end(text, position)
        elif name == "tikzstyle":
            position = _skip_space(text, _group_end(text, position))
            if position >= len(text) or text[position] != "=":
                raise TikzValidationError("Expected '=' in tikzstyle definition.")
            position = _group_end(text, position + 1, "[")
        else:
            raise TikzValidationError("Unsupported command before tikzpicture: \\" + name)


def validate_tikz(fragment: str) -> None:
    """Check the narrow model-output contract and reject obvious unsafe input.

    A denylist cannot make arbitrary TeX safe. In particular it cannot prove
    semantic editability or detect every disguised operation/path tracing.
    """
    if not isinstance(fragment, str) or not fragment.strip():
        raise TikzValidationError("The model returned an empty TikZ fragment.")
    if len(fragment) > 250_000:
        raise TikzValidationError("TikZ fragment exceeds the 250000-character experiment limit.")
    if "\x00" in fragment or "^^" in fragment:
        raise TikzValidationError("NUL and TeX ^^ character-code escapes are not permitted.")
    code = _without_comments(fragment)
    for match in _COMMAND.finditer(code):
        if match.group(1).lower() in _FORBIDDEN_COMMANDS:
            line = code.count("\n", 0, match.start()) + 1
            raise TikzValidationError("Forbidden LaTeX command {} at fragment line {}.".format(match.group(0), line))
    for environment in re.finditer(r"\\(?:begin|end)\s*\{\s*([^{}]+?)\s*\}", code):
        if environment.group(1).lower() in _FORBIDDEN_ENVIRONMENTS:
            raise TikzValidationError("Forbidden LaTeX environment: " + environment.group(1))
    begins, ends = list(_BEGIN.finditer(code)), list(_END.finditer(code))
    if len(begins) != 1 or len(ends) != 1 or ends[0].start() <= begins[0].start():
        raise TikzValidationError("Return exactly one complete tikzpicture environment.")
    _validate_prelude(code[:begins[0].start()])
    if code[ends[0].end():].strip():
        raise TikzValidationError("Only comments/whitespace may follow the tikzpicture environment.")


def extract_tikz(text: str) -> str:
    """Extract one TikZ-bearing code block, or accept an unfenced fragment.

    Full documents are deliberately rejected rather than silently nested or
    stripped: all package/font configuration belongs to the fixed template.
    """
    if not isinstance(text, str):
        raise TikzValidationError("API content must be a string.")
    text = text.lstrip("\ufeff").strip()
    blocks = re.findall(r"```[^\n`]*\r?\n(.*?)```", text, flags=re.DOTALL)
    if "```" in text:
        candidates = [block.strip() for block in blocks if _BEGIN.search(_without_comments(block))]
        if len(candidates) != 1:
            raise TikzValidationError("Expected exactly one code block containing tikzpicture.")
        fragment = candidates[0]
    else:
        fragment = text
    validate_tikz(fragment)
    return fragment


def wrap_document(fragment: str) -> str:
    """Wrap a validated fragment once using portable TeX Live Fandol fonts."""
    validate_tikz(fragment)
    preamble = r"""\documentclass[border=6pt]{standalone}
\usepackage[UTF8,fontset=fandol]{ctex}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc,positioning,fit,backgrounds,matrix,decorations.pathreplacing,decorations.markings,shapes.geometric,shapes.misc,patterns,3d}
\pagestyle{empty}
\begin{document}
"""
    return preamble + fragment.strip() + "\n\\end{document}\n"
