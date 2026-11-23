"""Conservative source diagnostics, not a TikZ renderer or overlap detector.

The report is advisory. It never rewrites source, compiles TeX, or invokes a model.
Only ordinary explicit declarations and literal coordinates are inspected.
"""
from __future__ import annotations

import math
import re
from typing import Optional, Tuple


_MACRO = re.compile(r"\\(?:newcommand|providecommand|renewcommand)\b\*?")
_PATH = re.compile(r"\\(?:node|coordinate|draw|path|filldraw|fill|shade|clip)\b")
_NAME = r"[A-Za-z_][A-Za-z0-9_:-]*"
_NUMBER = re.compile(r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan)", re.I)
_BUILTIN_NAMES = {"current bounding box", "current path bounding box", "current page", "current point"}


def _comments_blank(text: str) -> str:
    """Replace comments with spaces, preserving all character and line offsets."""
    result = list(text)
    cursor = 0
    while cursor < len(text):
        if text[cursor] == "\\":
            cursor += 2
            continue
        if text[cursor] == "%":
            end = text.find("\n", cursor)
            end = len(text) if end < 0 else end
            result[cursor:end] = " " * (end - cursor)
            cursor = end
        else:
            cursor += 1
    return "".join(result)


def _space(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _group_end(text: str, start: int, opener: str = "{") -> Optional[int]:
    if start >= len(text) or text[start] != opener:
        return None
    closer = "}" if opener == "{" else "]"
    depth, braces, cursor = 1, 0, start + 1
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
        if not braces:
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if not depth:
                    return cursor + 1
        cursor += 1
    return None


def _blank_range(chars: list, start: int, end: int) -> None:
    for index in range(start, end):
        if chars[index] not in "\r\n":
            chars[index] = " "


def _statement_end(text: str, start: int) -> int:
    cursor, depth = start, 0
    while cursor < len(text):
        char = text[cursor]
        if char == "\\":
            cursor += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == ";" and depth == 0:
            return cursor + 1
        cursor += 1
    return len(text)


def validate_layout(tikz_text: str, *, canvas_bounds: Optional[Tuple[float, float, float, float]] = None) -> dict:
    """Return deterministic/advisory findings for the statically supported subset.

    canvas_bounds uses the caller's TikZ coordinate units, NOT normalized image
    coordinates. Without explicit bounds no canvas size is inferred. Scope,
    loop, macro expansion and transformed coordinate placement are not resolved.
    """
    if not isinstance(tikz_text, str):
        raise TypeError("TikZ source must be a string.")
    if canvas_bounds is not None:
        if (len(canvas_bounds) != 4 or any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) for value in canvas_bounds)
                or canvas_bounds[0] >= canvas_bounds[2] or canvas_bounds[1] >= canvas_bounds[3]):
            raise ValueError("canvas_bounds must be finite (xmin, ymin, xmax, ymax) with positive extent.")
    code = _comments_blank(tikz_text)
    issues, macros, coordinates, declarations, references = [], [], [], [], []
    limitations = [
        "Advisory source inspection only; no guarantee of compilability or faithful layout.",
        "No text extents, overlap, arrow direction, z-order or rendered geometry are inferred.",
        "Macro expansion, foreach loops, scoped placement, relative positioning, coordinate arithmetic and node dimensions are not evaluated.",
    ]

    def issue(severity, name, position, message):
        issues.append({"severity": severity, "code": name,
                       "line": code.count("\n", 0, position) + 1, "message": message})

    stack, cursor = [], 0
    while cursor < len(code):
        if code[cursor] == "\\":
            cursor += 2
            continue
        if code[cursor] == "{":
            stack.append(cursor)
        elif code[cursor] == "}":
            if stack:
                stack.pop()
            else:
                issue("error", "unbalanced_brace", cursor, "Closing brace has no matching opening brace.")
        cursor += 1
    for position in stack:
        issue("error", "unbalanced_brace", position, "Opening brace has no matching closing brace.")

    # Macro bodies are checked for simple parameter consistency, then masked so
    # nodes inside definitions are never confused with instantiated declarations.
    visible = list(code)
    cursor = 0
    while True:
        match = _MACRO.search(code, cursor)
        if not match:
            break
        position = _space(code, match.end())
        if position < len(code) and code[position] == "{":
            end = _group_end(code, position)
            name = code[position + 1:end - 1].strip() if end else ""
            position = _space(code, end) if end else len(code)
        else:
            name_match = re.match(r"\\[A-Za-z]+", code[position:])
            name = name_match.group() if name_match else ""
            position = _space(code, position + len(name))
        count = 0
        if position < len(code) and code[position] == "[":
            end = _group_end(code, position, "[")
            raw = code[position + 1:end - 1].strip() if end else ""
            count = int(raw) if raw.isdigit() else None
            position = _space(code, end) if end else len(code)
            if position < len(code) and code[position] == "[":
                end = _group_end(code, position, "[")
                position = _space(code, end) if end else len(code)
        end = _group_end(code, position)
        if end is None or not re.fullmatch(r"\\[A-Za-z]+", name):
            limitations.append("An unsupported or incomplete macro declaration was not analyzed.")
            cursor = max(match.end(), position)
            continue
        body = code[position + 1:end - 1]
        entry = {"name": name, "line": code.count("\n", 0, match.start()) + 1,
                 "declared_arguments": count, "parameter_check": "checked"}
        if count is None or re.search(r"\\(?:newcommand|renewcommand|providecommand|def|gdef|edef|xdef|csname)\b|##", body):
            entry["parameter_check"] = "skipped_dynamic_or_nested_definition"
            limitations.append("Nested/dynamic macro parameter definitions were skipped.")
        else:
            if count > 9:
                issue("error", "macro_parameter_count", match.start(), "Ordinary newcommand supports at most 9 parameters.")
            body_cursor, used, reported_parameters = 0, [], set()
            while body_cursor < len(body):
                if body[body_cursor] == "\\":
                    body_cursor += 2
                    continue
                if body[body_cursor] == "#":
                    marker = body[body_cursor + 1:body_cursor + 2]
                    if marker.isdigit() and 1 <= int(marker) <= 9:
                        used.append(int(marker))
                        if int(marker) > count and marker not in reported_parameters:
                            issue("error", "macro_parameter_out_of_range", position + 1 + body_cursor,
                                  "{} declares {} arguments but uses #{}.".format(name, count, marker))
                            reported_parameters.add(marker)
                    else:
                        issue("error", "invalid_macro_parameter", position + 1 + body_cursor,
                              "A simple macro parameter marker must be #1 through #9; literal hashes require escaping.")
                body_cursor += 1
            entry["referenced_parameters"] = sorted(set(used))
        macros.append(entry)
        _blank_range(visible, match.start(), end)
        cursor = end
    static = "".join(visible)

    # Skip complete scope blocks; local transforms/name prefixes cannot be
    # interpreted as global coordinates or global explicit names here.
    scope_stack = []
    for match in re.finditer(r"\\(begin|end)\s*\{scope\}", static):
        if match.group(1) == "begin":
            scope_stack.append(match.start())
        elif scope_stack:
            start = scope_stack.pop()
            if not scope_stack:
                _blank_range(visible, start, match.end())
    if scope_stack:
        _blank_range(visible, scope_stack[0], len(visible))
        limitations.append("An unfinished scope prevents analysis of its remaining contents.")
    static = "".join(visible)
    cursor = 0
    while True:
        loop = re.search(r"\\foreach\b", static[cursor:])
        if not loop:
            break
        start = cursor + loop.start()
        in_match = re.search(r"\bin\s*", static[start:])
        body_start = _space(static, start + in_match.end()) if in_match else start
        if body_start < len(static) and static[body_start] == "{":
            group_end = _group_end(static, body_start)
            body_start = _space(static, group_end) if group_end else body_start
        end = _group_end(static, body_start) or _statement_end(static, start)
        _blank_range(visible, start, end)
        cursor = end
    static = "".join(visible)
    transformed = bool(re.search(r"\b(?:shift|xshift|yshift|scale|xscale|yscale|rotate|cm|x|y|z)\s*=", static))
    dynamic_names = bool(re.search(r"name\s+(?:prefix|suffix)\s*=", static))
    if dynamic_names:
        limitations.append("Node-name checks skipped because name prefixes or suffixes require expansion.")
    if transformed and canvas_bounds is not None:
        limitations.append("Canvas-bound checks skipped because coordinate transforms or basis settings are present.")
    # Brace-enclosed label text and styles cannot provide node references.
    cursor = 0
    while cursor < len(static):
        if static[cursor] == "\\":
            cursor += 2
            continue
        if static[cursor] == "{":
            end = _group_end(static, cursor)
            if end:
                _blank_range(visible, cursor, end)
                cursor = end
                continue
        cursor += 1
    static = "".join(visible)

    declaration = re.compile(r"(?:\\(?:node|coordinate)\b|\b(?:node|coordinate)\b)\s*(?:\[[^\]]*\]\s*)?(?:at\s*\([^()]*\)\s*)?\((" + _NAME + r")\)")
    cursor = 0
    while True:
        path = _PATH.search(static, cursor)
        if not path:
            break
        end = _statement_end(static, path.start())
        statement = static[path.start():end]
        decl_ranges = []
        for match in declaration.finditer(statement):
            absolute = path.start() + match.start(1)
            if not dynamic_names:
                declarations.append((match.group(1), absolute))
            decl_ranges.append(match.span(1))
        for match in re.finditer(r"\(([^()]*)\)", statement):
            inner = match.group(1).strip()
            absolute = path.start() + match.start()
            fields = [field.strip() for field in inner.split(",")]
            if len(fields) in (2, 3) and all(_NUMBER.fullmatch(field) for field in fields):
                values = [float(field) for field in fields]
                if not all(math.isfinite(value) for value in values):
                    issue("error", "nonfinite_coordinate", absolute, "Literal coordinate contains NaN, infinity or numeric overflow.")
                    continue
                relative = statement[:match.start()].rstrip().endswith("+")
                coordinates.append({"line": code.count("\n", 0, absolute) + 1,
                                    "values": values, "relative": relative})
                if any(abs(value) > 10000 for value in values):
                    issue("warning", "extreme_coordinate", absolute,
                          "Unusually large literal coordinate; this is a heuristic, not an inferred canvas boundary.")
                if canvas_bounds is not None and not transformed and not relative and len(values) == 2:
                    xmin, ymin, xmax, ymax = canvas_bounds
                    if not (xmin <= values[0] <= xmax and ymin <= values[1] <= ymax):
                        issue("warning", "coordinate_outside_declared_bounds", absolute,
                              "Literal coordinate is outside the caller-provided canvas bounds.")
            elif not any(start >= match.start() and finish <= match.end() for start, finish in decl_ranges):
                name = inner.split(".", 1)[0]
                if not dynamic_names and (re.fullmatch(_NAME, name) or name in _BUILTIN_NAMES):
                    references.append((name, absolute))
        cursor = end
    names = {}
    for name, absolute in declarations:
        if name in names:
            issue("warning", "duplicate_node_name", absolute,
                  "Explicit node/coordinate name '{}' is declared repeatedly; later definitions may shadow earlier ones.".format(name))
        else:
            names[name] = absolute
    for name, absolute in references:
        if name not in names and name not in _BUILTIN_NAMES:
            issue("warning", "unresolved_node_reference", absolute,
                  "Reference '{}' has no explicit declaration in the inspected subset; macros/scopes may define it.".format(name))
    issues.sort(key=lambda item: (item["line"], item["code"]))
    return {
        "schema_version": "1.0", "advisory_only": True, "line_basis": "1-based input source lines",
        "checks": [{"name": "brace_balance", "status": "performed"},
                   {"name": "ordinary_macro_parameters", "status": "partial"},
                   {"name": "explicit_node_names_and_references", "status": "partial"},
                   {"name": "literal_coordinates", "status": "partial"},
                   {"name": "canvas_bounds", "status": "partial" if canvas_bounds is not None and not transformed else "skipped"},
                   {"name": "visual_overlap_and_topology", "status": "not_implemented"}],
        "canvas_bounds": list(canvas_bounds) if canvas_bounds is not None else None,
        "explicit_node_names": sorted(names), "macros": macros, "literal_coordinates": coordinates,
        "issues": issues, "limitations": list(dict.fromkeys(limitations)),
        "summary": {"errors": sum(item["severity"] == "error" for item in issues),
                    "warnings": sum(item["severity"] == "warning" for item in issues)},
    }
