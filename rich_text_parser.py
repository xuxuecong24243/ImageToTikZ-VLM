"""Parse explicit inline math only; never infer mathematics from a command.

Prose is literal except for the conventional escaped dollar (\\$ -> $).
The caller retains the original source. Math contents are kept byte-for-byte
as Python strings and checked with the renderer's existing command allowlist.
Display math, nested/mismatched delimiters and empty math are rejected.
"""
from __future__ import annotations

from structured_tikz.validator import validate_math, validate_rich_segments


class RichTextParseError(ValueError):
    """The complete input cannot be safely segmented; keep its literal source."""


def _math_end(text, start, closer):
    index = start
    while index < len(text):
        if text.startswith(closer, index):
            return index
        if text[index] == '$' or text.startswith((r'\(', r'\)'), index):
            raise RichTextParseError('Nested or mismatched math delimiter at character {}.'.format(index))
        # Skip an escaped character, so \\) does not close a \(...\) span.
        index += 2 if text[index] == '\\' else 1
    raise RichTextParseError('Unclosed math delimiter at character {}.'.format(start))


def parse_rich_text(text: str) -> list[dict]:
    """Return ordered text/math spans, or fail atomically with an explanation."""
    if not isinstance(text, str) or len(text) > 4000:
        raise RichTextParseError('Input must be a string of at most 4000 characters.')
    segments, prose = [], []

    def flush():
        if prose:
            segments.append({'type': 'text', 'content': ''.join(prose)})
            prose.clear()

    index = 0
    while index < len(text):
        char = text[index]
        if text.startswith(r'\)', index):
            raise RichTextParseError('Unexpected closing delimiter at character {}.'.format(index))
        opener = '$' if char == '$' else r'\(' if text.startswith(r'\(', index) else None
        if opener:
            flush()
            closer = '$' if opener == '$' else r'\)'
            start = index + len(opener)
            end = _math_end(text, start, closer)
            content = text[start:end]
            try:
                validate_math(content)
            except ValueError as error:
                raise RichTextParseError('Invalid math at character {}: {}'.format(start, error)) from error
            segments.append({'type': 'math', 'content': content})
            index = end + len(closer)
        elif char == '\\' and index + 1 < len(text) and text[index+1] in ('$', '\\'):
            # Pair backslashes to respect delimiter escape parity; preserve both.
            prose.append('$' if text[index+1] == '$' else text[index:index+2])
            index += 2
        else:
            prose.append(char)
            index += 1
    flush()
    if not segments:  # Empty plain text has no mathematical content.
        return []
    try:
        validate_rich_segments(segments)
    except ValueError as error:
        raise RichTextParseError(str(error)) from error
    return segments
