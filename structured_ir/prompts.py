"""Replaceable stage-one interface. No HTTP, model, configuration or Scene calls."""
import json
from .schema import ir_schema


def build_ir_prompt(image_size, region_plan=None):
    if (not isinstance(image_size, (tuple, list)) or len(image_size) != 2
            or any(type(n) is not int or not 1 <= n <= 20000 for n in image_size)):
        raise ValueError('image_size must contain two positive integer source dimensions.')
    context = None
    if region_plan is not None:
        # Measurements are candidate evidence, not a compulsory node inventory.
        context = {key: region_plan[key] for key in ('canvas', 'candidates', 'overlaps', 'diagnostics')
                   if key in region_plan}
    requirements = """Recognize the entire technical image into independent image structure IR.
Return exactly one complete JSON object satisfying the IR Schema below. No
Markdown fences, explanatory prose, TikZ or final Scene. Stage two is a local
deterministic conversion; do not emit a sample or summarize the image as a few
example nodes. Preserve all reliably visible text, code, mathematical notation,
containers, headings and meaningful connections without inventing unseen data.
Use raw_text exactly as observed; do not silently correct code or mathematics.
Record text_type and reading_order for each text-bearing element. Use confidence
and uncertainty for ambiguous observations; describe unexpressible or unreadable
items in diagnostics rather than silently omitting or completing their content.
Coordinates are ORIGINAL full-image pixels, origin top-left, x right and y down.
Bounding boxes are [left,top,width,height]. Region candidates may overlap or be
wrong; they are geometric evidence, NOT recognized semantic regions or a required
number of elements. Text-like components are not real text-node counts.
Keep region parent/child ownership explicit. Keep semantic connections separate
from geometric-only line elements: source_id, target_id, direction and anchors
must identify the actual relationship. Preserve distinct information flows,
decision diamonds, branch labels and merge relationships. Do not flatten branches
into one sequential column. Use kind=branch for decision exits and kind=merge for
converging connections. Path endpoints must agree with their semantic anchors.
If branch text position is known, include label_bbox. Retain diamond in IR even
though the downstream renderer may not support it. Never substitute a rectangle
to conceal missing renderer support. Valid IR does not establish content coverage
or visual correctness. Do not invent nodes just to satisfy a coverage check.
"""
    return (requirements + '\nRequired source canvas dimensions: ' + json.dumps(list(image_size))
            + '\nIR_SCHEMA_JSON:\n' + json.dumps(ir_schema(), ensure_ascii=False, sort_keys=True)
            + '\nREGION_CANDIDATE_EVIDENCE_JSON:\n' + json.dumps(context, ensure_ascii=False, sort_keys=True))


def parse_ir_response(text):
    """Reuse strict JSON safety without Scene provenance edits or recovery.

The caller saves raw bytes first and runs inspect_ir separately so JSON parsing
errors remain distinguishable from structural or content-quality failures.
"""
    from auto_scene_generator import _strict_json
    return _strict_json(text)
