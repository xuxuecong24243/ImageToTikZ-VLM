"""Deterministic geometric candidates, not OCR or a semantic diagram parser.

Scores rank heuristic evidence; they are not calibrated probabilities. Coordinates
refer to the original stored pixel array. Nothing in this module calls an API.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


MAX_PIXELS = 25_000_000
MAX_RECTANGLES = 60
MAX_LINES = 180


class GeometryError(RuntimeError):
    """Analysis failed; callers must not silently use partial geometry."""


def _dependencies():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise GeometryError(
            "Geometry analysis needs OpenCV and NumPy. Use the project .venv and "
            "install its requirements-geometry.txt; no dependency is installed automatically."
        ) from exc
    return cv2, np


def _load_image(path):
    cv2, np = _dependencies()
    from PIL import Image
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise GeometryError("Image file does not exist: " + str(source))
    try:
        with Image.open(source) as image:
            width, height = image.size
            if width < 1 or height < 1 or width * height > MAX_PIXELS:
                raise GeometryError("Image must contain 1 to 25000000 pixels; it is not silently resized.")
            if getattr(image, "n_frames", 1) != 1:
                raise GeometryError("Geometry analysis requires one image frame.")
            metadata = {"width": width, "height": height, "aspect_ratio": round(width / height, 8),
                        "channels": len(image.getbands()), "color_mode": image.mode,
                        "analysis_color_mode": "BGR (alpha composited onto white)",
                        "exif_orientation": int(image.getexif().get(274, 1)),
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", image.size, "white")
            rgb = Image.alpha_composite(background, rgba).convert("RGB")
            pixels = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    except GeometryError:
        raise
    except Exception as exc:
        raise GeometryError("Cannot decode image: " + str(source)) from exc
    return pixels, metadata


def normalized_bbox(box, width, height):
    x, y, w, h = box
    return {"x": round(x / width, 6), "y": round(y / height, 6),
            "width": round(w / width, 6), "height": round(h / height, 6)}


def _pixel_bbox(box):
    return dict(zip(("x", "y", "width", "height"), (int(v) for v in box)))


def _intersection(a, b):
    x = max(a[0], b[0]); y = max(a[1], b[1])
    return max(0, min(a[0] + a[2], b[0] + b[2]) - x) * max(0, min(a[1] + a[3], b[1] + b[3]) - y)


def deduplicate_rectangles(candidates, image_area):
    """Collapse near-duplicate edges and suppress contained grid-cell candidates.

    Large enclosing panels (>20% image area) do not erase their contents.
    Keeping an outer bounding box does not claim it is one semantic object.
    """
    kept = []
    for candidate in sorted(candidates, key=lambda c: (-c["box"][2] * c["box"][3], -c["score"], c["box"])):
        box = candidate["box"]
        area = box[2] * box[3]
        discard = False
        for other in kept:
            outer = other["box"]
            outer_area = outer[2] * outer[3]
            overlap = _intersection(box, outer)
            iou = overlap / max(1, area + outer_area - overlap)
            near_edges = max(abs(box[i] - outer[i]) for i in range(4)) <= 4
            if iou >= .83 or near_edges:
                other["methods"] = sorted(set(other["methods"] + candidate["methods"]))
                other["score"] = max(other["score"], candidate["score"])
                discard = True
                break
            if overlap / area >= .97 and area < .9 * outer_area and outer_area <= .20 * image_area:
                other["suppressed_inner_candidates"] = other.get("suppressed_inner_candidates", 0) + 1
                discard = True
                break
        if not discard:
            kept.append(dict(candidate))
    return kept


def _rectangles(pixels, edges, width, height, cv2, np):
    minimum_w, minimum_h = max(24, width * .014), max(16, height * .016)
    candidates = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        quad = cv2.approxPolyDP(contour, .02 * perimeter, True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        box = cv2.boundingRect(quad)
        x, y, w, h = box
        if w < minimum_w or h < minimum_h or not .08 <= w / h <= 16:
            continue
        if w * h > width * height * .85:
            continue
        points = quad.reshape(4, 2).astype(float)
        sides = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(sides, axis=1)
        if np.min(lengths) == 0:
            continue
        cosines = np.abs(np.sum(sides * np.roll(sides, -1, axis=0), axis=1)
                         / (lengths * np.roll(lengths, -1)))
        rectangularity = float(cv2.contourArea(quad)) / (w * h)
        axis_error = float(np.max(np.min(np.abs(sides), axis=1) / lengths))
        if float(np.max(cosines)) <= .18 and rectangularity >= .82 and axis_error <= .12:
            candidates.append({"box": box, "score": round(.7 + .2 * rectangularity, 4),
                               "methods": ["canny_four_corner_axis_aligned_contour"],
                               "rectangularity": round(rectangularity, 4)})

    # Filled, chromatic shapes often have broken outer edges due to grids/text.
    # This yields bounding boxes of components, not claims of rectangular shape.
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 22, 80]), np.array([179, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    colored, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in colored:
        box = cv2.boundingRect(contour)
        x, y, w, h = box
        area = float(cv2.contourArea(contour))
        if (w >= minimum_w and h >= minimum_h and .08 <= w / h <= 16
                and .65 <= area / (w * h) and w * h < width * height * .50):
            candidates.append({"box": box, "score": .72,
                               "methods": ["chromatic_component_bounding_box"],
                               "rectangularity": round(area / (w * h), 4)})
    result = deduplicate_rectangles(candidates, width * height)
    result = sorted(result, key=lambda c: (-c["score"], -c["box"][2] * c["box"][3]))[:MAX_RECTANGLES]
    result.sort(key=lambda c: (c["box"][1], c["box"][0]))
    objects = []
    for index, candidate in enumerate(result, 1):
        x, y, w, h = candidate["box"]
        objects.append({"id": "rect_%03d" % index, "bbox_px": _pixel_bbox(candidate["box"]),
                        "bbox": normalized_bbox(candidate["box"], width, height),
                        "center_px": {"x": x + w / 2, "y": y + h / 2},
                        "center": {"x": round((x + w / 2) / width, 6), "y": round((y + h / 2) / height, 6)},
                        "score": candidate["score"], "score_type": "heuristic_rank_not_probability",
                        "evidence": {"methods": candidate["methods"],
                                     "contour_area_to_bbox_ratio": candidate["rectangularity"],
                                     "suppressed_inner_candidates": candidate.get("suppressed_inner_candidates", 0),
                                     "meaning": "geometric_candidate_not_semantic_object"}})
    return objects, {"contours": len(contours), "rectangle_candidates_before_dedup": len(candidates),
                     "rectangle_candidates_removed_or_capped": len(candidates) - len(objects)}


def merge_line_segments(segments, *, distance_tolerance=3.0, max_gap=10.0):
    """Merge approximately collinear, nearby segments without assigning direction."""
    ordered = sorted(segments, key=lambda s: (-math.hypot(s[2] - s[0], s[3] - s[1]), tuple(s)))
    kept = []
    for raw in ordered:
        points = [(float(raw[0]), float(raw[1])), (float(raw[2]), float(raw[3]))]
        if points[0] > points[1]:
            points.reverse()
        length = math.dist(points[0], points[1])
        if length == 0:
            continue
        direction = ((points[1][0] - points[0][0]) / length, (points[1][1] - points[0][1]) / length)
        merged = False
        for segment in kept:
            a, b, support = segment
            old_length = math.dist(a, b)
            u = ((b[0] - a[0]) / old_length, (b[1] - a[1]) / old_length)
            if abs(u[0] * direction[0] + u[1] * direction[1]) < math.cos(math.radians(3)):
                continue
            projections = [(p[0] - a[0]) * u[0] + (p[1] - a[1]) * u[1] for p in points]
            distances = [(p[0] - a[0]) * -u[1] + (p[1] - a[1]) * u[0] for p in points]
            gap = max(0, min(projections) - old_length, -max(projections))
            if max(abs(v) for v in distances) > distance_tolerance or gap > max_gap:
                continue
            low, high = min(0, *projections), max(old_length, *projections)
            offset = sum(distances) / (2 * (support + 1))
            segment[0] = (a[0] + low * u[0] - offset * u[1], a[1] + low * u[1] + offset * u[0])
            segment[1] = (a[0] + high * u[0] - offset * u[1], a[1] + high * u[1] + offset * u[0])
            segment[2] += 1
            merged = True
            break
        if not merged:
            kept.append([points[0], points[1], 1])
    return kept


def _inside(point, box, padding=3):
    x, y, w, h = (box[k] for k in ("x", "y", "width", "height"))
    return x - padding <= point[0] <= x + w + padding and y - padding <= point[1] <= y + h + padding


def _lines(edges, rectangles, width, height, cv2, np):
    min_length = max(22, int(min(width, height) * .03))
    raw = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=max(18, min_length),
                          minLineLength=min_length, maxLineGap=6)
    segments = [] if raw is None else [tuple(int(v) for v in line[0]) for line in raw]
    raw_count = len(segments)
    segments = sorted(segments, key=lambda s: -math.hypot(s[2] - s[0], s[3] - s[1]))[:800]
    merged = merge_line_segments(segments, max_gap=max(6, min(width, height) * .012))
    supported_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    candidates = []
    for a, b, support in merged:
        a = (min(width - 1, max(0, a[0])), min(height - 1, max(0, a[1])))
        b = (min(width - 1, max(0, b[0])), min(height - 1, max(0, b[1])))
        length = math.dist(a, b)
        angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180
        orientation = "horizontal" if min(angle, 180 - angle) <= 4 else "vertical" if abs(angle - 90) <= 4 else "diagonal"
        samples = max(2, int(length) + 1)
        xs = np.linspace(a[0], b[0], samples).round().astype(int)
        ys = np.linspace(a[1], b[1], samples).round().astype(int)
        coverage = supported_edges[ys, xs] > 0
        longest = run = 0
        for present in coverage:
            run = run + 1 if present else 0
            longest = max(longest, run)
        edge_fraction, continuous_fraction = float(np.mean(coverage)), longest / samples
        contained = []
        for rectangle in rectangles:
            box = rectangle["bbox_px"]
            within = ((xs >= box["x"] - 3) & (xs <= box["x"] + box["width"] + 3)
                      & (ys >= box["y"] - 3) & (ys <= box["y"] + box["height"] + 3))
            if float(np.mean(within)) >= .8:
                contained.append(rectangle["id"])
        score = min(.95, .4 + length / math.hypot(width, height) * 1.7 + min(support, 4) * .05)
        # Hough can stitch many separate text strokes into a misleading long line.
        # Downrank weak continuity, but keep candidates (including true dashed lines).
        score *= edge_fraction * (.35 + .65 * continuous_fraction)
        if contained:
            score *= .65
        candidates.append({"a": a, "b": b, "support": support, "length": length,
                           "orientation": orientation, "contained": contained, "score": round(score, 4),
                           "edge_fraction": round(edge_fraction, 4), "continuous_fraction": round(continuous_fraction, 4)})
    candidates.sort(key=lambda c: (-c["score"], -c["length"], c["a"]))
    objects = []
    for index, candidate in enumerate(candidates[:MAX_LINES], 1):
        a, b = candidate["a"], candidate["b"]
        normalized = [round(a[0] / width, 6), round(a[1] / height, 6),
                      round(b[0] / width, 6), round(b[1] / height, 6)]
        item = dict(zip(("x1", "y1", "x2", "y2"), normalized))
        item.update(id="line_%03d" % index,
                    pixel=dict(zip(("x1", "y1", "x2", "y2"), [round(v, 2) for v in (*a, *b)])),
                    orientation=candidate["orientation"], direction="unknown", length_px=round(candidate["length"], 2),
                    endpoint_candidates=[{"x": normalized[0], "y": normalized[1], "role": "unknown"},
                                         {"x": normalized[2], "y": normalized[3], "role": "unknown"}],
                    score=candidate["score"], score_type="heuristic_rank_not_probability",
                    evidence={"method": "canny_hough_lines_p_collinear_merge", "support_segments": candidate["support"],
                              "contained_in": candidate["contained"],
                              "possible_role": "border_or_grid" if candidate["contained"] else "unclassified_line",
                              "edge_coverage_fraction": candidate["edge_fraction"],
                              "longest_continuous_support_fraction": candidate["continuous_fraction"],
                              "arrowhead_detection": "not_performed"})
        objects.append(item)
    return objects, {"raw_hough_segments": raw_count, "hough_segments_processed": len(segments),
                     "merged_segments": len(merged), "line_candidates_capped": max(0, len(merged) - MAX_LINES)}


def _regions(rectangles, lines, width, height):
    groups = [[r] for r in rectangles]
    relations = []

    def bounds(group):
        boxes = [r["bbox_px"] for r in group]
        x, y = min(b["x"] for b in boxes), min(b["y"] for b in boxes)
        return x, y, max(b["x"] + b["width"] for b in boxes) - x, max(b["y"] + b["height"] for b in boxes) - y

    # Merge nearby aligned groups, while capping extent to avoid one whole-image group.
    for left in rectangles:
        a = left["bbox_px"]
        for right in rectangles:
            if left["id"] >= right["id"]:
                continue
            b = right["bbox_px"]
            dx = max(0, a["x"] - b["x"] - b["width"], b["x"] - a["x"] - a["width"])
            dy = max(0, a["y"] - b["y"] - b["height"], b["y"] - a["y"] - a["height"])
            overlap_x = max(0, min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"]))
            overlap_y = max(0, min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"]))
            nearby = ((dx <= width * .055 and overlap_y >= .3 * min(a["height"], b["height"]))
                      or (dy <= height * .05 and overlap_x >= .3 * min(a["width"], b["width"])))
            if not nearby:
                continue
            first = next(g for g in groups if left in g)
            second = next(g for g in groups if right in g)
            if first is second:
                continue
            box = bounds(first + second)
            if box[2] > width * .48 or box[3] > height * .48:
                continue
            first.extend(second)
            groups.remove(second)
            relations.append([left["id"], right["id"]])
    result = []
    for index, group in enumerate(groups, 1):
        box = bounds(group)
        members = [r["id"] for r in group]
        line_ids = [line["id"] for line in lines
                    if _inside((line["pixel"]["x1"], line["pixel"]["y1"]), _pixel_bbox(box))
                    and _inside((line["pixel"]["x2"], line["pixel"]["y2"]), _pixel_bbox(box))]
        result.append({"id": "region_%03d" % index, "bbox_px": _pixel_bbox(box),
                       "bbox": normalized_bbox(box, width, height), "member_ids": members + line_ids,
                       "evidence": {"method": "proximity_alignment_grouping", "semantic_label": None,
                                    "aligned_pairs": [pair for pair in relations if all(v in members for v in pair)],
                                    "meaning": "bounds_of_detected_geometry_only_not_text_or_semantic_region"}})
    return result


def _segment_box_interval(pixel, box):
    """Clip a segment to a box; return its parameter interval in [0,1]."""
    x, y, w, h = box
    low, high = 0.0, 1.0
    for start, delta, minimum, maximum in (
            (pixel["x1"], pixel["x2"] - pixel["x1"], x, x + w),
            (pixel["y1"], pixel["y2"] - pixel["y1"], y, y + h)):
        if abs(delta) < 1e-12:
            if not minimum <= start <= maximum:
                return None
        else:
            a, b = sorted(((minimum - start) / delta, (maximum - start) / delta))
            low, high = max(low, a), min(high, b)
            if high <= low:
                return None
    return (low, high) if high > low else None


def classify_text_overlap(lines, text_regions, width, height):
    """Annotate all original candidates; never remove lines or change their score.

    Union of clipped intervals gives a length ratio without double counting
    overlapping boxes. Center containment alone is never a deletion criterion.
    """
    result = []
    short_limit, long_limit = max(32, min(width, height) * .08), max(80, min(width, height) * .14)
    for original in lines:
        line = dict(original)
        intervals, touched = [], []
        for region in text_regions:
            interval = _segment_box_interval(line["pixel"], region["bbox_px"])
            if interval:
                intervals.append(interval)
                touched.append(region)
        union = []
        for start, end in sorted(intervals):
            if union and start <= union[-1][1]:
                union[-1][1] = max(end, union[-1][1])
            else:
                union.append([start, end])
        overlap = min(1.0, sum(end - start for start, end in union))
        evidence = line.get("evidence", {})
        protected = bool(evidence.get("contained_in")) or evidence.get("possible_role") == "border_or_grid"
        length = line["length_px"]
        classification, reason = "geometry", "no_text_regions" if not text_regions else "no_text_overlap"
        if overlap > 0:
            classification = "uncertain"
            reason = "partial_text_overlap"
            if protected:
                classification, reason = "geometry", "protected_border_or_grid"
            elif length >= long_limit:
                reason = "long_line_with_text_overlap"
            elif (line["orientation"] == "horizontal" and touched
                  and length > 2.5 * max(r["bbox_px"][3] for r in touched)):
                reason = "possible_dimension_or_formula_rule"
            elif overlap >= .85 and length <= short_limit:
                classification, reason = "text_stroke_candidate", "short_line_inside_text_region"
        multiplier = .1 if classification == "text_stroke_candidate" else .65 if classification == "uncertain" else 1.0
        line.update(classification=classification, text_overlap_ratio=round(overlap, 6),
                    text_region_ids=[r["id"] for r in touched], filter_reason=reason,
                    context_priority=round(line["score"] * multiplier, 6))
        result.append(line)
    return result


def analyze_image(image_path, *, text_regions=None, text_detector=None):
    cv2, np = _dependencies()
    pixels, metadata = _load_image(image_path)
    width, height = metadata["width"], metadata["height"]
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 60, 160)
    rectangles, rectangle_counts = _rectangles(pixels, edges, width, height, cv2, np)
    lines, line_counts = _lines(edges, rectangles, width, height, cv2, np)
    from text_region_detector import detect_text_regions
    detected_text, text_detection = detect_text_regions(
        pixels, supplied_regions=text_regions, detector=text_detector)
    lines = classify_text_overlap(lines, detected_text, width, height)
    regions = _regions(rectangles, lines, width, height)
    warnings = ["All detections are geometric candidates, not semantic objects or OCR.",
                "Scores are heuristic ranks, not calibrated confidence probabilities.",
                "Lines may be text strokes, grids or borders. Arrow direction is always unknown.",
                "Weak edge continuity lowers line rank; real dashed connectors may therefore be under-ranked.",
                "Chromatic component boxes may enclose a whole perspective shape, not one rectangular face.",
                "Contained rectangles are suppressed to reduce grid-cell detections; some valid nested objects may be lost.",
                "Regions group detected geometry only and do not locate text labels or semantic modules."]
    if metadata["exif_orientation"] != 1:
        warnings.append("EXIF rotation is not applied: measurements use stored pixels; check orientation against the VLM's view.")
    if not rectangles:
        warnings.append("No rectangle candidates passed the filters.")
    if not lines:
        warnings.append("No line candidates passed the filters.")
    if line_counts["raw_hough_segments"] > 800 or line_counts["line_candidates_capped"]:
        warnings.append("Line candidate resource caps were applied; see counts.")
    warnings.extend(text_detection["warnings"])
    warnings.append("Line classification is heuristic: geometry does not prove a connector; uncertain lines are retained.")
    classifications = {kind: sum(line["classification"] == kind for line in lines)
                       for kind in ("geometry", "text_stroke_candidate", "uncertain")}
    return {"schema_version": "1.0", "image": metadata,
            "coordinate_system": "top-left-normalized", "axes": {"x": "right", "y": "down"},
            "rectangles": rectangles, "lines": lines, "regions": regions, "warnings": warnings,
            "text_regions": detected_text, "text_detection": text_detection,
            "text_filter": {"policy": "retain_all_candidates; exclude_only_text_stroke_candidates_from_context",
                            "short_line_max_px": max(32, min(width, height) * .08),
                            "long_line_min_px": max(80, min(width, height) * .14), "strong_overlap_min": .85,
                            "classifications": classifications},
            "parameters": {"canny": [60, 160], "gaussian_kernel": [3, 3], "polygon_epsilon_fraction": .02,
                           "chromatic_saturation_min": 22, "max_rectangles": MAX_RECTANGLES, "max_lines": MAX_LINES,
                           "opencv_version": cv2.__version__, "debug_line_label_limit": 0},
            "counts": dict(rectangle_counts, **line_counts, rectangles=len(rectangles), lines=len(lines), regions=len(regions),
                           text_regions=len(detected_text), **classifications)}


def _debug_image(pixels, geometry, cv2, np, *, before=False):
    debug = pixels.copy()
    for region in geometry["regions"]:
        b = region["bbox_px"]
        x, y, w, h = (int(b[k]) for k in ("x", "y", "width", "height"))
        color = (200, 0, 180)
        for offset in range(0, w, 14):
            cv2.line(debug, (x + offset, y), (x + min(offset + 7, w), y), color, 1)
            cv2.line(debug, (x + offset, y + h), (x + min(offset + 7, w), y + h), color, 1)
        for offset in range(0, h, 14):
            cv2.line(debug, (x, y + offset), (x, y + min(offset + 7, h)), color, 1)
            cv2.line(debug, (x + w, y + offset), (x + w, y + min(offset + 7, h)), color, 1)
    if not before:
        for region in geometry.get("text_regions", []):
            x, y, w, h = (round(v) for v in region["bbox_px"])
            cv2.rectangle(debug, (x, y), (x + w, y + h), (220, 100, 30), 1)
    for index, line in enumerate(geometry["lines"]):
        p = line["pixel"]
        a, b = (round(p["x1"]), round(p["y1"])), (round(p["x2"]), round(p["y2"]))
        classification = "geometry" if before else line.get("classification", "geometry")
        if classification == "uncertain":
            length = max(1, math.dist(a, b))
            for offset in range(0, math.ceil(length), 10):
                start, end = offset / length, min(1, (offset + 5) / length)
                p1 = tuple(round(a[i] + (b[i] - a[i]) * start) for i in (0, 1))
                p2 = tuple(round(a[i] + (b[i] - a[i]) * end) for i in (0, 1))
                cv2.line(debug, p1, p2, (130, 130, 130), 1, cv2.LINE_AA)
        else:
            color = (30, 30, 230) if classification == "text_stroke_candidate" else (30, 165, 30)
            cv2.line(debug, a, b, color, 1, cv2.LINE_AA)
    for rectangle in geometry["rectangles"]:
        b = rectangle["bbox_px"]
        x, y, w, h = (int(b[k]) for k in ("x", "y", "width", "height"))
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 80, 240), 2)
        cv2.putText(debug, rectangle["id"], (x + 2, min(debug.shape[0] - 3, y + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, .40, (0, 50, 200), 1, cv2.LINE_AA)
    header = np.full((70, debug.shape[1], 3), 255, dtype=np.uint8)
    labels = (("BEFORE: all original line candidates GREEN | ORANGE: rectangles | MAGENTA DASH: geometric groups",
               "Same coordinates and candidates as AFTER. No line IDs; inspect geometry.json for measurements.") if before else
              ("AFTER: GREEN geometry | RED text-stroke candidates | GRAY DASH uncertain | ORANGE rectangles",
               "BLUE text boxes (source in JSON; fallback is NOT OCR) | No line IDs; all candidates retained."))
    for index, label in enumerate(labels):
        cv2.putText(header, label, (8, 24 + index * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    min(.5, max(.15, debug.shape[1] / 2100)), (20, 20, 20), 1, cv2.LINE_AA)
    return np.concatenate((header, debug), axis=0)


def save_geometry_artifacts(image_path, output_dir, geometry=None):
    """Persist JSON and a labeled overlay without overwriting any prior artifact."""
    cv2, np = _dependencies()
    geometry = analyze_image(image_path) if geometry is None else geometry
    pixels, metadata = _load_image(image_path)
    if geometry.get("image", {}).get("sha256") != metadata["sha256"]:
        raise GeometryError("Geometry belongs to a different image; refusing a misleading overlay.")
    output = Path(output_dir).expanduser().resolve()
    paths = {"geometry": output / "geometry.json", "debug": output / "geometry_debug.png",
             "before": output / "geometry_before.png", "after": output / "geometry_after.png"}
    if any(path.exists() for path in paths.values()):
        raise GeometryError("Geometry output already exists; use a new run directory.")
    try:
        data = json.dumps(geometry, ensure_ascii=False, indent=2, allow_nan=False)
        debug = _debug_image(pixels, geometry, cv2, np)
        success, encoded = cv2.imencode(".png", debug)
        if not success:
            raise GeometryError("OpenCV could not encode the geometry debug PNG.")
        success, encoded_before = cv2.imencode(".png", _debug_image(pixels, geometry, cv2, np, before=True))
        if not success:
            raise GeometryError("OpenCV could not encode the before PNG.")
        output.mkdir(parents=True, exist_ok=True)
        with paths["geometry"].open("x", encoding="utf-8") as handle:
            handle.write(data)
        with paths["debug"].open("xb") as handle:
            handle.write(encoded.tobytes())
        with paths["after"].open("xb") as handle:
            handle.write(encoded.tobytes())
        with paths["before"].open("xb") as handle:
            handle.write(encoded_before.tobytes())
    except (OSError, TypeError, ValueError) as exc:
        raise GeometryError("Geometry artifacts could not be saved completely: " + str(exc)) from exc
    return paths
