"""Review scene candidates over stored image pixels; never infer semantics.

Exporting a verified scene requires a caller's explicit review declaration in
addition to schema validation. This records a declaration, not authentication or
proof that a person actually inspected the image. Auxiliary candidates are not
rendered objects and can remain unverified until selected into nodes/edges/labels.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from structured_tikz.generator import load_scene
from structured_tikz.validator import SceneValidationError, validate_scene


HEADER_HEIGHT = 56
STATUS_COLORS = {"unverified": (235, 135, 20), "verified": (25, 150, 65), "rejected": (215, 45, 45)}


def _status(item):
    metadata = item.get("metadata", {})
    if not isinstance(metadata, dict):
        raise SceneValidationError("Review metadata must be an object.")
    status = metadata.get("review_status", "unverified")
    if not isinstance(status, str) or status not in STATUS_COLORS:
        raise SceneValidationError("review_status must be unverified, verified or rejected.")
    return status


def validate_review_status(scene):
    """Validate data and declarations without changing any review state."""
    validate_scene(scene)
    _status(scene)
    for kind in ("nodes", "edges", "labels"):
        for item in scene[kind]:
            _status(item)


def _edge_points(edge, nodes):
    if "points" in edge:
        return [tuple(p) for p in edge["points"]]

    def anchor(identifier, name):
        x, y, w, h = nodes[identifier]["bbox"]
        return {"north": (x + w / 2, y), "south": (x + w / 2, y + h),
                "west": (x, y + h / 2), "east": (x + w, y + h / 2)}[name]

    start = anchor(edge["from"], edge.get("from_anchor", "east"))
    end = anchor(edge["to"], edge.get("to_anchor", "west"))
    points = [start]
    if edge.get("routing", "straight") == "orthogonal":
        points.append((end[0], start[1]) if edge.get("first_axis", "x") == "x" else (start[0], end[1]))
    points.append(end)
    return [p for i, p in enumerate(points) if i == 0 or p != points[i - 1]]


def _draw_edge(draw, points, edge):
    color = STATUS_COLORS["verified"]
    points = [(x, y + HEADER_HEIGHT) for x, y in points]
    for start, end in zip(points, points[1:]):
        if edge.get("dashed", False):
            length = math.dist(start, end)
            for offset in range(0, math.ceil(length), 13):
                a, b = offset / length, min(1, (offset + 8) / length)
                draw.line([(start[0] + (end[0] - start[0]) * a, start[1] + (end[1] - start[1]) * a),
                           (start[0] + (end[0] - start[0]) * b, start[1] + (end[1] - start[1]) * b)],
                          fill=color, width=2)
        else:
            draw.line([start, end], fill=color, width=2)

    def arrow(start, end):
        length = math.dist(start, end)
        if length == 0:
            return
        ux, uy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
        size = min(10, length / 2)
        base = (end[0] - ux * size, end[1] - uy * size)
        draw.polygon([end, (base[0] - uy * size / 2, base[1] + ux * size / 2),
                      (base[0] + uy * size / 2, base[1] - ux * size / 2)], fill=color)

    if len(points) >= 2:
        if edge.get("arrow", "none") in {"start", "both"}:
            arrow(points[1], points[0])
        if edge.get("arrow", "none") in {"end", "both"}:
            arrow(points[-2], points[-1])


def save_scene_review(scene, image_path, output_path):
    """Save one ID overlay. Orange is unverified, green verified, red rejected.

    Bounds are scene bboxes, including only the front face of a tensor. This is
    an annotation overlay, not a rendering of the finished scene or its labels.
    """
    from PIL import Image, ImageDraw, ImageFont

    validate_review_status(scene)
    image_path, output_path = Path(image_path).resolve(), Path(output_path).resolve()
    if output_path.suffix.lower() != ".png":
        raise SceneValidationError("Review output must end in .png.")
    if output_path.exists():
        raise FileExistsError("Review output already exists: " + str(output_path))
    source = scene.get("metadata", {}).get("source_image", {})
    if not isinstance(source, dict):
        raise SceneValidationError("metadata.source_image must be an object.")
    expected_hash = source.get("sha256")
    if expected_hash and hashlib.sha256(image_path.read_bytes()).hexdigest() != expected_hash:
        raise SceneValidationError("Scene source image hash does not match the supplied image.")
    with Image.open(image_path) as image:
        expected_size = (scene["canvas"]["width"], scene["canvas"]["height"])
        if image.size != expected_size:
            raise SceneValidationError("Scene canvas dimensions do not match the supplied image.")
        if getattr(image, "n_frames", 1) != 1 or image.width * image.height > 25_000_000:
            raise SceneValidationError("Review needs one image frame of at most 25000000 pixels.")
        rgba = image.convert("RGBA")  # Deliberately no EXIF rotation: match geometry's stored pixels.
        original = Image.alpha_composite(Image.new("RGBA", image.size, "white"), rgba).convert("RGB")
    preview = Image.new("RGB", (original.width, original.height + HEADER_HEIGHT), "white")
    preview.paste(original, (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(preview)
    try:
        font = ImageFont.load_default(size=12)
    except TypeError:
        font = ImageFont.load_default()
    nodes = {node["id"]: node for node in scene["nodes"]}
    confirmed_edges = [edge for edge in scene["edges"] if _status(edge) == "verified"]
    for edge in confirmed_edges:
        _draw_edge(draw, _edge_points(edge, nodes), edge)
    for node in scene["nodes"]:
        x, y, w, h = node["bbox"]
        y += HEADER_HEIGHT
        status = _status(node)
        color = STATUS_COLORS[status]
        draw.rectangle((x, y, x + w, y + h), outline=color, width=2)
        if status == "rejected":
            draw.line((x, y, x + w, y + h), fill=color, width=2)
            draw.line((x + w, y, x, y + h), fill=color, width=2)
        label = node["id"]
        label_box = draw.textbbox((x + 3, y + 3), label, font=font)
        draw.rectangle(label_box, fill="white")
        draw.text((x + 3, y + 3), label, font=font, fill=color)
    draw.text((6, 6), "ORANGE unverified | GREEN verified | RED rejected | Scene bbox / ID overlay", font=font, fill="black")
    draw.text((6, 29), "Confirmed edges: %d | Other edge candidates and text are not inferred or drawn." % len(confirmed_edges),
              font=font, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        preview.save(handle, format="PNG")
    return output_path


def export_verified_scene(scene, output_path, *, confirm_reviewed=False, reviewer=""):
    """Export only caller-declared reviewed objects; never auto-approve a draft."""
    validate_review_status(scene)
    if confirm_reviewed is not True or not isinstance(reviewer, str) or not reviewer.strip():
        raise SceneValidationError("Export needs --confirm-reviewed and a nonempty --reviewer declaration.")
    if len(reviewer) > 200 or any(ord(c) < 32 for c in reviewer):
        raise SceneValidationError("Reviewer must be a short single-line name.")
    if _status(scene) != "verified":
        raise SceneValidationError("Scene metadata.review_status must explicitly be verified.")
    pending = [item["id"] for kind in ("nodes", "edges", "labels") for item in scene[kind]
               if _status(item) == "unverified"]
    if pending:
        raise SceneValidationError("Drawing objects still need review: " + ", ".join(pending[:20]))
    output = copy.deepcopy(scene)
    rejected = {kind: [item for item in output[kind] if _status(item) == "rejected"]
                for kind in ("nodes", "edges", "labels")}
    for kind in rejected:
        output[kind] = [item for item in output[kind] if _status(item) == "verified"]
    validate_scene(output)  # Reject edges left pointing at a removed node.
    output["metadata"]["review_declaration"] = {
        "reviewer": reviewer.strip(), "confirmed_by_caller": True,
        "meaning": "Human review declaration supplied by caller; not independently authenticated.",
        "rejected_objects": rejected,
        "auxiliary_candidates": "Unselected candidates remain auxiliary and are not rendered.",
    }
    encoded = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)
    output_path = Path(output_path).resolve()
    if output_path.suffix.lower() != ".json":
        raise SceneValidationError("Verified output must end in .json.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--image", type=Path, help="Original image for the ID overlay")
    parser.add_argument("--output", type=Path, help="New PNG path; default scene_review.png beside scene")
    parser.add_argument("--export-verified", type=Path, help="New verified JSON path; requires explicit review")
    parser.add_argument("--confirm-reviewed", action="store_true")
    parser.add_argument("--reviewer", default="")
    args = parser.parse_args(argv)
    if args.image is None and args.export_verified is None:
        parser.error("Provide --image for a preview or --export-verified for a reviewed export.")
    if args.output is not None and args.image is None:
        parser.error("--output needs --image.")
    try:
        scene, _ = load_scene(args.scene)
        if args.image is not None:
            preview = save_scene_review(scene, args.image, args.output or args.scene.parent / "scene_review.png")
            print("Review preview:", preview)
        if args.export_verified is not None:
            path = export_verified_scene(scene, args.export_verified,
                                         confirm_reviewed=args.confirm_reviewed, reviewer=args.reviewer)
            print("Caller-declared verified scene:", path)
    except (OSError, ValueError) as exc:
        parser.exit(2, "Scene review error: " + str(exc) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
