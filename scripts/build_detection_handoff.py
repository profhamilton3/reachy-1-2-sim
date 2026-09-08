#!/usr/bin/env python3
"""Assemble a simulator detection/classification set in the lab's own layout.

    python3 scripts/build_detection_handoff.py --out <dir> [--detection 300]
                                                           [--per-class 80]

Runs generate_dataset.py several times and reshapes the output into the two
things the vision work actually consumes:

  detection/       YOLO — images/, labels/, classes.txt, annotations/
  classification/  one folder per class, the layout the Coral on-device
                   imprinting flow trains from (data/calibration/annotation/
                   {cube,cylinder,empty}/ in IITG-Reachy-Project)

WHY TWO LAYOUTS.  The classifier in the lab today learns from folder names and
needs no boxes; detection is where the work is going, because a class label
with no pixel location cannot drive a grasp.  Shipping both means the existing
model can be evaluated against simulator frames immediately, without waiting
for the detector, and the same render run backs both.

WHY SEPARATE RUNS PER CLASS.  A classification frame carries ONE label for the
whole image, so a board holding a cube and a cylinder at once has no valid
folder to go in.  Each classification run therefore drops every object but one;
the empty run drops all of them, which is a real class (`empty/`) and not a
failure.

PORTRAIT.  Frames are rendered 640x480 landscape, because that is the
orientation the measured calibration is expressed in (fy 407 -> 61.1 deg
vertical over 480 rows), then rotated to 480x640 — the shape
`reachy.left_camera.last_frame` actually returns and the shape every real
capture in the lab set is stored in.  Boxes are rotated with the pixels; a
label file that still described the landscape frame would be silently wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

from PIL import Image

_REPO = pathlib.Path(__file__).resolve().parents[1]
_GEN = _REPO / "native_mujoco" / "cli" / "generate_dataset.py"
_SCENE = _REPO / "scenes" / "FWDCenterLabSiva.yaml"

# Exposure that puts rendered luminance on the real feed's.  Measured against
# the lab captures: board 130 vs 130, object 59 vs 65, object/board ratio 0.473
# vs 0.50, no clipped pixels.  See the scene file for where those come from.
_EXPOSURE = 0.40

# Every frame at the calibrated zoom, for two reasons that happen to agree.
# The barrel profile is measured at INTER and nowhere else, and the render
# margin that keeps the warp inside its source buffer is derived from that
# field — at OUT (98 deg vs 61) the warp reads past the buffer and folds the
# frame, which showed up as three boxes covering 15x their object.  And the
# real captures this set is meant to be compared against were all taken at one
# zoom, so mixing fields in would confound the comparison anyway.  Per-zoom
# calibration is an open ask with the operator; revisit when it lands.
_ZOOM = "inter"

# id -> folder/class name.  The ids are the scene's; the names are the lab's.
_CLASSES = {
    "red_cube": "cube",
    "blue_cylinder": "cylinder",
    "soda_can": "can",
    "foam_block": "foam",
}


def _run(out: pathlib.Path, count: int, seed: int, drop: list[str]) -> dict:
    cmd = [sys.executable, str(_GEN),
           "--scene", str(_SCENE), "--out", str(out),
           "--count", str(count), "--seed", str(seed),
           "--distort", "--exposure", str(_EXPOSURE), "--zoom", _ZOOM]
    for d in drop:
        cmd += ["--drop", d]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.loads((out / "manifest.json").read_text())


def _to_portrait(src: pathlib.Path, dst: pathlib.Path) -> tuple[int, int]:
    """Rotate a landscape frame to the stored portrait orientation."""
    im = Image.open(src).rotate(-90, expand=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=92)
    return im.size


def _rotate_box(b: dict, w: int, h: int) -> tuple[int, int, int, int]:
    """Landscape (w x h) pixel box -> the same object in the rotated frame.

    A -90 deg (clockwise) rotation maps (x, y) -> (h - 1 - y, x), so the new
    frame is h wide and w tall.  Corners are remapped and re-cornered rather
    than assumed, because the mapping swaps which extreme is min and which
    is max on one axis.
    """
    xs = [h - 1 - b["y_min"], h - 1 - b["y_max"]]
    ys = [b["x_min"], b["x_max"]]
    return min(xs), min(ys), max(xs), max(ys)


def build_detection(work: pathlib.Path, out: pathlib.Path, count: int,
                    seed: int) -> dict:
    raw = work / "detection_raw"
    manifest = _run(raw, count, seed, drop=[])

    images, labels, annots = out / "images", out / "labels", out / "annotations"
    for d in (images, labels, annots):
        d.mkdir(parents=True, exist_ok=True)

    classes = manifest["classes"]
    index = {c: i for i, c in enumerate(classes)}
    (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    kept = 0
    per_class = {c: 0 for c in classes}
    for ann_path in sorted((raw / "annotations").glob("*.json")):
        rec = json.loads(ann_path.read_text())
        stem = rec["stem"]
        pw, ph = _to_portrait(raw / "images" / f"{stem}.jpg",
                              images / f"{stem}.jpg")
        lines = []
        for b in rec["boxes"]:
            x0, y0, x1, y1 = _rotate_box(b, rec["width"], rec["height"])
            b["x_min"], b["y_min"], b["x_max"], b["y_max"] = x0, y0, x1, y1
            cx = (x0 + x1 + 1) / 2.0 / pw
            cy = (y0 + y1 + 1) / 2.0 / ph
            lines.append(f"{index[b['semantic_class']]} {cx:.6f} {cy:.6f} "
                         f"{(x1 - x0 + 1) / pw:.6f} {(y1 - y0 + 1) / ph:.6f}")
            per_class[b["semantic_class"]] += 1
        (labels / f"{stem}.txt").write_text(
            ("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        rec["width"], rec["height"] = pw, ph
        rec["orientation"] = "portrait (rotated -90 from render)"
        (annots / f"{stem}.json").write_text(json.dumps(rec, indent=2),
                                             encoding="utf-8")
        kept += 1

    return {"frames": kept, "resolution": [pw, ph], "classes": classes,
            "boxes_per_class": per_class, "seed": seed,
            "empty_frames": manifest["empty_frames"]}


def build_classification(work: pathlib.Path, out: pathlib.Path, per_class: int,
                         seed: int) -> dict:
    counts: dict[str, int] = {}
    runs = [(oid, name, [o for o in _CLASSES if o != oid])
            for oid, name in _CLASSES.items()]
    runs.append((None, "empty", list(_CLASSES)))

    for i, (_oid, name, drop) in enumerate(runs):
        raw = work / f"cls_{name}"
        _run(raw, per_class, seed + 100 + i, drop=drop)
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        n = 0
        for src in sorted((raw / "images").glob("*.jpg")):
            _to_portrait(src, folder / f"{name}_{n:04d}.jpg")
            n += 1
        counts[name] = n
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--detection", type=int, default=300)
    ap.add_argument("--per-class", type=int, default=80)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--keep-work", action="store_true",
                    help="keep the landscape intermediates for inspection")
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out).resolve()
    work = out / "_work"
    out.mkdir(parents=True, exist_ok=True)

    print(f"scene    : {_SCENE.name}")
    print(f"exposure : {_EXPOSURE}")
    print(f"zoom     : {_ZOOM}")
    print(f"detection: {args.detection} frames")
    det = build_detection(work, out / "detection", args.detection, args.seed)
    print(f"  -> {det['frames']} frames, {sum(det['boxes_per_class'].values())} boxes")

    print(f"classification: {args.per_class} frames x {len(_CLASSES) + 1} classes")
    cls = build_classification(work, out / "classification", args.per_class,
                               args.seed)
    print(f"  -> {cls}")

    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)

    summary = {"detection": det, "classification": cls,
               "scene": _SCENE.name, "exposure": _EXPOSURE, "zoom": _ZOOM,
               "orientation": "480x640 portrait",
               "generator": "native_mujoco/cli/generate_dataset.py"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2),
                                      encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
