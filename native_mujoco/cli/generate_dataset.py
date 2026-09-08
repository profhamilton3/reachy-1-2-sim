#!/usr/bin/env python3
"""Render a synthetic labelled detection dataset (R12-606).

Needs a GL context, so CI can only exercise the pure-function parts.  It does
NOT need mjpython: offscreen rendering runs under plain python3 on macOS
(verified 2026-09-02).  mjpython is only required for the interactive viewer,
and using it here costs a launcher indirection for nothing.

  python3 native_mujoco/cli/generate_dataset.py \
      --scene scenes/FWDCenterLabSiva.yaml \
      --out /tmp/reachy_dataset --count 500 --distort --exposure 0.40

Writes images/, labels/ (YOLO), annotations/ (JSON with head pose, zoom and
object poses), classes.txt and manifest.json.  See native_mujoco/dataset.py.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration import load_calibration, synthetic_defaults
from dataset import DatasetGenerator
from objects import build_scene_model_xml
from scene_io import load_scene
from zoom import ZoomLevel

_REPO = pathlib.Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default=str(_REPO / "scenes" / "FWDCenterLabMCC.yaml"))
    ap.add_argument("--model", default=str(_REPO / "native_mujoco" / "model" / "reachy_1_2.xml"))
    ap.add_argument("--calibration",
                    default=str(_REPO / "scenes" / "calibration_measured_2026_08_27.yaml"),
                    help="camera profile; anchors the zoom levels")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--camera", default="left_camera",
                    choices=("left_camera", "right_camera"))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--distort", action="store_true",
                    help="warp frames to match the real lens's barrel — what a "
                         "detector destined for the real feed should train on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--object-tag", default="detector-target")
    ap.add_argument("--pitch-range", type=float, nargs=2, default=(15.0, 55.0),
                    metavar=("MIN", "MAX"),
                    help="neck pitch sampling range in degrees.  Sets how the "
                         "board is framed, which decides how many pixels an "
                         "object covers — match it to the captures the frames "
                         "will be compared against.")
    ap.add_argument("--zoom", choices=("in", "inter", "out"), default=None,
                    help="pin every frame to one zoom instead of sampling.  "
                         "Use with --distort: the barrel profile is measured "
                         "at INTER only, and the render margin that keeps the "
                         "warp inside its source buffer is derived from that "
                         "field, so OUT frames fold back on themselves.")
    ap.add_argument("--drop", action="append", default=[], metavar="OBJECT_ID",
                    help="remove an object from the scene before rendering; "
                         "repeatable.  Building a single-class run (the folder "
                         "layout an imprinting classifier trains on) means "
                         "rendering a board with only one kind of thing on it, "
                         "and dropping the rest is the whole difference "
                         "between runs.  Dropping every target gives empty "
                         "board frames.")
    ap.add_argument("--exposure", type=float, default=1.0,
                    help="scale every light before rendering.  MuJoCo has no "
                         "auto-exposure, so a scene lit for a viewer window "
                         "clips to pure white on the board, while the real "
                         "camera — stopped down by a bright glossy floor — "
                         "photographs that same board at about half scale.  "
                         "Below 1.0 brings rendered luminance onto the real "
                         "feed's. Affects this run only; nothing is written "
                         "back to the model or the scene.")
    args = ap.parse_args(argv)

    scene_doc = load_scene(args.scene)
    if args.drop:
        present = {o.get("id") for o in (scene_doc.get("objects") or [])}
        unknown = [d for d in args.drop if d not in present]
        if unknown:
            print(f"--drop names objects not in the scene: {sorted(unknown)}",
                  file=sys.stderr)
            return 2
        scene_doc = dict(scene_doc)
        scene_doc["objects"] = [o for o in scene_doc["objects"]
                                if o.get("id") not in set(args.drop)]

    model = mujoco.MjModel.from_xml_string(
        build_scene_model_xml(scene_doc, args.model))

    if args.exposure != 1.0:
        if args.exposure <= 0.0:
            print("--exposure must be > 0", file=sys.stderr)
            return 2
        model.light_diffuse[:] = model.light_diffuse * args.exposure
        model.light_ambient[:] = model.light_ambient * args.exposure
        model.light_specular[:] = model.light_specular * args.exposure
        # The headlight is not one of the scene's lights and does not appear in
        # the arrays above.  It is on by default and camera-mounted, so leaving
        # it out puts a floor under the exposure sweep — scaling the named
        # lights alone moved the board only 206 -> 178 across a 0.35 -> 0.20
        # range, because the headlight was supplying the rest.
        hl = model.vis.headlight
        hl.ambient[:] = hl.ambient * args.exposure
        hl.diffuse[:] = hl.diffuse * args.exposure
        hl.specular[:] = hl.specular * args.exposure

    if args.calibration:
        cal = load_calibration(args.calibration)
    else:
        cal = synthetic_defaults(args.width, args.height)
        print("WARNING: using synthetic defaults; frames will not match the "
              "real camera's field of view", file=sys.stderr)

    gen = DatasetGenerator(
        model, scene_doc, cal,
        width=args.width, height=args.height,
        distort=args.distort, seed=args.seed, object_tag=args.object_tag,
        allow_no_targets=bool(args.drop),
        pitch_range=tuple(args.pitch_range),
    )
    print(f"scene       : {scene_doc.get('name')}")
    print(f"calibration : {cal.provenance}")
    print(f"classes     : {', '.join(gen.class_names)}")
    print(f"distortion  : {'ON' if args.distort else 'off'}")
    print(f"exposure    : {args.exposure:g}")
    print(f"zoom        : {args.zoom or 'sampled'}")
    if args.drop:
        print(f"dropped     : {', '.join(sorted(args.drop))}")
    print(f"rendering   : {args.count} frames -> {args.out}")

    manifest = gen.generate(args.out, args.count, camera=args.camera,
                            zoom=ZoomLevel(args.zoom) if args.zoom else None)
    gen.close()

    print(json.dumps(manifest, indent=2))
    if manifest["empty_frames"]:
        pct = 100.0 * manifest["empty_frames"] / manifest["count"]
        print(f"\nNote: {manifest['empty_frames']} frames ({pct:.1f}%) contain no "
              f"labelled object. Some are legitimate negatives; a high share "
              f"usually means the head-pose or zoom range needs narrowing.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
