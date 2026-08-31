#!/usr/bin/env python3
"""Render a synthetic labelled detection dataset (R12-606).

Must run under mjpython on macOS — MuJoCo needs a real GL context, which is
also why this cannot be exercised in CI beyond its pure-function parts.

  mjpython native_mujoco/cli/generate_dataset.py \
      --scene scenes/FWDCenterLabMCC.yaml \
      --out /tmp/reachy_dataset --count 500 --distort

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
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration import load_calibration, synthetic_defaults
from dataset import DatasetGenerator
from objects import build_scene_model_xml

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
    args = ap.parse_args(argv)

    scene_doc = yaml.safe_load(pathlib.Path(args.scene).read_text())
    model = mujoco.MjModel.from_xml_string(
        build_scene_model_xml(scene_doc, args.model))

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
    )
    print(f"scene       : {scene_doc.get('name')}")
    print(f"calibration : {cal.provenance}")
    print(f"classes     : {', '.join(gen.class_names)}")
    print(f"distortion  : {'ON' if args.distort else 'off'}")
    print(f"rendering   : {args.count} frames -> {args.out}")

    manifest = gen.generate(args.out, args.count, camera=args.camera)
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
