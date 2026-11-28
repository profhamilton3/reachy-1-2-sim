#!/usr/bin/env python3
"""sim #212 D6: the PART families' RENDERED reference colours (CIELAB).

    python3 measure_rendered_colours.py [--out rendered_colours.json]

What is measured, per owner decision D6 (2026-11-27):
  camera     left_camera, 640x480, calibration_measured_2026_08_27 (the
             launcher's default), no distortion, no sensor effects
  head       gaze.BOARD_VIEW (neck 0, 64, 0 deg); arm at the start keyframe (REST)
  lighting   whatever the compiled scene has (robot model lights and headlight)
  pixels     the frame as the server would send it: StereoRenderer's JPEG, decoded
  which      the part's TOP and OUTER SIDE faces.  Each pixel is placed on the
             part from MuJoCo's segmentation and float32 depth buffers; the
             inside of a bore is left out, and so is every pixel with a
             neighbour that is not the same part (edge blending)
  sampling   one part at a time on each of the nine cells, yaw 0; both designs
             of a family (solid and holed, copy _a)

The reference is the median of L*, a*, b* over all of a family's pixels.  The
spread is how far single placements' medians lie from it (CIE76 delta E), which
is what lighting across the board does.

Uses the server's own world and renderer classes, offline.  Reads no STL/STEP.
"""
import argparse
import io
import json
import math
import os
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "native_mujoco"), str(REPO / "src")]

import mujoco                                                     # noqa: E402
import numpy as np                                                # noqa: E402
from PIL import Image                                             # noqa: E402

from calibration import load_calibration                          # noqa: E402
from joint_map import JOINT_TABLE                                 # noqa: E402
from reachy_ai.motion import gaze                                 # noqa: E402
from renderer import StereoRenderer                               # noqa: E402
from scene_compiler import interactive_specs, tracked_object_ids  # noqa: E402
from scene_io import load_scene                                   # noqa: E402
from server import ReachyMujocoServer, SimState                   # noqa: E402

CALIBRATION = REPO / "scenes" / "calibration_measured_2026_08_27.yaml"
SCENES = {"family": "FWDCenterLabSivaPool", "black": "FWDCenterLabSivaPoolBlack"}
FAMILIES = {"11": ("block, short", "mustard yellow"), "12": ("block, tall", "burnt orange"),
            "21": ("cylinder, short", "dusty-blue grey"), "22": ("cylinder, tall", "grass green")}
CELLS = [f"r{r}c{c}" for r in (1, 2, 3) for c in (1, 2, 3)]
W, H = 640, 480
HALF_SIDE, OUTER_R = 0.024605, 0.024606
FACE_TOL = 0.0004          # covers the 0.1 mm recess and the depth buffer's error


def srgb_to_lab(rgb):
    """N x 3 sRGB (0..255) -> CIELAB, D65."""
    c = rgb.astype(float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]]).T
    t = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]),
                     200 * (f[:, 1] - f[:, 2])], axis=1)


def face_pixels(model, data, flat, body, cylinder, half_height):
    """(top mask, side mask, mask of the whole part less edges) in the left image."""
    flat.update_scene(data, camera="left_camera")
    flat.enable_segmentation_rendering()
    seg = flat.render()
    flat.disable_segmentation_rendering()
    flat.enable_depth_rendering()
    depth = flat.render().astype(float)
    flat.disable_depth_rendering()

    geom = seg[:, :, 0]
    mine = (geom >= 0) & (np.where(geom >= 0, model.geom_bodyid[np.maximum(geom, 0)], -1) == body)
    inner = mine.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            inner &= np.roll(mine, (dy, dx), axis=(0, 1))

    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "left_camera")
    focal = (H / 2) / math.tan(math.radians(model.cam_fovy[cam]) / 2)
    v, u = np.nonzero(inner)
    d = depth[v, u]
    in_camera = np.stack([(u + 0.5 - W / 2) / focal * d, -(v + 0.5 - H / 2) / focal * d, -d], axis=1)
    world = data.cam_xpos[cam] + in_camera @ data.cam_xmat[cam].reshape(3, 3).T
    local = (world - data.xpos[body]) @ data.xmat[body].reshape(3, 3)

    top = local[:, 2] > half_height - FACE_TOL
    if cylinder:
        outer = np.hypot(local[:, 0], local[:, 1]) > OUTER_R - FACE_TOL
    else:
        outer = np.max(np.abs(local[:, :2]), axis=1) > HALF_SIDE - FACE_TOL
    top_mask, side_mask = np.zeros_like(inner), np.zeros_like(inner)
    top_mask[v[top], u[top]] = True
    side_mask[v[outer & ~top], u[outer & ~top]] = True
    return top_mask, side_mask, inner


def measure(scene_name):
    scene_path = REPO / "scenes" / f"{scene_name}.yaml"
    doc = load_scene(scene_path)
    server = ReachyMujocoServer(str(REPO / "native_mujoco" / "model" / "reachy_1_2.xml"),
                                "127.0.0.1", 0, scene_path=str(scene_path),
                                calibration=load_calibration(str(CALIBRATION)))
    model = server._model
    sim = SimState(model, tracked_ids=tracked_object_ids(doc),
                   interactive_specs=interactive_specs(doc), scene_doc=doc)
    dt = model.opt.timestep

    def step(seconds):
        for _ in range(int(round(seconds / dt))):
            sim.apply_pending()
            sim.control_step(dt)
            mujoco.mj_step(model, sim.data)

    neck = {e.sdk_name: e.mjcf_index for e in JOINT_TABLE if e.sdk_name.startswith("neck")}
    for joint, deg in gaze.BOARD_VIEW.items():
        sim.controller.set_goal_position(neck[joint], math.radians(deg))
    step(4.0)
    reached = {j: round(math.degrees(float(sim.data.qpos[i])), 2) for j, i in neck.items()}
    assert all(abs(reached[j] - deg) < 1.0 for j, deg in gaze.BOARD_VIEW.items()), reached

    renderer = StereoRenderer(model, width=W, height=H)
    flat = mujoco.Renderer(model, H, W)
    families = {}
    for family in FAMILIES:
        placements = []
        for design in (family + "0", family + "1"):
            oid = f"part{design}_a"
            obj = next(o for o in doc["objects"] if o["id"] == oid)
            half_height = obj["pose"]["position"][2]          # parked on the floor
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, oid)
            for cell in CELLS:
                sim.submit_place({"object_id": oid, "cell": cell})
                sim.apply_pending()
                assert sim.drain_place_results()[0]["accepted"]
                step(0.6)
                frame = renderer.render_stereo(sim.copy_data(), cameras=("left_camera",))
                rgb = np.asarray(Image.open(io.BytesIO(frame["left_camera"].jpeg_bytes)).convert("RGB"))
                top, side, inner = face_pixels(model, sim.data, flat, body, family[0] == "2", half_height)
                sim.placer.stow(oid)
                faces = top | side
                assert faces.sum() >= 50, f"{oid} on {cell}: only {faces.sum()} face pixels visible"
                placements.append({
                    "object_id": oid, "cell": cell, "hole": design[2] == "1",
                    "rgb_top": rgb[top], "rgb_side": rgb[side],
                    "excluded_fraction": 1.0 - faces.sum() / inner.sum(),
                })
        every = srgb_to_lab(np.concatenate([np.concatenate([p["rgb_top"], p["rgb_side"]])
                                            for p in placements]))
        reference = np.median(every, axis=0)
        def distances(pick, centre):
            """Each placement's own median, as CIE76 delta E from `centre`."""
            return [float(np.linalg.norm(np.median(srgb_to_lab(pick(p)), axis=0) - centre))
                    for p in placements if len(pick(p)) >= 25]

        spread = distances(lambda p: np.concatenate([p["rgb_top"], p["rgb_side"]]), reference)
        top_px = np.concatenate([p["rgb_top"] for p in placements])
        side_px = np.concatenate([p["rgb_side"] for p in placements])
        top_ref = np.median(srgb_to_lab(top_px), axis=0)
        side_ref = np.median(srgb_to_lab(side_px), axis=0)
        top_spread = distances(lambda p: p["rgb_top"], top_ref)
        side_spread = distances(lambda p: p["rgb_side"], side_ref)
        rounded = lambda a: [round(float(x), 1) for x in a]       # noqa: E731
        placement_medians = [
            [round(float(x), 2) for x in np.median(srgb_to_lab(
                np.concatenate([q["rgb_top"], q["rgb_side"]])), axis=0)] for q in placements]
        families[family] = {
            "placement_lab_medians": placement_medians,
            "shape": FAMILIES[family][0], "palette_name": FAMILIES[family][1],
            "material_rgba": next(o for o in doc["objects"]
                                  if o["id"] == f"part{family}0_a")["material"]["rgba"],
            "lab_median": rounded(reference),
            "lab_median_top": rounded(top_ref),
            "lab_median_side": rounded(side_ref),
            "top_pixel_share": round(len(top_px) / (len(top_px) + len(side_px)), 3),
            "top_clipped_fraction": round(float((top_px >= 255).any(axis=1).mean()), 3),
            "top_placement_delta_e_max": round(max(top_spread), 1),
            "side_placement_delta_e_max": round(max(side_spread), 1),
            "placement_delta_e_median": round(float(np.median(spread)), 1),
            "placement_delta_e_max": round(max(spread), 1),
            "pixels": int(len(every)), "placements": len(placements),
            "max_excluded_fraction_solid": round(max(p["excluded_fraction"] for p in placements
                                                     if not p["hole"]), 4),
            "max_excluded_fraction_holed": round(max(p["excluded_fraction"] for p in placements
                                                     if p["hole"]), 4),
        }
    renderer.close()
    flat.close()
    # Can the references tell the families apart?  Each placement's median goes
    # to the nearest family reference: in L*a*b*, and in a*b* alone.
    refs = {f: np.array(m["lab_median"]) for f, m in families.items()}
    for family, m in families.items():
        for key, axes in (("nearest_reference_correct_lab", slice(0, 3)),
                          ("nearest_reference_correct_ab", slice(1, 3))):
            m[key] = sum(
                min(refs, key=lambda f: np.linalg.norm(np.array(x)[axes] - refs[f][axes])) == family
                for x in m["placement_lab_medians"])
    return {"scene": scene_name, "neck_deg_reached": reached, "families": families}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "rendered_colours.json")
    args = ap.parse_args()
    args.out = args.out.resolve()
    # MuJoCo appends its warnings to MUJOCO_LOG.TXT in the working directory.
    os.chdir(tempfile.mkdtemp())
    report = {"camera": "left_camera 640x480", "calibration": CALIBRATION.name,
              "board_view_deg": gaze.BOARD_VIEW, "mujoco": mujoco.__version__,
              "colour_space": "CIELAB (D65) from the sRGB of the decoded JPEG",
              "variants": {label: measure(scene) for label, scene in SCENES.items()}}
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    for label, variant in report["variants"].items():
        for family, m in variant["families"].items():
            print(f"{label:7s} PART{family}x {m['palette_name']:16s} Lab {m['lab_median']} "
                  f"top {m['lab_median_top']} side {m['lab_median_side']} "
                  f"dE placements med {m['placement_delta_e_median']} max {m['placement_delta_e_max']} "
                  f"top share {m['top_pixel_share']} top clipped {m['top_clipped_fraction']} "
                  f"dE max top-only {m['top_placement_delta_e_max']} side-only {m['side_placement_delta_e_max']} "
                  f"nearest-ref correct lab {m['nearest_reference_correct_lab']}/{m['placements']} "
                  f"ab {m['nearest_reference_correct_ab']}/{m['placements']} "
                  f"px {m['pixels']} excluded solid<= {m['max_excluded_fraction_solid']} "
                  f"holed<= {m['max_excluded_fraction_holed']}")


if __name__ == "__main__":
    main()
