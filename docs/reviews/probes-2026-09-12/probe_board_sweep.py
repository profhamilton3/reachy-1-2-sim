"""Board sweep probe (2026-09-12, offline, no simulator): A1 descent, A3 lift.

For a standing box (6 cm cube, and a 6x6x12 cm block) at every point of a
4 cm grid over the FWDCenterLabSivaPool table top, compute the worst
clearance along each leg under
  capsule  -- the panel guard's model with the A2 aperture rule
  shell    -- MJCF shells + forearm/upper-arm capsules + torso box
              (sampled closed-form distances, see geomdist.py)
for the legs:
  descent  PRESENT -> REST_SHUT   (STOW_FROM_SIDE[0] / LOWER_TO_REST[0], A1)
  lift     REST -> PRESENT        (LIFT_TO_PRESENT, unguarded, A3)
  tail     REST -> REST_SHUT -> HOVER (STOW_ROUTE's checked legs)
Reports, per leg and model, how many grid positions read negative, and the
positions where the two models disagree about the sign.  The object is
pool_box_1 reshaped with ObjectPlacer.reshape (the supported path; it fixes
geom_rbound), moved by writing its free-joint qpos.  mj_forward only.

    PYTHONPATH=src:native_mujoco python3 probe_board_sweep.py <repo_root>
"""
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
sys.path.insert(0, os.path.join(root, "src"))
sys.path.insert(0, os.path.join(root, "native_mujoco"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(root)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import geomdist  # noqa: E402
from objects import build_scene_model_xml  # noqa: E402
from placement import ObjectPlacer, cells_from_scene  # noqa: E402
from scene_io import load_scene  # noqa: E402
from reachy_ai.motion import rig_routes as RR  # noqa: E402
from reachy_ai.motion.kinematics import (  # noqa: E402
    hand_radius, joint_path, link_capsules)
from reachy_ai.scene.awareness import SceneModel  # noqa: E402

SCENE = "scenes/FWDCenterLabSivaPool.yaml"
OID = "pool_box_1"
STEPS = 13            # the guard's own sampling
GRID_M = 0.04

doc = load_scene(SCENE)
m = mujoco.MjModel.from_xml_string(build_scene_model_xml(doc))
d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)
for jid in range(m.njnt):
    if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
        adr = m.jnt_qposadr[jid]
        d.qpos[adr:adr + 7] = m.qpos0[adr:adr + 7]
mujoco.mj_forward(m, d)
placer = ObjectPlacer(m, d, doc)
cells = cells_from_scene(doc)
top_z = next(iter(cells.values())).top_z

tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table_top")
tg = int(m.body_geomadr[tb])
tc = d.geom_xpos[tg]
th = m.geom_size[tg]
xs = np.arange(tc[0] - th[0] + 0.03, tc[0] + th[0] - 0.03 + 1e-9, GRID_M)
ys = np.arange(tc[1] - th[1] + 0.03, tc[1] + th[1] - 0.03 + 1e-9, GRID_M)

gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
ARM_SHELL = {n: gid(n) for n in ("r_upper_arm_col", "r_forearm_col",
                                 "r_thumb_body", "r_finger_body",
                                 "r_wrist_ball", "torso_col")}
ob = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OID)
og = int(m.body_geomadr[ob])
oj = int(m.body_jntadr[ob])
oadr = int(m.jnt_qposadr[oj])
JQ = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
      for n in RR.R_JOINTS}

scene = SceneModel.from_yaml(SCENE)


def set_arm(q7, gripper_deg):
    for name, val in zip(RR.ARM7, q7):
        d.qpos[JQ[name]] = np.radians(val)
    d.qpos[JQ["r_gripper"]] = np.radians(gripper_deg)


def put_box(x, y, full_h):
    d.qpos[oadr:oadr + 3] = (x, y, top_z + full_h / 2 + 0.001)
    d.qpos[oadr + 3:oadr + 7] = (1, 0, 0, 0)


LEGS = {
    "descent PRESENT->REST_SHUT (A1)": [(RR.PRESENT, RR.REST_SHUT)],
    "lift REST->PRESENT (A3, unguarded)": [(RR.REST, RR.PRESENT)],
    "tail REST->REST_SHUT->HOVER (checked)": [(RR.REST, RR.REST_SHUT),
                                             (RR.REST_SHUT, RR.HOVER)],
}

# Precompute arm samples per leg (joints + commanded gripper + guard gripper)
SAMPLES = {}
for leg, pairs in LEGS.items():
    rows = []
    for a, b in pairs:
        qa = [a[j] for j in RR.ARM7]
        qb = [b[j] for j in RR.ARM7]
        ga, gb = a["r_gripper"], b["r_gripper"]
        gg = max(ga, gb, key=hand_radius)
        for k, q in enumerate(joint_path(qa, qb, STEPS)):
            t = k / (STEPS - 1)
            rows.append((q, ga + t * (gb - ga), gg))
    SAMPLES[leg] = rows

results = {"grid_m": GRID_M, "top_z": float(top_z), "table_center": [float(v) for v in tc],
           "table_half": [float(v) for v in th], "sizes": {}}
for full in ((0.06, 0.06, 0.06), (0.06, 0.06, 0.12)):
    placer.reshape(OID, size=list(full))
    tag = f"{int(full[2]*100)}cm"
    per_leg = {}
    for leg, rows in LEGS.items():
        cap_neg, shell_neg, disagree, grid = [], [], [], []
        for x in xs:
            for y in ys:
                put_box(float(x), float(y), full[2])
                scene.update_poses({OID: (float(x), float(y),
                                          top_z + full[2] / 2 + 0.001)})
                # scene's object size must match the reshaped box
                o = scene._objects[OID]
                from dataclasses import replace
                scene._objects[OID] = replace(o, size=tuple(full))
                cw = sw = None
                for q, gcmd, gg in SAMPLES[leg]:
                    c = scene.clearances(link_capsules(q, "right", gg),
                                         ids=[OID])[OID].distance
                    cw = c if cw is None else min(cw, c)
                    set_arm(q, gcmd)
                    mujoco.mj_forward(m, d)
                    s = min(geomdist.distance(m, d, g, og)
                            for g in ARM_SHELL.values())
                    sw = s if sw is None else min(sw, s)
                grid.append((float(x), float(y), cw, sw))
                if cw < 0:
                    cap_neg.append((float(x), float(y)))
                if sw < 0:
                    shell_neg.append((float(x), float(y)))
                if (cw < 0) != (sw < 0):
                    disagree.append((float(x), float(y), cw, sw))
        per_leg[leg] = {"capsule_negative": cap_neg, "shell_negative": shell_neg,
                        "sign_disagreements": disagree, "grid": grid}
        print(f"[{tag}] {leg:40s} positions={len(grid):3d}  capsule<0: {len(cap_neg):3d}  "
              f"shell<0: {len(shell_neg):3d}  sign-disagree: {len(disagree):3d}")
    results["sizes"][tag] = per_leg

# Positions the checked tail clears but the descent / lift hit (the A1/A3 sets)
for tag, per_leg in results["sizes"].items():
    tail = {(x, y) for x, y, cw, sw in per_leg["tail REST->REST_SHUT->HOVER (checked)"]["grid"] if cw < 0}
    for leg in ("descent PRESENT->REST_SHUT (A1)", "lift REST->PRESENT (A3, unguarded)"):
        g = per_leg[leg]["grid"]
        only_cap = sorted((x, y) for x, y, cw, sw in g if cw < 0 and (x, y) not in tail)
        only_shell = sorted((x, y) for x, y, cw, sw in g if sw < 0 and (x, y) not in tail)
        print(f"[{tag}] {leg}: negative here but clear on the checked tail -> "
              f"capsule {len(only_cap)}, shell {len(only_shell)}")
        if only_shell:
            xs_ = [p[0] for p in only_shell]; ys_ = [p[1] for p in only_shell]
            print(f"        shell-negative region x {min(xs_):.2f}..{max(xs_):.2f}  y {min(ys_):.2f}..{max(ys_):.2f}")
        if only_cap:
            xs_ = [p[0] for p in only_cap]; ys_ = [p[1] for p in only_cap]
            print(f"        capsule-negative region x {min(xs_):.2f}..{max(xs_):.2f}  y {min(ys_):.2f}..{max(ys_):.2f}")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "probe_board_sweep.json"), "w") as f:
    json.dump(results, f, indent=1, default=float)
