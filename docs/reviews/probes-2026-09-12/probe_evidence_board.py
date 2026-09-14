"""Evidence-board reconciliation probe (2026-09-12, offline, no simulator).

Puts ADR-0002's certified board (soda_can@r1c1, foam_block@r2c3) into the
real MuJoCo scene model that native_mujoco/server.py compiles, walks the
arm through every FOOTPRINT_LEGS leg plus the unguarded REST->PRESENT lift
(A3) at the SAME joint-space samples the panel guard uses, and at each
sample compares three geometric answers for "how close is the arm to the
object":

  capsule   what web/panel_executor._footprint_refusal computes
            (kinematics.link_capsules -> SceneModel.clearances), with the
            leg's commanded aperture the way the A2 fix picks it
  physics   MuJoCo's own signed distance between the arm's COLLISION geoms
            (r_upper_arm_col, r_forearm_col, r_thumb_col, r_finger_col) and
            the object geom (mujoco.mj_geomDistance) -- this is the geometry
            "board undisturbed" was measured against
  shell     the same, but using the gripper's VISUAL shells (r_thumb_body,
            r_finger_body, r_wrist_ball) in place of the two fingertip
            collision boxes -- the closest offline stand-in for the real
            hand's envelope, which the physics does not collide with at all

Nothing is stepped: mj_forward only, so no dynamics, no contact response,
no motion.  Run from a checkout/export of the repository root:

    PYTHONPATH=src:native_mujoco python3 probe_evidence_board.py <repo_root>

Writes probe_evidence_board.json and prints a summary.
"""
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
sys.path.insert(0, os.path.join(root, "src"))
sys.path.insert(0, os.path.join(root, "native_mujoco"))
os.chdir(root)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from objects import build_scene_model_xml  # noqa: E402
from placement import ObjectPlacer  # noqa: E402
from scene_io import load_scene  # noqa: E402
from reachy_ai.motion import rig_routes as RR  # noqa: E402
from reachy_ai.motion.kinematics import (  # noqa: E402
    hand_radius, joint_path, link_capsules)
from reachy_ai.scene.awareness import SceneModel  # noqa: E402

SCENE = "scenes/FWDCenterLabSivaPool.yaml"
BOARD = {"soda_can": "r1c1", "foam_block": "r2c3"}

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
for oid, cell in BOARD.items():
    placer.place(oid, cell)

# The capsule-model scene, with the objects where the placer actually put them.
scene = SceneModel.from_yaml(SCENE)
live = {}
for oid in BOARD:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, oid)
    live[oid] = tuple(float(v) for v in d.xpos[bid])
scene.update_poses(live)


def gid(name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)


def body_geom(body):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    return int(m.body_geomadr[b])


ARM_COL = {n: gid(n) for n in ("r_upper_arm_col", "r_forearm_col",
                               "r_thumb_col", "r_finger_col")}
ARM_SHELL = {n: gid(n) for n in ("r_upper_arm_col", "r_forearm_col",
                                 "r_thumb_body", "r_finger_body",
                                 "r_wrist_ball")}
TORSO = gid("torso_col")
OBJ = {oid: body_geom(oid) for oid in BOARD}
RAILS = {n: body_geom(n) for n in ("rig_rail_outer_right",
                                   "rig_rail_inner_right")}
JQ = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
      for n in RR.R_JOINTS}


def set_arm(q7, gripper_deg):
    for name, val in zip(RR.ARM7, q7):
        d.qpos[JQ[name]] = np.radians(val)
    d.qpos[JQ["r_gripper"]] = np.radians(gripper_deg)
    mujoco.mj_forward(m, d)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geomdist  # noqa: E402

MJ_DISAGREE = []   # (leg, t, arm geom, target geom, mj, sampled)


def geom_dist(g1, g2):
    """Sampled closed-form distance (see geomdist.py); mj_geomDistance is
    recorded beside it wherever the two differ by more than 2 mm."""
    sampled = geomdist.distance(m, d, g1, g2)
    mj = geomdist.mj_distance(m, d, g1, g2)
    if abs(mj - sampled) > 0.002:
        MJ_DISAGREE.append((g1, g2, mj, sampled))
    return sampled


def min_dist(arm_geoms, target):
    best = None
    for name, g in arm_geoms.items():
        dist = geom_dist(g, target)
        if best is None or dist < best[0]:
            best = (dist, name)
    return best


def capsule_clear(q7, gripper_deg):
    caps = link_capsules(q7, "right", gripper_deg)
    return {oid: (c.distance, c.link) for oid, c in
            scene.clearances(caps, ids=list(BOARD)).items()}


LEGS = {}
for route, wps in RR.FOOTPRINT_LEGS.items():
    for i, (a, b) in enumerate(zip(wps, wps[1:])):
        LEGS[f"{route}[{i}]"] = (a, b)
LEGS["LIFT_TO_PRESENT[0] (unguarded, A3)"] = (RR.REST, RR.PRESENT)
LEGS["SWING_2->SWING_1 (rail, #74)"] = (RR.SWING_2, RR.SWING_1)
LEGS["SWING_1->TUCK (rail, #74)"] = (RR.SWING_1, RR.TUCK)

results = {"board": {k: v for k, v in live.items()}, "legs": {}}
for leg, (a, b) in LEGS.items():
    qa = [a[j] for j in RR.ARM7]
    qb = [b[j] for j in RR.ARM7]
    ga, gb = a["r_gripper"], b["r_gripper"]
    guard_grip = max(ga, gb, key=hand_radius)      # the A2 rule
    out = {"guard_gripper_deg": guard_grip, "samples": {}}
    for steps in (13, 101):
        rows = []
        for k, q in enumerate(joint_path(qa, qb, steps)):
            t = k / (steps - 1)
            g_interp = ga + t * (gb - ga)              # min-jerk is monotone
            cap_guard = capsule_clear(q, guard_grip)   # what the guard uses
            cap_open = capsule_clear(q, None)          # pre-A2 assumption
            cap_interp = capsule_clear(q, g_interp)    # commanded at t
            set_arm(q, g_interp)
            row = {"t": round(t, 4)}
            for oid, og in OBJ.items():
                row[oid] = {
                    "capsule_guard": cap_guard[oid][0],
                    "capsule_guard_link": cap_guard[oid][1],
                    "capsule_open": cap_open[oid][0],
                    "capsule_interp": cap_interp[oid][0],
                    "physics_col": min_dist(ARM_COL, og),
                    "shell": min_dist(ARM_SHELL, og),
                    "torso": geom_dist(TORSO, og),
                }
            for rn, rg in RAILS.items():
                row[rn] = {"physics_col": min_dist(ARM_COL, rg),
                           "shell": min_dist(ARM_SHELL, rg)}
            rows.append(row)
        out["samples"][steps] = rows
    results["legs"][leg] = out

results["mj_geomDistance_disagreements"] = [
    {"arm_geom": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, a),
     "target_geom": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, b)
     or mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[b]),
     "mj": mj, "sampled": smp} for a, b, mj, smp in MJ_DISAGREE]
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "probe_evidence_board.json"), "w") as f:
    json.dump(results, f, indent=1, default=float)
print(f"mj_geomDistance disagreed with sampled distance (>2 mm) at "
      f"{len(MJ_DISAGREE)} of the geom-pair evaluations")


def worst(rows, oid, key, sub=None):
    vals = [(r[oid][key] if sub is None else r[oid][key][0]) for r in rows]
    i = int(np.argmin(vals))
    return vals[i], rows[i]["t"]


print(f"board: {live}")
print(f"{'leg':40s} {'grip':>6s} {'obj':11s} {'caps(open)':>11s} "
      f"{'caps(guard)':>11s} {'caps(cmd)':>10s} {'physics':>9s} "
      f"{'shell':>8s} {'torso':>7s}   [13 samples | 101 samples]")
for leg, out in results["legs"].items():
    for oid in OBJ:
        line = f"{leg:40s} {out['guard_gripper_deg']:6.1f} {oid:11s}"
        for key, sub in (("capsule_open", None), ("capsule_guard", None),
                         ("capsule_interp", None), ("physics_col", 0),
                         ("shell", 0), ("torso", None)):
            w13, _ = worst(out["samples"][13], oid, key, sub)
            w101, _ = worst(out["samples"][101], oid, key, sub)
            line += f" {100*w13:+6.1f}|{100*w101:+6.1f}"
        print(line)
print()
print("rails (cm), worst over 101 samples:")
for leg, out in results["legs"].items():
    for rn in RAILS:
        w_col = min(r[rn]["physics_col"][0] for r in out["samples"][101])
        w_sh = min(r[rn]["shell"][0] for r in out["samples"][101])
        print(f"  {leg:40s} {rn:22s} physics {100*w_col:+6.1f}  shell {100*w_sh:+6.1f}")
