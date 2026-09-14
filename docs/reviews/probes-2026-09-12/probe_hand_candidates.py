"""Hand-model candidates (2026-09-12, offline, no simulator).

The evidence-board probe showed every capsule-vs-physics disagreement comes
from the hand: the upper-arm and forearm capsules ARE the MJCF collision
geoms (agree to <1 mm), while the hand is one isotropic tube of radius
5.2-8.3 cm wrapped around two shells that are 5x5.6x7.6 cm (thumb) and
2.4x2x7.6 cm (finger).  This probe evaluates candidate hand models against
the MJCF shells on the guarded tabletop legs, so the design can say what a
better hand model would read BEFORE anyone changes kinematics.py:

  tube        today's link_capsules hand (commanded aperture, A2 rule)
  two-caps    one capsule per shell: thumb_body -> capsule (r = half-diagonal
              of its 5x5.6 cross-section = 3.75 cm, half-length 3.8 cm),
              finger_body -> capsule (r = 1.56 cm, half-length 3.8 cm), plus
              the wrist ball (r 2.8 cm).  Still a bound on the shells, still
              plain capsules for SceneModel.segment_object_distance.
  shells      the MJCF boxes themselves (reference; sampled distance)

Capsule axes for the candidates are taken from MuJoCo's forward kinematics
(geom_xpos/geom_xmat after mj_forward) so the probe tests the GEOMETRY of the
candidate, not a re-implementation of the chain.  A real implementation must
derive the same frames from kinematics.link_frames plus the finger hinge and
be tested against these numbers.

    PYTHONPATH=src:native_mujoco python3 probe_hand_candidates.py <repo_root>
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
from dataclasses import replace  # noqa: E402

import geomdist  # noqa: E402
from objects import build_scene_model_xml  # noqa: E402
from placement import ObjectPlacer, cells_from_scene  # noqa: E402
from scene_io import load_scene  # noqa: E402
from reachy_ai.motion import rig_routes as RR  # noqa: E402
from reachy_ai.motion.kinematics import (  # noqa: E402
    hand_radius, joint_path, link_capsules)
from reachy_ai.scene.awareness import SceneModel  # noqa: E402

SCENE = "scenes/FWDCenterLabSivaPool.yaml"
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
top_z = next(iter(cells_from_scene(doc).values())).top_z
scene = SceneModel.from_yaml(SCENE)

gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
G = {n: gid(n) for n in ("r_upper_arm_col", "r_forearm_col", "r_thumb_body",
                         "r_finger_body", "r_wrist_ball")}
JQ = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
      for n in RR.R_JOINTS}


def set_arm(q7, g):
    for name, val in zip(RR.ARM7, q7):
        d.qpos[JQ[name]] = np.radians(val)
    d.qpos[JQ["r_gripper"]] = np.radians(g)
    mujoco.mj_forward(m, d)


def box_as_capsule(g):
    """Capsule along the box's local z through its centre, bounding it."""
    h = m.geom_size[g]
    R = d.geom_xmat[g].reshape(3, 3)
    c = d.geom_xpos[g]
    axis = R @ np.array([0.0, 0.0, 1.0])
    r = float(np.hypot(h[0], h[1]))
    p0 = c - axis * h[2]
    p1 = c + axis * h[2]
    return tuple(map(float, p0)), tuple(map(float, p1)), r


def two_caps():
    caps = []
    for name in ("r_upper_arm_col", "r_forearm_col"):
        g = G[name]
        h = m.geom_size[g]
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g]
        axis = R @ np.array([0.0, 0.0, 1.0])
        caps.append((name, tuple(map(float, c - axis * h[1])),
                     tuple(map(float, c + axis * h[1])), float(h[0])))
    for name in ("r_thumb_body", "r_finger_body"):
        p0, p1, r = box_as_capsule(G[name])
        caps.append((name, p0, p1, r))
    g = G["r_wrist_ball"]
    c = tuple(map(float, d.geom_xpos[g]))
    caps.append(("r_wrist_ball", c, c, float(m.geom_size[g][0])))
    return caps


def worst_over(samples, oid, fn):
    return min(fn(q, g) for q, g in samples)


def samples_for(pairs, steps=13):
    out = []
    for a, b in pairs:
        qa = [a[j] for j in RR.ARM7]
        qb = [b[j] for j in RR.ARM7]
        ga, gb = a["r_gripper"], b["r_gripper"]
        gg = max(ga, gb, key=hand_radius)
        for k, q in enumerate(joint_path(qa, qb, steps)):
            out.append((q, ga + k / (steps - 1) * (gb - ga), gg))
    return out


def tube(q, gcmd, gg, oid):
    return scene.clearances(link_capsules(q, "right", gg), ids=[oid])[oid].distance


def twocaps(q, gcmd, gg, oid):
    set_arm(q, gcmd)
    return scene.clearances(two_caps(), ids=[oid])[oid].distance


def shells(q, gcmd, gg, og):
    set_arm(q, gcmd)
    return min(geomdist.distance(m, d, g, og) for g in G.values())


LEGS = {
    "PLACE_ROUTE tail HOVER->REST_SHUT->REST": [(RR.HOVER, RR.REST_SHUT), (RR.REST_SHUT, RR.REST)],
    "LOWER_TO_REST PRESENT->REST_SHUT->REST": [(RR.PRESENT, RR.REST_SHUT), (RR.REST_SHUT, RR.REST)],
    "LIFT REST->PRESENT (A3)": [(RR.REST, RR.PRESENT)],
}
results = {}

# 1. Evidence board and incident-like boards with the real objects
boards = {
    "evidence (soda r1c1, foam r2c3)": {"soda_can": ("cell", "r1c1"), "foam_block": ("cell", "r2c3")},
    "incident (foam 7 cm right of r2c3)": {"foam_block": ("xy", (0.4318, -0.1524 - 0.07))},
    "foam on r3c3": {"foam_block": ("cell", "r3c3")},
    "foam on r1c3": {"foam_block": ("cell", "r1c3")},
    "soda_can on r2c3": {"soda_can": ("cell", "r2c3")},
}
print(f"{'board':38s} {'leg':40s} {'obj':11s} {'tube':>7s} {'two-caps':>9s} {'shells':>7s}")
for bname, objs in boards.items():
    for oid in ("soda_can", "foam_block"):
        placer.stow(oid)
    for oid, (kind, where) in objs.items():
        if kind == "cell":
            placer.place(oid, where)
        else:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, oid)
            adr = int(m.jnt_qposadr[int(m.body_jntadr[b])])
            hh = float(m.geom_size[int(m.body_geomadr[b])][2])
            d.qpos[adr:adr + 3] = (where[0], where[1], top_z + hh + 0.001)
            d.qpos[adr + 3:adr + 7] = (1, 0, 0, 0)
            mujoco.mj_forward(m, d)
    live = {}
    for oid in objs:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, oid)
        live[oid] = tuple(float(v) for v in d.xpos[b])
    scene.update_poses(live)
    results[bname] = {}
    for leg, pairs in LEGS.items():
        S = samples_for(pairs)
        for oid in objs:
            og = int(m.body_geomadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, oid)])
            t = min(tube(q, gc, gg, oid) for q, gc, gg in S)
            c2 = min(twocaps(q, gc, gg, oid) for q, gc, gg in S)
            sh = min(shells(q, gc, gg, og) for q, gc, gg in S)
            results[bname][f"{leg} / {oid}"] = {"tube": t, "two_caps": c2, "shells": sh}
            print(f"{bname:38s} {leg:40s} {oid:11s} {100*t:+7.1f} {100*c2:+9.1f} {100*sh:+7.1f}")

# 2. Sweep: is two-caps always >= shells (conservative) and how much tighter than the tube?
for oid in ("soda_can", "foam_block"):
    placer.stow(oid)
placer.reshape("pool_box_1", size=[0.06, 0.06, 0.06])
scene._objects["pool_box_1"] = replace(scene._objects["pool_box_1"], size=(0.06, 0.06, 0.06))
b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pool_box_1")
og = int(m.body_geomadr[b])
adr = int(m.jnt_qposadr[int(m.body_jntadr[b])])
tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table_top")
tc = d.geom_xpos[int(m.body_geomadr[tb])]
th = m.geom_size[int(m.body_geomadr[tb])]
xs = np.arange(tc[0] - th[0] + 0.03, tc[0] + th[0] - 0.03 + 1e-9, 0.04)
ys = np.arange(tc[1] - th[1] + 0.03, tc[1] + th[1] - 0.03 + 1e-9, 0.04)
S = samples_for(LEGS["PLACE_ROUTE tail HOVER->REST_SHUT->REST"])
sweep = []
for x in xs:
    for y in ys:
        d.qpos[adr:adr + 3] = (float(x), float(y), top_z + 0.031)
        d.qpos[adr + 3:adr + 7] = (1, 0, 0, 0)
        scene.update_poses({"pool_box_1": (float(x), float(y), top_z + 0.031)})
        t = min(tube(q, gc, gg, "pool_box_1") for q, gc, gg in S)
        c2 = min(twocaps(q, gc, gg, "pool_box_1") for q, gc, gg in S)
        sh = min(shells(q, gc, gg, og) for q, gc, gg in S)
        sweep.append((float(x), float(y), t, c2, sh))
n = len(sweep)
print(f"\nsweep, 6 cm cube, PLACE_ROUTE tail, {n} positions:")
print(f"  tube<0: {sum(1 for r in sweep if r[2] < 0)}   two-caps<0: {sum(1 for r in sweep if r[3] < 0)}   shells<0: {sum(1 for r in sweep if r[4] < 0)}")
print(f"  two-caps less conservative than shells by >2 mm: {sum(1 for r in sweep if r[3] > r[4] + 0.002)} positions")
print(f"  sign disagreements two-caps vs shells: {sum(1 for r in sweep if (r[3] < 0) != (r[4] < 0))}   tube vs shells: {sum(1 for r in sweep if (r[2] < 0) != (r[4] < 0))}")
gap_tube = max(r[4] - r[2] for r in sweep)
gap_2c = max(r[4] - r[3] for r in sweep)
print(f"  max over-conservatism (shells - model): tube {100*gap_tube:+.1f} cm, two-caps {100*gap_2c:+.1f} cm")
results["sweep"] = sweep
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "probe_hand_candidates.json"), "w") as f:
    json.dump(results, f, indent=1, default=float)
