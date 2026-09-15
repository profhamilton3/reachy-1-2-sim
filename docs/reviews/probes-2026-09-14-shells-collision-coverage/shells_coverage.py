""""shells" coverage of the MJCF collision pads (Slice 2, ADR-0003 item 4),
mirroring probes-2026-09-14/tube_coverage.py's method but against the
`thumb`/`thumb_pad`/`finger`/`wrist_ball` shells capsules instead of the
single tube.  Run from the repo root:

    PYTHONPATH=src:native_mujoco python3 \
        docs/reviews/probes-2026-09-14-shells-collision-coverage/shells_coverage.py

Offline: mj_forward only, no server, no SDK, no motion.
"""
import math, sys, numpy as np, mujoco
sys.path.insert(0, "tests/unit"); sys.path.insert(0, "src"); sys.path.insert(0, "native_mujoco")
import test_footprint_boards as T
from objects import build_scene_model_xml
from scene_io import load_scene
from reachy_ai.motion import rig_routes as R
from reachy_ai.motion.kinematics import (
    _GRIPPER_OPEN_LIMIT_DEG, _GRIPPER_SHUT_LIMIT_DEG, _WRIST_ROLL_LIMIT_DEG,
    _FINGER_RADIUS, _FINGER_SHELL_RADIUS, _THUMB_PAD_RADIUS,
    hand_radius, joint_path, link_capsules, link_frames,
)

print(f"finger radius: {_FINGER_RADIUS * 100:.4f} cm (was shell-only "
      f"{_FINGER_SHELL_RADIUS * 100:.4f} cm)")
print(f"thumb_pad radius: {_THUMB_PAD_RADIUS * 100:.4f} cm")

doc = load_scene(T._SCENE_PATH)
m = mujoco.MjModel.from_xml_string(build_scene_model_xml(doc))
d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)
for jid in range(m.njnt):
    if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
        a = m.jnt_qposadr[jid]; d.qpos[a:a + 7] = m.qpos0[a:a + 7]
mujoco.mj_forward(m, d)
jq = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
      for n in R.R_JOINTS}
G = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
HAND_BOXES = ("r_thumb_body", "r_finger_body", "r_thumb_col", "r_finger_col")
SHELLS_HAND_CAPS = ("thumb", "thumb_pad", "finger", "wrist_ball")


def corners(g):
    gid = G(g); h = m.geom_size[gid]; Rm = d.geom_xmat[gid].reshape(3, 3); c = d.geom_xpos[gid]
    return [c + Rm @ np.array([sx * h[0], sy * h[1], sz * h[2]])
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1) if np.dot(ab, ab) > 0 else 0.0
    return np.linalg.norm(p - (a + t * ab))


def set_arm(q7, g):
    for n, v in zip(R.ARM7, q7): d.qpos[jq[n]] = math.radians(v)
    d.qpos[jq["r_gripper"]] = math.radians(g)
    mujoco.mj_forward(m, d)


def excess_at(q7, g):
    """Worst (corner distance - radius) over every shells capsule -- the
    corner only needs to clear the nearest one, mirroring TestShellsCoverMJCFHand."""
    set_arm(q7, g)
    caps = [(np.array(a), np.array(b), r) for n, a, b, r in
            link_capsules(q7, "right", g, hand="shells") if n in SHELLS_HAND_CAPS]

    def excess(p):
        return min(seg_dist(p, a, b) - r for a, b, r in caps)

    worst = max(excess(p) for gname in HAND_BOXES for p in corners(gname))
    ball = d.geom_xpos[G("r_wrist_ball")]
    worst = max(worst, excess(ball) + float(m.geom_size[G("r_wrist_ball")][0]))
    return worst


# 1. coverage over named waypoints + FOOTPRINT_LEGS samples
GRIPPER_SAMPLES = (-68.8, -45.0, 0.0, 20.0, _GRIPPER_SHUT_LIMIT_DEG)
poses = {}
for name, p in R.POSTURES.items(): poses[name] = p
for route_name in ("PLACE_ROUTE", "STOW_ROUTE", "LIFT_TO_PRESENT",
                   "LOWER_TO_REST", "RAISE_TO_SIDE", "STOW_FROM_SIDE"):
    for wp in R.route_named(route_name): poses.setdefault(wp.name, wp.pose)
poses["WAVE_A"] = R.WAVE_A
poses["WAVE_B"] = R.WAVE_B

print("== coverage: worst excess over named waypoints + FOOTPRINT_LEGS samples ==")
worst, label = -1e9, None
for name, pose in poses.items():
    q7 = [pose[j] for j in R.ARM7]
    for g in GRIPPER_SAMPLES:
        e = excess_at(q7, g)
        if e > worst: worst, label = e, (name, g)
for route_name, waypoints in R.FOOTPRINT_LEGS.items():
    for i, (a, b) in enumerate(zip(waypoints, waypoints[1:])):
        qa = [a[j] for j in R.ARM7]; qb = [b[j] for j in R.ARM7]
        for k, q7 in enumerate(joint_path(qa, qb, steps=13)):
            for g in GRIPPER_SAMPLES:
                e = excess_at(q7, g)
                if e > worst: worst, label = e, (f"{route_name}[{i}]#{k}", g)
print(f"worst excess (must be <= 1e-9): {worst:.3e} m at {label}")

# 2. coverage over the full r_gripper x r_wrist_roll grid, on HOVER
print("== coverage: full r_gripper x r_wrist_roll grid, HOVER base ==")
ROLL_GRID = tuple(float(r) for r in range(-45, 46, 5))
GRIPPER_GRID = (_GRIPPER_OPEN_LIMIT_DEG, -68.75) + tuple(float(g) for g in range(-65, 21, 5)) + (_GRIPPER_SHUT_LIMIT_DEG,)
base = [R.HOVER[j] for j in R.ARM7]
grid_worst, grid_label = -1e9, None
for roll in ROLL_GRID:
    q7 = list(base); q7[6] = roll
    for g in GRIPPER_GRID:
        e = excess_at(q7, g)
        if e > grid_worst: grid_worst, grid_label = e, (roll, g)
print(f"worst excess (must be <= 1e-9): {grid_worst:.3e} m at roll,g={grid_label}")

# 3. the thumb/finger pad regression case: old miss vs new margin
print("== thumb_pad / finger_pad margin, now (was 2.05 / 1.25 cm outside; 0.05 cm outside for the finger pad) ==")
for posture, g, old in [("HOVER", -68.8, 2.05), ("HOVER", -45.0, 2.05), ("HOVER", 0.0, 2.05),
                         ("HOVER", 20.0, 1.25), ("REST", -68.8, 2.05), ("REST", -45.0, 2.05),
                         ("REST", 0.0, 2.05), ("REST", 20.0, 1.25)]:
    q7 = [getattr(R, posture)[j] for j in R.ARM7]
    set_arm(q7, g)
    caps = [(np.array(a), np.array(b), r) for n, a, b, r in
            link_capsules(q7, "right", g, hand="shells") if n in SHELLS_HAND_CAPS]

    def outside(p):
        return min(seg_dist(p, a, b) - r for a, b, r in caps)

    thumb_pad = max(outside(p) for p in corners("r_thumb_col"))
    finger_pad = max(outside(p) for p in corners("r_finger_col"))
    print(f"  {posture:6s} g={g:6.1f}  thumb_pad {thumb_pad * 100:8.5f} cm "
          f"(was +{old})  finger_pad {finger_pad * 100:8.5f} cm")

# 4. TestTubeVsShellsCapsules re-derivation: tube radius - shells surface/centreline
print("== tube radius - shells capsule axis-distance (surface / centreline), cm ==")
for pose_name in ("REST", "REST_SHUT", "HOVER", "PRESENT", "HOME"):
    q7 = [getattr(R, pose_name)[j] for j in R.ARM7]
    for g in (-68.8, -45.0, 0.0, 20.0):
        _s, _e, wrist, Rm = link_frames(q7)
        ax = Rm @ np.array([0, 0, -1.0]); ax /= np.linalg.norm(ax)
        rt = hand_radius(g, q7[6])
        ws = wc = 1e9
        for name, p0, p1, r in link_capsules(q7, "right", g, hand="shells"):
            if name in ("upper_arm", "forearm"): continue
            for p in (np.array(p0), np.array(p1)):
                rel = p - wrist
                perp = np.linalg.norm(rel - np.dot(rel, ax) * ax)
                ws = min(ws, rt - perp - r); wc = min(wc, rt - perp)
        print(f"  {pose_name:9s} {g:6.1f} surface {ws * 100:7.4f}  centreline {wc * 100:7.4f}")

# 5. fixture table: tube (corrected) / shells-with-pads, the four named boards
# plus the two 4 cm pool objects on r2c3
print("== fixture table (tube new / shells-with-pads), cm ==")
def worst_any(model, oid, hand):
    worst = 1e9
    for _name, wps in R.FOOTPRINT_LEGS.items():
        for a_, b_ in zip(wps, wps[1:]):
            qa = [a_[j] for j in R.ARM7]; qb = [b_[j] for j in R.ARM7]
            gg = max(a_["r_gripper"], b_["r_gripper"], key=hand_radius)
            for q in joint_path(qa, qb, 13):
                caps = link_capsules(q, "right", gg, hand=hand)
                worst = min(worst, model.clearances(caps, ids=[oid])[oid].distance)
    return worst

boards = {
    "evidence": ({"soda_can": T._cell_xy("cell_r1c1"), "foam_block": T._cell_xy("cell_r2c3")}, "foam_block"),
    "incident": ({"foam_block": (T._cell_xy("cell_r2c3")[0], T._cell_xy("cell_r2c3")[1] - 0.07)}, "foam_block"),
    "foam_r3c3": ({"foam_block": T._cell_xy("cell_r3c3")}, "foam_block"),
    "soda_r2c3": ({"soda_can": T._cell_xy("cell_r2c3")}, "soda_can"),
    "pool_box_1_r2c3": ({"pool_box_1": T._cell_xy("cell_r2c3")}, "pool_box_1"),
    "pool_cyl_1_r2c3": ({"pool_cyl_1": T._cell_xy("cell_r2c3")}, "pool_cyl_1"),
}
for name, (objs, oid) in boards.items():
    model = T._board(objs)
    print(f"  {name:16s} tube {worst_any(model, oid, 'tube') * 100:7.3f}  "
          f"shells-with-pads {worst_any(model, oid, 'shells') * 100:7.3f}")
