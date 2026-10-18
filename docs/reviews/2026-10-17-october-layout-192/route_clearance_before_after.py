"""Planned worst clearance of each full route, before and after #192.

Every waypoint of the route from its start posture, joint-space legs sampled
at 61 points, tube and shells hand models (each leg at the wider aperture of
its ends for the tube; per-sample aperture for the shells), against three
groups in FWDCenterLabMCC with its default objects on the board:
objects (red_cube r1c3, foam_block r1c2, blue_cylinder r2c2, soda_can r2c1),
the five rig rails, and table_top (which REST touches by design).

    PYTHONPATH=src python3 docs/reviews/2026-10-17-october-layout-192/route_clearance_before_after.py
"""
import subprocess
import sys
import types

from reachy_ai.motion import kinematics as K
from reachy_ai.motion import rig_routes as R
from reachy_ai.scene.awareness import SceneModel

SCENE = "scenes/FWDCenterLabMCC.yaml"
STEPS = 61

# The August routes as they were at main (db0c0ef), before #192 retired them.
_src = subprocess.run(["git", "show", "db0c0ef:src/reachy_ai/motion/rig_routes.py"],
                      capture_output=True, text=True, check=True).stdout
A = types.ModuleType("rig_routes_august")
sys.modules[A.__name__] = A
exec(compile(_src, "rig_routes@db0c0ef", "exec"), A.__dict__)


def poses(start, route):
    return (start,) + tuple(w.pose for w in route)


BEFORE = {
    "PLACE_ROUTE": poses(A.HOME, A.PLACE_ROUTE),
    "STOW_ROUTE": poses(A.REST, A.STOW_ROUTE),
    "RAISE_TO_SIDE": poses(A.HOME, A.RAISE_TO_SIDE),
    "STOW_FROM_SIDE": poses(A.PRESENT, A.STOW_FROM_SIDE),
    "LIFT_TO_PRESENT": poses(A.REST, A.LIFT_TO_PRESENT),
    "LOWER_TO_REST": poses(A.PRESENT, A.LOWER_TO_REST),
    "WAVE": (A.PRESENT, A.WAVE_A, A.WAVE_B, A.PRESENT),
}
AFTER = {
    "LIFT_TO_PRESENT": poses(R.REST, R.LIFT_TO_PRESENT),
    "LOWER_TO_REST": poses(R.PRESENT, R.LOWER_TO_REST),
    "WAVE": (R.PRESENT, R.WAVE_A, R.WAVE_B, R.PRESENT),
}


def worst(seq, base):
    saved = K._BASE.copy(); K._BASE[:] = base
    try:
        m = SceneModel.from_yaml(SCENE)
        groups = {"objects": m.obstacle_ids(),
                  "rails": [i for i in m.obstacle_ids(include_static=True) if i.startswith("rig_rail")],
                  "table": ["table_top"]}
        out = {}
        for hand in ("tube", "shells"):
            for a, b in zip(seq, seq[1:]):
                qa = [a[j] for j in R.ARM7]; qb = [b[j] for j in R.ARM7]
                tube_g = max(a["r_gripper"], b["r_gripper"], key=K.hand_radius)
                for i, q in enumerate(K.joint_path(qa, qb, STEPS)):
                    g = tube_g if hand == "tube" else a["r_gripper"] + i / (STEPS - 1) * (b["r_gripper"] - a["r_gripper"])
                    caps = K.link_capsules(q, "right", g, hand=hand)
                    for name, ids in groups.items():
                        c = m.clearance(caps, ids=ids)
                        if c is not None and ((hand, name) not in out or c.distance < out[(hand, name)].distance):
                            out[(hand, name)] = c
        return out
    finally:
        K._BASE[:] = saved


def row(label, route, res):
    cells = []
    for hand in ("tube", "shells"):
        for g in ("objects", "rails", "table"):
            c = res[(hand, g)]
            cells.append(f"{c.distance*100:+.1f} ({c.object_id if g != 'table' else c.link})")
    print(f"| {label} | {route} | " + " | ".join(cells) + " |")


print("| mount | route | tube: objects | tube: rails | tube: table | shells: objects | shells: rails | shells: table |")
print("|---|---|---|---|---|---|---|---|")
for r, seq in BEFORE.items():
    row("August", r, worst(seq, (0.0, 0.0, 1.0)))
for r, seq in AFTER.items():
    row("October", r, worst(seq, (0.206, 0.0, 1.15)))
