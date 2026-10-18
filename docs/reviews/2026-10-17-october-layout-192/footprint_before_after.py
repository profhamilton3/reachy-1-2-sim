"""REST footprint vs an object on each column-3 cell, before and after #192.

The live guard's own computation (`footprint.route_footprint_clearances`: the
legs in `FOOTPRINT_LEGS`, tube hand, each leg at the wider aperture of its
ends), one probe object at a time standing on the cell centre, nothing else on
the board.  BEFORE: torso at (0, 0, 1.0), the August REST and every route that
landed on or left it.  AFTER: torso at (0.206, 0, 1.15), the October REST and
the two routes that reach it now.  A negative number is a refusal
(`FOOTPRINT_MARGIN` = 0).

    PYTHONPATH=src python3 docs/reviews/2026-10-17-october-layout-192/footprint_before_after.py
"""
import numpy as np

from reachy_ai.motion import footprint as F
from reachy_ai.motion import kinematics as K
from reachy_ai.motion import rig_routes as R
from reachy_ai.scene.awareness import SceneModel

SCENE = "scenes/FWDCenterLabSivaPool.yaml"
CELLS = ("cell_r1c3", "cell_r2c3", "cell_r3c3")
PROBES = ("pool_box_3", "foam_block", "red_cube", "pool_cyl_1", "blue_cylinder", "soda_can")
# The August poses, as they were in rig_routes before #192 retired them.
HOVER = R.pose(r_gripper=R.SHUT, r_shoulder_pitch=-40.0, r_shoulder_roll=-10.0,
               r_elbow_pitch=-60.0, r_wrist_pitch=-15.0)
REST_SHUT_AUGUST = R.pose(r_gripper=R.SHUT, r_shoulder_pitch=-40.0,
                          r_shoulder_roll=-10.0, r_elbow_pitch=-45.0,
                          r_wrist_pitch=-10.0, r_wrist_roll=30.0)
REST_AUGUST = dict(REST_SHUT_AUGUST, r_gripper=R.OPEN)
BEFORE_LEGS = {
    "PLACE_ROUTE": (HOVER, REST_SHUT_AUGUST, REST_AUGUST),
    "RAISE_TO_SIDE": (HOVER, REST_SHUT_AUGUST, REST_AUGUST),
    "STOW_ROUTE": (REST_AUGUST, REST_SHUT_AUGUST, HOVER),
    "STOW_FROM_SIDE": (R.PRESENT, REST_SHUT_AUGUST, HOVER),
    "LOWER_TO_REST": (R.PRESENT, REST_SHUT_AUGUST, REST_AUGUST),
    "LIFT_TO_PRESENT": (REST_AUGUST, R.PRESENT),
}
AFTER_ROUTES = ("LOWER_TO_REST", "LIFT_TO_PRESENT")


def table(base, legs, routes):
    saved_base, saved_legs = K._BASE.copy(), dict(R.FOOTPRINT_LEGS)
    K._BASE[:] = base
    R.FOOTPRINT_LEGS.clear(); R.FOOTPRINT_LEGS.update(legs)
    try:
        out = {}
        for cell in CELLS:
            for oid in PROBES:
                m = SceneModel.from_yaml(SCENE)
                x, y, z = m.cell_center(cell)
                m.update_poses({oid: (x, y, z + m.get(oid).size[2] / 2)})
                c = F.route_footprint_clearances(m, routes)[oid]
                out[(cell, oid)] = c
        return out
    finally:
        K._BASE[:] = saved_base
        R.FOOTPRINT_LEGS.clear(); R.FOOTPRINT_LEGS.update(saved_legs)


before = table((0.0, 0.0, 1.0), BEFORE_LEGS, list(BEFORE_LEGS))
after = table(tuple(K._BASE), dict(R.FOOTPRINT_LEGS), AFTER_ROUTES)
print("| cell | object (height) | before: clearance, link | after: clearance, link |")
print("|---|---|---|---|")
for cell in CELLS:
    for oid in PROBES:
        b, a = before[(cell, oid)], after[(cell, oid)]
        h = SceneModel.from_yaml(SCENE).get(oid).size[2]
        print(f"| {cell[5:]} | {oid} ({h*100:.1f} cm) | {b.distance*100:+.1f} cm, {b.link}"
              f"{' **refused**' if b.distance < R.FOOTPRINT_MARGIN else ''} | "
              f"{a.distance*100:+.1f} cm, {a.link}{' **refused**' if a.distance < R.FOOTPRINT_MARGIN else ''} |")
