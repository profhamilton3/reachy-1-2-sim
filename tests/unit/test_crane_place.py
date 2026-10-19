"""#174: planning a placement -- the carry with the held object, the exit
check, the destination, and the place-only grasp attitude.

Preflight only: IK is the host mirror of the simulator's service
(`host_ik_arm.py`), nothing moves.

At the October mount (#192) the crane plans only from cell_r3c2 (it was
r2c2, where #174 was flown), so that is the origin here.  Its offline flights
halt at the hover until the crane is re-tuned for the October mount (a
follow-up to #192), so they are not tested here.
"""

import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "."):
    _abs = os.path.join(_HERE, _p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

pytest.importorskip("scipy")

import numpy as np  # noqa: E402

from host_ik_arm import HostIKArm  # noqa: E402
from reachy_ai.motion import grasp_alignment as G  # noqa: E402
from reachy_ai.motion.kinematics import CartesianPlanner  # noqa: E402
from reachy_ai.scene.awareness import SceneModel, SceneObject  # noqa: E402
from reachy_ai.tasks import crane_pick_live as CP  # noqa: E402
from reachy_ai.tasks import crane_place_live as PL  # noqa: E402

SCENE = os.path.join(_HERE, "../../scenes/FWDCenterLabSivaPool.yaml")


def _world(**others):
    """The pool scene, red_cube upright on r3c2; ``others`` stand where given."""
    scene = SceneModel.from_yaml(SCENE)
    x, y, z = scene.cell_center("cell_r3c2")
    pos = (x, y, z + 0.03)
    scene.update_poses({"red_cube": pos, **others})
    seen = CP.Observation(pos, (1.0, 0.0, 0.0, 0.0), 0.0, False)
    return scene, CartesianPlanner(HostIKArm(), scene=scene), seen


def _on(scene, cell, oid):
    x, y, z = scene.cell_center(f"cell_{cell}")
    return (x, y, z + scene.get(oid).size[2] / 2.0)


def test_r3c2_to_r3c3_is_planned_end_to_end_at_the_shared_attitude():
    scene, planner, seen = _world()
    pp = PL.plan_crane_place(planner, scene, "red_cube", "r3c3", seen)
    # October mount (#195): the placement and the lift share one attitude
    assert pp.pick.spec.tilt_deg == PL.PLACE_GRASP.tilt_deg == 22.5
    assert pp.pick.spec.lean_bearing_deg == PL.PLACE_GRASP.lean_bearing_deg == 87.5
    assert pp.dest.spec.release_retreat_m == pytest.approx(0.009)
    centre = np.asarray(scene.cell_center("cell_r3c3")[:2])
    assert np.hypot(*(pp.centre[:2] - centre)) < 1e-9          # D6 measures from here
    assert np.hypot(*(pp.target[:2] - centre)) == pytest.approx(0.0, abs=1e-9)   # no aim offset
    held = pp.checks["held"]
    assert set(held) == {f"carry: climb to +{pp.carry_rung} (pickup ladder)",
                         f"carry: to r3c3 +{pp.carry_rung}",
                         f"set-down: r3c3 +{pp.carry_rung} -> resting"}
    for rep in held.values():                     # the held box, checked
        assert rep["arm_mm"] > 0 and rep["others_mm"] > 0
    assert all(e["open"] for e in pp.checks["exits"].values())


def test_the_lift_shares_the_placement_attitude_but_has_no_release_retreat():
    spec = CP.CraneSpec()
    assert (spec.tilt_deg, spec.lean_bearing_deg) == (22.5, 87.5)
    assert (spec.tilt_deg, spec.lean_bearing_deg) == (PL.PLACE_GRASP.tilt_deg,
                                                      PL.PLACE_GRASP.lean_bearing_deg)
    assert spec.release_retreat_m == 0.0 and spec.release_backoff_m == 0.0


def test_a_placement_that_leaves_no_exit_is_refused(monkeypatch):
    """The exit check is the executor's footprint gate.  At the October mount
    no cell blocks LOWER_TO_REST (#192), so the gate is made stricter than
    the placed object's clearance (+9 cm or so on r2c1) to exercise it."""
    from reachy_ai.motion import rig_routes as R
    scene, planner, seen = _world()
    pp = PL.plan_crane_place(planner, scene, "red_cube", "r2c1", seen)
    assert pp.checks["exits"]["rest"]["open"]
    monkeypatch.setattr(R, "FOOTPRINT_MARGIN", 1.0)
    with pytest.raises(CP.CraneRefused) as e:
        PL.check_place(planner, scene, pp)
    assert e.value.stage == "exit"
    assert "no validated exit from PRESENT" in e.value.reason


def test_an_occupied_destination_is_refused():
    scene, _planner, _seen = _world()
    scene, planner, seen = _world(foam_block=_on(scene, "r2c1", "foam_block"))
    with pytest.raises(CP.CraneRefused) as e:
        PL.plan_crane_place(planner, scene, "red_cube", "r2c1", seen)
    assert e.value.stage == "destination" and "not free: foam_block" in e.value.reason


def test_the_carry_is_checked_with_the_held_object_in_it(monkeypatch):
    """An obstacle the HELD cube would cross on the carry is refused by the
    held-object check -- not only by the arm's.  (The whole-arm check is
    stubbed here so the held box is what is under test.)"""
    scene, planner, seen = _world()
    clean = PL.plan_crane_place(planner, scene, "red_cube", "r2c1", seen)
    c = clean.carry_rung
    # where the held box is halfway across the carry
    mid = [0.5 * (a + b) for a, b in zip(*clean.carry())]
    box = G.carried_box(G.hand_frames(mid, clean.dest.spec.close_deg), clean.pick.obj,
                        clean.rel)
    wall = SceneObject(id="wall", kind="box", center=tuple(box.center), size=(0.02, 0.02, 0.02),
                       dynamic=False, tracked=False)
    scene._objects["wall"] = wall
    monkeypatch.setattr(planner, "check_arm_path", lambda *a, **k: None)
    with pytest.raises(CP.CraneRefused) as e:
        PL.check_place(planner, scene, clean)
    assert e.value.stage == "carried"
    assert "would hit wall" in e.value.reason and f"carry: to r2c1 +{c}" in e.value.reason


def test_the_held_object_is_checked_against_the_arm():
    """`carried_arm_clearance`: a held box pushed into the forearm reads
    negative; where the grasp holds it, positive."""
    scene, planner, seen = _world()
    pp = PL.plan_crane_place(planner, scene, "red_cube", "r2c1", seen)
    q = pp.carry()[0]
    ok, _link, _i = G.carried_arm_clearance([q], -5.0, pp.pick.obj, pp.rel)
    assert ok > 0.0
    h = G.hand_frames(q, -5.0)
    from reachy_ai.motion.kinematics import link_capsules
    elbow_to_wrist = next(c for c in link_capsules(q, "right", -5.0) if c[0] == "forearm")
    on_forearm = 0.5 * (np.asarray(elbow_to_wrist[1]) + np.asarray(elbow_to_wrist[2]))
    rel = (pp.rel[0], h.thumb_rot.T @ (on_forearm - h.thumb.center))
    bad, link, _i = G.carried_arm_clearance([q], -5.0, pp.pick.obj, rel)
    assert bad < 0.0 and link == "forearm"


def test_a_measured_hold_re_plans_the_destination_and_is_re_checked():
    scene, planner, seen = _world()
    pp = PL.plan_crane_place(planner, scene, "red_cube", "r2c1", seen)
    rot, t = pp.rel
    moved = (rot, t + np.array([0.0, 0.0, 0.002]))
    again = PL.replan_from_hold(planner, scene, pp, moved)
    assert again.hold_source == "measured"
    assert again.checks["exits"]["rest"]["open"]
    # the object, held 2 mm differently, still stands on the cell centre
    b = G.carried_box(G.hand_frames(again.dest.rungs[0], again.dest.spec.close_deg),
                      pp.pick.obj, moved)
    assert np.hypot(*(b.center[:2] - again.target[:2])) < 0.001


def test_headings_are_tried_nearest_first():
    assert PL.headings_from(10.0)[:5] == [10.0, 25.0, -5.0, 40.0, -20.0]
    assert len(PL.headings_from(0.0)) == 24
