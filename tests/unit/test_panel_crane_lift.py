"""Issue #55: the crane lift reaches the panel (lift_object).

Focused on the joins, not on the arm logic (test_crane_grasp_alignment.py):
the simulator link carries what the grasp needs (orientation, grip force),
the request maps onto the ability and reads back, the motion process gets
live feedback over its existing pipe without a socket of its own, a missing
feed is treated as stale (which halts), and the panel judges a lift by its
measured criteria rather than by "the motion finished".
"""

import json
import os
import sys
import textwrap
import time

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../web"))
sys.path.insert(0, os.path.join(_HERE, "../../src"))

import motion_worker  # noqa: E402
import panel_abilities as abilities  # noqa: E402
from panel_executor import LIFT_CRITERIA, MotionWorker, _judge_lift  # noqa: E402
from panel_language import Action  # noqa: E402
from panel_sim_link import SimLink  # noqa: E402
from tasks import Proposal  # noqa: E402


def test_the_link_keeps_orientation_and_grip_force():
    link = SimLink("ws://unused")
    link._ingest_state({
        "seq": 7, "objects": [
            {"object_id": "red_cube", "pos_xyz": [0.43, 0.0, 0.77],
             "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
            {"object_id": "soda_can", "pos_xyz": [0.5, 0.9, 0.06],
             "quat_wxyz": [float("nan"), 0.0, 0.0, 0.0]}],
        "grippers": [{"side": "right", "grip_force_n": 0.25, "grasping": False},
                     {"side": "left", "grip_force_n": float("inf")}]})
    snap = link.snapshot()
    assert snap.quats == {"red_cube": (1.0, 0.0, 0.0, 0.0)}   # NaN dropped, not identity
    assert snap.grippers == {"right": {"grip_force_n": 0.25, "grasping": False}}
    assert snap.objects["soda_can"] == (0.5, 0.9, 0.06)       # positions unchanged


def test_the_request_maps_onto_the_lift_and_reads_back():
    m = abilities.match("pick up the red cube and put it back")
    assert m.name == "lift_object" and m.slots[abilities.SLOT_OBJECT] == "red cube"
    assert abilities.REGISTRY["lift_object"].end_posture == "present"
    act = Action(ability="lift_object", arguments={abilities.SLOT_OBJECT: "red cube"})
    ok, why = act.reads_back()
    assert ok, why
    # pointing is unchanged
    assert abilities.match("point to the red cube").name == "point_object"


def test_no_forwarded_feedback_reads_as_stale(monkeypatch):
    monkeypatch.setattr(motion_worker, "OBSERVATIONS", motion_worker.Observations())
    observe = motion_worker._observer("red_cube")
    assert observe().age_s > 1.0                       # halts a monitored move
    motion_worker.OBSERVATIONS.put({
        "object_id": "red_cube", "position": [0.43, 0.0, 0.77],
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0], "age_s": 0.01,
        "grippers": {"right": {"grip_force_n": 0.4, "grasping": False}}})
    o = observe()
    assert o.position == (0.43, 0.0, 0.77) and o.grip_force_n == 0.4
    assert o.age_s < 0.5
    # an observation of a different object is not feedback on this one
    assert motion_worker._observer("soda_can")().age_s > 1.0


def test_the_parent_forwards_observations_over_the_worker_pipe(tmp_path):
    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent("""
        import json, sys
        seen = 0
        for line in sys.stdin:
            msg = json.loads(line)
            if "observe" in msg:
                seen += 1
                if seen >= 3:
                    print(json.dumps({"result": {"status": "moved", "seen": seen,
                                                 "last": msg["observe"]}}), flush=True)
            elif "job" in msg:
                print(json.dumps({"phase": "started"}), flush=True)
    """))
    worker = MotionWorker(argv=[sys.executable, "-u", str(child)], deadline_s=10)
    n = {"i": 0}

    def feed():
        n["i"] += 1
        time.sleep(0.01)
        return {"seq": n["i"], "object_id": "red_cube"}
    try:
        out = worker.run({"kind": "ability"}, feed=feed)
    finally:
        worker.close()
    assert out["status"] == "moved" and out["seen"] >= 3
    assert out["last"]["object_id"] == "red_cube"


def _proposal():
    return Proposal(plan_id="p", plan_version=0, task_type="lift_object",
                    target_id=None, destination=None, destination_kind="",
                    brief_reason="", arm="right", route="CRANE_LIFT",
                    route_version=1, object_id="red_cube", summary="lift")


def test_a_lift_is_judged_by_its_criteria_not_by_finishing():
    crane = {"criteria": {k: True for k, _ in LIFT_CRITERIA},
             "lifted_and_held": {"rise_start_mm": 42.4, "rise_end_mm": 42.3,
                                 "in_hand_slip_mm": 0.05},
             "replaced": {"offset_from_start_mm": 0.8}}
    assert _judge_lift(_proposal(), crane, "present", {}).status == "completed"
    crane["criteria"]["slip_within_resolution"] = False
    res = _judge_lift(_proposal(), crane, "present", {})
    assert res.status == "failed" and \
        "did not keep it from slipping more than 0.1 mm" in res.detail
    assert res.evidence["crane"] is crane
    stopped = _judge_lift(_proposal(), {"halt": "grip force 0.3 N"}, "present", {})
    assert stopped.status == "failed" and "grip force" in stopped.detail


def test_the_corrected_path_is_shown_and_kept():
    """The measured correction, the way-out judgement and the hold are phases
    the panel shows, and the key events stay in the evidence afterwards."""
    shown, kept = [], []
    on_event = motion_worker._crane_phases(shown.append, kept)
    on_event("MEASURE", label="descent rung", rung=5, gap_error_mm=[0, 0, 0],
             pads={"thumb_side_mm": 1, "finger_side_mm": 1})
    on_event("CORRECTION", label="hover", iteration=1, because="finger route -1.6 mm",
             shift_mm=[3.4, 0.4, 8.8], cumulative_mm=[3.4, 0.4, 8.8], predicted_route={})
    on_event("WITHDRAWAL_JUDGED", ok=False, route={"route_thumb_mm": -0.46,
                                                   "route_finger_mm": 5.65})
    on_event("HELD", rise_end_mm=40.5, in_hand_slip_mm=1.69)
    assert any("correcting the commanded path by (+3.4, +0.4, +8.8) mm" in p for p in shown)
    assert any("-0.5 / 5.7 mm (NOT clear)" in p for p in shown)
    assert any("slip in the hand 1.69 mm" in p for p in shown)
    assert [e["kind"] for e in kept] == ["CORRECTION", "WITHDRAWAL_JUDGED", "HELD"]
