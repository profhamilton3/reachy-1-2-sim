"""#172: rebuilding a recorded lift from its states and panel task, offline.

A synthetic recording, constructed here: HOME -> REST -> PRESENT on
RAISE_TO_SIDE, then the crane's lift of red_cube, with the crane's events
stamped on a clock 50 ms ahead of the states' (as the recorded ones are).
"""

import math
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco"):
    _abs = os.path.join(_HERE, _p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

pytest.importorskip("numpy")

from reachy_ai.evaluation import contact_rules as CR  # noqa: E402
from reachy_ai.evaluation import lift_replay as LR  # noqa: E402
from reachy_ai.evaluation.panel_routes import (  # noqa: E402
    LIFT_PUT_BACK_KEY, LIFT_RISE_KEY, LIFT_SLIP_KEY, WITHDRAWAL_KEY,
    contact_samples, evaluate,
)
from reachy_ai.motion import rig_routes as R  # noqa: E402

P = CR.Phase
SKEW = 0.05          # event clock minus state clock
T0 = 1000.0
Z0 = 0.77


def _joints(posture):
    pose = dict(R.POSTURES[posture])
    pose.setdefault("r_gripper", -65.0)
    for j in LR.ARM_JOINTS:
        pose.setdefault(j, 0.0)
    return [{"name": n, "position_rad": math.radians(d)} for n, d in pose.items()]


def _state(i, t, posture, cube):
    return {"type": "state", "sim_step": 1000 + 10 * i, "sim_time_s": 2.0 + 0.02 * i,
            "host_time": t, "joints": _joints(posture),
            "objects": [{"object_id": "red_cube", "pos_xyz": list(cube),
                         "quat_wxyz": [1, 0, 0, 0]},
                        {"object_id": "soda_can", "pos_xyz": [0.55, 0.95, 0.06],
                         "quat_wxyz": [1, 0, 0, 0]}]}


#: (start s, end s, posture, cube z offset, cube x offset)
TIMELINE = [
    (0, 2, "home", 0.0, 0.0),
    (2, 4, "rest", 0.0, 0.0),           # RAISE_TO_SIDE passes REST
    (4, 10, "present", 0.0, 0.0),       # at PRESENT: the crane starts, approaches
    (10, 14, "present", 0.0, 0.0),      # grasp: close + lift-off
    (14, 16, "present", 0.05, 0.0),     # hold, 5 cm up
    (16, 20, "present", 0.0, 0.0),      # set-down
    (20, 22, "present", 0.0, 0.003),    # release; the drag moved it 3 mm
    (22, 30, "present", 0.0, 0.003),    # withdrawal, back at PRESENT
]


def recording():
    states, i = [], 0
    for a, b, posture, dz, dx in TIMELINE:
        t = float(a)
        while t < b:
            # A resting object in MuJoCo jitters at ~1e-12 m, so no two
            # recorded states hold bit-identical poses; neither do these.
            states.append(_state(i, T0 + t, posture,
                                 (0.4318 + dx + 1e-12 * i, 0.0, Z0 + dz)))
            i += 1
            t += 0.25
    return states


def ev(kind, at, **kw):
    return dict(kind=kind, t=T0 + at + SKEW, **kw)


def _cube_at(at):
    """The cube's recorded position in the state at ``at`` seconds."""
    return LR.object_pos(min(recording(), key=lambda s: abs(s["host_time"]
                                                            - (T0 + at))),
                         "red_cube")


def task(*, phase_events=False, clean=True):
    events = [ev("STRADDLE", 10.0, ok=True),
              ev("CLOSED", 12.0, object=_cube_at(12.0)),
              ev("HELD", 16.0, ok=True, object=_cube_at(15.75)),
              ev("SUPPORTED", 20.0, reached=True),
              ev("RELEASED", 22.0, object=_cube_at(22.0))]
    if phase_events:
        events += [ev("phase", 5.0, name="ATTEMPT_START"),
                   ev("phase", 10.0, name="close"),
                   ev("phase", 12.0, name="lift"),
                   ev("phase", 14.0, name="hold (2 s)"),
                   ev("phase", 16.0, name="replace"),
                   ev("phase", 22.0, name="withdrawal")]
    return {
        "task_id": "t1", "state": "completed" if clean else "failed",
        "events": [{"role": "user", "text": "lift it", "at": T0 - 1},
                   {"role": "reachy", "text": "Executing in simulation.",
                    "at": T0 + SKEW},
                   {"role": "reachy", "text": "done", "at": T0 + 29.9 + SKEW}],
        "proposal": {"object_id": "red_cube", "route": "CRANE_LIFT",
                     "route_version": 1, "expected_start_posture": "present"},
        "execution_evidence": {
            "start_posture": "home", "final_posture": "present",
            "crane": {"events": events,
                      "lifted_and_held": {"rise_end_mm": 50.0,
                                          "in_hand_slip_mm": 0.0},
                      "replaced": {"offset_from_start_mm": 3.0},
                      "withdrawal": {"clean": clean}}}}


def contacts(state):
    """Forearm on the board at REST; pads on the cube while it is up."""
    pose = R.posture_of(LR.joints_deg(state))
    z = LR.object_pos(state, "red_cube")[2]
    out = []
    if pose == "rest":
        out.append(("r_forearm", "table_top", -0.0004))
    if z > Z0 + 0.01:
        out.append(("r_gripper_thumb", "red_cube", -0.0001))
    else:
        out.append(("red_cube", "table_top", -0.00001))
    return out


def test_the_two_clocks_are_aligned_on_the_objects_measured_position():
    t = task()
    assert LR.clock_offset(t["execution_evidence"]["crane"]["events"],
                           recording(), "red_cube") == pytest.approx(SKEW)


def test_the_approach_leg_and_the_attempt_start_are_read_from_the_arm():
    job = recording()
    legs, leg_of, arrived = LR.approach_legs(job, "home", "present")
    assert legs == ["RAISE_TO_SIDE"]
    assert set(leg_of) == {"RAISE_TO_SIDE"}
    assert arrived == pytest.approx(T0 + 4.0)
    assert LR.approach_legs(job, "present", "present")[0] == []


@pytest.mark.parametrize("phase_events", [False, True])
def test_the_phase_windows(phase_events):
    rep = LR.replay_lift(task(phase_events=phase_events), recording(),
                         contacts_at=contacts)
    got = [(w.phase, round(w.start - T0, 2), round(w.end - T0, 2))
           for w in rep.windows]
    start = 5.0 if phase_events else 4.0
    assert got == [(P.APPROACH, start, 10.0), (P.GRASP, 10.0, 14.0),
                   (P.HOLD, 14.0, 16.0), (P.SET_DOWN, 16.0, 20.0),
                   (P.RELEASE, 20.0, 22.0), (P.WITHDRAW, 22.0, 29.9)]
    assert rep.approach_legs == ["RAISE_TO_SIDE"]


def test_the_lift_is_re_measured_from_the_states():
    rep = LR.replay_lift(task(), recording(), contacts_at=contacts)
    m = rep.result.metrics
    assert m[LIFT_RISE_KEY] == pytest.approx(0.05)
    assert m[LIFT_SLIP_KEY] == pytest.approx(0.0, abs=1e-9)
    assert m[LIFT_PUT_BACK_KEY] == pytest.approx(0.003)
    assert m[WITHDRAWAL_KEY] == 1.0
    assert rep.result.start_sim_step == 1000
    assert rep.result.end_sim_step > rep.result.start_sim_step
    assert rep.spec.target_object_id == "red_cube"
    assert rep.crane_reported["rise_end_mm"] == 50.0


def test_contacts_carry_their_phase_and_the_approach_its_route():
    rep = LR.replay_lift(task(), recording(), contacts_at=contacts)
    got = {(c.body1, c.body2, c.phase, c.pose, c.route)
           for c in contact_samples(rep.result)}
    assert ("r_forearm", "table_top", P.APPROACH, "rest", "RAISE_TO_SIDE") in got
    assert ("r_gripper_thumb", "red_cube", P.HOLD, "present", "") in got
    assert not any(c[2] is P.HOLD and c[1] == "table_top" for c in got)


def test_a_replayed_full_cycle_and_a_replayed_refusal():
    rep = LR.replay_lift(task(), recording(), contacts_at=contacts)
    v = evaluate("lift_object", rep.result, rep.spec, plan=rep.plan)
    assert v.is_successful and v.reported["full_cycle"], v.explanation
    assert v.reported["approach"]["RAISE_TO_SIDE"], \
        "the forearm at REST on RAISE_TO_SIDE is reported, under that route"

    rep = LR.replay_lift(task(clean=False), recording(), contacts_at=contacts)
    v = evaluate("lift_object", rep.result, rep.spec, plan=rep.plan)
    assert v.is_successful and not v.reported["full_cycle"]
    assert v.reported["withdrawal"] == "refused"


def test_without_contacts_the_lift_cannot_be_cleared():
    rep = LR.replay_lift(task(), recording())
    v = evaluate("lift_object", rep.result, rep.spec)
    assert not v.is_valid and not v.is_successful
