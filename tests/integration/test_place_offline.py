"""#174: the placement flown offline, end to end, and judged as a live run is.

`tools.place_sweep.fly` flies `execute_crane_pick(..., place=...)` -- the
crane module unchanged -- against the #172 offline world through an SDK-shaped
arm whose every sleep advances the physics.  A flown run is recorded in the
live recorder's shape, replayed (`lift_replay.replay_place`, contacts from the
kinematically posed world) and judged by `evaluate_place` (D6).

  * r3c2 -> r3c3 places the cube within D6, withdraws, and ends at PRESENT;
  * a stop during the carry flies the carry back, sets the cube down where it
    was picked up -- never opening the hand before it rests on the table --
    withdraws and ends at PRESENT;
  * the lift (pick, hold, put back) is unchanged: put back within 5 mm, no
    release retreat, every criterion met.

October mount (#192, #195).  r3c2 is the crane's one origin for red_cube:
the palm (`grasp_alignment.PALM`, MJCF r_palm_col) cannot be held clear of the
cube while straddling it from r2c2 at any reachable attitude (see the
CraneSpec docstring in reachy_ai/tasks/crane_pick_live.py).  r3c3 is the clean
destination (r2c2 places but leaves residual grip force at the release; r2c1
aborts when re-planning from the measured hold).  The lift and the placement
share one attitude (tilt 22.5, lean 87.5).

Needs MuJoCo and scipy; each flight takes ~1-2 min.
"""

import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytest.importorskip("mujoco")
pytest.importorskip("scipy")

import tools.place_sweep  # noqa: E402,F401  (paths)
import tools.withdrawal_replay  # noqa: E402,F401
from tools.place_sweep.fly import as_task, fly  # noqa: E402

_world = {}


def _judge(run, dest):
    from replay_lift_evaluation import MODEL, SCENE, KinematicWorld
    from reachy_ai.evaluation.lift_replay import replay_lift, replay_place
    from reachy_ai.evaluation.panel_routes import evaluate

    if "w" not in _world:
        _world["w"] = KinematicWorld(str(MODEL), str(SCENE))
    w = _world["w"]
    replay = replay_place if dest else replay_lift
    r = replay(as_task(run, dest), run["states"], contacts_at=w.contacts,
               collidable=w.collidable, episode_id="offline")
    return evaluate("place_on_cell" if dest else "lift_object", r.result, r.spec,
                    plan=r.plan), r


def _kinds(run):
    return [k for k, _ in run["events"]]


def test_r3c2_to_r3c3_meets_d6_offline():
    run = fly(dest="r3c3", record=True)
    assert "refused" not in run, run.get("refused")
    v, r = _judge(run, "r3c3")
    assert v.is_successful, v.explanation
    rep = v.reported
    assert rep["placement_error_mm"] <= 10.0 and rep["tilt_deg"] <= 5.0
    assert rep["others_moved_max_mm"] <= 1.0
    assert rep["withdrawal"] == "completed" and rep["full_cycle"] is True
    # the carry and the set-down are their own phases, judged by the rule table
    assert {"carry", "set_down", "release", "withdraw"} <= {w.phase.value for w in r.windows}
    # the hand backed off as it let go, and the held object was corrected to
    # the cell centre before it was set down
    retreat = [kw for k, kw in run["events"] if k == "RELEASE_RETREAT"]
    assert retreat and retreat[0]["ok"] is True
    last = [kw for k, kw in run["events"] if k == "SETDOWN_MEASURE"][-1]
    assert last["error_mm"] <= 1.0
    assert run["criteria"]["placed_on_target"] is True


def test_a_stop_during_the_carry_sets_it_down_where_it_was_picked_up():
    run = fly(dest="r3c3", record=True,
              abort_when=lambda ev: any(k == "IN_HAND" for k, _ in ev))
    kinds = _kinds(run)
    assert "ABORT" in kinds
    abort = kinds.index("ABORT")
    # never released in the air: nothing opened between the hold and the
    # support at the pickup
    sup = [i for i, (k, kw) in enumerate(run["events"]) if k == "SUPPORTED"]
    assert len(sup) == 1 and sup[0] > abort and run["events"][sup[0]][1]["reached"]
    assert kinds.index("RELEASED") > sup[0]
    out = run["outcome"]
    start = out["place"]
    assert start["aborted"] == "aborted by the caller"
    assert out["final"] == "PRESENT" and out["withdrawal"]["clean"] is True
    assert run["criteria"]["placed_on_target"] is False
    # back on r3c2, within the lift's own put-back bound of where it was
    assert out["replaced"]["offset_from_start_mm"] <= 5.0
    v, _r = _judge(run, "r3c3")
    assert not v.is_successful and v.reported["aborted"] is True


def test_the_lift_is_unchanged():
    run = fly(dest=None, record=True)
    out = run["outcome"]
    assert all(out["criteria"].values()), out["criteria"]
    assert out["replaced"]["offset_from_start_mm"] <= 5.0
    assert out["replaced"]["release_retreat_mm"] == 0.0
    assert "RELEASE_RETREAT" not in _kinds(run) and "place" not in out
    v, _r = _judge(run, None)
    assert v.is_successful, v.explanation
    assert v.reported["full_cycle"] is True
