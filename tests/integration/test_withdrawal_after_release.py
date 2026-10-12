"""#173: the ten Phase 1.5 lifts, released and withdrawn offline as the crane
now does it (O1+O2), in the #172 offline world.

Each lift's state just before the hand opened is rebuilt from its fixture
(`tools.withdrawal_replay`): the plan re-planned with the host IK mirror and
checked against the recorded commanded pose, the physics posed from the
recorded joints and object poses.  Then the release and the withdrawal are
flown in MuJoCo with the crane's own decisions (`plan_way_out`).

  * the four refused live (A1, A4, B4, B5) must now withdraw;
  * the six completed live must still withdraw;
  * the new withdrawal step, from the RECORDED release state of every lift
    (the live release, not a rebuilt one), must withdraw too.

Needs MuJoCo and scipy; ~5 s a lift.
"""

import json
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

pytest.importorskip("mujoco")
pytest.importorskip("scipy")

from tools.withdrawal_replay import options as O  # noqa: E402
from tools.withdrawal_replay.rebuild import Rebuilt  # noqa: E402
from tools.withdrawal_replay.record import load  # noqa: E402

FIXTURES = _REPO / "tests" / "fixtures" / "withdrawal_173"
REFUSED_LIVE = ("A1", "A4", "B4", "B5")
COMPLETED_LIVE = ("A2", "A3", "A5", "B1", "B2", "B3")

_cache = {}


def rebuilt(name):
    if name not in _cache:
        _cache[name] = Rebuilt(load(str(FIXTURES / f"{name}.json")))
    return _cache[name]


def test_the_fixtures_are_the_ten_lifts_and_their_recorded_outcomes():
    names = sorted(p.stem for p in FIXTURES.glob("*.json"))
    assert names == sorted(REFUSED_LIVE + COMPLETED_LIVE)
    for name in names:
        rec = json.loads((FIXTURES / f"{name}.json").read_text())
        assert rec["recorded"]["withdrawal_clean"] is (name in COMPLETED_LIVE)


@pytest.mark.parametrize("name", REFUSED_LIVE + COMPLETED_LIVE)
def test_the_rebuild_matches_the_recording(name):
    rb = rebuilt(name)
    assert rb.plan_err_deg < 0.01
    assert rb.cmd_fit_deg < 0.1


@pytest.mark.parametrize("name", REFUSED_LIVE)
def test_a_lift_refused_live_now_withdraws(name):
    r = O.run_option(rebuilt(name), "O1+O2")
    assert not r["refused"], r
    assert r.get("halt") is None
    assert r["kind"] in ("direct", "sideways", "separating")
    assert r["residual_mm"] <= 0.1, r
    assert r["final_opening_deg"] == pytest.approx(O.SPEC.wide_opening_deg, abs=0.5)


@pytest.mark.parametrize("name", COMPLETED_LIVE)
def test_a_lift_completed_live_still_withdraws(name):
    r = O.run_option(rebuilt(name), "O1+O2")
    assert not r["refused"], r
    assert r.get("halt") is None
    assert r["residual_mm"] <= 0.1, r


@pytest.mark.parametrize("name", REFUSED_LIVE + COMPLETED_LIVE)
def test_the_new_withdrawal_from_the_recorded_release(name):
    r = O.withdraw_from_recorded(rebuilt(name))
    assert not r["refused"], r
    assert r.get("halt") is None


def test_the_code_before_173_still_refuses_offline():
    """Control: the base option on the same rebuilt states refuses where the
    live code refused (A1, A4) -- the rebuild is not simply easier."""
    for name in ("A1", "A4"):
        assert O.run_option(rebuilt(name), "base")["refused"]
