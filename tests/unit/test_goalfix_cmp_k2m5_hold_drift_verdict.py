"""K2-M5 (2026-09-26 pr144-k2-k4 assignment): a dedicated kill for
``cycle.py``'s own cycle-level ``any_hold_target_drift`` (~line 990),
never previously depended on by any test.

At 77d6dab, ``any_hold_target_drift = False`` (the mutation) survives
the whole focused suite untouched -- no test's assertions change,
confirmed empirically (see the handoff for the mutation run). This is
not because the mutant is equivalent: it is because nothing exercises a
REAL (non-lead-in) hold-window drift at the cycle-verdict level at all.

This fixture isolates it: a real, two-leg B cycle (PLACE_ROUTE then
LIFT_TO_PRESENT, via ``make_fixtures.FlightSim``, zero lag -- matching
this suite's own T4/F4/F5 realistic-fixture convention) where the W-blk
window is valid (``segment_indeterminate`` is False), every C0-C8 check
passes on both legs, there is no lead-in violation and no unassigned-
non-carry command on either leg -- and the ONLY failing condition is a
single stray, off-path, non-carry command appended to the FLIGHT leg's
own tail, well after LIFT_TO_PRESENT's single waypoint (``PRESENT``) has
already been reached and held. That stray falls inside the PARKED-TAIL
window (``holds.PARKED_TAIL_GOAL_INDEX``, built from ``leg.sidecar_t_hi``
-- the sidecar's own aligned span extends to include it), and its target
differs from ``PRESENT``'s own goal, giving a non-zero
``target_drift`` there.

Two things keep this window's own violation from being caught by
anything else:

- ``pathcheck.check_c6`` (evaluated in ``evaluate_leg``) explicitly
  SKIPS any hold window whose ``goal_index < 0`` (lead-in/parked-tail
  are "gated separately, not part of C6's own scope" -- cycle.py
  ~449-459) -- so C6 stays clean.
- ``evaluate_leg``'s own unassigned-non-carry scan is bounded above by
  ``final_local`` (the route's own final-waypoint-assigned command's
  local index) -- our stray sits strictly AFTER that, so it is never
  even considered by that scan, and (since it does not match any goal
  within ``ON_SEGMENT_TOL_DEG``/1 ULP admission) ``assign_goals`` leaves
  it indeterminate (``None``), which is not itself an assignment
  violation (``segments.assign_goals``'s own ``ok`` flag is unaffected
  by an indeterminate tail sample) and is excluded from C0's own goal
  sequence (``None`` entries are skipped).

So only ``cycle.py``'s own cycle-level hold-drift gate
(``any_hold_target_drift``, via the PARKED-TAIL window's generic
``any(v != 0.0 for v in target_drift.values())`` branch of
``_hold_violates`` -- the LEAD_IN-specific "any command present"
branch is not this one) ever reports it -- confirmed not equivalent to
C6 or the unassigned-non-carry gate, for at least this real
construction."""
import json
import os
import sys
from unittest import mock

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import holds  # noqa: E402
from tools.goalfix_cmp import pathcheck as pc  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402

CYCLE = "k2m5-flight-parked-tail-drift"


def _write_sidecar(control_dir, kind, first_seq, last_seq):
    doc = {"alignment": [{"server_seq": first_seq}, {"server_seq": last_seq}]}
    path = control_dir / f"{CYCLE}-{kind}.link.json"
    path.write_text(json.dumps(doc))
    return path


def _build(tmp_path):
    ev_dir = tmp_path / "ev"
    control_dir = tmp_path / "control"
    control_dir.mkdir()
    home = dict(R.HOME)
    with mock.patch.object(mf, "_LAG_RAD", [0.0] * 8):
        sim = mf.FlightSim(home, pose_units="deg", restream_passes=1)
        sim.fly(R.PLACE_ROUTE)
        setup_last_seq = sim.state_rows[-1]["seq"]
        setup_first_seq = 0
        for _ in range(25):  # 0.5s settle, present position continuously reported
            sim._hold_ticks(1, dict(sim.pose))
        flight_first_seq = sim.state_rows[-1]["seq"]
        sim.pose = dict(R.REST)
        sim.fly(R.LIFT_TO_PRESENT)
        # A genuine settle after PRESENT's own arrival (no commands) --
        # the leg's own final-waypoint hold, unrelated to the stray below.
        for _ in range(15):
            sim._hold_ticks(1, dict(sim.pose))
        # ONE extra, off-path, non-carry command well after PRESENT's own
        # arrival: 15 deg PAST the goal, continuing the SAME direction
        # LIFT_TO_PRESENT's own goto already moved in -- far beyond
        # ON_SEGMENT_TOL_DEG (1e-4 deg)/1 ULP on the goal side, so
        # `assign_goals` leaves it indeterminate rather than admitting it.
        stray_target = dict(sim.pose)
        stray_target["r_shoulder_pitch"] = sim.pose["r_shoulder_pitch"] - 15.0
        sim._tick_command_then_state(stray_target)
        flight_last_seq = sim.state_rows[-1]["seq"]

    result = sim.result()
    mf.write_evidence(ev_dir, result.state_rows, result.command_rows)
    _write_sidecar(control_dir, "setup", setup_first_seq, setup_last_seq)
    _write_sidecar(control_dir, "flight", flight_first_seq, flight_last_seq)
    return ev_dir, control_dir


class TestHoldDriftAloneStopsTheBVerdict:
    def test_isolating_premise(self, tmp_path):
        """Every OTHER check is clean on both legs; only the flight
        leg's own parked-tail hold reports a violation, and only via
        the generic (non-LEAD_IN) drift branch."""
        ev_dir, control_dir = _build(tmp_path)
        evd = ev.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        setup_leg = cyc.resolve_leg(evd, control_dir / f"{CYCLE}-setup.link.json",
                                     "setup", R.PLACE_ROUTE, R.CRITICAL_JOINTS)
        flight_leg = cyc.resolve_leg(evd, control_dir / f"{CYCLE}-flight.link.json",
                                      "flight", R.LIFT_TO_PRESENT, R._PRESENT_GUARD)
        setup_lr = cyc.evaluate_leg(evd, setup_leg)
        flight_lr = cyc.evaluate_leg(evd, flight_leg)

        for lr in (setup_lr, flight_lr):
            failing = [cid for cid in pc.ALL_CHECKS if not lr.pathcheck[cid].passed]
            assert failing == [], {cid: lr.pathcheck[cid].detail for cid in failing}
            assert lr.lead_in_violation is False, lr.lead_in_detail
            assert lr.unassigned_non_carry_violation is False, lr.unassigned_non_carry_detail

        # setup: no hold window (lead-in/parked-tail) reports a violation.
        for hs in setup_lr.hold_stats:
            if hs.window.goal_index == holds.LEAD_IN_GOAL_INDEX:
                assert not hs.window_command_indices, hs.window_command_indices
            else:
                assert not any(v != 0.0 for v in hs.target_drift.values()), hs.target_drift

        # flight: every window is clean EXCEPT the parked-tail one, which
        # carries exactly our one stray command and a non-zero drift on
        # r_shoulder_pitch only.
        parked_tail = [hs for hs in flight_lr.hold_stats
                       if hs.window.goal_index == holds.PARKED_TAIL_GOAL_INDEX]
        assert len(parked_tail) == 1, flight_lr.hold_stats
        pt = parked_tail[0]
        assert len(pt.window_command_indices) == 1, pt.window_command_indices
        assert pt.target_drift["r_shoulder_pitch"] != 0.0, pt.target_drift
        for hs in flight_lr.hold_stats:
            if hs is pt:
                continue
            if hs.window.goal_index == holds.LEAD_IN_GOAL_INDEX:
                assert not hs.window_command_indices, hs.window_command_indices
            else:
                assert not any(v != 0.0 for v in hs.target_drift.values()), hs.target_drift

    def test_hold_drift_alone_stops_the_b_verdict_at_rc_2(self, tmp_path):
        ev_dir, control_dir = _build(tmp_path)
        evd = ev.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        setup_leg = cyc.resolve_leg(evd, control_dir / f"{CYCLE}-setup.link.json",
                                     "setup", R.PLACE_ROUTE, R.CRITICAL_JOINTS)
        flight_leg = cyc.resolve_leg(evd, control_dir / f"{CYCLE}-flight.link.json",
                                      "flight", R.LIFT_TO_PRESENT, R._PRESENT_GUARD)

        cv, _ = cyc.evaluate_cycle(CYCLE, "B", evd, [setup_leg, flight_leg], skip_gates=True)
        assert cv.segment_indeterminate is False, cv.reasons
        assert cv.genuine_echo_count == 0
        assert len(cv.reasons) == 1, cv.reasons
        assert cv.reasons[0].startswith("flight: hold at goal -2 drifted"), cv.reasons
        assert cv.verdict == cyc.VERDICT_STOP, cv.reasons
        assert cv.rc() == 2
