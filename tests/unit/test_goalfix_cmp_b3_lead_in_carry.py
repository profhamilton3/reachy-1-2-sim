"""B3 (H3): the lead-in gate, isolated from C1'/N3 (merge verdict §2;
coordinator Stage B authorization).

H3 found that the previously-tried "lead-in violation" fixture also
failed setup C1/flight C1/C2, so asserting the reason string alone could
not prove the lead-in gate is independently load-bearing (a mutant
removing it would still STOP through those other checks). The agent's
claimed equivalence ("every lead-in violation also fails C1") was
unverified, and is in fact FALSE for the specific shape this module
builds: a CARRY (bit-exact to the immediately preceding command) is
EXEMPT from N3/C1' by construction (R-carry, B16) -- so a stray command
that is itself a carry can precede the real turn_on match while every
other check (C0-C8, including the new C1'/N3) stays clean. Only the
lead-in gate catches it.

Construction (hand-built evidence, real bracket/epoch semantics via
``evidence.verify_and_load``, no simulated plant): a single-waypoint
route, right-arm ``r_shoulder_pitch`` only.

- command 0: target = X (a few 1e-5 deg forward of ``S``={start_pose8},
  well inside C1' (0.05deg) and the admission box) -- NOT a carry (its
  own reference is ``start_pose8``), but passes N3/C1' on its own
  merits. The plant has NOT yet caught up (state stays at S) so this
  does not match the turn_on present-position property.
- command 1: target = X again -- a bit-exact CARRY of command 0.
  EXEMPT from N3/C1'. The plant still has not caught up.
- command 2: target = X again -- ALSO a bit-exact carry of command 1
  (exempt too), but by now the plant HAS caught up to X, so command 2's
  own target matches the present-position reading at its own bracket --
  the REAL, correctly-matched turn_on command.

Two commands (0 and 1) precede the real match -- a lead-in violation --
while C0-C8 all pass (verified explicitly below).
"""
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
from reachy_ai.motion.rig_routes import Waypoint as RWaypoint  # noqa: E402

BASE = 0.0
X = np.radians(3e-5)  # ~1e-4 deg admission tolerance's own scale, well under it

# A REAL rig_routes.Waypoint (degrees) -- resolve_leg's own route_rad()
# conversion needs .tol/.guard, which the local fixture-only mf.Waypoint
# does not carry.
ROUTE = [RWaypoint("A", {"r_shoulder_pitch": 90.0}, 2.0, 6.0)]
START = {j: 0.0 for j in mf.R_JOINTS}


def _pose(v):
    return {j: (v if j == "r_shoulder_pitch" else 0.0) for j in mf.R_JOINTS}


GOAL = np.radians(90.0)


def _build(tmp_path):
    # commands 0, 1, 2: the lead-in shape (0/1 are carries preceding the
    # real match at 2). Command 3 completes the goto (arrives exactly at
    # G) so C4' has a real, passing last setpoint to check -- this
    # fixture is not just a lead-in window, it is a whole (minimal) leg.
    targets = [X, X, X, GOAL]
    present_before = [BASE, X, GOAL, GOAL]

    rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                          wall_time_ns=1, position_rad21=mf.full21(_pose(BASE)))]
    cmds = []
    for i, (tgt, present) in enumerate(zip(targets, present_before)):
        cmds.append(mf.command_row_joint(seq=i, target_rad21=mf.full21(_pose(tgt))))
        rows.append(mf.state_row(seq=i + 1, sim_step=i + 1, sim_time_s=(i + 1) * 0.02,
                                  cmd_seq=i, wall_time_ns=2 + i,
                                  position_rad21=mf.full21(_pose(present))))

    mf.write_evidence(tmp_path, rows, cmds)
    control_dir = tmp_path.parent / (tmp_path.name + "-control")
    control_dir.mkdir(exist_ok=True)
    sidecar = control_dir / "cyc-setup.link.json"
    import json as _json
    sidecar.write_text(_json.dumps({
        "alignment": [{"server_seq": 0}, {"server_seq": len(rows) - 1}]}))
    return tmp_path, control_dir, sidecar


class TestCarryBeforeCorrectlyMatchedTurnOn:
    def test_only_the_lead_in_gate_fails(self, tmp_path):
        ev_dir, control_dir, sidecar = _build(tmp_path)
        evd = ev.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        leg = cyc.resolve_leg(evd, sidecar, "setup", ROUTE, ())
        lr = cyc.evaluate_leg(evd, leg)

        # The isolating claim: every C0-C8 check passes.
        failing = [cid for cid in pc.ALL_CHECKS if not lr.pathcheck[cid].passed]
        assert failing == [], {cid: lr.pathcheck[cid].detail for cid in failing}
        assert not lr.unassigned_non_carry_violation

        # ... yet the lead-in gate DOES fire: two commands (0 and 1)
        # precede the real turn_on match (local index 2).
        turn_on_match = holds.find_turn_on_command(evd, leg.command_indices)
        assert turn_on_match is not None
        assert turn_on_match.local_index == 2
        assert turn_on_match.violation is True
        assert lr.lead_in_violation is True

    def test_the_two_leading_commands_are_carries_exempt_from_c1_prime(self, tmp_path):
        """Confirms WHY C1 does not (and cannot) catch this: commands 0
        and 1 are carries under R-carry's own definition, hence outside
        C1'/N3's scope entirely -- this is not a coincidence of the
        chosen tolerance, it is what a carry always does."""
        from tools.goalfix_cmp import segments as seg
        ev_dir, control_dir, sidecar = _build(tmp_path)
        evd = ev.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        leg = cyc.resolve_leg(evd, sidecar, "setup", ROUTE, ())
        idx = leg.command_indices
        targets8 = [{name: float(evd.commands.target_rad[i, k])
                     for k, name in enumerate(pc.R_JOINTS)} for i in idx]
        carries = seg.carry_mask(targets8, leg.start_pose8)
        assert carries[1]["r_shoulder_pitch"] is True   # command 1: carry of command 0
        assert carries[0]["r_shoulder_pitch"] is False  # command 0: NOT a carry (leg's own first)

    def test_mutation_lead_in_gate_removed_would_authorize(self, tmp_path):
        """Mutation guard: with the lead-in gate's own contribution
        removed (find_turn_on_command's violation flag ignored), this
        exact fixture has no OTHER failing check to fall back on --
        reproduced directly: `lead_in_violation` is the ONLY True
        boolean this leg reports among lead-in/C0-C8/hold-drift/
        unassigned-non-carry."""
        ev_dir, control_dir, sidecar = _build(tmp_path)
        evd = ev.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        leg = cyc.resolve_leg(evd, sidecar, "setup", ROUTE, ())
        lr = cyc.evaluate_leg(evd, leg)
        any_pathcheck_fail = any(not lr.pathcheck[cid].passed for cid in pc.ALL_CHECKS)
        any_hold_drift = any(
            (hs.window.goal_index == holds.LEAD_IN_GOAL_INDEX and hs.window_command_indices)
            or any(v != 0.0 for v in hs.target_drift.values())
            for hs in lr.hold_stats if hs.window.goal_index != holds.LEAD_IN_GOAL_INDEX)
        # The lead-in window itself DOES carry the two stray commands
        # (that is what makes it a violation for the LEAD_IN window
        # specifically) -- but with the lead-in gate's own signal
        # (`lead_in_violation`/the LEAD_IN window's own commands check)
        # removed, nothing else in this fixture is true.
        assert not any_pathcheck_fail
        assert not any_hold_drift
        assert not lr.unassigned_non_carry_violation
        # i.e. the mutant (drop lead_in_violation and the LEAD_IN
        # window's own "any command inside it" rule) would authorize
        # this leg cleanly -- the shipped code does not, because
        # lr.lead_in_violation is True.
        assert lr.lead_in_violation is True


# ---------------------------------------------------------------------------
# K2-M1 (2026-09-26 pr144-k2-k4 assignment): a genuine B VERDICT assertion,
# not just `lr.lead_in_violation`. Neither this module's own tests above
# nor test_goalfix_cmp_f6_surviving_mutants.py's TestA1LeadInCarryThroughCli
# kill cycle.py:1193's `or any_lead_in_violation` term specifically:
#
# - The tests above never reach `evaluate_cycle`/the cycle-level verdict
#   at all -- they stop at `lr.lead_in_violation`/`evaluate_leg`.
# - F6's own CLI test deliberately offsets its stray command by 0.05 rad
#   (~2.9 deg), which ALSO fails C1 (its own docstring: "C1 co-firing is
#   reported, not hidden"). With C1 also failing, `any_pathcheck_fail` is
#   True too, so deleting `or any_lead_in_violation` from the STOP
#   condition would not flip that fixture's verdict -- and the fixture's
#   OTHER assertion (a "lead-in violation" substring in `reasons`) is
#   unaffected by the mutation either way, since `lead_in_reasons` is
#   built unconditionally (cycle.py ~1004), independent of the STOP
#   formula the mutation touches.
#
# This class isolates the mutation at the CYCLE-VERDICT level: a real,
# two-leg B cycle (the shipped PLACE_ROUTE/LIFT_TO_PRESENT routes, via
# make_fixtures.FlightSim -- not a hand-built single-waypoint route)
# where the W-blk window is VALID (segment_indeterminate is False,
# unlike the single-waypoint fixture above, where it always is True and
# would mask the mutation by itself forcing STOP regardless), every
# C0-C8 check passes on both legs, hold-drift and the unassigned-non-
# carry gate are both clean, and the ONLY failing condition is the setup
# leg's own lead-in gate -- so cycle.py:1193 read verdict = STOP
# (rc 2), and with `or any_lead_in_violation` deleted, verdict would be
# OK (rc 0).
#
# `cyc._cli`'s own `--validation-mode` cannot show "rc 2" at all: `_cli`
# unconditionally sets `cv.validation_only = True` in that mode, and
# `CycleVerdict.rc()` maps `validation_only` to `RC_INCONCLUSIVE` (3)
# BEFORE even consulting `verdict` -- so no `--validation-mode` CLI run
# can ever report rc 2, whatever the verdict. Getting a real rc 2
# through the full (non-validation) CLI needs the operator-supplied
# provenance/compliance/arm_map/versions inputs (MB2), which are
# unrelated to the lead-in gate and would only add unrelated noise to
# what this test isolates. This is the documented reason `evaluate_cycle`
# is used directly below, exactly as the assignment allows.
#
# The setup leg's own commanded MINUS realised offset (a per-joint
# constant, applied only while flying PLACE_ROUTE) is what keeps
# `holds.find_turn_on_command` from ever matching -- every command's
# target is a few float32 ULPs away from the present-position reading
# one tick before it, on every joint, for the whole leg (never a
# bit-exact match, so `find_turn_on_command` returns `None`, and
# `lead_in_violation` is set via the "no in-span command matches..."
# path -- the ONLY branch that never also builds a LEAD_IN hold window,
# so this cannot co-fire the hold-drift gate the way a "matched late"
# construction, like the single-waypoint fixture above, provably does
# (that fixture's own LEAD_IN hold window always contains the leading
# strays, so `any_hold_target_drift` is True there too -- see the
# handoff for the worked-out trace). The offset is far below both the
# 1-float32-ULP N3 bound and the 0.05 deg C1' bound for every joint's
# own FIRST non-carry sample of PLACE_ROUTE's own first goto (HOME ->
# GRIP_SHUT, every right-arm joint moves in the qualifying direction
# for this offset's sign -- verified below, not assumed) -- so C1/N3
# never fires either. The flight leg is flown with ZERO offset (its own
# first goto, REST -> LIFT_TO_PRESENT's first waypoint, has at least
# one joint moving the OTHER direction, which this same offset WOULD
# trip -- so flight is simply kept clean and contributes nothing, never
# hidden).

CYCLE_K2M1 = "k2m1-setup-lead-in"
#: Per-joint offset (rad): commanded target minus realised state, held
#: constant for every tick of the setup leg's own flight. ~8-16 float32
#: ULPs near these joints' own magnitudes (never rounds away to an exact
#: match), and ~0.00006 deg -- far under both the 1-ULP N3 bound and the
#: 0.05 deg C1' bound.
_K2M1_OFFSET = 1e-6


def _write_k2m1_sidecar(control_dir, kind, first_seq, last_seq):
    doc = {"alignment": [{"server_seq": first_seq}, {"server_seq": last_seq}]}
    path = control_dir / f"{CYCLE_K2M1}-{kind}.link.json"
    path.write_text(json.dumps(doc))
    return path


def _build_k2m1_cycle(tmp_path):
    """A real two-leg B cycle (PLACE_ROUTE then LIFT_TO_PRESENT, both
    flown by `make_fixtures.FlightSim` exactly as
    test_goalfix_cmp_t4_end_to_end.py's own `_build_cycle` does) where
    the setup leg's own commanded-vs-realised offset is held at
    `_K2M1_OFFSET` (never 0, and never this module's usual zeroed lag)
    for every tick of PLACE_ROUTE specifically, then reset to exactly
    0 before the settle hold and the flight leg -- so ONLY the setup
    leg's own turn_on match ever fails, and the flight leg is completely
    clean (matched turn_on, no lead-in violation, no pathcheck failure).
    """
    ev_dir = tmp_path / "ev"
    control_dir = tmp_path / "control"
    control_dir.mkdir()
    home = dict(R.HOME)
    with mock.patch.object(mf, "_LAG_RAD", [_K2M1_OFFSET] * 8):
        sim = mf.FlightSim(home, pose_units="deg", restream_passes=1)
        sim.fly(R.PLACE_ROUTE)
    setup_last_seq = sim.state_rows[-1]["seq"]
    setup_first_seq = 0
    with mock.patch.object(mf, "_LAG_RAD", [0.0] * 8):
        for _ in range(25):  # 0.5s settle, present position continuously reported
            sim._hold_ticks(1, dict(sim.pose))
        flight_first_seq = sim.state_rows[-1]["seq"]
        sim.pose = dict(R.REST)
        sim.fly(R.LIFT_TO_PRESENT)
    flight_last_seq = sim.state_rows[-1]["seq"]

    result = sim.result()
    mf.write_evidence(ev_dir, result.state_rows, result.command_rows)
    _write_k2m1_sidecar(control_dir, "setup", setup_first_seq, setup_last_seq)
    _write_k2m1_sidecar(control_dir, "flight", flight_first_seq, flight_last_seq)
    return ev_dir, control_dir


class TestB3IsolatedThroughCycleVerdict:
    """K2-M1: the B verdict itself, not `lr.lead_in_violation` alone,
    and not just a reason-string substring."""

    def test_lead_in_alone_stops_the_b_verdict_at_rc_2(self, tmp_path):
        ev_dir, control_dir = _build_k2m1_cycle(tmp_path)
        evd = ev.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        setup_leg = cyc.resolve_leg(evd, control_dir / f"{CYCLE_K2M1}-setup.link.json",
                                     "setup", R.PLACE_ROUTE, R.CRITICAL_JOINTS)
        flight_leg = cyc.resolve_leg(evd, control_dir / f"{CYCLE_K2M1}-flight.link.json",
                                      "flight", R.LIFT_TO_PRESENT, R._PRESENT_GUARD)
        setup_lr = cyc.evaluate_leg(evd, setup_leg)
        flight_lr = cyc.evaluate_leg(evd, flight_leg)

        # The isolating claim, at LEG level for both legs first (same
        # style as TestCarryBeforeCorrectlyMatchedTurnOn above): every
        # C0-C8 check passes on both legs, no hold-drift, no
        # unassigned-non-carry -- ONLY the setup leg's lead-in gate
        # fails, and it fails via the "no match found at all" path
        # (never `edge_window_stats`, so no LEAD_IN hold window is even
        # built for it).
        for lr in (setup_lr, flight_lr):
            failing = [cid for cid in pc.ALL_CHECKS if not lr.pathcheck[cid].passed]
            assert failing == [], {cid: lr.pathcheck[cid].detail for cid in failing}
            assert not lr.unassigned_non_carry_violation, lr.unassigned_non_carry_detail
            for hs in lr.hold_stats:
                if hs.window.goal_index == holds.LEAD_IN_GOAL_INDEX:
                    assert not hs.window_command_indices, hs.window_command_indices
                else:
                    assert not any(v != 0.0 for v in hs.target_drift.values()), hs.target_drift

        assert setup_lr.lead_in_violation is True
        assert setup_lr.lead_in_detail == (
            "no in-span command matches the turn_on present-position property")
        assert flight_lr.lead_in_violation is False

        # The cycle-verdict level (K2-M1's own target): with
        # `skip_gates=True` (no provenance/compliance/start_variant
        # supplied, exactly as this construction has none), the ONLY
        # thing that can make this a B STOP is `any_lead_in_violation`.
        cv, _ = cyc.evaluate_cycle(CYCLE_K2M1, "B", evd, [setup_leg, flight_leg],
                                    skip_gates=True)
        assert cv.segment_indeterminate is False, cv.reasons
        assert cv.genuine_echo_count == 0
        assert cv.reasons == [
            "setup: lead-in violation: no in-span command matches the "
            "turn_on present-position property"], cv.reasons
        assert cv.verdict == cyc.VERDICT_STOP, cv.reasons
        assert cv.rc() == 2

    def test_setup_first_goto_direction_makes_the_offset_safe(self, tmp_path):
        """Confirms the construction's own premise (not assumed): for
        every right-arm joint that PLACE_ROUTE's first waypoint moves
        (HOME -> GRIP_SHUT), the direction is such that subtracting
        `_K2M1_OFFSET` from the commanded target never reads as
        "behind S" (N3) -- this is what keeps setup's own goto 0 free of
        a C1 failure despite the offset applied on every one of its
        commands."""
        home = dict(R.HOME)
        goal = dict(home)
        goal.update(R.PLACE_ROUTE[0].pose)
        for j in mf.R_JOINTS:
            a, b = home.get(j, 0.0), goal.get(j, 0.0)
            if a == b:
                continue
            # `_LAG_RAD` is subtracted from the commanded (degree) value
            # before conversion; degrees is the route's own native unit,
            # so comparing signs here is unit-independent.
            assert b > a, (j, a, b)
