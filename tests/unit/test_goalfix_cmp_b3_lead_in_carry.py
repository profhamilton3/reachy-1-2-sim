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
import os
import sys

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
