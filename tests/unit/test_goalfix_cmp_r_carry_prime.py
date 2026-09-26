"""R-carry' provenance acceptance tests (D-2; decision report
``outputs/decision-2026-09-26-pr144-option-c-window-proof.md`` §3.4/§6;
assignment ``outputs/assignment-2026-09-26-sonnet-pr144-wblk-window.md``
Part R, §R2).

R-AC1 is library-level (assignment: "All are CLI tests except R-AC1").

R-AC2/R-AC3 are NOT run through ``cyc._cli``'s own argument parsing, a
disclosed deviation: ``cycle._cli`` hardcodes ``R.PLACE_ROUTE`` for the
setup leg, and embedding R-AC2's exact numeric scenario (a genuine echo
whose OWN command stays under the 30ms C2 tolerance, while a later
bit-exact repeat of it would not) inside PLACE_ROUTE's own real joint
geometry runs into a real mathematical wall: echo.py's near-end exemption
and C2's own tau-spread check share the SAME 30ms tolerance and the SAME
``implied_tau`` machinery, so on any REAL PLACE_ROUTE goto, escaping the
comparison joint's own near-end zone (set by its real range) and keeping
the injection-tick spread under 30ms turn out to be mutually exclusive at
that goto's own real ``seconds`` duration (checked against every
multi-joint goto in PLACE_ROUTE). Both tests instead call
``cycle.evaluate_cycle`` directly -- the exact same shipped call site the
CLI itself invokes -- on a small, hand-built synthetic route, which the
assignment's own precedent (``test_goalfix_cmp_f6_surviving_mutants.py``'s
``TestMutantQIsolatedProof``, "ISOLATION PROOF (library level,
`evaluate_cycle` directly...)") already treats as an acceptable level for
isolating one specific mechanism. R-AC4 goes through the real CLI (it
reuses the existing, already-real, venv-gated Stage A slice).
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
from tools.goalfix_cmp import echo  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import pathcheck as pc  # noqa: E402
from tools.goalfix_cmp import segments as seg  # noqa: E402


def full_pose(**kw) -> dict:
    base = {j: 0.0 for j in mf.R_JOINTS}
    base.update(kw)
    return base


class TestRAC1LabelsAndCounts:
    """R-AC1: a mid-goto injected genuine echo followed by one bit-equal
    repeat. Labels GENUINE_ECHO then ECHO_CARRY; echo_carry == 1;
    genuine_echo unchanged (== 1, not 2 -- the repeat is not double
    counted as its own echo)."""

    def _build(self, tmp_path):
        start = full_pose()
        # A single-joint goto, r_shoulder_pitch 0.0 -> -1.0 rad, so a
        # value outside that range can never be produced by the goto's
        # own minimum-jerk curve at any tau (echo.py's on-path inversion
        # clips to the [0, -1.0] endpoint, so it can never round-trip) --
        # a clean, tau-independent way to force GENUINE_ECHO regardless
        # of timing.
        route = [mf.Waypoint("G", full_pose(r_shoulder_pitch=-1.0), 2.0)]
        targets = [
            full_pose(r_shoulder_pitch=0.0),     # t0: anchor
            full_pose(r_shoulder_pitch=-0.1),    # t1: fresh, on-path
            full_pose(r_shoulder_pitch=0.5),     # t2: the future echo SOURCE -- 0.5 is outside
                                                  #     [0, -1.0] (this goto's own range), so it can
                                                  #     never round-trip through implied_tau/pose_at
                                                  #     -- off-path regardless of tau agreement
            full_pose(r_shoulder_pitch=-0.3),    # t3: fresh, on-path
            full_pose(r_shoulder_pitch=0.5),     # t4: GENUINE ECHO -- exact repeat of t2's own
                                                  #     recorded STATE
            full_pose(r_shoulder_pitch=0.5),     # t5: ECHO_CARRY -- bit-exact repeat of t4
            full_pose(r_shoulder_pitch=-0.4),    # t6: fresh again, breaks the chain
        ]
        rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                              wall_time_ns=1, position_rad21=mf.full21(start))]
        cmds = []
        for i, tgt in enumerate(targets):
            cmds.append(mf.command_row_joint(seq=i, target_rad21=mf.full21(tgt)))
            rows.append(mf.state_row(seq=i + 1, sim_step=i + 1, sim_time_s=(i + 1) * 0.02,
                                      cmd_seq=i, wall_time_ns=2 + i,
                                      position_rad21=mf.full21(tgt)))
        mf.write_evidence(tmp_path, rows, cmds)
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        return evd, route, start

    def test_genuine_echo_then_echo_carry(self, tmp_path):
        evd, route, start = self._build(tmp_path)
        ctx = echo.GotoContext(start8=start, goal8=route[0].pose, seconds=route[0].seconds)
        goto_context = [ctx] * len(evd.commands)
        results = echo.classify_commands(evd, goto_context, leg_turn_on_state_index={0: 0})

        assert results[4].joints["r_shoulder_pitch"].label == echo.GENUINE_ECHO
        assert results[5].joints["r_shoulder_pitch"].label == echo.ECHO_CARRY

        counts = echo.count_labels(results)
        assert counts.genuine_echo == 1
        assert counts.echo_carry == 1


class TestRAC2ProvenanceRecheckStopsB:
    """R-AC2 (B): a genuine echo (r_shoulder_pitch, out-of-range value,
    implied tau pinned to 0 by C2's own clipping inversion) whose OWN
    command's tau spread against a second, normally-moving joint
    (r_elbow_pitch) stays under 30ms; one tick later, a bit-exact repeat
    of the SAME echo value -- excluded from tau as an ordinary carry on
    the first pass -- would, with its exemption withdrawn (its chain
    origin is genuine_echo), see r_elbow_pitch's own tau have advanced
    far enough to exceed 30ms. First pass: C2 passes. With Part R (the
    provenance recheck baked into evaluate_cycle): C2 fails with an
    ``R-carry'`` reason, verdict STOP."""

    def _build(self, tmp_path):
        joints = list(mf.R_JOINTS)
        start = full_pose()
        goal = full_pose(r_shoulder_pitch=-1.0, r_elbow_pitch=-1.0)
        route = [mf.Waypoint("G", goal, 3.0)]

        from tools.goalfix_cmp._minjerk import pose_at
        dt = 0.1
        targets = [
            full_pose(),  # t0: anchor
            full_pose(r_shoulder_pitch=1.0,
                      r_elbow_pitch=pose_at(0.0, -1.0, 0.02)),   # t1: injection (tau_A pinned to 0)
            full_pose(r_shoulder_pitch=1.0,
                      r_elbow_pitch=pose_at(0.0, -1.0, 0.15)),   # t2: repeat of t1's shoulder value
            full_pose(r_shoulder_pitch=pose_at(0.0, -1.0, 0.30),
                      r_elbow_pitch=pose_at(0.0, -1.0, 0.30)),   # t3: fresh again
        ]
        # t0's own reported STATE plants the echo source: r_shoulder_pitch
        # = 1.0 (a stale value from before this goto -- never actually
        # reachable on the CURRENT [0, -1.0] path, so it is off-path
        # regardless of tau agreement).
        state0 = full_pose(r_shoulder_pitch=1.0)

        rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                              wall_time_ns=1, position_rad21=mf.full21(state0))]
        cmds = []
        for i, tgt in enumerate(targets):
            cmds.append(mf.command_row_joint(seq=i, target_rad21=mf.full21(tgt)))
            rows.append(mf.state_row(seq=i + 1, sim_step=i + 1, sim_time_s=(i + 1) * dt,
                                      cmd_seq=i, wall_time_ns=2 + i,
                                      position_rad21=mf.full21(tgt)))
        mf.write_evidence(tmp_path, rows, cmds)
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        leg = cyc.LegSpec("setup", route_rad=route, guard=(), start_pose8=start,
                           command_indices=list(range(len(cmds))))
        return evd, leg

    def test_first_pass_c2_passes(self, tmp_path):
        """Pinned as the b59404c-equivalent (pre-D-2) behaviour: with no
        provenance recheck at all, this exact input's C2 check passes --
        checked directly against pathcheck.check_c2_c3's own default
        (``carry_withdraw=None``, unchanged by D-2)."""
        evd, leg = self._build(tmp_path)
        targets8 = [{name: float(v) for name, v in zip(mf.R_JOINTS, evd.commands.target_rad[i, :8])}
                    for i in leg.command_indices]
        assignment = seg.assign_goals(leg.route_rad, leg.start_pose8, targets8)
        t_hi_s = [evd.brackets[i].t_hi for i in leg.command_indices]
        c2, _c3 = pc.check_c2_c3(targets8, t_hi_s, leg.start_pose8, leg.route_rad, assignment)
        assert c2.passed, c2.detail

    def test_provenance_recheck_stops_b(self, tmp_path):
        evd, leg = self._build(tmp_path)
        cv, leg_results = cyc.evaluate_cycle("rac2", "B", evd, [leg], skip_gates=True)
        assert cv.verdict == cyc.VERDICT_STOP, cv.as_dict()
        c2_result = leg_results["setup"].pathcheck["C2"]
        assert not c2_result.passed
        assert c2_result.detail.startswith("R-carry′:"), c2_result.detail
        assert any(r.startswith("setup:C2") for r in cv.reasons), cv.reasons


class TestRAC3FreshOriginRepeatStaysExcluded:
    """R-AC3: a fresh-origin, depth-1, mid-goto repeat -- no genuine echo
    involved at all -- stays excluded from tau exactly as before (its
    chain origin is FRESH, never withdrawn). The frozen r3 G-a boundary
    test (B16, test_goalfix_cmp_r_carry.py) is unaffected by D-2 (its own
    file is untouched; confirmed green in the full focused-suite run --
    carry_mask's default behaviour, withdraw=None, is exactly what B16
    exercises)."""

    def _build(self, tmp_path):
        start = full_pose()
        goal = full_pose(r_shoulder_pitch=-1.0)
        route = [mf.Waypoint("G", goal, 2.0)]
        # A mid-goto value that has no past-state match anywhere (never
        # planted as an echo source) repeats once, bit-exact, purely by
        # construction -- its own chain origin is FRESH.
        targets = [
            full_pose(),                          # t0: anchor
            full_pose(r_shoulder_pitch=-0.4),      # t1: fresh
            full_pose(r_shoulder_pitch=-0.4),      # t2: repeat of t1 -- fresh-origin carry
            full_pose(r_shoulder_pitch=-0.5),      # t3: fresh, breaks the chain
        ]
        rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                              wall_time_ns=1, position_rad21=mf.full21(start))]
        cmds = []
        for i, tgt in enumerate(targets):
            cmds.append(mf.command_row_joint(seq=i, target_rad21=mf.full21(tgt)))
            rows.append(mf.state_row(seq=i + 1, sim_step=i + 1, sim_time_s=(i + 1) * 0.02,
                                      cmd_seq=i, wall_time_ns=2 + i,
                                      position_rad21=mf.full21(tgt)))
        mf.write_evidence(tmp_path, rows, cmds)
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        leg = cyc.LegSpec("setup", route_rad=route, guard=(), start_pose8=start,
                           command_indices=list(range(len(cmds))))
        return evd, leg

    def test_fresh_origin_repeat_is_carry_not_echo_carry(self, tmp_path):
        evd, leg = self._build(tmp_path)
        ctx = echo.GotoContext(start8=leg.start_pose8, goal8=leg.route_rad[0].pose,
                                seconds=leg.route_rad[0].seconds)
        goto_context = [ctx] * len(evd.commands)
        results = echo.classify_commands(evd, goto_context, leg_turn_on_state_index={0: 0})
        assert results[2].joints["r_shoulder_pitch"].label == echo.CARRY
        counts = echo.count_labels(results)
        assert counts.echo_carry == 0
        assert counts.genuine_echo == 0

    def test_c2_unaffected_through_evaluate_cycle(self, tmp_path):
        evd, leg = self._build(tmp_path)
        cv, leg_results = cyc.evaluate_cycle("rac3", "B", evd, [leg], skip_gates=True)
        assert leg_results["setup"].pathcheck["C2"].passed
        assert cv.echo_carry_by_leg["setup"] == 0
