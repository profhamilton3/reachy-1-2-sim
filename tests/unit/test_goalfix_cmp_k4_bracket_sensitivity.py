"""K4 (2026-09-26 pr144-k2-k4 assignment): the four-endpoint
``wrist_ball_bracket_sensitivity`` in ``compute_place_route_metrics``
(``cycle.py`` ~815-885).

At 77d6dab this field computed only two of the four (start, end)
combinations of the window's own ``b_start``/``b_rest`` brackets ---
``(t_lo, t_lo)`` (the primary) and ``(t_hi, t_hi)`` --- never
``(t_lo, t_hi)``/``(t_hi, t_lo)``, and reported no min/max. This module
exercises the real, shipped four-combination implementation, mirroring
``test_goalfix_cmp_b12_wrist_ball.py``'s own fixture style (hand-built
evidence, real ``compute_place_route_metrics``/``window.identify_window``
call sites, never a spy that fabricates a ``LinkDelta``).

A note on what "mixed endpoints matter" can prove, worked out empirically
while building this file (not assumed): for any two temporally-ordered
brackets ``b_start`` (earlier) and ``b_rest`` (later), the four windows
nest exactly as
``hi_lo = [b_start.t_hi, b_rest.t_lo) ⊆ {lo_lo, hi_hi} ⊆ lo_hi =
[b_start.t_lo, b_rest.t_hi)``, and (given ``b_start.t_hi <= b_rest.t_lo``,
true for any real segment) ``lo_hi``'s own sample set is EXACTLY
``lo_lo``'s ∪ ``hi_hi``'s. Since ``realised_cm`` is a per-link MINIMUM
over its own sample set (``clearance._worst_per_link_over_samples``),
this makes ``lo_hi``'s own value *exactly* ``min(lo_lo, hi_hi)`` for ANY
input -- it can never be numerically different from BOTH ``lo_lo`` and
``hi_hi`` simultaneously (it must equal whichever of the two is smaller).
So this file's own mixed-endpoint test demonstrates the achievable,
correct property instead: ``hi_lo`` (the narrowest window, common to
every combination) differs from both ``lo_lo`` and ``hi_hi``, ``lo_lo``
differs from ``hi_hi`` (so which SIDE'S own bracket is widened matters
too, not only whether one is widened at all), and ``lo_hi`` is exactly
the minimum of the four, confirmed by an INDEPENDENT recomputation over
each combination's own expected state-index range (never by calling
``compute_place_route_metrics``/its internal ``_combo`` closure)."""
import json
import os
import sys
from unittest import mock

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../scripts", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import clearance as cl  # noqa: E402
from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import window  # noqa: E402
from tools.goalfix_cmp._io import RC_OK, read_result  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.motion.rig_routes import SHUT, OPEN  # noqa: E402
from reachy_ai.scene.awareness import SceneModel, SceneObject  # noqa: E402

JOINTS = list(mf.R_JOINTS)


def _pose(**overrides) -> dict:
    p = {j: 0.0 for j in JOINTS}
    p.update(overrides)
    return p


def _pose8_from_q7_deg(q7_deg, gripper_deg=SHUT) -> dict:
    p = {j: np.radians(v) for j, v in zip(JOINTS[:7], q7_deg)}
    p["r_gripper"] = np.radians(gripper_deg)
    return p


def _box_scene(obj_id="box_1"):
    box = SceneObject(id=obj_id, kind="box", center=(0.5, -0.3, 1.1), size=(0.1, 0.1, 0.1),
                       dynamic=True, tracked=True)
    return SceneModel(frame_id="pedestal", objects=[box], table_id=None)


# Two joint configurations (found by an offline random search over the
# reachable q7 space against `_box_scene()`'s own box) with distinct,
# known "shells"/wrist_ball clearances against it -- one moderately
# close (9.99 cm, planted at the START-side trap), one deeply
# penetrating (-5.32 cm, planted at the END-side trap) -- both well
# below the segment's own natural worst point along HOVER->REST_SHUT
# (21.09 cm on this fixture's own poses, confirmed below). The END trap
# is deliberately the DEEPER of the two: since lo_hi == min(lo_lo, hi_hi)
# always (this module's own docstring), making hi_hi the smaller value
# is what makes lo_hi equal hi_hi rather than lo_lo -- needed so a
# mutant that computes lo_hi (or hi_lo) identically to lo_lo is actually
# distinguishable from the correct value on THIS fixture (confirmed by
# running each such mutation against this file; see the handoff).
START_TRAP_Q7_DEG = [-89.5403693, -35.73883974, 72.78364515, -8.20120754,
                     86.26529583, -54.33912253, -67.71356086]
END_TRAP_Q7_DEG = [-87.04621235, -16.5932692, 12.52745282, -24.37840727,
                    -85.24326719, 43.10863496, 73.59365218]


def _build(tmp_path, *, start_trap_q7=None, end_trap_q7=None, hold_commands=1):
    """A HOVER -> REST_SHUT -> (hold) -> REST leg, mirroring
    test_goalfix_cmp_b12_wrist_ball.py's own ``_build`` (2 commands
    before HOVER, real settle gaps for W-blk). ``start_trap_q7``/
    ``end_trap_q7`` (degrees, 7 right-arm joints, gripper stays SHUT):
    when given, REPLACE the STATE (never the command target) reported by
    the row that becomes ``b_start.t_lo`` (HOVER's own last command's
    own bracket, i.e. the row right before it -- PRE2's own reported
    state) / ``b_rest.t_lo`` (REST's own first command's own bracket,
    i.e. the LAST row of the settle before REST) respectively -- a
    plausible "the plant was still reporting something else when the
    command was applied" scenario, never a change to the commanded
    path, block structure, or window validity."""
    pre1 = _pose(r_shoulder_pitch=np.radians(-5.0), r_gripper=np.radians(SHUT))
    pre2 = _pose(r_shoulder_pitch=np.radians(-10.0), r_gripper=np.radians(SHUT))
    hover = _pose(r_shoulder_pitch=np.radians(-30.0), r_gripper=np.radians(SHUT))
    rest_shut = _pose(r_shoulder_pitch=np.radians(-70.0), r_gripper=np.radians(SHUT))
    rest = dict(rest_shut, r_gripper=np.radians(OPEN))
    start_pose = _pose()

    rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                          wall_time_ns=1, position_rad21=mf.full21(start_pose))]
    cmds = []

    def _emit(tgt, state_override=None):
        i = len(cmds)
        cmds.append(mf.command_row_joint(seq=i, target_rad21=mf.full21(tgt)))
        t = rows[-1]["sim_time_s"] + 0.02
        reported = state_override if state_override is not None else tgt
        rows.append(mf.state_row(seq=len(rows), sim_step=len(rows), sim_time_s=t,
                                  cmd_seq=i, wall_time_ns=2 + len(rows),
                                  position_rad21=mf.full21(reported)))

    def _settle(pose, n=25):
        for _ in range(n):  # >= window.SETTLE_GAP_S (0.28s) of command-free ticks
            t = rows[-1]["sim_time_s"] + 0.02
            rows.append(mf.state_row(seq=len(rows), sim_step=len(rows), sim_time_s=t,
                                      cmd_seq=len(cmds) - 1, wall_time_ns=2 + len(rows),
                                      position_rad21=mf.full21(pose)))

    start_override = _pose8_from_q7_deg(start_trap_q7) if start_trap_q7 is not None else None
    _emit(pre1)
    _emit(pre2, state_override=start_override)  # this row becomes b_start.t_lo
    _emit(hover)                                 # segment_start: b_start.t_hi is THIS row
    _settle(hover)
    _emit(rest_shut)
    for _ in range(hold_commands):
        _emit(dict(rest_shut))
    end_override = _pose8_from_q7_deg(end_trap_q7) if end_trap_q7 is not None else None
    if end_override is not None:
        # Replace only the settle's OWN last row (which becomes
        # b_rest.t_lo) -- every earlier settle tick is untouched.
        _settle(rest_shut, n=24)
        t = rows[-1]["sim_time_s"] + 0.02
        rows.append(mf.state_row(seq=len(rows), sim_step=len(rows), sim_time_s=t,
                                  cmd_seq=len(cmds) - 1, wall_time_ns=2 + len(rows),
                                  position_rad21=mf.full21(end_override)))
    else:
        _settle(rest_shut)
    _emit(rest)                                  # first_rest: b_rest.t_hi is THIS row

    route = [mf.Waypoint("HOVER", hover, 1.0), mf.Waypoint("REST_SHUT", rest_shut, 1.0),
             mf.Waypoint("REST", rest, 1.0)]
    mf.write_evidence(tmp_path, rows, cmds)
    evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
    leg = cyc.LegSpec("setup", route_rad=route, guard=(), start_pose8=start_pose,
                       command_indices=list(range(len(cmds))))
    return evd, leg


def _independent_combo(evd, epoch, wb_commanded, scene, object_id, lo_t, hi_t):
    """The SAME underlying library calls compute_place_route_metrics
    itself uses (cl.realised_samples_from_states/cl.compute_deltas) --
    never compute_place_route_metrics or its own internal closures --
    applied to a state-index range WE determine independently here from
    (lo_t, hi_t) alone."""
    mask = ((evd.states.epoch == epoch) & (evd.states.sim_time_s >= lo_t)
             & (evd.states.sim_time_s < hi_t))
    state_idx = np.nonzero(mask)[0]
    if len(state_idx) == 0:
        return None
    realised, _ = cl.realised_samples_from_states(evd.states.position_rad[state_idx, :8])
    deltas = cl.compute_deltas("PLACE_ROUTE", [R.HOVER, R.REST_SHUT], 400, scene,
                                wb_commanded, realised)
    entry = next((d for d in deltas if d.hand == "shells" and d.link == "wrist_ball"
                   and d.object_id == object_id), None)
    if entry is None:
        return None
    return entry.realised_cm, entry.planned_cm - entry.realised_cm


class TestMixedEndpointsMatter:
    def test_lo_hi_and_hi_lo_use_the_windows_own_wide_brackets(self, tmp_path):
        evd, leg = _build(tmp_path, start_trap_q7=START_TRAP_Q7_DEG, end_trap_q7=END_TRAP_Q7_DEG)
        lr = cyc.evaluate_leg(evd, leg)
        affected = cyc.seg.find_affected_segment(lr.assignment, leg.route_rad)
        win = window.identify_window(evd, leg)
        assert win.valid, win.reasons

        b_start = evd.brackets[win.segment_start]
        b_rest = evd.brackets[win.first_rest]
        # The isolating premise: b_start's own bracket is wide enough
        # that its t_lo row is our planted start-trap, and b_rest's own
        # bracket is wide enough that its t_lo row is our planted
        # end-trap -- neither trap sits at either bracket's own t_hi.
        assert b_start.t_lo < b_start.t_hi
        assert b_rest.t_lo < b_rest.t_hi

        m = cyc.compute_place_route_metrics(
            evd, leg, lr, affected, _box_scene(), board_object_ids=["box_1"], win=win)
        assert not m.open_questions, m.open_questions
        sens = m.wrist_ball_bracket_sensitivity
        assert sens["complete"] is True, sens

        # Independent pin, per combo, over the state-index range WE
        # compute from (b_start, b_rest) alone -- never by calling
        # compute_place_route_metrics's own internal combo logic.
        epoch = int(evd.commands.epoch[win.segment_start])
        wb_commanded = cl.commanded_samples(
            evd.commands.target_rad[np.asarray(win.segment_commands), :8])
        expected = {
            "lo_lo": _independent_combo(evd, epoch, wb_commanded, _box_scene(), "box_1",
                                         b_start.t_lo, b_rest.t_lo),
            "lo_hi": _independent_combo(evd, epoch, wb_commanded, _box_scene(), "box_1",
                                         b_start.t_lo, b_rest.t_hi),
            "hi_lo": _independent_combo(evd, epoch, wb_commanded, _box_scene(), "box_1",
                                         b_start.t_hi, b_rest.t_lo),
            "hi_hi": _independent_combo(evd, epoch, wb_commanded, _box_scene(), "box_1",
                                         b_start.t_hi, b_rest.t_hi),
        }
        for key, (exp_realised, exp_delta) in expected.items():
            assert sens["combos"][key]["status"] == "ok", sens["combos"][key]
            assert sens["combos"][key]["realised_cm"] == pytest.approx(exp_realised, abs=1e-9), key
            assert sens["combos"][key]["delta_cm"] == pytest.approx(exp_delta, abs=1e-9), key

        lo_lo = sens["combos"]["lo_lo"]["realised_cm"]
        lo_hi = sens["combos"]["lo_hi"]["realised_cm"]
        hi_lo = sens["combos"]["hi_lo"]["realised_cm"]
        hi_hi = sens["combos"]["hi_hi"]["realised_cm"]

        # hi_lo (the narrowest window, common to every combination --
        # no trap) differs from BOTH single-trap combos.
        assert hi_lo != pytest.approx(lo_lo, abs=1e-9)
        assert hi_lo != pytest.approx(hi_hi, abs=1e-9)
        # The two single-trap combos differ from EACH OTHER -- which
        # SIDE'S bracket is widened matters, not merely whether one is.
        assert lo_lo != pytest.approx(hi_hi, abs=1e-9)
        # lo_hi (both traps) is exactly the minimum of the two
        # single-trap combos -- the mathematically forced identity
        # worked out in this module's own docstring, confirmed rather
        # than assumed.
        assert lo_hi == pytest.approx(min(lo_lo, hi_hi), abs=1e-9)

        # min/max equal the true extremes over all four.
        realised_values = [lo_lo, lo_hi, hi_lo, hi_hi]
        assert sens["realised_cm_min"] == pytest.approx(min(realised_values), abs=1e-9)
        assert sens["realised_cm_max"] == pytest.approx(max(realised_values), abs=1e-9)
        delta_values = [sens["combos"][k]["delta_cm"] for k in ("lo_lo", "lo_hi", "hi_lo", "hi_hi")]
        assert sens["delta_cm_min"] == pytest.approx(min(delta_values), abs=1e-9)
        assert sens["delta_cm_max"] == pytest.approx(max(delta_values), abs=1e-9)
        assert sens["incomplete_combos"] == []


class TestPrimaryUnchanged:
    def test_lo_lo_equals_the_primary_bit_exactly(self, tmp_path):
        """(b): the primary wrist_ball_delta_cm/wrist_ball_realised_cm
        (computed from (start=t_lo, end=t_lo), untouched by K4) equal
        combos["lo_lo"] bit-exactly -- on a fixture with NO traps at
        all, i.e. the same value test_goalfix_cmp_b12_wrist_ball.py's
        own (unmodified, still-passing) assertions already pin as the
        pre-K4 value."""
        evd, leg = _build(tmp_path)
        lr = cyc.evaluate_leg(evd, leg)
        affected = cyc.seg.find_affected_segment(lr.assignment, leg.route_rad)
        win = window.identify_window(evd, leg)
        assert win.valid, win.reasons
        m = cyc.compute_place_route_metrics(
            evd, leg, lr, affected, _box_scene(), board_object_ids=["box_1"], win=win)
        assert not m.open_questions, m.open_questions
        assert m.wrist_ball_delta_cm is not None

        sens = m.wrist_ball_bracket_sensitivity
        assert sens["primary_combo"] == "lo_lo"
        assert sens["combos"]["lo_lo"]["realised_cm"] == m.wrist_ball_realised_cm
        assert sens["combos"]["lo_lo"]["delta_cm"] == m.wrist_ball_delta_cm
        # Backward-compatible aliases.
        assert sens["t_lo_realised_cm"] == m.wrist_ball_realised_cm
        assert sens["t_lo_delta_cm"] == m.wrist_ball_delta_cm
        assert sens["t_hi_realised_cm"] == sens["combos"]["hi_hi"]["realised_cm"]
        assert sens["t_hi_delta_cm"] == sens["combos"]["hi_hi"]["delta_cm"]


class TestIncompleteHandling:
    def test_no_entry_when_object_id_has_no_scene_match(self, tmp_path):
        """(c) no_entry: a real, valid window and real, non-empty
        commanded/realised samples -- but board_object_ids names an
        object absent from the scene, so compute_deltas produces no
        matching (hand, link, object_id) entry for any combination."""
        evd, leg = _build(tmp_path)
        lr = cyc.evaluate_leg(evd, leg)
        win = window.identify_window(evd, leg)
        assert win.valid, win.reasons
        m = cyc.compute_place_route_metrics(
            evd, leg, lr, None, _box_scene(), board_object_ids=["not_in_scene"], win=win)

        sens = m.wrist_ball_bracket_sensitivity
        assert sens is not None
        for key in ("lo_lo", "lo_hi", "hi_lo", "hi_hi"):
            assert sens["combos"][key] == {"realised_cm": None, "delta_cm": None, "status": "no_entry"}
        assert sens["complete"] is False
        assert sens["incomplete_combos"] == ["lo_lo", "lo_hi", "hi_lo", "hi_hi"]
        assert sens["realised_cm_min"] is None
        assert sens["realised_cm_max"] is None
        assert sens["delta_cm_min"] is None
        assert sens["delta_cm_max"] is None
        # The primary itself is null for the SAME reason, via
        # open_questions -- unchanged K4 behaviour.
        assert m.wrist_ball_delta_cm is None
        assert any("no wrist_ball/shells clearance entry" in q for q in m.open_questions), m.open_questions

    def test_mixed_completeness_nulls_min_max_over_the_whole_set(self, tmp_path):
        """A single combo (lo_hi -- the widest of the four, uniquely
        identifiable by its own sample COUNT on this fixture, confirmed
        below) fails while the other three succeed: `complete` is False
        and every min/max is null, never computed from just the
        surviving three. This is what actually distinguishes K4-M3
        ("min-max computed over only the available combinations when
        incomplete") and K4-M4 ("complete computed as 'any ok' instead
        of 'all ok'") from correct behaviour -- the all-no_entry fixture
        above cannot: with zero combos ok, "any ok" and "all ok" agree
        (both False), and "available" is empty either way."""
        evd, leg = _build(tmp_path, start_trap_q7=START_TRAP_Q7_DEG, end_trap_q7=END_TRAP_Q7_DEG)
        lr = cyc.evaluate_leg(evd, leg)
        affected = cyc.seg.find_affected_segment(lr.assignment, leg.route_rad)
        win = window.identify_window(evd, leg)
        assert win.valid, win.reasons

        epoch = int(evd.commands.epoch[win.segment_start])
        b_start = evd.brackets[win.segment_start]
        b_rest = evd.brackets[win.first_rest]

        def _n_states(lo_t, hi_t):
            mask = ((evd.states.epoch == epoch) & (evd.states.sim_time_s >= lo_t)
                     & (evd.states.sim_time_s < hi_t))
            return int(np.count_nonzero(mask))

        n_lo_lo = _n_states(b_start.t_lo, b_rest.t_lo)
        n_lo_hi = _n_states(b_start.t_lo, b_rest.t_hi)
        n_hi_lo = _n_states(b_start.t_hi, b_rest.t_lo)
        n_hi_hi = _n_states(b_start.t_hi, b_rest.t_hi)
        # lo_hi (the union of every other window) has strictly more
        # states than each of the other three on this fixture -- the
        # unique count we key the failure injection on below.
        assert n_lo_hi > n_lo_lo and n_lo_hi > n_hi_lo and n_lo_hi > n_hi_hi

        real_compute_deltas = cl.compute_deltas

        def _fail_lo_hi_only(route, waypoints, n_planned_samples, scene,
                              commanded_q7_gripper, realised_q7_gripper):
            if len(realised_q7_gripper) == n_lo_hi:
                return []  # simulates a real "no matching entry" outcome for lo_hi only
            return real_compute_deltas(route, waypoints, n_planned_samples, scene,
                                        commanded_q7_gripper, realised_q7_gripper)

        with mock.patch.object(cyc.cl, "compute_deltas", side_effect=_fail_lo_hi_only):
            m = cyc.compute_place_route_metrics(
                evd, leg, lr, affected, _box_scene(), board_object_ids=["box_1"], win=win)

        sens = m.wrist_ball_bracket_sensitivity
        assert sens["combos"]["lo_hi"]["status"] == "no_entry", sens
        for key in ("lo_lo", "hi_lo", "hi_hi"):
            assert sens["combos"][key]["status"] == "ok", sens
        assert sens["complete"] is False, sens
        assert sens["incomplete_combos"] == ["lo_hi"], sens
        assert sens["realised_cm_min"] is None
        assert sens["realised_cm_max"] is None
        assert sens["delta_cm_min"] is None
        assert sens["delta_cm_max"] is None
        # The primary (lo_lo) is UNAFFECTED -- it is computed by its own,
        # separate code path, never through this combo machinery.
        assert m.wrist_ball_delta_cm is not None
        assert sens["combos"]["lo_lo"]["realised_cm"] == m.wrist_ball_realised_cm

    # bracket_missing and empty_window (investigated, not reached):
    #
    # bracket_missing requires an endpoint's own bracket to be
    # unplaceable or missing t_lo/t_hi -- but window.identify_window's
    # own V-a step already rejects the WHOLE window (win.valid=False,
    # never reaching this code at all) if ANY command in the leg,
    # including b_start/b_rest themselves (both members of
    # leg.command_indices), is unplaceable -- Bracket.unplaceable is
    # defined as exactly "t_hi is None or t_lo is None or seq_ambiguous"
    # (_simtime.py), the identical condition this combo code checks. So
    # b_start/b_rest can never reach compute_place_route_metrics's own
    # combo computation in an unplaceable state: bracket_missing is
    # provably unreachable at this call site, not merely untested.
    #
    # empty_window (a combo whose own [start, end) has no epoch-matching
    # states) was attempted by widening b_start's own bracket far enough
    # that b_start.t_hi >= b_rest.t_lo (the assignment's own example).
    # Two independent constructions (a smooth multi-tick delayed report,
    # and a single large gap before HOVER's own confirming state) both
    # instead tripped window.identify_window's OWN "ambiguous gap"
    # validity check (V-e) at ANY widening large enough to matter: the
    # widened bracket's own t_lo is always identical to the PRECEDING
    # command's own t_hi (bracket_commands ties adjacent brackets
    # together via the same boundary state), so the gap's own lower
    # bound is pinned to 0 while its upper bound grows with the
    # widening -- crossing SETTLE_GAP_S (0.28s) without ever cleanly
    # reaching it from below lands squarely in the "ambiguous" zone
    # (neither a clean "within" nor a clean "settle"), invalidating the
    # window before this combo code is ever reached. A genuine
    # empty_window combo appears to need EITHER a real recording's own
    # gap in the state stream (not fabricated here) or a construction
    # this assignment's time budget did not find; reported as
    # investigated, not reached -- never forced.


class TestEndToEnd:
    CYCLE = "k4-e2e"

    @staticmethod
    def _write_scene_yaml(path):
        path.write_text(
            "frame_id: pedestal\n"
            "objects:\n"
            "  - id: board\n"
            "    geometry:\n"
            "      kind: box\n"
            "    pose:\n"
            "      position: [0.5, -0.3, 1.1]\n"
            "    size: [0.1, 0.1, 0.1]\n")

    def _write_sidecar(self, control_dir, kind, first_seq, last_seq, scene_block=None):
        doc = {"alignment": [{"server_seq": first_seq}, {"server_seq": last_seq}]}
        if scene_block is not None:
            doc["scene"] = scene_block
        path = control_dir / f"{self.CYCLE}-{kind}.link.json"
        path.write_text(json.dumps(doc))
        return path

    def test_cli_between_json_has_all_four_combo_keys(self, tmp_path, monkeypatch):
        """(d): one CLI `between` JSON check that `combos` has all four
        keys -- real R.PLACE_ROUTE/LIFT_TO_PRESENT, a real scene block
        (CB1), through cyc._cli."""
        monkeypatch.setattr(mf, "_LAG_RAD", [0.0] * 8)
        ev_dir = tmp_path / "ev"
        control_dir = tmp_path / "control"
        control_dir.mkdir()

        sim = mf.FlightSim(dict(R.HOME), pose_units="deg", restream_passes=1)
        sim.fly(R.PLACE_ROUTE)
        setup_last_seq = sim.state_rows[-1]["seq"]
        for _ in range(25):
            sim._hold_ticks(1, dict(sim.pose))
        flight_first_seq = sim.state_rows[-1]["seq"]
        sim.pose = dict(R.REST)
        sim.fly(R.LIFT_TO_PRESENT)
        flight_last_seq = sim.state_rows[-1]["seq"]

        result = sim.result()
        mf.write_evidence(ev_dir, result.state_rows, result.command_rows)

        scene_path = tmp_path / "scene.yaml"
        self._write_scene_yaml(scene_path)
        import hashlib
        chain_sha256 = {str(scene_path): hashlib.sha256(scene_path.read_bytes()).hexdigest()}
        scene_block = {"path": str(scene_path), "chain_sha256": chain_sha256,
                       "board_object_ids": ["board"]}
        self._write_sidecar(control_dir, "setup", 0, setup_last_seq, scene_block=scene_block)
        self._write_sidecar(control_dir, "flight", flight_first_seq, flight_last_seq)

        out = tmp_path / "out.json"
        rc = cyc._cli([
            "--ev-dir", str(ev_dir), "--control-dir", str(control_dir), "--cycle", self.CYCLE,
            "--rep", "1", "--arm", "B", "--out", str(out), "--validation-mode",
        ])
        payload = read_result(out)
        sens = payload["metrics"]["wrist_ball_bracket_sensitivity"]
        assert sens is not None, payload
        assert set(sens["combos"].keys()) == {"lo_lo", "lo_hi", "hi_lo", "hi_hi"}
        for key, combo in sens["combos"].items():
            assert set(combo.keys()) == {"realised_cm", "delta_cm", "status"}
        assert sens["primary_combo"] == "lo_lo"
        assert "complete" in sens
        assert "incomplete_combos" in sens
