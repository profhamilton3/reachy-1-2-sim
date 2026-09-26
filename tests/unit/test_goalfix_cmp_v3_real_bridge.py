"""V3 real-bridge fixture tests (Stage 1 assignment, 2026-09-25, §2).

Drives the REAL client/bridge stack, offline:

    reachy_sdk 0.7.0 -> loopback gRPC -> fake_reachy_server.FakeJointService
      -> KinematicBridge(MujocoRemoteBackend) -> tests/integration/native_stub.NativeStub

via ``tests/fixtures/goalfix_cmp/real_bridge.py`` (which reuses the
CONSTRUCTION PATTERN, not the fixture function, of
``tests/integration/test_goal_reporting_sdk_path.py:92-150`` -- ``Rig``,
``make_rig``), and records F1-F7 (the assignment's "Recorded findings"):
what is ACTUALLY observed on this path, never a tuned/loosened check.

No Docker, no native MuJoCo server, no hardware, no motion. Loopback
sockets only, inside this in-process harness.

Skips (``importorskip``) if ``reachy_sdk``/``grpc``/``websockets``/``scipy``
are missing (system ``python3``). With ``REQUIRE_REACHY_SDK=1`` (run under
``~/goalfix-venv``) these FAIL instead of skipping.
"""
from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "native_mujoco"),
           os.path.join(_ROOT, "scripts"), os.path.join(_ROOT, "tests", "integration"),
           os.path.join(_ROOT, "tests", "fixtures", "goalfix_cmp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if os.environ.get("REQUIRE_REACHY_SDK") == "1":
    import grpc  # noqa: F401
    import numpy as np
    import reachy_sdk  # noqa: F401
    import websockets  # noqa: F401
    import scipy  # noqa: F401
else:
    grpc = pytest.importorskip("grpc")
    np = pytest.importorskip("numpy")
    pytest.importorskip("reachy_sdk")
    pytest.importorskip("websockets")
    pytest.importorskip("scipy")

import real_bridge as rb  # noqa: E402
from native_stub import IDX  # noqa: E402
from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import echo  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import pathcheck as pc  # noqa: E402
from tools.goalfix_cmp._io import RC_INCONCLUSIVE, read_result  # noqa: E402
from tools.goalfix_cmp._units import route_rad  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")

CYCLE_NAMES = ["S2-B4-c-r1", "S2-B4-c-r2"]
ARM8 = R.ARM7 + ("r_gripper",)
ARM_IDX = [IDX[n] for n in ARM8]


@pytest.fixture(scope="module")
def v3(tmp_path_factory):
    """Runs the whole V3 scenario ONCE for this module -- Stage 0
    ``turn_on``, then twice: container-recreate equivalent -> reset (through
    the bridge's own reset path) -> a fresh client + ``turn_on`` +
    ``fly_route(PLACE_ROUTE)`` (product defaults) -> a fresh client +
    ``turn_on`` + ``fly_route(LIFT_TO_PRESENT)`` (assignment §2.3) -- and
    emits its evidence to a module-scoped ``tmp_path`` (never committed,
    per §2.2). Every test below OBSERVES this same run (the assignment's
    F1-F7 framing), rather than building an independent scenario per test:
    each leg's own ``fly_route`` call is several real wall-clock seconds,
    so re-running the whole scenario per test would multiply an already
    multi-minute cost for no additional evidence."""
    session = rb.build_session()
    try:
        rb.run_two_cycles(session)
        tmp = tmp_path_factory.mktemp("v3_real_bridge")
        ev_dir, control_dir = tmp / "ev", tmp / "control"
        rb.emit_evidence(session, ev_dir, control_dir, CYCLE_NAMES)
        evidence = ev.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        yield {
            "session": session, "ev_dir": ev_dir, "control_dir": control_dir,
            "evidence": evidence, "names": CYCLE_NAMES,
        }
    finally:
        session.close()


def _resolve_legs(v3, name):
    evd = v3["evidence"]
    setup = cyc.resolve_leg(evd, v3["control_dir"] / f"{name}-setup.link.json",
                             "setup", R.PLACE_ROUTE, R.CRITICAL_JOINTS)
    flight = cyc.resolve_leg(evd, v3["control_dir"] / f"{name}-flight.link.json",
                              "flight", R.LIFT_TO_PRESENT, R._PRESENT_GUARD)
    return setup, flight


# ---------------------------------------------------------------------------
# F5: seq restart at each recreate, native cmd_seq carried, no unplaceable
# ---------------------------------------------------------------------------

def test_f5_bridge_sessions_restart_and_carry_with_zero_unplaceable(v3):
    evd = v3["evidence"]
    assert evd.n_epochs() == 3, "Stage 0 + 2 recreate/reset cycles => 3 epochs"
    sessions = evd.bridge_sessions
    assert sessions[0].restart is False, "epoch 0 (Stage 0, no recreate yet): no restart"
    assert sessions[1].restart is True, "epoch 1 (after cycle 1's recreate): bridge seq restarted"
    assert sessions[2].restart is True, "epoch 2 (after cycle 2's recreate): bridge seq restarted"
    assert not any(s.ambiguous for s in sessions.values()), (
        "no epoch's restart made a seq ambiguous on this harness")
    assert evd.unplaceable_command_indices == [], (
        "every joint_command was placeable across both real recreates")


# ---------------------------------------------------------------------------
# F1: re-stream-pass exactness -- INAPPLICABLE on this toy plant (no
# re-stream pass was ever triggered), a documented finding, not a
# loosened check.
# ---------------------------------------------------------------------------

def test_f1_toy_plant_never_triggers_a_restream_pass(v3):
    """``fly_route``'s retry loop (``rig_motion.py``) issues a re-stream
    pass only when the GUARDED joints' worst tracking error exceeds
    ``TRACK_TOL`` (6 deg) after the initial goto. NativeStub's toy plant
    (first-order, tau=0.05 s, no bias/sag by default -- ``build_session``
    passes ``bias=None``) converges within that tolerance from a single
    goto, for every one of the 11 PLACE_ROUTE and 1 LIFT_TO_PRESENT
    waypoints, in both legs of both cycles: exactly one ``move()`` call per
    flown waypoint, confirmed via ``fly_route``'s own (unmodified) ``move=``
    extension point. F1 ("are re-stream-pass targets bit-exact
    ``route_rad(wp)``") therefore has NO SAMPLE to check on this harness --
    it needs a biased/sagging plant (out of Stage 1's toy-plant scope) to
    ever produce a re-stream pass at all."""
    session = v3["session"]
    for cb in session.cycles:
        for leg in (cb.setup, cb.flight):
            assert len(leg.move_calls) == len(leg.flown), (
                leg.flown, [mc.pose for mc in leg.move_calls])
            # Every move() call's pose is DISTINCT (no two consecutive calls
            # share a target) -- the direct evidence that no waypoint needed
            # a second (re-stream) pass to the same pose.
            poses = [tuple(sorted(mc.pose.items())) for mc in leg.move_calls]
            assert len(poses) == len(set(poses)), "a repeated pose would be a re-stream pass"


# ---------------------------------------------------------------------------
# F2: a goto's last setpoint is usually NOT exactly route_rad(wp) -- the
# T-10ms-style minimum-jerk residual is real and observable, and whether a
# given waypoint lands exactly is not stable across repeated cycles.
# ---------------------------------------------------------------------------

def test_f2_last_setpoint_is_usually_not_exactly_route_rad(v3):
    """Across 5 runs of this scenario during Stage 1 development, WHICH
    waypoint's goto ends bit-exact on ``route_rad(wp)`` varied from run to
    run (real wall-clock/100 Hz-tick scheduling, not simulation-tick
    deterministic) -- no single waypoint name is safe to pin as "always
    exact" or "always non-exact". What is robust across every run is the
    aggregate: over 12 flown waypoints x 2 legs x 2 cycles, at least one
    goto's last commanded sample is NOT bit-exact ``route_rad(wp)`` (the
    minimum-jerk / 100 Hz-tick residual the SDK-path test's own comment
    describes: "goto stops at the last 100 Hz tick before duration"), and
    it is also common (not universal) for every waypoint in a leg to land
    exactly. Both are recorded; neither is asserted per-waypoint."""
    session = v3["session"]
    with session.stub.lock:
        commands = list(session.stub.commands)

    def last_target_diffs(leg_bounds, route):
        rr = route_rad(route)
        out = {}
        for name, call, wp in zip(leg_bounds.flown, leg_bounds.move_calls, rr):
            last8 = [commands[call.last_command_index - 1]["target"][i] for i in ARM_IDX]
            want8 = [wp.pose_rad8[j] for j in ARM8]
            out[name] = max(abs(a - b) for a, b in zip(last8, want8))
        return out

    all_diffs = []
    for cb in session.cycles:
        all_diffs.append(last_target_diffs(cb.setup, R.PLACE_ROUTE))
        all_diffs.append(last_target_diffs(cb.flight, R.LIFT_TO_PRESENT))

    assert any(d > 0.0 for diffs in all_diffs for d in diffs.values()), (
        "expected at least one non-exact goto ending across the whole "
        f"session; got all-exact: {all_diffs}")


# ---------------------------------------------------------------------------
# F3: no joint_commands during a settle_s hold; a small, turn_on-attributed
# gap between legs; commands do follow immediately after turn_on's own
# settle.
# ---------------------------------------------------------------------------

def test_f3_no_commands_during_settle_s_holds(v3):
    """Every consecutive pair of ``move()`` calls WITHIN a leg is
    command-contiguous (the next call's first stub-log index equals the
    previous call's last) -- i.e. the intervening ``time.sleep(settle_s)``
    inside ``fly_route`` (0.3 s default, between every waypoint including
    the last) carried NO ``joint_command`` of its own, on this harness."""
    session = v3["session"]
    for cb in session.cycles:
        for leg in (cb.setup, cb.flight):
            for prev, nxt in zip(leg.move_calls, leg.move_calls[1:]):
                assert nxt.first_command_index == prev.last_command_index, (
                    "a command landed between two waypoints' own settle_s window")


def test_f3_gap_between_legs_is_small_and_attributable_to_turn_on(v3):
    """The gap between the setup leg's last logged command and the flight
    leg's first ``fly_route`` move call (the harness's OWN inter-leg
    settle, not ``fly_route``'s ``settle_s``) is non-zero -- ``turn_on``
    itself issues at least one ``joint_command`` -- but small (a handful
    of ticks, not a sustained stream)."""
    for cb in v3["session"].cycles:
        gap = cb.flight.move_calls[0].first_command_index - cb.setup.move_calls[-1].last_command_index
        assert 0 < gap <= 10, f"unexpected inter-leg command gap: {gap}"


# ---------------------------------------------------------------------------
# F4: at each leg's own turn_on, the first setpoint equals the
# present-position state in force at turn_on (validates T7/C1's start
# source).
# ---------------------------------------------------------------------------

def test_f4_first_setpoint_equals_present_at_turn_on(v3):
    evd = v3["evidence"]
    for name in v3["names"]:
        setup_leg, flight_leg = _resolve_legs(v3, name)
        for leg in (setup_leg, flight_leg):
            turn_on_idx = leg.turn_on_state_index
            assert turn_on_idx is not None and turn_on_idx >= 0
            first_cmd_i = leg.command_indices[0]
            first_target8 = evd.commands.target_rad[first_cmd_i, :8]
            present_at_turnon8 = evd.states.position_rad[turn_on_idx, :8]
            maxdiff = float(np.max(np.abs(first_target8 - present_at_turnon8)))
            # float64 arithmetic noise only (< 1e-9 rad); the SDK/bridge
            # round trip is float32, so anything at this scale is exact.
            assert maxdiff < 1e-8, (
                f"{name}/{leg.name}: first setpoint {maxdiff:.3e} rad from "
                "the present-position reading in force at turn_on")


# ---------------------------------------------------------------------------
# F6: the shipped `cycle` CLI, in --validation-mode, on both real-bridge
# legs of both cycles -- rc, verdict, every C0-C8 result, hold stats.
# ---------------------------------------------------------------------------

def test_f6_cycle_cli_on_real_bridge_cycles(v3, tmp_path):
    """Runs the SHIPPED ``cycle`` CLI (not a library call) on each
    generated cycle. ``--validation-mode`` always exits rc 3 by design
    (the provenance/compliance/start_variant gates do not exist for this
    harness -- there is no Docker image, no host SHA, no start-variant
    file); the payload's own ``verdict`` is what carries the finding.

    UPDATED (Stage B, owner rulings 2026-09-25: N1-N3, C1'/C4', R-carry,
    R-over, carry-aware C2): this test originally pinned "STOP, purely
    from real signal characteristics" (see test_f6_c0_c8_breakdown's own
    docstring: C0/C1/C2/C3/C4/C5/C7 were each observed to fail OR pass on
    different runs, at the OLD literal tolerances -- C1_TOL_RAD=1e-6 rad,
    C2's non-carry-aware 30ms check, C4's T-10ms residual budget, C0's
    strict tie-break exactness). Those exact tolerances are what the
    owner's rulings replaced, precisely because they mistook real
    float32/scheduling noise on a genuinely clean signal for a
    violation. Re-observed after the fix (repeatedly, including outside
    pytest): both real-bridge B cycles now verdict "ok", with EVERY
    C0-C8 check passing on both legs, reasons=[], genuine_echo_count=0 --
    not a loosened check, the same real captured data now correctly read.
    Pinned to this new, correct baseline; a future STOP here on real data
    is a new finding worth its own investigation, not a sign this
    assertion needs loosening again."""
    for name in v3["names"]:
        out = tmp_path / f"between_{name}.json"
        argv = ["--ev-dir", str(v3["ev_dir"]), "--control-dir", str(v3["control_dir"]),
                "--cycle", name, "--rep", "1" if name.endswith("r1") else "2",
                "--arm", "B", "--out", str(out), "--validation-mode"]
        rc = cyc._cli(argv)
        assert rc == RC_INCONCLUSIVE, "validation-mode always exits 3 by design"
        payload = read_result(out)
        assert payload["validation_only"] is True
        assert payload["verdict"] == cyc.VERDICT_OK, payload["reasons"]
        assert payload["reasons"] == []
        # Not pinned at exactly 0 in general -- see test_f7's docstring:
        # this same segment-scoped count was observed nonzero on real
        # data on at least one (pre-fix) run, from real signal
        # coincidence, not manipulation. Currently observed at 0.
        assert 0 <= payload["genuine_echo_count"] <= 20, payload
        assert payload["segment_indeterminate"] is False
        assert payload["unplaceable_command_indices"] == []


def test_f6_c0_c8_breakdown_after_r_tie_r_const_q_hold_q_echo(v3):
    """The full per-check C0-C8 table (not just the CLI's failure-only
    ``reasons`` list), via ``evaluate_cycle`` directly (the same call the
    CLI makes; this reads the checks it does not serialize).

    HISTORY: after R-tie/R-const/Q-hold/Q-echo (2026-09-25 stage-1
    rulings), C0/C1/C2/C3/C4/C5/C7 each still varied run to run at their
    OLD literal tolerances (``C1_TOL_RAD=1e-6`` rad,
    ``C2_SKEW_TOL_S=0.030`` s non-carry-aware, C4's T-10ms residual
    budget, C5's CB6 gating, C7's exact-equality, C0's tie-break
    exactness) -- reported as-observed, none of them asserted either
    way, per the "no invented definition" instruction.

    UPDATED (Stage B, owner rulings 2026-09-25: N1-N3, C1'=0.05deg,
    C4'=0.01deg, R-carry, R-over, carry-aware C2): repeated re-observation
    on this same real-bridge scenario now shows EVERY C0-C8 check passing
    on both legs of both cycles (see test_f6_cycle_cli_on_real_bridge_cycles's
    own updated docstring) -- the variance above was the OLD tolerances'
    own false positives on real float32/scheduling noise, not
    unavoidable plant noise. Only C6/C8 (and the empty-hold-window facts)
    are asserted directly here, unchanged, since this test predates the
    fix and a full C0-C8 assertion belongs with the CLI-level test that
    already carries it."""
    for name in v3["names"]:
        setup_leg, flight_leg = _resolve_legs(v3, name)
        evd = v3["evidence"]
        cv, leg_results = cyc.evaluate_cycle(name, "B", evd, [setup_leg, flight_leg], skip_gates=True)
        # `.pathcheck` also carries non-CheckResult bookkeeping keys
        # ("_seed_deviation_rad", "_assignment" -- pathcheck.run_all's own
        # internals); only pc.ALL_CHECKS ("C0".."C8") are CheckResults.
        setup_checks = {cid: leg_results["setup"].pathcheck[cid].passed for cid in pc.ALL_CHECKS}
        assert setup_checks["C6"] is True, "passes vacuously: see docstring (F3: empty hold windows)"
        assert setup_checks["C8"] is True
        # Every hold window found had NO commands of its own on this
        # harness (F3's finding, restated at the hold level: target drift
        # is 0 by construction because the window is empty, never because
        # drift was computed and happened to be 0).
        for hs in leg_results["setup"].hold_stats + leg_results["flight"].hold_stats:
            assert hs.window_command_indices == []
            assert all(v == 0.0 for v in hs.target_drift.values())


# ---------------------------------------------------------------------------
# F7: echo classification of the generated (B-like) cycle.
# ---------------------------------------------------------------------------

def test_f7_echo_classification_of_generated_cycle(v3):
    evd = v3["evidence"]
    for name in v3["names"]:
        setup_leg, flight_leg = _resolve_legs(v3, name)
        cv, leg_results = cyc.evaluate_cycle(name, "B", evd, [setup_leg, flight_leg], skip_gates=True)
        full_goto_context = [None] * len(evd.commands)
        leg_turn_on_map = {}
        for leg in (setup_leg, flight_leg):
            lr = leg_results[leg.name]
            for local_i, global_i in enumerate(leg.command_indices):
                full_goto_context[global_i] = lr.goto_context[local_i]
            leg_turn_on_map[leg.command_indices[0]] = leg.turn_on_state_index
        results = echo.classify_commands(evd, full_goto_context, leg_turn_on_state_index=leg_turn_on_map)
        both_legs_idx = list(setup_leg.command_indices) + list(flight_leg.command_indices)
        counts = echo.count_labels(results, both_legs_idx)
        # NOT pinned at exactly 0: across repeated runs of this same
        # uninjected-B scenario, genuine_echo_count was observed as BOTH 0
        # and a small nonzero value (4, on one run, before the Q-echo
        # fix). Real, fast-converging plant dynamics can put a genuinely
        # commanded FUTURE target within float32-exact reach of a real
        # PAST state sample purely by coincidence, with no injected
        # manipulation. The Q-echo fix reclassifies the near-end/on-path
        # shape of that coincidence as `path_coincidence` (also observed
        # nonzero here after the fix, where it was always 0 before) --
        # but does not guarantee genuine_echo is always exactly 0, since
        # an off-path near-end (or a non-near-end on-path-by-fluke)
        # coincidence is still possible in principle. This remains a
        # finding for the coordinator: the B-verdict's "any genuine echo
        # -> STOP" rule (cycle.py) has no margin against real coincidence
        # on real data. Bounded loosely here only to catch a gross
        # regression (e.g. hundreds of echoes), never tuned to make this
        # test pass.
        assert 0 <= counts.genuine_echo <= 20, counts.as_dict()
        assert 0 <= counts.path_coincidence <= 20, counts.as_dict()
        assert counts.timing_ambiguous == 0
        assert counts.unavailable == 0
        # start_coincidence fires (both legs' own turn_on, per F4) --
        # exact count recorded as observed, not asserted from a formula.
        assert counts.start_coincidence == 8, counts.as_dict()
