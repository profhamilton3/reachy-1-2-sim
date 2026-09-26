"""W-blk acceptance tests (D-1; decision report
``outputs/decision-2026-09-26-pr144-option-c-window-proof.md`` §2/§6;
assignment ``outputs/assignment-2026-09-26-sonnet-pr144-wblk-window.md``
Part W, §W3): the settle-gap measurement window in ``window.py``, wired
into ``cycle.py`` (W2), exercised through the real ``cycle`` CLI
(``cyc._cli``, the same in-process call every other CLI test in this
suite uses -- there is no separate ``python -m`` subprocess anywhere in
this test suite) on ``make_fixtures.FlightSim(pose_units="deg")`` flights
of the real ``R.PLACE_ROUTE`` -- never a swapped-in local route, since
``cycle._cli`` hardcodes ``R.PLACE_ROUTE``/``R.CRITICAL_JOINTS`` for the
setup leg (T4's own design). Every scenario therefore manipulates the
ACTUAL RECORDING (settle timing, a deleted command run, a deleted state,
an out-of-order flight), never the route object passed to
``identify_window``.

Every CLI call here runs in NON-validation mode with the same
manifest/reset-binding/provenance/compliance/start_variant gate inputs
``test_goalfix_cmp_f4_f5_manifest_reset_binding.py``'s own
``TestGatesThroughTheCli``-equivalent tests supply (assignment W3's own
"actual CLI test... in non-validation mode, with the same gate inputs the
existing CLI tests supply") -- reused directly from that module rather
than re-implemented, so a genuine rc (0/2/3, never --validation-mode's
own forced rc 3) is observed throughout.
"""
import json
import os
import random
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in (".", "../../src", "../../scripts", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
import test_goalfix_cmp_f4_f5_manifest_reset_binding as f4f5  # noqa: E402
from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import provenance as pv  # noqa: E402
from tools.goalfix_cmp import segments as seg  # noqa: E402
from tools.goalfix_cmp import _units as units  # noqa: E402
from tools.goalfix_cmp import window  # noqa: E402
from tools.goalfix_cmp._io import RC_INCONCLUSIVE, RC_OK, RC_STOP, read_result  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402

CYCLE = "wblk"
REP_A = 1                 # 'A' in pv.ARM_MAP_ORDER
REP_B = f4f5.GATE_REP     # 2, 'B' in pv.ARM_MAP_ORDER


@pytest.fixture(autouse=True)
def _zero_lag(monkeypatch):
    monkeypatch.setattr(mf, "_LAG_RAD", [0.0] * 8)


def _pose8_rad(route_deg_pose):
    return units.pose8_rad(route_deg_pose)


def _flight_leg_tail(sim):
    """Appends a clean flight leg (settle -> LIFT_TO_PRESENT) after the
    setup leg currently in `sim`, and returns (flight_first_seq,
    flight_last_seq). Every cycle here needs a second leg -- `cycle`
    always resolves both."""
    for _ in range(25):
        sim._hold_ticks(1, dict(sim.pose))
    flight_first_seq = sim.state_rows[-1]["seq"]
    sim.pose = dict(R.REST)
    sim.fly(R.LIFT_TO_PRESENT)
    flight_last_seq = sim.state_rows[-1]["seq"]
    return flight_first_seq, flight_last_seq


def _build_reset_and_setup(fly_fn=None):
    """The SAME single-reset dance
    `f4f5._build_cycle_n_resets(tmp_path, 1)` uses (so its own
    reset-record text -- "resets_recorded 0->1 sim_step 0->0" --
    matches), but with `fly_fn(sim)` driving the setup leg's own flight
    instead of a plain `sim.fly(R.PLACE_ROUTE)`. Returns
    `(sim, setup_first_seq)`; the caller flies the setup leg and appends
    the flight leg itself."""
    home = dict(R.HOME)
    sim = mf.FlightSim(home, pose_units="deg", restream_passes=1)
    sim._hold_ticks(1, home)
    sim.reset(seed=1)
    sim._hold_ticks(1, home)
    setup_first_seq = sim.state_rows[-1]["seq"]
    if fly_fn is not None:
        fly_fn(sim)
    else:
        sim.fly(R.PLACE_ROUTE)
    return sim, setup_first_seq


def _write_scene_yaml(path):
    """The same board (position/size) test_goalfix_cmp_cb1_scene.py and
    test_goalfix_cmp_b12_wrist_ball.py's own `_box_scene` use, PLUS
    `tracked`/`physics.dynamic` (which cb1_scene.py's own YAML omits --
    its tests never check for a non-null clearance) -- without both,
    `SceneModel.obstacle_ids()` (manipulable_ids(): "tracked AND dynamic
    objects") never includes it at all, so `clearances()` finds nothing
    to measure against regardless of geometry."""
    path.write_text(
        "frame_id: pedestal\n"
        "objects:\n"
        "  - id: board\n"
        "    tracked: true\n"
        "    physics:\n"
        "      dynamic: true\n"
        "    geometry:\n"
        "      kind: box\n"
        "    pose:\n"
        "      position: [0.5, -0.3, 1.1]\n"
        "    size: [0.1, 0.1, 0.1]\n")


def _write_evidence_and_sidecars(tmp_path, ev_dir_name, state_rows, command_rows,
                                  setup_first_seq, setup_last_seq, flight_first_seq, flight_last_seq,
                                  *, with_scene=False):
    import hashlib as _hashlib
    ev_dir = tmp_path / ev_dir_name
    control_dir = tmp_path / "control"
    control_dir.mkdir(exist_ok=True)
    mf.write_evidence(ev_dir, state_rows, command_rows)
    run_dir = str(ev_dir.resolve())
    f4f5._write_sidecar(control_dir, CYCLE, "setup", setup_first_seq, setup_last_seq,
                         route_name="PLACE_ROUTE", server_run_dir=run_dir)
    f4f5._write_sidecar(control_dir, CYCLE, "flight", flight_first_seq, flight_last_seq,
                         route_name="LIFT_TO_PRESENT", server_run_dir=run_dir)
    if with_scene:
        scene_path = tmp_path / "scene.yaml"
        _write_scene_yaml(scene_path)
        sha = _hashlib.sha256(scene_path.read_bytes()).hexdigest()
        setup_sidecar_path = control_dir / f"{CYCLE}-setup.link.json"
        doc = json.loads(setup_sidecar_path.read_text())
        doc["scene"] = {"path": str(scene_path), "chain_sha256": {str(scene_path.resolve()): sha},
                         "board_object_ids": ["board"]}
        setup_sidecar_path.write_text(json.dumps(doc))
    return ev_dir, control_dir


def _build_cycle(tmp_path, *, fly_fn=None, ev_dir_name="ev", with_scene=False):
    """A full setup+flight cycle, reset epoch 1, sidecars naming the real
    route/run (needed for the manifest's own route/run checks outside
    --validation-mode). `fly_fn(sim)` customises the setup leg's own
    flight; the flight leg is always a plain LIFT_TO_PRESENT."""
    sim, setup_first_seq = _build_reset_and_setup(fly_fn)
    setup_last_seq = sim.state_rows[-1]["seq"]
    flight_first_seq, flight_last_seq = _flight_leg_tail(sim)
    result = sim.result()
    return _write_evidence_and_sidecars(
        tmp_path, ev_dir_name, result.state_rows, result.command_rows,
        setup_first_seq, setup_last_seq, flight_first_seq, flight_last_seq, with_scene=with_scene)


def _write_full_gates_for_arm(control_dir, arm, *, cycle=CYCLE):
    """Like `f4f5._write_full_gates`, but the versions document's own
    `bridge_arm`/`bridge_sha` match WHICHEVER arm is under test (that
    helper hardcodes 'B' -- every existing caller in
    test_goalfix_cmp_f4_f5_manifest_reset_binding.py/test_goalfix_cmp_
    t4_end_to_end.py only ever exercises arm B through the full,
    non-validation-mode gate path). Returns the CLI's own extra argv."""
    bridge_sha = f4f5.BRIDGE_SHA_A if arm == "A" else f4f5.BRIDGE_SHA_B
    arm_map = [
        {"rep": i + 1, "arm": pv.ARM_MAP_ORDER[i],
         "image_tag": "img-A" if pv.ARM_MAP_ORDER[i] == "A" else "img-B",
         "image_id": "sha256:A" if pv.ARM_MAP_ORDER[i] == "A" else "sha256:B",
         "bridge_sha": f4f5.BRIDGE_SHA_A if pv.ARM_MAP_ORDER[i] == "A" else f4f5.BRIDGE_SHA_B,
         "opt_hashes": {f: ("a" * 64 if pv.ARM_MAP_ORDER[i] == "A" else "b" * 64)
                        for f in pv.EXPECTED_DIFF_FILES}}
        for i in range(12)
    ]
    (control_dir / "arm_map.json").write_text(json.dumps(arm_map))
    (control_dir / f"versions_{cycle}.json").write_text(json.dumps({
        "host_native_kernel_sha": "M-sha", "host_tree_dirty": False,
        "bridge_arm": arm, "bridge_sha": bridge_sha,
        "running_image_id": f"sha256:{arm}",
        "opt_hashes": {f: ("a" * 64 if arm == "A" else "b" * 64) for f in pv.EXPECTED_DIFF_FILES},
        "supervisor_start_times": {"bridge": 200.0},
        "recreate_timestamp": 100.0,
    }))
    (control_dir / f"prep_{cycle}.json").write_text(json.dumps({
        "expected_compliant21": [True] * 21, "actual_compliant21": [True] * 21,
    }))
    (control_dir / f"start_variant_{cycle}.json").write_text(json.dumps({
        "cycle": cycle, "start_variant": "stiff-zero", "pose": {j: 0.0 for j in R.R_JOINTS},
    }))
    return [
        "--arm-map", str(control_dir / "arm_map.json"),
        "--expected-host-sha", "M-sha",
        "--required-supervisor-programs", "bridge",
        "--expected-bridge-sha-a", f4f5.BRIDGE_SHA_A,
        "--expected-bridge-sha-b", f4f5.BRIDGE_SHA_B,
    ]


def _run_full(ev_dir, control_dir, arm, out):
    """Non-validation-mode CLI run with a genuine, matching manifest,
    reset binding and provenance/compliance/start_variant gates for
    `arm` -- a real rc (0/2/3), never --validation-mode's own forced 3."""
    rep = REP_A if arm == "A" else REP_B
    f4f5._write_manifest(control_dir, cycle=CYCLE, rep=rep, reset_gen=1, reset_record="reset_record.txt")
    f4f5._write_reset_record(control_dir, "reset_record.txt",
                              "reset gen=1 ack=1 resets_recorded 0->1 sim_step 0->0\n")
    gate_args = _write_full_gates_for_arm(control_dir, arm, cycle=CYCLE)
    return f4f5._run_cli(ev_dir, control_dir, arm, out, rep=rep, cycle=CYCLE, extra=gate_args)


class TestWAC1CleanB:
    """W-AC1: window.valid; segment_start/first_rest sit immediately on
    either side of their own settle gap; t_lo_start precedes the first
    REST_SHUT command's t_lo by >= SETTLE_GAP_S (the HOVER hold is inside
    the window); wrist-ball fields are non-null."""

    def test_window_and_segment_boundaries(self, tmp_path):
        ev_dir, control_dir = _build_cycle(tmp_path, with_scene=True)
        evd = ev.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        out = tmp_path / "out.json"
        rc = _run_full(ev_dir, control_dir, "B", out)
        payload = read_result(out)
        assert payload["verdict"] == cyc.VERDICT_OK, payload
        assert rc == RC_OK
        w = payload["window"]
        assert w["valid"], w["reasons"]
        assert len(w["blocks"]) == len(R.PLACE_ROUTE) == 11

        seg_start, first_rest = w["segment_start"], w["first_rest"]
        # segment_start is the command IMMEDIATELY BEFORE the HOVER ->
        # REST_SHUT settle gap: a real settle gap follows it, and no
        # settle gap precedes it.
        lb_after = evd.brackets[seg_start + 1].t_lo - evd.brackets[seg_start].t_hi
        assert lb_after >= window.SETTLE_GAP_S
        ub_before = evd.brackets[seg_start].t_hi - evd.brackets[seg_start - 1].t_lo
        assert ub_before < window.SETTLE_GAP_S
        # first_rest is the command IMMEDIATELY AFTER the REST_SHUT
        # settle gap.
        lb_before_rest = evd.brackets[first_rest].t_lo - evd.brackets[first_rest - 1].t_hi
        assert lb_before_rest >= window.SETTLE_GAP_S

        # t_lo_start precedes the first REST_SHUT command's own t_lo
        # (segment_start + 1) by >= SETTLE_GAP_S -- the HOVER hold is
        # inside the window.
        assert evd.brackets[seg_start + 1].t_lo - w["t_lo_start"] >= window.SETTLE_GAP_S

        m = payload["metrics"]
        assert m["wrist_ball_delta_cm"] is not None
        assert m["wrist_ball_planned_cm"] is not None
        assert m["wrist_ball_commanded_cm"] is not None
        assert m["wrist_ball_realised_cm"] is not None


class TestWAC2ManipulatedAWithBrokenSegmentation:
    """W-AC2: the case the old W4 threw away. A genuinely lagged plant
    plus a strong echo rate makes >=1 injected echo off-path enough that
    `assign_goals` cannot place it (an unassigned, non-carry command) --
    independently confirmed here via `place_lr.unassigned_non_carry_
    violation` / `find_affected_segment(...).indeterminate`, the OLD W4
    trigger. W-blk's window does not depend on goal assignment, so it
    stays valid; the cycle is correctly `manipulated`, not
    `inconclusive_baseline`."""

    def test_manipulated_despite_broken_goal_assignment(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mf, "_LAG_RAD", [0.0021 + 0.0001 * j for j in range(8)])

        def fly(sim):
            sim.fly(R.PLACE_ROUTE, echo_rate=0.4, echo_rng=random.Random(7))

        ev_dir, control_dir = _build_cycle(tmp_path, fly_fn=fly, with_scene=True)

        # Confirm the OLD W4 trigger is genuinely present on this fixture
        # (library level, same evidence/route/leg the CLI itself resolves).
        evd = ev.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        route_r = cyc.route_rad(R.PLACE_ROUTE)
        setup_sidecar = json.loads((control_dir / f"{CYCLE}-setup.link.json").read_text())
        setup_leg_span = ev.leg_from_sidecar("setup", setup_sidecar, evd.states)
        setup_idx = ev.commands_in_leg(evd, setup_leg_span)
        leg = cyc.LegSpec("setup", route_r, guard=R.CRITICAL_JOINTS, start_pose8=_pose8_rad(R.HOME),
                           command_indices=setup_idx)
        lr = cyc.evaluate_leg(evd, leg)
        affected = seg.find_affected_segment(lr.assignment, leg.route_rad)
        old_w4_would_fire = lr.unassigned_non_carry_violation or (
            affected is None or affected.indeterminate)
        assert old_w4_would_fire, (
            "fixture must reproduce the old W4 trigger for this to be W-AC2's own case")

        out = tmp_path / "out.json"
        rc = _run_full(ev_dir, control_dir, "A", out)
        payload = read_result(out)
        assert payload["verdict"] == cyc.VERDICT_MANIPULATED, payload
        assert rc == RC_OK
        assert payload["window"]["valid"] is True
        assert payload["metrics"]["wrist_ball_delta_cm"] is not None
        assert any("unassigned non-carry" in r for r in payload["reasons"]), payload["reasons"]


class TestWAC3InvalidWindows:
    """Each construction once as A (`inconclusive_baseline`, rc 3) and
    once as B (`STOP`, rc 2), carrying the named `wblk:` reason."""

    def _assert_both_arms(self, ev_dir, control_dir, tmp_path, reason_substr):
        out_a = tmp_path / "out_a.json"
        rc_a = _run_full(ev_dir, control_dir, "A", out_a)
        payload_a = read_result(out_a)
        assert payload_a["verdict"] == cyc.VERDICT_INCONCLUSIVE_BASELINE, payload_a
        assert rc_a == RC_INCONCLUSIVE
        assert any(reason_substr in r for r in payload_a["reasons"]), payload_a["reasons"]

        out_b = tmp_path / "out_b.json"
        rc_b = _run_full(ev_dir, control_dir, "B", out_b)
        payload_b = read_result(out_b)
        assert payload_b["verdict"] == cyc.VERDICT_STOP, payload_b
        assert rc_b == RC_STOP
        assert any(reason_substr in r for r in payload_b["reasons"]), payload_b["reasons"]

    def test_a_no_settle_gap_between_two_waypoints(self, tmp_path):
        """(a): merging the GRIP_SHUT->BACK boundary (settle_s=0 there
        only) collapses 11 waypoints into 10 blocks."""
        def fly(sim):
            sim.fly(R.PLACE_ROUTE[:1])
            sim.settle_s = 0.0
            sim.fly(R.PLACE_ROUTE[1:2])
            sim.settle_s = 0.3
            sim.fly(R.PLACE_ROUTE[2:])

        ev_dir, control_dir = _build_cycle(tmp_path, fly_fn=fly)
        self._assert_both_arms(ev_dir, control_dir, tmp_path, "wblk:block_count=10")

    def test_b_extra_command_free_interval_inside_a_goto(self, tmp_path):
        """(b): 20 mid-goto commands (well inside CURL's own approach,
        never touching a real boundary) are deleted outright -- states
        stay continuous, so the surviving commands bracket normally, but
        a >= 0.28s command-free interval now sits INSIDE what was one
        goto, splitting it into two blocks (11 -> 12)."""
        sim, setup_first_seq = _build_reset_and_setup(lambda s: s.fly(R.PLACE_ROUTE))
        setup_last_seq = sim.state_rows[-1]["seq"]
        flight_first_seq, flight_last_seq = _flight_leg_tail(sim)
        result = sim.result()

        # The setup leg's own joint_command rows only (the reset row that
        # opened epoch 1 has no target_rad at all, and is not part of
        # any goto's own assignment).
        joint_global = [i for i, c in enumerate(result.command_rows) if c["type"] == "joint_command"]
        route_r = cyc.route_rad(R.PLACE_ROUTE)
        start_rad = _pose8_rad(R.HOME)
        targets8 = [{name: float(v) for name, v in zip(mf.R_JOINTS, result.command_rows[i]["target_rad"][:8])}
                    for i in joint_global]
        assignment = seg.assign_goals(route_r, start_rad, targets8)
        curl_positions = [i for i, g in enumerate(assignment.goal_index) if g == 2]
        mid = joint_global[curl_positions[len(curl_positions) // 2]]
        cmds2 = result.command_rows[:mid] + result.command_rows[mid + 20:]

        ev_dir, control_dir = _write_evidence_and_sidecars(
            tmp_path, "ev", result.state_rows, cmds2,
            setup_first_seq, setup_last_seq, flight_first_seq, flight_last_seq)
        self._assert_both_arms(ev_dir, control_dir, tmp_path, "wblk:block_count=12")

    def test_c_ambiguous_gap(self, tmp_path):
        """(c): settle_s=0.26 at the first boundary only gives lb=0.26
        (< 0.28) and ub=0.30 (>= 0.28) -- neither settle nor within."""
        def fly(sim):
            sim.fly(R.PLACE_ROUTE[:1])
            sim.settle_s = 0.26
            sim.fly(R.PLACE_ROUTE[1:2])
            sim.settle_s = 0.3
            sim.fly(R.PLACE_ROUTE[2:])

        ev_dir, control_dir = _build_cycle(tmp_path, fly_fn=fly)
        self._assert_both_arms(ev_dir, control_dir, tmp_path, "wblk:ambiguous_gap")

    def test_d_two_waypoints_flown_in_swapped_order(self, tmp_path):
        """(d): BACK and CURL are physically flown in swapped order --
        `cycle._cli` still resolves the real, unswapped R.PLACE_ROUTE, so
        block 1's actual content (CURL's own motion) mismatches its
        expected ordinal (BACK), and vice versa."""
        def fly(sim):
            swapped = list(R.PLACE_ROUTE)
            swapped[1], swapped[2] = swapped[2], swapped[1]
            sim.fly(swapped)

        ev_dir, control_dir = _build_cycle(tmp_path, fly_fn=fly)
        self._assert_both_arms(ev_dir, control_dir, tmp_path, "wblk:label_mismatch")

    def test_e_missing_state_inside_the_window(self, tmp_path):
        """(e): one state row inside the wrist-ball measurement window
        (located from a clean, unperturbed window computation) is
        deleted -- V-d's contiguous-seq check fails."""
        ev_dir, control_dir = _build_cycle(tmp_path)
        evd = ev.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        route_r = cyc.route_rad(R.PLACE_ROUTE)
        setup_sidecar = json.loads((control_dir / f"{CYCLE}-setup.link.json").read_text())
        setup_leg_span = ev.leg_from_sidecar("setup", setup_sidecar, evd.states)
        setup_idx = ev.commands_in_leg(evd, setup_leg_span)
        leg = cyc.LegSpec("setup", route_r, guard=R.CRITICAL_JOINTS, start_pose8=_pose8_rad(R.HOME),
                           command_indices=setup_idx)
        win = window.identify_window(evd, leg)
        assert win.valid, win.reasons

        states_path = ev_dir / "states.jsonl"
        rows = [json.loads(line) for line in states_path.read_text().splitlines() if line.strip()]
        victim = win.state_indices[len(win.state_indices) // 2]
        del rows[victim]
        states_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        _recompute_sha(ev_dir, "states.jsonl")

        self._assert_both_arms(ev_dir, control_dir, tmp_path, "wblk:window_states")


def _recompute_sha(ev_dir, name):
    import hashlib
    p = ev_dir / name
    sums_path = ev_dir / "SHA256SUMS"
    lines = sums_path.read_text().splitlines()
    new_lines = []
    for line in lines:
        h, n = line.split(None, 1)
        if n == name:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
        new_lines.append(f"{h}  {n}")
    sums_path.write_text("\n".join(new_lines) + "\n")


class TestWAC4LeadInAfterRecreate:
    """W-AC4: a compliance command (a non-None flag among the 8 right-arm
    joints, the SDK turn_on split by the bridge) followed by a >= 0.28s
    gap before GRIP_SHUT is stripped as lead-in, never counted as its own
    block. Expect 11 blocks and a valid window."""

    def test_lead_in_stripped_gives_eleven_blocks(self, tmp_path):
        def fly(sim):
            # A synthetic turn_on compliance command (target == the
            # current epoch-start state -- a compliance command carries
            # no informative target of its own), immediately followed by
            # a real settle gap (the first hold tick's own state is what
            # reports it applied) before any real motion.
            compliance_cmd = mf.command_row_joint(
                seq=sim._cmd_seq,
                target_rad21=[j["position_rad"] for j in sim.state_rows[-1]["joints"]],
                compliant=[True] * 8 + [None] * 13)
            sim.command_rows.append(compliance_cmd)
            sim._cmd_seq += 1
            sim._native_cmd_seq = sim._cmd_seq - 1
            sim._hold_ticks(15, sim.pose)  # >= 0.28s command-free gap before GRIP_SHUT
            sim.fly(R.PLACE_ROUTE)

        ev_dir, control_dir = _build_cycle(tmp_path, fly_fn=fly)
        out = tmp_path / "out.json"
        rc = _run_full(ev_dir, control_dir, "B", out)
        payload = read_result(out)
        assert payload["verdict"] == cyc.VERDICT_OK, payload
        assert rc == RC_OK
        w = payload["window"]
        assert w["valid"], w["reasons"]
        assert w["lead_in"] == 1
        assert len(w["blocks"]) == 11


class TestWAC5SegmentStartOffByOneFromAffected:
    """W-AC5 (the §3.2 finding, pinned): `win.segment_start ==
    global(affected.start_command_index) - 1`; that command's successor
    is bit-equal to `float32(HOVER)`.

    The decision report's own §3.2 finding needs HOVER's REAL last
    approach tick to fall a hair short of its nominal goal (as real
    tracking lag does -- report §4; well inside C1'/C4''s own 0.05/0.01deg
    tolerances) so that the FOLLOWING tick -- REST_SHUT's own anchor
    sample, always bit-exact to the PREVIOUS goto's own goal by
    minimum-jerk construction -- is NOT a bit-exact carry of it. C0's own
    R-tie tie-break (segments.py) only advances `cur` to the next goto on
    a carry-of-an-already-exact value; without one, the anchor is
    ordinarily (and correctly, per C0's own rules) admitted to HOVER's
    OWN segment instead, extending `affected.start_command_index` one
    past W-blk's own, purely timing-based boundary (which never moves,
    regardless of goal assignment). `make_fixtures.FlightSim`'s own
    minimum-jerk plateaus bit-exact well before its very last tick (this
    file's other tests all show `win.segment_start == affected_start`,
    no offset at all) -- reproducing the real off-by-one needs this exact
    short-of-goal residual, built by hand here rather than via FlightSim.
    """

    def test_off_by_one_from_c0_boundary(self, tmp_path):
        joints = list(mf.R_JOINTS)

        def pose(**kw):
            p = dict(zip(joints, [0.0] * 8))
            p.update(kw)
            return p

        start = pose()
        hover_true = pose(r_shoulder_pitch=np.radians(-30.0))
        hover_last = pose(r_shoulder_pitch=np.radians(-30.0) + np.radians(0.005))  # 0.005deg short
        rest_shut_mid = pose(r_shoulder_pitch=np.radians(-50.0))
        rest_shut_goal = pose(r_shoulder_pitch=np.radians(-70.0))
        rest_pose = dict(rest_shut_goal, r_gripper=np.radians(-45.0))

        rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                              wall_time_ns=1, position_rad21=mf.full21(start))]
        cmds = []

        def emit(tgt):
            i = len(cmds)
            cmds.append(mf.command_row_joint(seq=i, target_rad21=mf.full21(tgt)))
            rows.append(mf.state_row(seq=len(rows), sim_step=len(rows), sim_time_s=len(rows) * 0.02,
                                      cmd_seq=i, wall_time_ns=2 + len(rows), position_rad21=mf.full21(tgt)))

        def settle(pose_, n=16):
            for _ in range(n):
                rows.append(mf.state_row(seq=len(rows), sim_step=len(rows), sim_time_s=len(rows) * 0.02,
                                          cmd_seq=len(cmds) - 1, wall_time_ns=2 + len(rows),
                                          position_rad21=mf.full21(pose_)))

        emit(hover_last)  # HOVER's own (only) real approach tick, short of goal
        settle(hover_last)
        emit(hover_true)  # REST_SHUT's own anchor tick -- bit-exact to HOVER's nominal goal
        emit(rest_shut_mid)
        emit(rest_shut_goal)
        settle(rest_shut_goal)
        emit(rest_pose)

        mf.write_evidence(tmp_path, rows, cmds)
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        route = [mf.Waypoint("HOVER", hover_true, 1.0), mf.Waypoint("REST_SHUT", rest_shut_goal, 1.0),
                 mf.Waypoint("REST", rest_pose, 1.0)]
        leg = cyc.LegSpec("setup", route, guard=(), start_pose8=start, command_indices=list(range(len(cmds))))

        targets8 = [{name: float(v) for name, v in zip(mf.R_JOINTS, c["target_rad"][:8])} for c in cmds]
        assignment = seg.assign_goals(route, start, targets8)
        affected = seg.find_affected_segment(assignment, route)
        assert affected is not None and not affected.indeterminate

        win = window.identify_window(evd, leg)
        assert win.valid, win.reasons

        affected_start_global = leg.command_indices[affected.start_command_index]
        assert win.segment_start == affected_start_global - 1

        successor_target = evd.commands.target_rad[win.segment_start + 1, :8]
        hover8 = np.array([hover_true[j] for j in mf.R_JOINTS], dtype=np.float32)
        assert np.array_equal(successor_target.astype(np.float32), hover8)


class TestWAC6Constants:
    """W-AC6: window.SETTLE_S is pinned to rig_motion.fly_route's own
    settle_s default."""

    def test_settle_s_matches_fly_route_default(self):
        import inspect
        from reachy_ai.tasks import rig_motion
        default = inspect.signature(rig_motion.fly_route).parameters["settle_s"].default
        assert window.SETTLE_S == default
