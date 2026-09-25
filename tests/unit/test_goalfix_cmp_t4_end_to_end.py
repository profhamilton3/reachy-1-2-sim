"""T4 acceptance tests (assignment §2, T4): the end-to-end `cycle` CLI on
realistic fixtures -- both legs derived from linker sidecars, start poses
from the turn_on goal (never R.HOME/R.REST), goto_context/path coincidence
live, hold drift measured, and the provenance/compliance/start_variant
gates called."""
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../scripts", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import provenance as pv  # noqa: E402
from tools.goalfix_cmp._io import RC_INCONCLUSIVE, RC_OK, RC_STOP, read_result  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402

# This module's fixtures test C1 (start continuity against the turn_on
# GOAL) at leg boundaries, which -- on the real robot/SDK -- IS whatever
# the state stream was most recently reporting (turn_on locally sets
# goal_position=present_position; report §1 step 7). make_fixtures's
# constant per-joint _LAG_RAD deliberately keeps "realised" off the exact
# commanded curve so OTHER tests never see an accidental bit-exact match;
# here it would instead make a leg's own first command mismatch its own
# turn_on reading by ~0.1-0.2 deg -- a fixture artefact, not a defect
# under test. Zeroed for every test in this module only (autouse
# monkeypatch, restored after each test -- module-level mutation of the
# shared make_fixtures list would leak into every OTHER test file pytest
# collects into the same process).


@pytest.fixture(autouse=True)
def _zero_lag(monkeypatch):
    monkeypatch.setattr(mf, "_LAG_RAD", [0.0] * 8)

CYCLE = "S2-B4-c-r1"


def _write_sidecar(control_dir, cycle, kind, first_seq, last_seq):
    path = control_dir / f"{cycle}-{kind}.link.json"
    path.write_text(json.dumps({
        "alignment": [{"server_seq": first_seq}, {"server_seq": last_seq}],
    }))
    return path


def _build_cycle(tmp_path, *, echo_rate=0.0, seed=0, home_offset_deg=0.0):
    """A realistic single-epoch cycle: setup PLACE_ROUTE (from HOME, with
    an optional stiff-zero offset) then flight LIFT_TO_PRESENT (from
    REST), both flown in real degrees through route_rad's own conversion.
    Returns (ev_dir, control_dir)."""
    ev_dir = tmp_path / "ev"
    control_dir = tmp_path / "control"
    control_dir.mkdir()

    home = dict(R.HOME)
    home["r_shoulder_pitch"] = home.get("r_shoulder_pitch", 0.0) + home_offset_deg
    sim = mf.FlightSim(home, pose_units="deg", restream_passes=1, settle_s=0.05)
    sim.fly(R.PLACE_ROUTE, echo_rate=echo_rate, echo_rng=__import__("random").Random(seed))
    setup_last_seq = sim.state_rows[-1]["seq"]
    setup_first_seq = 0

    # A settle before the flight leg's own turn_on (a fresh client + turn_on
    # before any post-reset motion, plan §5 P8) -- the flight leg's own
    # start pose must be read from ITS OWN turn_on, not the setup leg's.
    for _ in range(25):  # 0.5s settle, present position continuously reported
        sim._hold_ticks(1, dict(sim.pose))
    flight_first_seq = sim.state_rows[-1]["seq"]

    sim.pose = dict(R.REST)  # REST is LIFT_TO_PRESENT's own nominal start
    sim.fly(R.LIFT_TO_PRESENT, echo_rate=echo_rate, echo_rng=__import__("random").Random(seed + 1))
    flight_last_seq = sim.state_rows[-1]["seq"]

    result = sim.result()
    mf.write_evidence(ev_dir, result.state_rows, result.command_rows)
    _write_sidecar(control_dir, CYCLE, "setup", setup_first_seq, setup_last_seq)
    _write_sidecar(control_dir, CYCLE, "flight", flight_first_seq, flight_last_seq)
    return ev_dir, control_dir


def _run_cli(ev_dir, control_dir, arm, out, **extra):
    argv = [
        "--ev-dir", str(ev_dir), "--control-dir", str(control_dir), "--cycle", CYCLE,
        "--rep", "1", "--arm", arm, "--out", str(out), "--validation-mode",
    ]
    for k, v in extra.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return cyc._cli(argv)


class TestEndToEndFixtureTable:
    def test_clean_b_is_ok(self, tmp_path):
        ev_dir, control_dir = _build_cycle(tmp_path)
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        payload = read_result(out)
        assert payload["verdict"] == cyc.VERDICT_OK, payload

    def test_a_with_echoes_is_manipulated(self, tmp_path, monkeypatch):
        # Q-echo (coordinator ruling, 2026-09-25 stage-1 rulings, §3): with
        # this module's own zeroed `_LAG_RAD` (state == target exactly),
        # EVERY forced echo sourced from a past STATE of the SAME goto is,
        # by construction, mathematically exactly ON that goto's own
        # minimum-jerk curve -- so whenever it also lands near either end
        # (common: minimum-jerk spends many ticks near-flat there), it is
        # now correctly path_coincidence, never genuine_echo (this is the
        # exact bug the ruling fixes, not a defect in the fix). That
        # collapsed this fixture's segment-scoped genuine count to 0 for
        # every seed tried. A real echo is never on the CORRECT goto's own
        # path in the first place (it is a stale/manipulated report, not a
        # coincidental resample of the same curve) -- modelling that here
        # needs a genuinely lagged plant, restored for this ONE test only
        # (every other test in this module still isolates C1 from lag via
        # the file's own autouse fixture; lag never enters any C0-C8 check,
        # which reads only commanded targets, never states).
        monkeypatch.setattr(mf, "_LAG_RAD", [0.0021 + 0.0001 * j for j in range(8)])
        ev_dir, control_dir = _build_cycle(tmp_path, echo_rate=0.4, seed=7)
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "A", out)
        payload = read_result(out)
        assert payload["verdict"] == cyc.VERDICT_MANIPULATED, payload
        assert payload["genuine_echo_count"] > 0

    def test_a_without_echoes_is_inconclusive_baseline(self, tmp_path):
        ev_dir, control_dir = _build_cycle(tmp_path)
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "A", out)
        payload = read_result(out)
        assert payload["verdict"] == cyc.VERDICT_INCONCLUSIVE_BASELINE, payload

    def test_b_c8_violation_only_on_flight_leg_stops(self, tmp_path):
        ev_dir, control_dir = _build_cycle(tmp_path)
        commands_path = ev_dir / "commands.jsonl"
        rows = [json.loads(l) for l in commands_path.read_text().splitlines() if l.strip()]
        # Corrupt a non-right-arm joint on the LAST joint_command (which
        # belongs to the flight leg).
        last_jc = max(i for i, r in enumerate(rows) if r["type"] == "joint_command")
        rows[last_jc]["target_rad"][10] = 0.777
        commands_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        # Recompute SHA256SUMS for the mutated commands.jsonl.
        import hashlib
        sums_path = ev_dir / "SHA256SUMS"
        lines = sums_path.read_text().splitlines()
        new_lines = []
        for line in lines:
            h, name = line.split(None, 1)
            if name == "commands.jsonl":
                h = hashlib.sha256(commands_path.read_bytes()).hexdigest()
            new_lines.append(f"{h}  {name}")
        sums_path.write_text("\n".join(new_lines) + "\n")

        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        payload = read_result(out)
        assert payload["verdict"] == cyc.VERDICT_STOP, payload
        assert any("C8" in r for r in payload["reasons"])

    def test_stiff_zero_start_half_degree_off_home_is_ok(self, tmp_path):
        """C1 checks against the turn_on goal (the actual leg-start
        reading), not a hard-coded R.HOME -- so a 0.5deg stiff-zero offset
        from the nominal HOME pose must still pass."""
        ev_dir, control_dir = _build_cycle(tmp_path, home_offset_deg=0.5)
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        payload = read_result(out)
        assert payload["verdict"] == cyc.VERDICT_OK, payload

    def test_missing_flight_sidecar_gives_rc3(self, tmp_path):
        ev_dir, control_dir = _build_cycle(tmp_path)
        (control_dir / f"{CYCLE}-flight.link.json").unlink()
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE


class TestGatesThroughTheCli:
    """The provenance/compliance/start_variant gates, exercised WITHOUT
    --validation-mode -- so the CLI reads versions_<cycle>.json,
    prep_<cycle>.json and start_variant_<cycle>.json from control_dir."""

    def _write_gate_files(self, control_dir, *, host_sha="M-sha", compliant_ok=True,
                           write_start_variant=True):
        arm_map = [{
            "rep": 1, "arm": "B", "image_tag": "img-B", "image_id": "sha256:B",
            "bridge_sha": "67730a1ecf646544825fb60c00129cf46de6d307",
            "opt_hashes": {f: "b" * 64 for f in pv.EXPECTED_DIFF_FILES},
        }]
        (control_dir / "arm_map.json").write_text(json.dumps(arm_map))
        (control_dir / f"versions_{CYCLE}.json").write_text(json.dumps({
            "host_native_kernel_sha": host_sha, "host_tree_dirty": False,
            "bridge_arm": "B", "bridge_sha": "67730a1ecf646544825fb60c00129cf46de6d307",
            "running_image_id": "sha256:B",
            "opt_hashes": {f: "b" * 64 for f in pv.EXPECTED_DIFF_FILES},
            "supervisor_start_times": {"bridge": 200.0},
            "recreate_timestamp": 100.0,
        }))
        actual = [True] * 21
        if not compliant_ok:
            actual[5] = False
        (control_dir / f"prep_{CYCLE}.json").write_text(json.dumps({
            "expected_compliant21": [True] * 21, "actual_compliant21": actual,
        }))
        if write_start_variant:
            (control_dir / f"start_variant_{CYCLE}.json").write_text(json.dumps({
                "cycle": CYCLE, "start_variant": "stiff-zero",
                "pose": {j: 0.0 for j in R.R_JOINTS},
            }))

    def _run_with_gates(self, ev_dir, control_dir, arm, out, **extra):
        argv = [
            "--ev-dir", str(ev_dir), "--control-dir", str(control_dir), "--cycle", CYCLE,
            "--rep", "1", "--arm", arm, "--arm-map", str(control_dir / "arm_map.json"),
            "--expected-host-sha", "M-sha", "--out", str(out),
        ]
        for k, v in extra.items():
            argv += [f"--{k.replace('_', '-')}", str(v)]
        return cyc._cli(argv)

    def test_wrong_host_sha_stops_or_incomplete(self, tmp_path):
        ev_dir, control_dir = _build_cycle(tmp_path)
        self._write_gate_files(control_dir, host_sha="NOT-M")
        out = tmp_path / "out.json"
        rc = self._run_with_gates(ev_dir, control_dir, "B", out)
        assert rc in (RC_STOP, RC_INCONCLUSIVE), read_result(out)

    def test_compliance_vector_flip_stops(self, tmp_path):
        ev_dir, control_dir = _build_cycle(tmp_path)
        self._write_gate_files(control_dir, compliant_ok=False)
        out = tmp_path / "out.json"
        rc = self._run_with_gates(ev_dir, control_dir, "B", out)
        payload = read_result(out)
        assert payload["verdict"] == cyc.VERDICT_STOP, payload

    def test_missing_start_variant_gives_rc3(self, tmp_path):
        ev_dir, control_dir = _build_cycle(tmp_path)
        self._write_gate_files(control_dir, write_start_variant=False)
        out = tmp_path / "out.json"
        rc = self._run_with_gates(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE, read_result(out)

    def test_matching_gates_pass(self, tmp_path):
        ev_dir, control_dir = _build_cycle(tmp_path)
        self._write_gate_files(control_dir)
        out = tmp_path / "out.json"
        rc = self._run_with_gates(ev_dir, control_dir, "B", out)
        payload = read_result(out)
        assert payload["verdict"] == cyc.VERDICT_OK, payload


class TestValidationModeRefusal:
    def test_refused_under_b4_goalfix_cmp_prefix(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)
        ev_dir = fake_home / "b4-goalfix-cmp-s1-2026-09-24" / "ev"
        ev_dir.mkdir(parents=True)
        out = tmp_path / "out.json"
        rc = cyc._cli([
            "--ev-dir", str(ev_dir), "--control-dir", str(tmp_path), "--cycle", CYCLE,
            "--rep", "1", "--arm", "B", "--out", str(out), "--validation-mode",
        ])
        assert rc == RC_INCONCLUSIVE
        assert not read_result(out)["ok"]

    def test_always_exits_3_even_when_ok(self, tmp_path):
        ev_dir, control_dir = _build_cycle(tmp_path)
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        payload = read_result(out)
        assert rc == RC_INCONCLUSIVE
        assert payload["validation_only"] is True
        assert payload["verdict"] == cyc.VERDICT_OK  # the underlying verdict is still recorded
