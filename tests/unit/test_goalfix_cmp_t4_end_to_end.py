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
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import initial  # noqa: E402
from tools.goalfix_cmp import provenance as pv  # noqa: E402
from tools.goalfix_cmp._io import IntegrityError, RC_INCONCLUSIVE, RC_OK, RC_STOP, read_result  # noqa: E402
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

    def _build_truncated_flight_cycle(self, tmp_path, k=10):
        """Probe P7: drops the flight leg's own last `k` states.jsonl
        rows (their commands then have no state reporting them
        applied) and fixes up the flight sidecar's own last alignment
        entry so `leg_from_sidecar` still finds a valid server_seq."""
        import hashlib
        import json as _json

        ev_dir, control_dir = _build_cycle(tmp_path)
        states_path = ev_dir / "states.jsonl"
        rows = [_json.loads(line) for line in states_path.read_text().splitlines() if line.strip()]
        truncated = rows[:-k]
        states_path.write_text("\n".join(_json.dumps(r) for r in truncated) + "\n")
        sums_path = ev_dir / "SHA256SUMS"
        lines = sums_path.read_text().splitlines()
        new_lines = []
        for line in lines:
            h, name = line.split(None, 1)
            if name == "states.jsonl":
                h = hashlib.sha256(states_path.read_bytes()).hexdigest()
            new_lines.append(f"{h}  {name}")
        sums_path.write_text("\n".join(new_lines) + "\n")
        flight_sidecar = control_dir / f"{CYCLE}-flight.link.json"
        sc = _json.loads(flight_sidecar.read_text())
        sc["alignment"][-1]["server_seq"] = truncated[-1]["seq"]
        flight_sidecar.write_text(_json.dumps(sc))
        return ev_dir, control_dir

    def test_mb5_unplaceable_trailing_flight_commands_gives_rc3(self, tmp_path):
        """MB5 (merge verdict, 2026-09-25 stage-repairs assignment §3;
        probe P7): the flight leg's own last 10 commands have no state
        reporting them applied -- `commands_in_leg` silently SKIPS
        unplaceable commands when building `leg.command_indices`, so
        they were never checked against any range at all, and
        da3a81c's CLI gives verdict `ok` (rc 3 here only because
        `--validation-mode` always forces it, masking the verdict --
        checked directly below) while the payload's own
        unplaceable_command_indices lists exactly them."""
        ev_dir, control_dir = self._build_truncated_flight_cycle(tmp_path)
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE
        payload = read_result(out)
        # The fix short-circuits with an EvidenceError before any
        # verdict is computed at all -- never a verdict of "ok".
        assert payload.get("verdict") != cyc.VERDICT_OK, payload
        assert "unplaceable" in payload.get("reason", ""), payload

    def test_mutation_unplaceable_gate_removed_would_authorize(self, tmp_path):
        """Mutation (drop the `_check_no_unplaceable_in_legs` call from
        the CLI, as at da3a81c): the SAME truncated fixture above gives
        verdict `ok` with the unplaceable indices merely reported, not
        gated -- verified directly against a da3a81c-equivalent
        (commit 922b5f7, the last head before MB5) copy of cycle.py,
        which returns exactly `{"verdict": "ok", "reasons": [], ...,
        "unplaceable_command_indices": [...]}` for this fixture."""
        ev_dir, control_dir = self._build_truncated_flight_cycle(tmp_path)
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        payload = read_result(out)
        # The shipped (fixed) CLI disagrees with the pre-MB5 CLI's own
        # verdict "ok" for this exact fixture.
        assert payload.get("verdict") != cyc.VERDICT_OK, payload


class TestGatesThroughTheCli:
    """The provenance/compliance/start_variant gates, exercised WITHOUT
    --validation-mode -- so the CLI reads versions_<cycle>.json,
    prep_<cycle>.json and start_variant_<cycle>.json from control_dir."""

    #: MB2 (merge verdict, 2026-09-25 stage-repairs assignment §3):
    #: `validate_arm_map` now runs unconditionally (not just when the
    #: `provenance arm-map` CLI is invoked directly) and requires a
    #: complete, 12-rep map in `pv.ARM_MAP_ORDER`'s own order
    #: ("ABBABAABABBA") -- rep 2 is the first 'B' slot, so these B-arm
    #: gate tests use rep 2, not rep 1 (which the order fixes as 'A').
    GATE_REP = 2
    BRIDGE_SHA_A = "8c0dad2d62dde791c8912fa723b8c2b7f190e1e8"
    BRIDGE_SHA_B = "67730a1ecf646544825fb60c00129cf46de6d307"

    def _write_gate_files(self, control_dir, *, host_sha="M-sha", compliant_ok=True,
                           write_start_variant=True):
        arm_map = [
            {"rep": i + 1, "arm": pv.ARM_MAP_ORDER[i],
             "image_tag": "img-A" if pv.ARM_MAP_ORDER[i] == "A" else "img-B",
             "image_id": "sha256:A" if pv.ARM_MAP_ORDER[i] == "A" else "sha256:B",
             "bridge_sha": self.BRIDGE_SHA_A if pv.ARM_MAP_ORDER[i] == "A" else self.BRIDGE_SHA_B,
             "opt_hashes": {f: ("a" * 64 if pv.ARM_MAP_ORDER[i] == "A" else "b" * 64)
                            for f in pv.EXPECTED_DIFF_FILES}}
            for i in range(12)
        ]
        (control_dir / "arm_map.json").write_text(json.dumps(arm_map))
        (control_dir / f"versions_{CYCLE}.json").write_text(json.dumps({
            "host_native_kernel_sha": host_sha, "host_tree_dirty": False,
            "bridge_arm": "B", "bridge_sha": self.BRIDGE_SHA_B,
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
            "--rep", str(self.GATE_REP), "--arm", arm, "--arm-map", str(control_dir / "arm_map.json"),
            "--expected-host-sha", "M-sha",
            "--required-supervisor-programs", "bridge",
            "--expected-bridge-sha-a", self.BRIDGE_SHA_A,
            "--expected-bridge-sha-b", self.BRIDGE_SHA_B,
            "--out", str(out),
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


class TestMB2ProvenanceGate(TestGatesThroughTheCli):
    """MB2 (merge verdict, 2026-09-25 stage-repairs assignment §3):
    P3-P5c, through the shipped `cycle` CLI. Each demonstrated `rc 0`
    at `da3a81c`; each is `rc 3`/STOP here (T6: an operator-input error
    is `evidence_incomplete`; a data-vs-map contradiction is a failed
    provenance gate -> STOP for B)."""

    def test_p3_missing_expected_host_sha_flag_is_rc3(self, tmp_path):
        ev_dir, control_dir = _build_cycle(tmp_path)
        self._write_gate_files(control_dir)
        out = tmp_path / "out.json"
        argv = [
            "--ev-dir", str(ev_dir), "--control-dir", str(control_dir), "--cycle", CYCLE,
            "--rep", str(self.GATE_REP), "--arm", "B", "--arm-map", str(control_dir / "arm_map.json"),
            "--required-supervisor-programs", "bridge",
            "--expected-bridge-sha-a", self.BRIDGE_SHA_A,
            "--expected-bridge-sha-b", self.BRIDGE_SHA_B,
            "--out", str(out),
        ]
        rc = cyc._cli(argv)
        assert rc == RC_INCONCLUSIVE, read_result(out)

    def test_p4_supervisor_list_mismatch_stops(self, tmp_path):
        """The document's own supervisor_start_times key is irrelevant --
        the CLI's `--required-supervisor-programs` names the program;
        naming one absent from the document must fail, never derive the
        requirement FROM the document (the P4 bug: previously
        `list(doc["supervisor_start_times"])`, so any document passed
        trivially)."""
        ev_dir, control_dir = _build_cycle(tmp_path)
        self._write_gate_files(control_dir)
        out = tmp_path / "out.json"
        rc = self._run_with_gates(ev_dir, control_dir, "B", out,
                                   **{"required_supervisor_programs": "not_in_doc"})
        payload = read_result(out)
        assert payload["verdict"] == cyc.VERDICT_STOP, payload

    def test_p4b_missing_recreate_timestamp_stops_or_incomplete(self, tmp_path):
        ev_dir, control_dir = _build_cycle(tmp_path)
        self._write_gate_files(control_dir)
        doc_path = control_dir / f"versions_{CYCLE}.json"
        doc = json.loads(doc_path.read_text())
        doc.pop("recreate_timestamp")
        doc_path.write_text(json.dumps(doc))
        out = tmp_path / "out.json"
        rc = self._run_with_gates(ev_dir, control_dir, "B", out)
        assert rc in (RC_STOP, RC_INCONCLUSIVE), read_result(out)

    def test_p5_arm_flag_contradicts_arm_map_stops_or_incomplete(self, tmp_path):
        ev_dir, control_dir = _build_cycle(tmp_path)
        self._write_gate_files(control_dir)
        out = tmp_path / "out.json"
        # GATE_REP (2) is 'B' in ARM_MAP_ORDER; pass --arm A instead.
        rc = self._run_with_gates(ev_dir, control_dir, "A", out)
        assert rc in (RC_STOP, RC_INCONCLUSIVE), read_result(out)
        payload = read_result(out)
        assert payload.get("verdict") != cyc.VERDICT_MANIPULATED

    def test_p5c_versions_bridge_fields_contradict_arm_map_stops(self, tmp_path):
        ev_dir, control_dir = _build_cycle(tmp_path)
        self._write_gate_files(control_dir)
        doc_path = control_dir / f"versions_{CYCLE}.json"
        doc = json.loads(doc_path.read_text())
        doc["bridge_arm"] = "A"
        doc["bridge_sha"] = self.BRIDGE_SHA_A
        doc_path.write_text(json.dumps(doc))
        out = tmp_path / "out.json"
        rc = self._run_with_gates(ev_dir, control_dir, "B", out)
        payload = read_result(out)
        assert payload["verdict"] == cyc.VERDICT_STOP, payload

    def test_mutation_arm_map_validation_removed_would_authorize(self, tmp_path):
        """Mutation (drop the `validate_arm_map`/entry cross-check block
        entirely, falling back to the pre-MB2 `versions_<cycle>.json`
        read as the only source of truth): the P5c fixture above would
        pass silently, since nothing there ever reads bridge_arm/
        bridge_sha against the map. Verified directly against a
        `da3a81c` copy of cycle.py, which returns rc 0 `ok` for this
        exact fixture (before MB1's own unrelated gate-completeness fix
        even applies -- MB1 is orthogonal and was independently killed
        in the same run)."""
        ev_dir, control_dir = _build_cycle(tmp_path)
        self._write_gate_files(control_dir)
        doc_path = control_dir / f"versions_{CYCLE}.json"
        doc = json.loads(doc_path.read_text())
        doc["bridge_arm"] = "A"
        doc["bridge_sha"] = self.BRIDGE_SHA_A
        doc_path.write_text(json.dumps(doc))
        out = tmp_path / "out.json"
        rc = self._run_with_gates(ev_dir, control_dir, "B", out)
        assert rc != cyc.RC_OK


class TestMB6NoEscapingExceptions(TestGatesThroughTheCli):
    """MB6 (merge verdict, 2026-09-25 stage-repairs assignment §3; probes
    P6a-c): malformed control JSON must give rc 3 JSON, never an
    untranslated traceback (invariant 1)."""

    def test_p6a_malformed_arm_map_gives_rc3_json(self, tmp_path):
        ev_dir, control_dir = _build_cycle(tmp_path)
        self._write_gate_files(control_dir)
        (control_dir / "arm_map.json").write_text("{not json")
        out = tmp_path / "out.json"
        rc = self._run_with_gates(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE
        assert "reason" in read_result(out)

    def test_p6b_arm_map_entry_missing_key_gives_rc3_json(self, tmp_path):
        ev_dir, control_dir = _build_cycle(tmp_path)
        self._write_gate_files(control_dir)
        arm_map_path = control_dir / "arm_map.json"
        entries = json.loads(arm_map_path.read_text())
        del entries[self.GATE_REP - 1]["image_id"]
        arm_map_path.write_text(json.dumps(entries))
        out = tmp_path / "out.json"
        rc = self._run_with_gates(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE
        assert "reason" in read_result(out)

    def test_p6c_malformed_versions_doc_gives_rc3_json(self, tmp_path):
        ev_dir, control_dir = _build_cycle(tmp_path)
        self._write_gate_files(control_dir)
        (control_dir / f"versions_{CYCLE}.json").write_text("[")
        out = tmp_path / "out.json"
        rc = self._run_with_gates(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE
        assert "reason" in read_result(out)

    def test_mutation_catch_all_removed_would_raise(self):
        """Mutation (remove the catch-all `except Exception`): a bare
        `json.JSONDecodeError` from malformed control JSON is not an
        instance of any of the specific exception types the CLI's OTHER
        except clause names -- so removing the catch-all lets it escape
        uncaught. Verified structurally (not by re-running a reverted
        CLI, to avoid a second real subprocess-equivalent invocation):
        none of the specific types below is a JSONDecodeError, and
        JSONDecodeError is not a subclass of any of them."""
        import json as _json

        specific = (ev.EvidenceError, IntegrityError, initial.StartVariantGateUnavailable)
        assert not issubclass(_json.JSONDecodeError, specific)


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
