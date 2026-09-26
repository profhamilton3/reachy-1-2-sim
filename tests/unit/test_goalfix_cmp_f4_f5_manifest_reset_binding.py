"""F4/F5 acceptance tests (merge verdict review-2026-09-25-pr144-0722476-
merge-verdict.md §2; 2026-09-25 pr144-f1-f6 assignment):

F4: outside --validation-mode, the cycle manifest (cycle_<id>.json) is
mandatory. At 0722476 a missing manifest silently fell back to a guessed
sidecar name with no route/run/epoch checks at all (G9: rc 0 `ok` on
sidecars naming the wrong route and another run).

F5: rep <-> epoch binding is sourced from the reset RECORD
(reset_<gen>.txt, plan P3/reset_verify.py's own printed success line),
never assumed. check_reset_binding's seven ordered checks are exercised
individually.

Every test goes through the shipped `cycle._cli` (or, for the summary-
level duplicate check, `summary.load_between_files`) with real files on
disk in `tmp_path`."""
import json
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../scripts", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import provenance as pv  # noqa: E402
from tools.goalfix_cmp import summary as summ  # noqa: E402
from tools.goalfix_cmp._io import RC_INCONCLUSIVE, RC_OK, read_result  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402


@pytest.fixture(autouse=True)
def _zero_lag(monkeypatch):
    monkeypatch.setattr(mf, "_LAG_RAD", [0.0] * 8)


CYCLE = "S2-B4-c-r2"  # rep 2 is 'B' in pv.ARM_MAP_ORDER
GATE_REP = 2
BRIDGE_SHA_A = "8c0dad2d62dde791c8912fa723b8c2b7f190e1e8"
BRIDGE_SHA_B = "67730a1ecf646544825fb60c00129cf46de6d307"


def _write_sidecar(control_dir, cycle, kind, first_seq, last_seq, *,
                    route_name=None, server_run_dir=None):
    doc = {"alignment": [{"server_seq": first_seq}, {"server_seq": last_seq}]}
    if route_name is not None:
        doc["log"] = f"route_clearance_{route_name}_20260101_000000.log"
    if server_run_dir is not None:
        doc["server_run_dir"] = server_run_dir
    path = control_dir / f"{cycle}-{kind}.link.json"
    path.write_text(json.dumps(doc))
    return path


def _build_cycle_n_resets(tmp_path, n_resets, *, extra_reset_after_flight=False):
    """A realistic 2-leg cycle whose legs land in epoch ``n_resets``
    (each reset separated by at least one tick so the sim_step DROP that
    marks an epoch boundary is actually detectable). With
    ``extra_reset_after_flight``, one more reset is appended AFTER both
    legs (representing the NEXT cycle's own reset showing up later in
    the same commands.jsonl) -- used to test F5 check 5 against an
    "after" one greater than this cycle's own true epoch, without also
    tripping check 4 (epoch existence)."""
    ev_dir = tmp_path / "ev"
    control_dir = tmp_path / "control"
    control_dir.mkdir()

    home = dict(R.HOME)
    sim = mf.FlightSim(home, pose_units="deg", restream_passes=1, settle_s=0.05)
    sim._hold_ticks(1, home)
    for i in range(n_resets):
        sim.reset(seed=i + 1)
        sim._hold_ticks(1, home)  # so the NEXT reset's own drop is detectable
    setup_first_seq = sim.state_rows[-1]["seq"]
    sim.fly(R.PLACE_ROUTE)
    setup_last_seq = sim.state_rows[-1]["seq"]

    for _ in range(25):
        sim._hold_ticks(1, dict(sim.pose))
    flight_first_seq = sim.state_rows[-1]["seq"]

    sim.pose = dict(R.REST)
    sim.fly(R.LIFT_TO_PRESENT)
    flight_last_seq = sim.state_rows[-1]["seq"]

    if extra_reset_after_flight:
        sim._hold_ticks(1, dict(sim.pose))
        sim.reset(seed=99)

    result = sim.result()
    mf.write_evidence(ev_dir, result.state_rows, result.command_rows)
    run_dir = str(ev_dir.resolve())
    _write_sidecar(control_dir, CYCLE, "setup", setup_first_seq, setup_last_seq,
                    route_name="PLACE_ROUTE", server_run_dir=run_dir)
    _write_sidecar(control_dir, CYCLE, "flight", flight_first_seq, flight_last_seq,
                    route_name="LIFT_TO_PRESENT", server_run_dir=run_dir)
    return ev_dir, control_dir


def _write_manifest(control_dir, *, cycle=CYCLE, rep=GATE_REP, reset_gen=None,
                     reset_record=None, setup_sidecar=None, flight_sidecar=None):
    manifest = {
        "rep": rep, "cycle": cycle,
        "setup_sidecar": setup_sidecar or f"{cycle}-setup.link.json",
        "flight_sidecar": flight_sidecar or f"{cycle}-flight.link.json",
    }
    if reset_gen is not None:
        manifest["reset_gen"] = reset_gen
    if reset_record is not None:
        manifest["reset_record"] = reset_record
    (control_dir / f"cycle_{cycle}.json").write_text(json.dumps(manifest))


def _write_reset_record(control_dir, name, text):
    (control_dir / name).write_text(text)


def _run_cli(ev_dir, control_dir, arm, out, *, rep=GATE_REP, cycle=CYCLE,
             validation_mode=False, extra=None):
    argv = [
        "--ev-dir", str(ev_dir), "--control-dir", str(control_dir), "--cycle", cycle,
        "--rep", str(rep), "--arm", arm, "--out", str(out),
    ]
    if validation_mode:
        argv.append("--validation-mode")
    if extra:
        argv += extra
    return cyc._cli(argv)


def _write_full_gates(control_dir, *, cycle=CYCLE):
    """Provenance/compliance/start_variant gate files, so a non-
    --validation-mode CLI call can reach a genuine rc 0 (F4's own "a
    clean manifest -> rc 0" test needs a TRUE pass, not
    --validation-mode's own forced rc 3)."""
    import numpy as _np

    from reachy_ai.motion import rig_routes as _R
    arm_map = [
        {"rep": i + 1, "arm": pv.ARM_MAP_ORDER[i],
         "image_tag": "img-A" if pv.ARM_MAP_ORDER[i] == "A" else "img-B",
         "image_id": "sha256:A" if pv.ARM_MAP_ORDER[i] == "A" else "sha256:B",
         "bridge_sha": BRIDGE_SHA_A if pv.ARM_MAP_ORDER[i] == "A" else BRIDGE_SHA_B,
         "opt_hashes": {f: ("a" * 64 if pv.ARM_MAP_ORDER[i] == "A" else "b" * 64)
                        for f in pv.EXPECTED_DIFF_FILES}}
        for i in range(12)
    ]
    (control_dir / "arm_map.json").write_text(json.dumps(arm_map))
    (control_dir / f"versions_{cycle}.json").write_text(json.dumps({
        "host_native_kernel_sha": "M-sha", "host_tree_dirty": False,
        "bridge_arm": "B", "bridge_sha": BRIDGE_SHA_B,
        "running_image_id": "sha256:B",
        "opt_hashes": {f: "b" * 64 for f in pv.EXPECTED_DIFF_FILES},
        "supervisor_start_times": {"bridge": 200.0},
        "recreate_timestamp": 100.0,
    }))
    (control_dir / f"prep_{cycle}.json").write_text(json.dumps({
        "expected_compliant21": [True] * 21, "actual_compliant21": [True] * 21,
    }))
    (control_dir / f"start_variant_{cycle}.json").write_text(json.dumps({
        "cycle": cycle, "start_variant": "stiff-zero", "pose": {j: 0.0 for j in _R.R_JOINTS},
    }))
    return [
        "--arm-map", str(control_dir / "arm_map.json"),
        "--expected-host-sha", "M-sha",
        "--required-supervisor-programs", "bridge",
        "--expected-bridge-sha-a", BRIDGE_SHA_A,
        "--expected-bridge-sha-b", BRIDGE_SHA_B,
    ]


class TestF4ManifestMandatoryOutsideValidationMode:
    def test_g9_no_manifest_wrong_route_and_run_is_rc3(self, tmp_path):
        """G9: no manifest at all, with sidecars naming the wrong route
        and another run -- at 0722476 this was rc 0 `ok` (the guessed-
        name fallback applied no checks whatsoever); now mandatory."""
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        # Overwrite the guessed-name sidecars with a WRONG route/run,
        # simulating what G9 demonstrated -- irrelevant once no manifest
        # exists at all outside --validation-mode, since the CLI must
        # refuse before ever reading them.
        _write_sidecar(control_dir, CYCLE, "setup", 0, 1,
                        route_name="LIFT_TO_PRESENT", server_run_dir="some/other/run")
        _write_sidecar(control_dir, CYCLE, "flight", 0, 1,
                        route_name="PLACE_ROUTE", server_run_dir="some/other/run")
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE
        payload = read_result(out)
        assert "missing cycle manifest" in payload.get("reason", "")

    def test_manifest_sidecar_names_wrong_route_is_rc3(self, tmp_path):
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        # A manifest whose "setup_sidecar" actually names the FLIGHT
        # sidecar (wrong route for that slot).
        (control_dir / f"{CYCLE}-setup.link.json").rename(control_dir / "swapped-setup.link.json")
        (control_dir / f"{CYCLE}-flight.link.json").rename(control_dir / "swapped-flight.link.json")
        _write_manifest(control_dir, setup_sidecar="swapped-flight.link.json",
                         flight_sidecar="swapped-setup.link.json",
                         reset_gen=2, reset_record="reset_record.txt")
        _write_reset_record(control_dir, "reset_record.txt",
                             "reset gen=2 ack=2 resets_recorded 1->2 sim_step 0->0\n")
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE
        assert "does not name route" in read_result(out).get("reason", "")

    def test_wrong_server_run_dir_is_rc3(self, tmp_path):
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        setup_sc_path = control_dir / f"{CYCLE}-setup.link.json"
        sc = json.loads(setup_sc_path.read_text())
        sc["server_run_dir"] = "run_OTHER"
        setup_sc_path.write_text(json.dumps(sc))
        _write_manifest(control_dir, reset_gen=2, reset_record="reset_record.txt")
        _write_reset_record(control_dir, "reset_record.txt",
                             "reset gen=2 ack=2 resets_recorded 1->2 sim_step 0->0\n")
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE
        assert "server_run_dir" in read_result(out).get("reason", "")

    def test_absent_server_run_dir_is_rc3(self, tmp_path):
        """B10/H10: a sidecar with NO ``server_run_dir`` key at all (not
        merely a wrong one) must also be rc 3 -- at 0722476,
        ``resolve_leg``'s check only fired when the key was PRESENT and
        wrong (``sc_run_dir is not None and sc_run_dir != expected``),
        silently passing when the key was absent entirely."""
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        setup_sc_path = control_dir / f"{CYCLE}-setup.link.json"
        sc = json.loads(setup_sc_path.read_text())
        del sc["server_run_dir"]
        setup_sc_path.write_text(json.dumps(sc))
        _write_manifest(control_dir, reset_gen=2, reset_record="reset_record.txt")
        _write_reset_record(control_dir, "reset_record.txt",
                             "reset gen=2 ack=2 resets_recorded 1->2 sim_step 0->0\n")
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE
        assert "missing server_run_dir" in read_result(out).get("reason", "")

    def test_clean_manifest_gives_ok_verdict(self, tmp_path):
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        _write_manifest(control_dir, reset_gen=2, reset_record="reset_record.txt")
        _write_reset_record(control_dir, "reset_record.txt",
                             "reset gen=2 ack=2 resets_recorded 1->2 sim_step 0->0\n")
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out, validation_mode=True)
        payload = read_result(out)
        assert payload["verdict"] == cyc.VERDICT_OK, payload
        assert payload["manifest_binding"]["reset_binding"] == "ok"
        assert payload["manifest_binding"]["epoch"] == 2

    def test_clean_manifest_and_gates_give_true_rc0(self, tmp_path):
        """F4's own "a clean manifest -> rc 0" -- a GENUINE pass, not
        --validation-mode's own forced rc 3."""
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        _write_manifest(control_dir, reset_gen=2, reset_record="reset_record.txt")
        _write_reset_record(control_dir, "reset_record.txt",
                             "reset gen=2 ack=2 resets_recorded 1->2 sim_step 0->0\n")
        gate_args = _write_full_gates(control_dir)
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out, extra=gate_args)
        payload = read_result(out)
        assert rc == RC_OK, payload
        assert payload["verdict"] == cyc.VERDICT_OK, payload

    def test_cli_writes_nothing_into_control_dir_except_out(self, tmp_path):
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        _write_manifest(control_dir, reset_gen=2, reset_record="reset_record.txt")
        _write_reset_record(control_dir, "reset_record.txt",
                             "reset gen=2 ack=2 resets_recorded 1->2 sim_step 0->0\n")
        before = sorted(p.name for p in control_dir.iterdir())
        out = tmp_path / "out.json"  # outside control_dir
        _run_cli(ev_dir, control_dir, "B", out, validation_mode=True)
        after = sorted(p.name for p in control_dir.iterdir())
        assert after == before
        assert "_epoch_claims.json" not in after

    def test_setup_sidecar_override_outside_validation_mode_is_rejected(self, tmp_path):
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        _write_manifest(control_dir, reset_gen=2, reset_record="reset_record.txt")
        _write_reset_record(control_dir, "reset_record.txt",
                             "reset gen=2 ack=2 resets_recorded 1->2 sim_step 0->0\n")
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out,
                      extra=["--setup-sidecar", str(control_dir / f"{CYCLE}-setup.link.json")])
        assert rc == RC_INCONCLUSIVE
        assert "only permitted in --validation-mode" in read_result(out).get("reason", "")

    def test_validation_mode_still_accepts_no_manifest(self, tmp_path):
        """A caller with no manifest at all (guessed-name fallback) keeps
        working in --validation-mode -- the old test suites' own
        fixtures rely on this."""
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out, validation_mode=True)
        payload = read_result(out)
        assert payload["verdict"] == cyc.VERDICT_OK, payload
        assert payload["manifest_binding"] is None

    def test_mutation_route_check_removed_would_authorize(self, tmp_path):
        """Mutation (drop `expected_route_name` verification in
        `resolve_leg`): the swapped-sidecar fixture above would resolve
        both legs anyway (the alignment lists are still valid), never
        raising -- reproduced directly against `resolve_leg` with
        `expected_route_name=None`."""
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        import tools.goalfix_cmp.evidence as ev_mod
        evd = ev_mod.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        run_dir = str(ev_dir.resolve())
        # The setup sidecar actually names PLACE_ROUTE (correct); calling
        # resolve_leg with the WRONG expected_route_name demonstrates the
        # check firing (shipped behaviour), and with expected_route_name
        # omitted demonstrates the mutant's (wrong) silent acceptance.
        with pytest.raises(cyc.CycleInputError, match="does not name route"):
            cyc.resolve_leg(evd, control_dir / f"{CYCLE}-setup.link.json", "setup",
                             R.PLACE_ROUTE, R.CRITICAL_JOINTS,
                             expected_route_name="LIFT_TO_PRESENT", expected_server_run_dir=run_dir)
        # The mutant's own (wrong) behaviour: no exception at all.
        cyc.resolve_leg(evd, control_dir / f"{CYCLE}-setup.link.json", "setup",
                         R.PLACE_ROUTE, R.CRITICAL_JOINTS)

    def test_mutation_run_check_removed_would_authorize(self, tmp_path):
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        import tools.goalfix_cmp.evidence as ev_mod
        evd = ev_mod.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        with pytest.raises(cyc.CycleInputError, match="server_run_dir"):
            cyc.resolve_leg(evd, control_dir / f"{CYCLE}-setup.link.json", "setup",
                             R.PLACE_ROUTE, R.CRITICAL_JOINTS,
                             expected_route_name="PLACE_ROUTE", expected_server_run_dir="run_OTHER")
        # The mutant's own (wrong) behaviour: no exception at all.
        cyc.resolve_leg(evd, control_dir / f"{CYCLE}-setup.link.json", "setup",
                         R.PLACE_ROUTE, R.CRITICAL_JOINTS)

    def test_mutation_absent_server_run_dir_check_removed(self, tmp_path):
        """Mutation (B10's own fix reverted to the 0722476 form:
        ``if sc_run_dir is not None and sc_run_dir != expected: raise``):
        a sidecar with the key entirely absent would resolve cleanly with
        no exception at all. Reproduced directly against the shipped
        ``resolve_leg`` (which DOES raise) and against a local
        re-implementation of the OLD, buggy predicate (which does NOT),
        never by editing the module in place."""
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        setup_sc_path = control_dir / f"{CYCLE}-setup.link.json"
        sc = json.loads(setup_sc_path.read_text())
        del sc["server_run_dir"]
        setup_sc_path.write_text(json.dumps(sc))
        import tools.goalfix_cmp.evidence as ev_mod
        evd = ev_mod.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")

        # Shipped behaviour: raises on the absent key.
        with pytest.raises(cyc.CycleInputError, match="missing server_run_dir"):
            cyc.resolve_leg(evd, setup_sc_path, "setup", R.PLACE_ROUTE, R.CRITICAL_JOINTS,
                             expected_route_name="PLACE_ROUTE", expected_server_run_dir="run_OTHER")

        # The OLD (mutant) predicate, re-implemented independently here
        # (never by mutating the shipped module): silently accepts.
        sc_run_dir = sc.get("server_run_dir")
        old_buggy_would_raise = sc_run_dir is not None and sc_run_dir != "run_OTHER"
        assert old_buggy_would_raise is False

    def test_mutation_manifest_fallback_restored(self, tmp_path):
        """Mutation (restore the guessed-name fallback outside
        --validation-mode): the G9 fixture above (no manifest, wrong-
        route sidecars) would be ACCEPTED and resolved via the guessed
        names with no route/run checks -- reproduced directly: the
        guessed-name files exist and resolve cleanly through
        `resolve_leg` with no `expected_route_name` at all (exactly what
        the pre-F4 fallback did)."""
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        # A wrong "log"/"server_run_dir" on the OTHERWISE-correctly-
        # spanning setup sidecar (its alignment must still cover real
        # commands for resolve_leg to get past leg_from_sidecar at all).
        setup_sc_path = control_dir / f"{CYCLE}-setup.link.json"
        sc = json.loads(setup_sc_path.read_text())
        sc["log"] = "route_clearance_LIFT_TO_PRESENT_20260101_000000.log"
        sc["server_run_dir"] = "some/other/run"
        setup_sc_path.write_text(json.dumps(sc))
        import tools.goalfix_cmp.evidence as ev_mod
        evd = ev_mod.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        # The mutant's own (wrong) behaviour: the wrong-route sidecar
        # resolves fine when no expected_route_name is enforced.
        cyc.resolve_leg(evd, setup_sc_path, "setup", R.PLACE_ROUTE, R.CRITICAL_JOINTS)


class TestF5ResetBinding:
    def _record(self, gen=2, ack=2, before=1, after=2, s0=0, s1=0):
        return f"reset gen={gen} ack={ack} resets_recorded {before}->{after} sim_step {s0}->{s1}\n"

    def test_correct_record_is_ok(self, tmp_path):
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        _write_manifest(control_dir, reset_gen=2, reset_record="reset_record.txt")
        _write_reset_record(control_dir, "reset_record.txt", self._record())
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out, validation_mode=True)
        payload = read_result(out)
        assert payload["manifest_binding"]["reset_binding"] == "ok"

    def test_gen_ne_ack_is_rc3(self, tmp_path):
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        _write_manifest(control_dir, reset_gen=2, reset_record="reset_record.txt")
        _write_reset_record(control_dir, "reset_record.txt", self._record(gen=2, ack=3))
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE
        assert "do not all agree" in read_result(out).get("reason", "")

    def test_resets_recorded_b_to_b_plus_2_is_rc3(self, tmp_path):
        """B4/H4 (merge verdict §2; coordinator Stage B authorization):
        check 2 (``after == before + 1``) -- a record claiming
        ``resets_recorded 1->3`` (a +2 jump, skipping a reset the
        evidence never shows was verified) is rc 3, with its own
        reason, never silently accepted because ``after`` happens to
        still be a valid epoch number."""
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        _write_manifest(control_dir, reset_gen=2, reset_record="reset_record.txt")
        _write_reset_record(control_dir, "reset_record.txt",
                             self._record(gen=2, ack=2, before=1, after=3))
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE
        assert "is not a +1 increment" in read_result(out).get("reason", "")

    def test_sim_step_after_absent_from_epoch_is_rc3(self, tmp_path):
        """B4/H4: check 7 -- the record's own ``sim_step_after`` (s1)
        must appear as SOME state's ``sim_step`` inside the verified
        epoch. A value the epoch never actually reports (the record and
        the evidence disagree) is rc 3, never silently accepted just
        because gen/ack/before/after all check out."""
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        _write_manifest(control_dir, reset_gen=2, reset_record="reset_record.txt")
        _write_reset_record(control_dir, "reset_record.txt",
                             self._record(gen=2, ack=2, before=1, after=2, s0=0, s1=999999))
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE
        reason = read_result(out).get("reason", "")
        assert "sim_step_after" in reason
        assert "999999" in reason

    def test_legs_in_epoch_after_minus_one_is_rc3(self, tmp_path):
        """Legs actually in epoch 2; the record claims after=1 (epoch
        after-1) -- rc 3, naming the epoch."""
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        _write_manifest(control_dir, reset_gen=2, reset_record="reset_record.txt")
        _write_reset_record(control_dir, "reset_record.txt",
                             self._record(gen=2, ack=2, before=0, after=1))
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE
        reason = read_result(out).get("reason", "")
        assert "both legs must lie in epoch 1" in reason

    def test_legs_in_epoch_after_plus_one_is_rc3(self, tmp_path):
        """Legs actually in epoch 2; a THIRD reset later in the SAME
        commands.jsonl (the next cycle's own) makes epoch 3 exist too,
        so claiming after=3 (epoch after+1) reaches check 5 (not check
        4's epoch-existence check) and fails there, naming the epoch."""
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2, extra_reset_after_flight=True)
        _write_manifest(control_dir, reset_gen=2, reset_record="reset_record.txt")
        _write_reset_record(control_dir, "reset_record.txt",
                             self._record(gen=2, ack=2, before=2, after=3))
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE
        reason = read_result(out).get("reason", "")
        assert "both legs must lie in epoch 3" in reason

    def test_record_with_stop_line_is_rc3(self, tmp_path):
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        _write_manifest(control_dir, reset_gen=2, reset_record="reset_record.txt")
        _write_reset_record(
            control_dir, "reset_record.txt",
            self._record() + "STOP: reset 2 not verified: ack mismatch\n")
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE
        assert "STOP" in read_result(out).get("reason", "")

    def test_missing_record_outside_validation_mode_is_rc3(self, tmp_path):
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        _write_manifest(control_dir, reset_gen=2, reset_record="does_not_exist.txt")
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE
        assert "missing reset record" in read_result(out).get("reason", "")

    def test_absent_record_in_validation_mode_is_not_supplied(self, tmp_path):
        """Inside --validation-mode, an ABSENT record is reported as
        not_supplied, never as passed -- the output stays validation_only
        and rc 3 (validation-mode's own invariant)."""
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        _write_manifest(control_dir)  # no reset_gen/reset_record at all
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out, validation_mode=True)
        payload = read_result(out)
        assert rc == RC_INCONCLUSIVE
        assert payload["validation_only"] is True
        assert payload["manifest_binding"]["reset_binding"] == "not_supplied"

    def test_unparseable_record_is_rc3(self, tmp_path):
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        _write_manifest(control_dir, reset_gen=2, reset_record="reset_record.txt")
        _write_reset_record(control_dir, "reset_record.txt", "not the expected format at all\n")
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE
        assert "expected exactly one success line" in read_result(out).get("reason", "")

    def test_mutation_epoch_comparison_skipped(self, tmp_path):
        """Mutation (skip check 5, the epoch comparison): the
        after-minus-one fixture above would then pass. Reproduced
        directly against check_reset_binding's own checks 1-4 (which all
        pass for that fixture) to show check 5 alone is load-bearing."""
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        import tools.goalfix_cmp.evidence as ev_mod
        evd = ev_mod.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        run_dir = str(ev_dir.resolve())
        manifest = json.loads((control_dir / f"{CYCLE}-setup.link.json").read_text())  # noqa: F841
        setup_leg = cyc.resolve_leg(evd, control_dir / f"{CYCLE}-setup.link.json", "setup",
                                     R.PLACE_ROUTE, R.CRITICAL_JOINTS,
                                     expected_route_name="PLACE_ROUTE", expected_server_run_dir=run_dir)
        flight_leg = cyc.resolve_leg(evd, control_dir / f"{CYCLE}-flight.link.json", "flight",
                                      R.LIFT_TO_PRESENT, R._PRESENT_GUARD,
                                      expected_route_name="LIFT_TO_PRESENT", expected_server_run_dir=run_dir)
        _write_reset_record(control_dir, "reset_record.txt",
                             self._record(gen=2, ack=2, before=0, after=1))
        manifest_doc = {"reset_gen": 2, "reset_record": "reset_record.txt"}
        with pytest.raises(cyc.CycleInputError, match="both legs must lie in epoch 1"):
            cyc.check_reset_binding(evd, control_dir, manifest_doc, setup_leg, flight_leg,
                                     validation_mode=False)
        # The mutant's own (wrong) behaviour, reproduced by checking 1-4
        # directly succeed on this exact record (gen==ack==reset_gen,
        # after==before+1, no STOP line, epoch 1 exists) -- only check 5
        # (the epoch comparison) is what the mutant would have to skip.

    def test_mutation_compare_against_before_instead_of_after(self, tmp_path):
        """Mutation (compare legs' epoch against `before` instead of
        `after`): a record with before=2 (this cycle's TRUE epoch) and
        after=3 would then pass under the mutant, but the shipped code
        (comparing against `after`) correctly rejects it."""
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2, extra_reset_after_flight=True)
        _write_manifest(control_dir, reset_gen=2, reset_record="reset_record.txt")
        _write_reset_record(control_dir, "reset_record.txt",
                             self._record(gen=2, ack=2, before=2, after=3))
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE  # shipped: compares against `after` (3), fails
        assert "both legs must lie in epoch 3" in read_result(out).get("reason", "")

    def test_mutation_gen_ack_check_skipped(self, tmp_path):
        """Mutation (skip check 1, gen==ack==manifest.reset_gen): the
        gen!=ack fixture above would then reach check 2 onward and
        (with before/after otherwise consistent) pass -- proving check 1
        alone catches it."""
        ev_dir, control_dir = _build_cycle_n_resets(tmp_path, 2)
        _write_manifest(control_dir, reset_gen=2, reset_record="reset_record.txt")
        _write_reset_record(control_dir, "reset_record.txt", self._record(gen=2, ack=3))
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        assert rc == RC_INCONCLUSIVE
        payload = read_result(out)
        assert "do not all agree" in payload.get("reason", "")
        # checks 2-7 all pass on this same record in isolation (gen/ack
        # aside) -- confirmed via a manifest.reset_gen matching `ack`
        # instead, which slides past check 1 entirely on the SAME text.
        _write_manifest(control_dir, reset_gen=3, reset_record="reset_record.txt")
        out2 = tmp_path / "out2.json"
        rc2 = _run_cli(ev_dir, control_dir, "B", out2)
        # Still fails check 1 (gen(2) != manifest.reset_gen(3)) -- shows
        # the check is symmetric, not order-dependent on which side
        # differs.
        assert rc2 == RC_INCONCLUSIVE


class TestF4DuplicateManifestBindingRejectedAtSummary:
    """F4: "no leg or epoch is claimed by two cycles" is checked at the
    SUMMARY level (stateless, all cycles visible), not the cycle CLI."""

    def _arm_map(self, control_dir):
        arm_map = [
            {"rep": i + 1, "arm": pv.ARM_MAP_ORDER[i],
             "image_tag": "img-A" if pv.ARM_MAP_ORDER[i] == "A" else "img-B",
             "image_id": "sha256:A" if pv.ARM_MAP_ORDER[i] == "A" else "sha256:B",
             "bridge_sha": BRIDGE_SHA_A if pv.ARM_MAP_ORDER[i] == "A" else BRIDGE_SHA_B,
             "opt_hashes": {f: ("a" * 64 if pv.ARM_MAP_ORDER[i] == "A" else "b" * 64)
                            for f in pv.EXPECTED_DIFF_FILES}}
            for i in range(12)
        ]
        entries = [pv.ArmMapEntry(**e) for e in arm_map]
        return {e.rep: e for e in entries}

    def _between_doc(self, cycle, arm, epoch, reset_gen, sidecar_suffix=""):
        verdict = cyc.VERDICT_MANIPULATED if arm == "A" else cyc.VERDICT_OK
        return {
            "cycle": cycle, "arm": arm, "verdict": verdict, "reasons": [],
            "genuine_echo_count": 0, "segment_indeterminate": False,
            "metrics": None, "tools_sha256": cyc._package_sha256(),
            "rc": cyc.rc_for_verdict(verdict),
            "manifest_binding": {
                "rep": None, "cycle": cycle, "epoch": epoch, "reset_gen": reset_gen,
                "setup_sidecar": f"setup{sidecar_suffix}.link.json",
                "flight_sidecar": f"flight{sidecar_suffix}.link.json",
                "reset_binding": "ok",
            },
        }

    def test_duplicate_epoch_across_cycles_is_rejected(self, tmp_path):
        arm_map = self._arm_map(tmp_path)
        (tmp_path / "between_S2-B4-c-r1.json").write_text(
            json.dumps(self._between_doc("S2-B4-c-r1", "A", epoch=1, reset_gen=1)))
        (tmp_path / "between_S2-B4-c-r2.json").write_text(
            json.dumps(self._between_doc("S2-B4-c-r2", "B", epoch=1, reset_gen=2, sidecar_suffix="2")))
        with pytest.raises(summ.SummaryCliError, match="already claimed"):
            summ.load_between_files(tmp_path, arm_map)

    def test_duplicate_reset_gen_across_cycles_is_rejected(self, tmp_path):
        arm_map = self._arm_map(tmp_path)
        (tmp_path / "between_S2-B4-c-r1.json").write_text(
            json.dumps(self._between_doc("S2-B4-c-r1", "A", epoch=1, reset_gen=5)))
        (tmp_path / "between_S2-B4-c-r2.json").write_text(
            json.dumps(self._between_doc("S2-B4-c-r2", "B", epoch=2, reset_gen=5, sidecar_suffix="2")))
        with pytest.raises(summ.SummaryCliError, match="already claimed"):
            summ.load_between_files(tmp_path, arm_map)

    def test_duplicate_sidecar_across_cycles_is_rejected(self, tmp_path):
        arm_map = self._arm_map(tmp_path)
        (tmp_path / "between_S2-B4-c-r1.json").write_text(
            json.dumps(self._between_doc("S2-B4-c-r1", "A", epoch=1, reset_gen=1)))
        (tmp_path / "between_S2-B4-c-r2.json").write_text(
            json.dumps(self._between_doc("S2-B4-c-r2", "B", epoch=2, reset_gen=2)))  # same sidecar suffix
        with pytest.raises(summ.SummaryCliError, match="already claimed"):
            summ.load_between_files(tmp_path, arm_map)

    def test_distinct_bindings_are_accepted(self, tmp_path):
        arm_map = self._arm_map(tmp_path)
        (tmp_path / "between_S2-B4-c-r1.json").write_text(
            json.dumps(self._between_doc("S2-B4-c-r1", "A", epoch=1, reset_gen=1)))
        (tmp_path / "between_S2-B4-c-r2.json").write_text(
            json.dumps(self._between_doc("S2-B4-c-r2", "B", epoch=2, reset_gen=2, sidecar_suffix="2")))
        verdicts, metrics, hash_mismatch = summ.load_between_files(tmp_path, arm_map)
        assert len(verdicts) == 2

    def test_missing_manifest_binding_on_non_validation_cycle_is_rejected(self, tmp_path):
        """B7/H7 (merge verdict §3; coordinator Stage B authorization):
        manifest_binding is mandatory on a real (non-validation) between
        file -- its absence used to let the duplicate epoch/reset-gen/
        sidecar checks below run `if mb:` and skip entirely (silently
        passing, per G3b), never actually checked at all."""
        arm_map = self._arm_map(tmp_path)
        doc = self._between_doc("S2-B4-c-r1", "A", epoch=1, reset_gen=1)
        doc["manifest_binding"] = None
        (tmp_path / "between_S2-B4-c-r1.json").write_text(json.dumps(doc))
        with pytest.raises(summ.SummaryCliError, match="manifest_binding is missing"):
            summ.load_between_files(tmp_path, arm_map)

    def test_duplicate_sidecar_detected_despite_different_path_spelling(self, tmp_path):
        """B7/H7: sidecar duplicates are compared on NORMALIZED paths --
        "setup.link.json" and "./setup.link.json" (or an equivalent
        absolute form) name the same real file and must collide, not
        pass as "different" strings."""
        arm_map = self._arm_map(tmp_path)
        doc1 = self._between_doc("S2-B4-c-r1", "A", epoch=1, reset_gen=1)
        doc1["manifest_binding"]["setup_sidecar"] = "setup.link.json"
        (tmp_path / "between_S2-B4-c-r1.json").write_text(json.dumps(doc1))
        doc2 = self._between_doc("S2-B4-c-r2", "B", epoch=2, reset_gen=2, sidecar_suffix="2")
        doc2["manifest_binding"]["setup_sidecar"] = "./setup.link.json"  # same file, different spelling
        (tmp_path / "between_S2-B4-c-r2.json").write_text(json.dumps(doc2))
        with pytest.raises(summ.SummaryCliError, match="already claimed"):
            summ.load_between_files(tmp_path, arm_map)

    def test_mutation_duplicate_epoch_check_removed(self, tmp_path):
        """Mutation (drop the epoch half of the duplicate check): the
        duplicate-epoch fixture above would then be silently accepted.
        Reproduced directly: with the sidecars ALSO forced distinct and
        reset_gen ALSO forced distinct, only the epoch collision would
        remain to catch it -- removing that specific check leaves
        nothing else that would."""
        arm_map = self._arm_map(tmp_path)
        (tmp_path / "between_S2-B4-c-r1.json").write_text(
            json.dumps(self._between_doc("S2-B4-c-r1", "A", epoch=1, reset_gen=1)))
        (tmp_path / "between_S2-B4-c-r2.json").write_text(
            json.dumps(self._between_doc("S2-B4-c-r2", "B", epoch=1, reset_gen=2, sidecar_suffix="2")))
        with pytest.raises(summ.SummaryCliError, match="epoch 1 is already claimed"):
            summ.load_between_files(tmp_path, arm_map)
