"""B6 (H6): the native SHA256SUMS layout (merge verdict; coordinator Stage
B authorization). A real session's own root ``SHA256SUMS`` sits at the
evidence ROOT, keyed ``./e1_server_runs/<run>/...``, while the sidecars'
own ``server_run_dir`` names the RUN directory -- two different
directories the old CLI could not satisfy at once (V1-d).

``--ev-dir`` may now be:

1. the evidence root, with ``--run <run>`` -- ``--states``/``--commands``
   default under ``e1_server_runs/<run>/``, and the sidecar check is
   against the run directory, never the root; or
2. the run directory itself, with ``--sha256sums`` pointing at the root
   file -- ``_io.verified`` falls back to the run-prefixed key, inferred
   from the run directory's own name.

Either way, entries are resolved with their own real keys; nothing is
copied or derived.
"""
import json
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp", "."):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
import test_goalfix_cmp_f4_f5_manifest_reset_binding as f4f5  # noqa: E402
from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp._io import IntegrityError, RC_OK, read_result, sha256_of  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402


RUN_NAME = "run_20260925_stage_b_b6"


def _build_real_layout(tmp_path):
    """<tmp>/ev/e1_server_runs/<RUN_NAME>/{states,commands}.jsonl, plus a
    ROOT SHA256SUMS keyed ./e1_server_runs/<RUN_NAME>/... -- exactly plan
    §6's own layout, as Stage A's own slice produced it. Two legs
    (non-overlapping seq ranges) so a setup/flight sidecar pair can be
    built without the two spans colliding."""
    ev_root = tmp_path / "ev"
    run_dir = ev_root / "e1_server_runs" / RUN_NAME
    sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS}, settle_s=0.05)
    sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.3}, 1.0)])
    setup_last_seq = sim.state_rows[-1]["seq"]
    sim.fly([mf.Waypoint("B", {"r_shoulder_pitch": -0.6}, 1.0)])
    flight_first_seq = setup_last_seq + 1
    result = sim.result()
    mf.write_evidence(run_dir, result.state_rows, result.command_rows)  # writes a LOCAL SHA256SUMS too (unused)
    states_path = run_dir / "states.jsonl"
    commands_path = run_dir / "commands.jsonl"
    root_sums = ev_root / "SHA256SUMS"
    root_sums.write_text(
        f"{sha256_of(states_path)}  ./e1_server_runs/{RUN_NAME}/states.jsonl\n"
        f"{sha256_of(commands_path)}  ./e1_server_runs/{RUN_NAME}/commands.jsonl\n")
    return ev_root, run_dir, setup_last_seq, flight_first_seq


RUN_NAME_FULL_GATES = "run_20260926_k2m2a_full_gates"
CYCLE_FULL_GATES = "S2-B4-c-r2"  # rep 2 is 'B' in pv.ARM_MAP_ORDER (f4f5.GATE_REP)


def _build_full_gates_root_run_layout(tmp_path):
    """K2-M2a (2026-09-26 pr144-k2-k4 assignment): the SAME native
    root/``--run`` layout as ``_build_real_layout`` above, but a
    realistic two-leg, two-reset B cycle (PLACE_ROUTE/LIFT_TO_PRESENT,
    via ``make_fixtures.FlightSim``, exactly as
    ``test_goalfix_cmp_f4_f5_manifest_reset_binding._build_cycle_n_resets``
    builds one) plus F4/F5's full manifest/reset-record/provenance/
    compliance/start_variant scaffolding (``f4f5._write_full_gates``) --
    so the shipped CLI, run OUTSIDE ``--validation-mode``, can reach a
    REAL rc 0 (``--validation-mode`` itself always forces rc 3
    regardless of verdict, so it could never demonstrate this)."""
    ev_root = tmp_path / "ev_root"
    run_dir = ev_root / "e1_server_runs" / RUN_NAME_FULL_GATES
    control_dir = tmp_path / "control"
    control_dir.mkdir()

    home = dict(R.HOME)
    sim = mf.FlightSim(home, pose_units="deg", restream_passes=1)
    sim._hold_ticks(1, home)
    for i in range(2):  # n_resets=2, matching f4f5.GATE_REP's own reset-record convention
        sim.reset(seed=i + 1)
        sim._hold_ticks(1, home)
    setup_first_seq = sim.state_rows[-1]["seq"]
    sim.fly(R.PLACE_ROUTE)
    setup_last_seq = sim.state_rows[-1]["seq"]
    for _ in range(25):
        sim._hold_ticks(1, dict(sim.pose))
    flight_first_seq = sim.state_rows[-1]["seq"]
    sim.pose = dict(R.REST)
    sim.fly(R.LIFT_TO_PRESENT)
    flight_last_seq = sim.state_rows[-1]["seq"]

    result = sim.result()
    mf.write_evidence(run_dir, result.state_rows, result.command_rows)
    states_path = run_dir / "states.jsonl"
    commands_path = run_dir / "commands.jsonl"
    root_sums = ev_root / "SHA256SUMS"
    root_sums.write_text(
        f"{sha256_of(states_path)}  ./e1_server_runs/{RUN_NAME_FULL_GATES}/states.jsonl\n"
        f"{sha256_of(commands_path)}  ./e1_server_runs/{RUN_NAME_FULL_GATES}/commands.jsonl\n")

    run_dir_str = str(run_dir.resolve())
    (control_dir / f"{CYCLE_FULL_GATES}-setup.link.json").write_text(json.dumps({
        "alignment": [{"server_seq": setup_first_seq}, {"server_seq": setup_last_seq}],
        "log": "route_clearance_PLACE_ROUTE_20260101_000000.log",
        "server_run_dir": run_dir_str,
    }))
    (control_dir / f"{CYCLE_FULL_GATES}-flight.link.json").write_text(json.dumps({
        "alignment": [{"server_seq": flight_first_seq}, {"server_seq": flight_last_seq}],
        "log": "route_clearance_LIFT_TO_PRESENT_20260101_000000.log",
        "server_run_dir": run_dir_str,
    }))
    (control_dir / f"cycle_{CYCLE_FULL_GATES}.json").write_text(json.dumps({
        "rep": f4f5.GATE_REP, "cycle": CYCLE_FULL_GATES,
        "setup_sidecar": f"{CYCLE_FULL_GATES}-setup.link.json",
        "flight_sidecar": f"{CYCLE_FULL_GATES}-flight.link.json",
        "reset_gen": 2, "reset_record": "reset_record.txt",
    }))
    (control_dir / "reset_record.txt").write_text(
        "reset gen=2 ack=2 resets_recorded 1->2 sim_step 0->0\n")
    gate_args = f4f5._write_full_gates(control_dir, cycle=CYCLE_FULL_GATES)
    return ev_root, control_dir, gate_args


class TestEvDirIsRootWithRunFlag:
    def test_verify_and_load_succeeds(self, tmp_path):
        ev_root, run_dir, _setup_last, _flight_first = _build_real_layout(tmp_path)
        evd = ev.verify_and_load(
            ev_root, f"e1_server_runs/{RUN_NAME}/states.jsonl",
            f"e1_server_runs/{RUN_NAME}/commands.jsonl", "SHA256SUMS")
        assert len(evd.commands) > 0

    @staticmethod
    def _write_manifest_and_sidecars(control_dir, run_dir, setup_last, flight_first):
        """The MANIFEST-driven path (F4), which is the one that actually
        calls resolve_leg with expected_server_run_dir -- the
        --setup-sidecar/--flight-sidecar override path (validation-mode,
        no manifest) never checks it at all, so a CLI test must go
        through a real cycle_<id>.json manifest to exercise this."""
        setup_sidecar = {
            "alignment": [{"server_seq": 0}, {"server_seq": setup_last}],
            "log": "route_clearance_PLACE_ROUTE_x.log",
            "server_run_dir": str(run_dir.resolve()),
        }
        flight_sidecar = {
            "alignment": [{"server_seq": flight_first}, {"server_seq": flight_first + 1}],
            "log": "route_clearance_LIFT_TO_PRESENT_x.log",
            "server_run_dir": str(run_dir.resolve()),
        }
        (control_dir / "setup.link.json").write_text(json.dumps(setup_sidecar))
        (control_dir / "flight.link.json").write_text(json.dumps(flight_sidecar))
        manifest = {"rep": 1, "cycle": "cyc", "setup_sidecar": "setup.link.json",
                    "flight_sidecar": "flight.link.json"}
        (control_dir / "cycle_cyc.json").write_text(json.dumps(manifest))

    def test_cli_end_to_end_with_run_flag(self, tmp_path, monkeypatch):
        """K2-M2a: the REAL shipped CLI entry point, --run given, run
        OUTSIDE --validation-mode (full manifest/reset-binding/
        provenance/compliance/start_variant gates) -- asserts the real
        expected outcome (rc 0, VERDICT_OK), not merely "not rc 3 with a
        server_run_dir reason" (which a --run-ignoring mutant can also
        satisfy, by failing with a DIFFERENT rc-3 reason -- states.jsonl/
        commands.jsonl not found at the root -- that never mentions
        server_run_dir at all, since resolve_leg's own check is never
        reached).

        --validation-mode itself cannot make this assertion: `_cli`
        unconditionally sets `cv.validation_only = True` in that mode,
        and `CycleVerdict.rc()` maps `validation_only` straight to
        `RC_INCONCLUSIVE` (3), before even consulting `verdict` -- so no
        --validation-mode run can ever report rc 0, whatever the
        verdict. Hence the switch to a full-gates, non-validation-mode
        fixture (test_goalfix_cmp_f4_f5_manifest_reset_binding's own
        `_write_full_gates`), matching that module's own
        `test_clean_manifest_and_gates_give_true_rc0`."""
        monkeypatch.setattr(mf, "_LAG_RAD", [0.0] * 8)
        ev_root, control_dir, gate_args = _build_full_gates_root_run_layout(tmp_path)
        out = tmp_path / "out.json"
        rc = cyc._cli([
            "--ev-dir", str(ev_root), "--run", RUN_NAME_FULL_GATES,
            "--control-dir", str(control_dir), "--cycle", CYCLE_FULL_GATES,
            "--rep", str(f4f5.GATE_REP), "--arm", "B", "--out", str(out),
        ] + gate_args)
        payload = read_result(out)
        assert rc == RC_OK, payload
        assert payload["verdict"] == cyc.VERDICT_OK, payload

    def test_run_dir_computation_binds_to_the_run_directory_not_the_root(self, tmp_path):
        """Isolates cycle._cli's own `run_dir` computation (the value
        `resolve_leg` checks the sidecar's `server_run_dir` against) --
        library-level, via `resolve_leg` directly, so a CLI-level overlap/
        manifest concern in a minimal 2-command fixture cannot mask the
        one thing under test here."""
        ev_root, run_dir, _setup_last, _flight_first = _build_real_layout(tmp_path)
        evd = ev.verify_and_load(
            ev_root, f"e1_server_runs/{RUN_NAME}/states.jsonl",
            f"e1_server_runs/{RUN_NAME}/commands.jsonl", "SHA256SUMS")
        sidecar_path = tmp_path / "setup.link.json"
        sidecar_path.write_text(json.dumps({
            "alignment": [{"server_seq": 0}, {"server_seq": int(evd.states.seq[-1])}],
            "log": "route_clearance_PLACE_ROUTE_x.log",
            "server_run_dir": str(run_dir.resolve()),
        }))
        # The shipped CLI's own computation when --run is given:
        real_run_dir = str((ev_root / "e1_server_runs" / RUN_NAME).resolve())
        leg = cyc.resolve_leg(evd, sidecar_path, "setup", R.PLACE_ROUTE, R.CRITICAL_JOINTS,
                               expected_server_run_dir=real_run_dir)
        assert leg is not None  # no CycleInputError -- the sidecar's own server_run_dir matched

    def test_mutation_run_dir_reverted_to_ev_dir_would_mismatch_the_sidecar(self, tmp_path):
        """Mutation guard: computing `run_dir` as `--ev-dir` itself
        (ignoring `--run`, the pre-B6 behaviour) makes it disagree with
        the sidecar's own `server_run_dir` (the RUN directory) --
        reproduced directly against `resolve_leg`'s own check."""
        ev_root, run_dir, _setup_last, _flight_first = _build_real_layout(tmp_path)
        evd = ev.verify_and_load(
            ev_root, f"e1_server_runs/{RUN_NAME}/states.jsonl",
            f"e1_server_runs/{RUN_NAME}/commands.jsonl", "SHA256SUMS")
        sidecar_path = tmp_path / "setup.link.json"
        sidecar_path.write_text(json.dumps({
            "alignment": [{"server_seq": 0}, {"server_seq": int(evd.states.seq[-1])}],
            "log": "route_clearance_PLACE_ROUTE_x.log",
            "server_run_dir": str(run_dir.resolve()),
        }))
        mutant_run_dir = str(ev_root.resolve())  # the mutant's own (wrong) value
        from tools.goalfix_cmp.cycle import CycleInputError
        with pytest.raises(CycleInputError, match="server_run_dir"):
            cyc.resolve_leg(evd, sidecar_path, "setup", R.PLACE_ROUTE, R.CRITICAL_JOINTS,
                             expected_server_run_dir=mutant_run_dir)
        # ... and the shipped code (fixed, using the REAL run_dir) disagrees -- it succeeds:
        real_run_dir = str((ev_root / "e1_server_runs" / RUN_NAME).resolve())
        leg = cyc.resolve_leg(evd, sidecar_path, "setup", R.PLACE_ROUTE, R.CRITICAL_JOINTS,
                               expected_server_run_dir=real_run_dir)
        assert leg is not None


class TestEvDirIsRunDirWithRootSha256sums:
    def test_verified_falls_back_to_the_prefixed_key(self, tmp_path):
        """K2-M2b acceptance: the real shipped `_io.verified`, against
        the real native layout (root SHA256SUMS keyed by the
        run-prefixed path, --ev-dir the RUN directory itself) -- loads
        and passes checksums. This is the one that actually needs the
        prefixed-key fallback (unlike TestEvDirIsRootWithRunFlag's own
        scenario, where --ev-dir is the ROOT and the plain
        `e1_server_runs/<run>/...` key already matches directly,
        independent of the fallback): removing the fallback here makes
        `verified` raise `IntegrityError("... no entry in SHA256SUMS")`
        on the very first lookup, which this test's own
        `len(evd.commands) > 0` assertion cannot satisfy."""
        ev_root, run_dir, _setup_last, _flight_first = _build_real_layout(tmp_path)
        # --ev-dir IS the run directory; --sha256sums points UP at the root file.
        evd = ev.verify_and_load(run_dir, "states.jsonl", "commands.jsonl", "../../SHA256SUMS")
        assert len(evd.commands) > 0

    def test_corrupted_root_entry_still_caught(self, tmp_path):
        """K2-M2b refusal (digest altered): the prefixed key IS found,
        but its digest no longer matches -- the SPECIFIC "sha256
        mismatch" reason (never "no entry in SHA256SUMS", which is what
        the fallback-removed mutant would ALSO raise here, for the wrong
        reason -- masking a real digest corruption as a missing entry)."""
        ev_root, run_dir, _setup_last, _flight_first = _build_real_layout(tmp_path)
        sums_path = ev_root / "SHA256SUMS"
        text = sums_path.read_text()
        # Flip the states.jsonl digest's first hex character.
        lines = text.splitlines()
        out = []
        for line in lines:
            digest, name = line.split(None, 1)
            if name.endswith("states.jsonl"):
                digest = ("0" if digest[0] != "0" else "1") + digest[1:]
            out.append(f"{digest}  {name}")
        sums_path.write_text("\n".join(out) + "\n")
        with pytest.raises(IntegrityError, match="sha256 mismatch"):
            ev.verify_and_load(run_dir, "states.jsonl", "commands.jsonl", "../../SHA256SUMS")

    def test_prefixed_entry_absent_is_rejected(self, tmp_path):
        """K2-M2b refusal (entry absent): the root SHA256SUMS has NO
        entry at all for this run's states.jsonl (prefixed or plain) --
        rejected with the "no entry" reason, against the real shipped
        `_io.verified` (never a local re-implementation of the removed
        fallback, which is what the deleted tautology
        `test_mutation_prefixed_fallback_removed_would_reject_the_real_layout`
        did -- it built its OWN `old_verified` function and asserted
        THAT returned `False`, never calling the shipped code under the
        mutation at all, so it could not fail regardless of whether the
        fallback was actually present in `_io.verified`)."""
        ev_root, run_dir, _setup_last, _flight_first = _build_real_layout(tmp_path)
        sums_path = ev_root / "SHA256SUMS"
        lines = sums_path.read_text().splitlines()
        kept = [line for line in lines if not line.split(None, 1)[1].endswith("states.jsonl")]
        assert len(kept) == len(lines) - 1, lines
        sums_path.write_text("\n".join(kept) + "\n")
        with pytest.raises(IntegrityError, match="no entry in SHA256SUMS"):
            ev.verify_and_load(run_dir, "states.jsonl", "commands.jsonl", "../../SHA256SUMS")
