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
for _p in ("../../src", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp._io import IntegrityError, sha256_of  # noqa: E402
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

    def test_cli_end_to_end_with_run_flag(self, tmp_path):
        """The REAL shipped CLI entry point, --run given, a real
        manifest with two non-overlapping sidecars (setup/flight) --
        never rc 3 for a server_run_dir mismatch."""
        ev_root, run_dir, setup_last, flight_first = _build_real_layout(tmp_path)
        control_dir = tmp_path / "control"
        control_dir.mkdir()
        self._write_manifest_and_sidecars(control_dir, run_dir, setup_last, flight_first)
        out = tmp_path / "out.json"
        rc = cyc._cli([
            "--ev-dir", str(ev_root), "--run", RUN_NAME,
            "--control-dir", str(control_dir), "--cycle", "cyc", "--rep", "1", "--arm", "B",
            "--out", str(out), "--validation-mode",
        ])
        payload = json.loads(out.read_text())
        assert not (rc == 3 and "server_run_dir" in payload.get("reason", "")), payload

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
        ev_root, run_dir, _setup_last, _flight_first = _build_real_layout(tmp_path)
        # --ev-dir IS the run directory; --sha256sums points UP at the root file.
        evd = ev.verify_and_load(run_dir, "states.jsonl", "commands.jsonl", "../../SHA256SUMS")
        assert len(evd.commands) > 0

    def test_corrupted_root_entry_still_caught(self, tmp_path):
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
        with pytest.raises(IntegrityError):
            ev.verify_and_load(run_dir, "states.jsonl", "commands.jsonl", "../../SHA256SUMS")

    def test_mutation_prefixed_fallback_removed_would_reject_the_real_layout(self, tmp_path):
        """Mutation guard: without the prefixed-key fallback, `verified`
        only ever tries the PLAIN key ("states.jsonl"), which the root
        file never has -- reproduced directly against a local
        re-implementation of the OLD (pre-B6) function."""
        ev_root, run_dir, _setup_last, _flight_first = _build_real_layout(tmp_path)
        from tools.goalfix_cmp._io import parse_sha256sums

        def old_verified(evidence_dir, rel_path, sums):
            import os as _os
            key = _os.path.normpath(rel_path)
            return key in sums

        sums = parse_sha256sums(ev_root / "SHA256SUMS")
        mutant_finds_it = old_verified(run_dir, "states.jsonl", sums)
        assert mutant_finds_it is False, "the reported bug: the old rule never finds the prefixed key"
        # ... and the shipped code (fixed) disagrees -- it succeeds:
        evd = ev.verify_and_load(run_dir, "states.jsonl", "commands.jsonl", "../../SHA256SUMS")
        assert len(evd.commands) > 0
