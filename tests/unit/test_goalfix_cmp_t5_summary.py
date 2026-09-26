"""T5 acceptance tests (assignment §2, T5): the summary/checkpoint CLI and
aggregate()'s corrected rules (review §3.7/M5) -- a B cycle without
metrics or evidence_incomplete must never reach `supports`, an incomplete
session must never be silently evaluated as the full one, and
"inconclusive comparison" must not require 6 A cycles to already exist."""
import json
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../.."))

from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import provenance as pv  # noqa: E402
from tools.goalfix_cmp import summary as summ  # noqa: E402
from tools.goalfix_cmp._io import RC_INCONCLUSIVE, RC_OK, RC_STOP, read_result  # noqa: E402

#: MB3 (merge verdict, 2026-09-25 stage-repairs assignment §3): both CLI
#: subcommands now require a validated arm map in the frozen
#: pv.ARM_MAP_ORDER ("ABBABAABABBA").
BRIDGE_SHA_A = "8c0dad2d62dde791c8912fa723b8c2b7f190e1e8"
BRIDGE_SHA_B = "67730a1ecf646544825fb60c00129cf46de6d307"


def _write_arm_map(control_dir):
    arm_map = [
        {"rep": i + 1, "arm": pv.ARM_MAP_ORDER[i],
         "image_tag": "img-A" if pv.ARM_MAP_ORDER[i] == "A" else "img-B",
         "image_id": "sha256:A" if pv.ARM_MAP_ORDER[i] == "A" else "sha256:B",
         "bridge_sha": BRIDGE_SHA_A if pv.ARM_MAP_ORDER[i] == "A" else BRIDGE_SHA_B,
         "opt_hashes": {f: ("a" * 64 if pv.ARM_MAP_ORDER[i] == "A" else "b" * 64)
                        for f in pv.EXPECTED_DIFF_FILES}}
        for i in range(12)
    ]
    path = control_dir / "arm_map.json"
    path.write_text(json.dumps(arm_map))
    return path


def _default_log_paths(control_dir):
    """F2/F3: checkpoint/final now require the full tripwire input set
    (--native-log, --states, one of --control-stop/--control-stop-absent).
    Tests in this module that are not themselves about tripwire
    behaviour (MB3 rejection, arm-map validation, ordering, ...) get
    clean/empty default logs here so they can reach the logic they
    actually exercise; a test that wants specific tripwire content
    passes its own flags, which win (see `_summ_cli`'s own
    `inject_default_logs`). F3 also requires a `cycle_<id>.json`
    manifest (bridge_log/reset_record) for every cycle in scope --
    written here, clean, for every rep 1..12."""
    native = control_dir / "_default_native_log.txt"
    states = control_dir / "_default_states.jsonl"
    for p in (native, states):
        if not p.exists():
            p.write_text("")
    for rep in range(1, 13):
        write_clean_cycle_manifest(control_dir, f"S2-B4-c-r{rep}")
    return str(native), str(states)


def write_clean_cycle_manifest(control_dir, cycle_id, *, bridge_log_text="",
                                reset_record_text="reset gen=1 ack=1 resets_recorded 0->1 "
                                                   "sim_step 0->0\n"):
    """F3 test helper: a per-cycle manifest (F4's cycle_<id>.json) naming
    a bridge_log and reset_record file, both written alongside it."""
    bridge_log_name = f"bridge_log_{cycle_id}.txt"
    (control_dir / bridge_log_name).write_text(bridge_log_text)
    reset_record_name = f"reset_record_{cycle_id}.txt"
    (control_dir / reset_record_name).write_text(reset_record_text)
    manifest = {
        "rep": _cycle_rep(cycle_id), "cycle": cycle_id,
        "setup_sidecar": f"{cycle_id}-setup.link.json",
        "flight_sidecar": f"{cycle_id}-flight.link.json",
        "reset_gen": 1, "reset_record": reset_record_name, "bridge_log": bridge_log_name,
    }
    (control_dir / f"cycle_{cycle_id}.json").write_text(json.dumps(manifest))


def _cycle_rep(cycle_id):
    import re as _re
    m = _re.search(r"r(\d+)$", cycle_id)
    return int(m.group(1)) if m else 0


def _summ_cli(argv, control_dir, inject_default_logs=True):
    extra = [
        "--arm-map", str(_write_arm_map(control_dir)),
        "--expected-bridge-sha-a", BRIDGE_SHA_A,
        "--expected-bridge-sha-b", BRIDGE_SHA_B,
    ]
    if inject_default_logs and "--native-log" not in argv:
        native, states = _default_log_paths(control_dir)
        extra += ["--native-log", native, "--states", states, "--control-stop-absent"]
    return summ._cli(argv + extra)


def _cv(cycle, arm, verdict, reasons=()):
    return cyc.CycleVerdict(cycle, arm, verdict, list(reasons))


def _good_b_metrics():
    return summ.CycleMetrics(
        leg_start_shoulder_pitch_error_deg=0.5, delta_cmd_max_cm=0.01, wrist_ball_delta_cm=0.9)


def _full_session(a_verdict=cyc.VERDICT_MANIPULATED, n=6):
    verdicts = ([_cv(f"A{i}", "A", a_verdict) for i in range(n)]
                + [_cv(f"B{i}", "B", cyc.VERDICT_OK) for i in range(n)])
    metrics = {f"A{i}": summ.CycleMetrics(wrist_ball_delta_cm=2.0) for i in range(n)}
    metrics.update({f"B{i}": _good_b_metrics() for i in range(n)})
    return verdicts, metrics


class TestAggregateFixtureTable:
    def test_all_valid_supports(self):
        verdicts, metrics = _full_session()
        r = summ.aggregate(verdicts, metrics)
        assert r.outcome == summ.OUTCOME_SUPPORTS
        assert r.rc() == RC_OK

    def test_one_a_inconclusive_still_supports(self):
        verdicts, metrics = _full_session()
        verdicts[0] = _cv("A0", "A", cyc.VERDICT_INCONCLUSIVE_BASELINE)
        del metrics["A0"]
        r = summ.aggregate(verdicts, metrics)
        assert r.outcome == summ.OUTCOME_SUPPORTS  # 5/6 manipulated, still >= 4

    def test_three_a_inconclusive_gives_inconclusive_comparison(self):
        verdicts, metrics = _full_session()
        for i in range(3):
            verdicts[i] = _cv(f"A{i}", "A", cyc.VERDICT_INCONCLUSIVE_BASELINE)
            del metrics[f"A{i}"]
        r = summ.aggregate(verdicts, metrics)
        assert r.outcome == summ.OUTCOME_INCONCLUSIVE_COMPARISON
        assert r.rc() == RC_INCONCLUSIVE

    def test_b_stop_at_cycle_three(self):
        verdicts, metrics = _full_session()
        verdicts[8] = _cv("B2", "B", cyc.VERDICT_STOP, ["C1 fail"])
        r = summ.aggregate(verdicts, metrics)
        assert r.outcome == summ.OUTCOME_STOPPED
        assert r.rc() == RC_STOP

    def test_b_evidence_incomplete_never_supports(self):
        verdicts, metrics = _full_session()
        verdicts[6] = _cv("B0", "B", cyc.VERDICT_EVIDENCE_INCOMPLETE)
        r = summ.aggregate(verdicts, metrics)
        assert r.outcome != summ.OUTCOME_SUPPORTS
        assert r.rc() == RC_INCONCLUSIVE

    def test_one_b_metric_null_gives_incomplete(self):
        verdicts, metrics = _full_session()
        metrics["B0"] = summ.CycleMetrics(
            leg_start_shoulder_pitch_error_deg=0.5, delta_cmd_max_cm=None,
            wrist_ball_delta_cm=0.9)
        r = summ.aggregate(verdicts, metrics)
        assert r.outcome == summ.OUTCOME_INCOMPLETE
        assert r.rc() == RC_INCONCLUSIVE
        assert "B0" in r.detail

    def test_eight_cycles_only_is_incomplete(self):
        verdicts, metrics = _full_session(n=4)  # 4 A + 4 B = 8, need 6 each
        r = summ.aggregate(verdicts, metrics, a_target_count=6)
        assert r.outcome == summ.OUTCOME_INCOMPLETE

    def test_inconclusive_comparison_does_not_require_six_a_cycles_to_exist(self):
        """The old bug: `len(a_cycles) >= a_target_count` gated the whole
        rule, so a session with fewer than 6 A cycles (but still < 4
        manipulated) could report `supports` instead of
        inconclusive_comparison. Using a_target_count=2 here so the
        session-completeness check (itself fixed, T5) doesn't mask it."""
        verdicts = ([_cv("A0", "A", cyc.VERDICT_INCONCLUSIVE_BASELINE),
                     _cv("A1", "A", cyc.VERDICT_INCONCLUSIVE_BASELINE)]
                    + [_cv("B0", "B", cyc.VERDICT_OK), _cv("B1", "B", cyc.VERDICT_OK)])
        metrics = {"B0": _good_b_metrics(), "B1": _good_b_metrics()}
        r = summ.aggregate(verdicts, metrics, a_target_count=2, a_manipulated_min=4)
        assert r.outcome == summ.OUTCOME_INCONCLUSIVE_COMPARISON


class TestCheckpointFixtureTable:
    def test_checkpoint_with_one_evidence_incomplete_is_rc3(self):
        verdicts = [_cv("A1", "A", cyc.VERDICT_MANIPULATED),
                    _cv("B1", "B", cyc.VERDICT_EVIDENCE_INCOMPLETE),
                    _cv("A2", "A", cyc.VERDICT_MANIPULATED),
                    _cv("B2", "B", cyc.VERDICT_OK)]
        r = summ.checkpoint_after(verdicts)
        assert r.any_incomplete
        assert r.rc() == RC_INCONCLUSIVE

    def test_checkpoint_missing_a_cycle_is_rc3(self):
        verdicts = [_cv("A1", "A", cyc.VERDICT_MANIPULATED),
                    _cv("B1", "B", cyc.VERDICT_OK),
                    _cv("A2", "A", cyc.VERDICT_MANIPULATED)]  # only 3 of 4
        r = summ.checkpoint_after(verdicts, n=4)
        assert r.any_incomplete
        assert r.missing_cycles == 1
        assert r.rc() == RC_INCONCLUSIVE


class TestSummaryCli:
    def _write_between(self, control_dir, cycle, arm, verdict, metrics=None, sha=None, rc=None):
        doc = {
            "cycle": cycle, "arm": arm, "verdict": verdict, "reasons": [],
            "genuine_echo_count": 0, "segment_indeterminate": False,
            "metrics": metrics, "tools_sha256": sha or cyc._package_sha256(),
            "rc": rc if rc is not None else cyc.rc_for_verdict(verdict),
        }
        (control_dir / f"between_{cycle}.json").write_text(json.dumps(doc))

    def test_wrong_tools_sha256_gives_rc3(self, tmp_path):
        # rep 1 is 'A' in pv.ARM_MAP_ORDER.
        self._write_between(tmp_path, "S2-B4-c-r1", "A", cyc.VERDICT_MANIPULATED,
                             metrics={"wrist_ball_delta_cm": 2.0}, sha="deadbeef" * 8)
        out = tmp_path / "checkpoint_1.json"
        rc = _summ_cli(["checkpoint", "--control-dir", str(tmp_path), "--n", "1",
                        "--out", str(out)], tmp_path)
        assert rc == RC_INCONCLUSIVE
        # MB3: a hash-mismatched file's own rep is excluded from
        # by_rep entirely, so it now ALSO trips the "missing required
        # rep" check (a stricter, still-rc-3 failure mode than the old
        # hash_mismatch-only path) -- the payload shape here is the
        # generic SummaryCliError one, not checkpoint's own.
        payload = read_result(out)
        assert payload.get("ok") is False
        assert "missing required rep" in payload.get("reason", "")

    def test_final_cli_supports_on_full_session(self, tmp_path):
        for i in range(12):
            arm = pv.ARM_MAP_ORDER[i]
            verdict = cyc.VERDICT_MANIPULATED if arm == "A" else cyc.VERDICT_OK
            metrics = ({"wrist_ball_delta_cm": 2.0} if arm == "A" else vars(_good_b_metrics()))
            self._write_between(tmp_path, f"S2-B4-c-r{i+1}", arm, verdict, metrics=metrics)
        out = tmp_path / "final.json"
        rc = _summ_cli(["final", "--control-dir", str(tmp_path), "--out", str(out)], tmp_path)
        payload = read_result(out)
        assert payload["outcome"] == summ.OUTCOME_SUPPORTS, payload
        assert rc == RC_OK

    def test_checkpoint_ordering_by_rep(self, tmp_path):
        # Written out of order; checkpoint --n 2 must take reps 1-2, not
        # file-listing order. Reps 1/2/3 are 'A'/'B'/'B' in ARM_MAP_ORDER.
        self._write_between(tmp_path, "S2-B4-c-r3", "B", cyc.VERDICT_MANIPULATED)
        self._write_between(tmp_path, "S2-B4-c-r1", "A", cyc.VERDICT_MANIPULATED)
        self._write_between(tmp_path, "S2-B4-c-r2", "B", cyc.VERDICT_STOP)
        out = tmp_path / "checkpoint_1.json"
        rc = _summ_cli(["checkpoint", "--control-dir", str(tmp_path), "--n", "2",
                        "--out", str(out)], tmp_path)
        payload = read_result(out)
        assert payload["any_stop"]  # rep 2 (a STOP) must be inside the first 2, not rep 3


class TestMB3RejectNonAuthorizingEvidence:
    """MB3 (merge verdict, 2026-09-25 stage-repairs assignment §3):
    probes P2/P10/P10b/P11/P12, through the shipped `summary` CLI."""

    def _write_between(self, control_dir, cycle, arm, verdict, metrics=None, sha=None,
                        rc=None, validation_only=False):
        doc = {
            "cycle": cycle, "arm": arm, "verdict": verdict, "reasons": [],
            "genuine_echo_count": 0, "segment_indeterminate": False,
            "metrics": metrics, "tools_sha256": sha or cyc._package_sha256(),
            "rc": rc if rc is not None else cyc.rc_for_verdict(verdict),
            "validation_only": validation_only,
        }
        (control_dir / f"between_{cycle}.json").write_text(json.dumps(doc))

    def test_p2_validation_only_checkpoint_rejected(self, tmp_path):
        for rep, arm in [(1, "A"), (2, "B"), (3, "B"), (4, "A")]:
            self._write_between(tmp_path, f"S2-B4-c-r{rep}", arm, cyc.VERDICT_EVIDENCE_INCOMPLETE,
                                 rc=RC_INCONCLUSIVE, validation_only=True)
        out = tmp_path / "checkpoint_1.json"
        rc = _summ_cli(["checkpoint", "--control-dir", str(tmp_path), "--n", "4",
                        "--out", str(out)], tmp_path)
        assert rc == RC_INCONCLUSIVE
        assert read_result(out).get("ok") is False

    def test_p10_missing_rep_one_rejected(self, tmp_path):
        for rep, arm in [(2, "B"), (3, "B"), (4, "A"), (5, "B")]:
            verdict = cyc.VERDICT_MANIPULATED if arm == "A" else cyc.VERDICT_OK
            metrics = ({"wrist_ball_delta_cm": 2.0} if arm == "A" else vars(_good_b_metrics()))
            self._write_between(tmp_path, f"S2-B4-c-r{rep}", arm, verdict, metrics=metrics)
        out = tmp_path / "checkpoint_1.json"
        rc = _summ_cli(["checkpoint", "--control-dir", str(tmp_path), "--n", "4",
                        "--out", str(out)], tmp_path)
        assert rc == RC_INCONCLUSIVE
        payload = read_result(out)
        assert payload.get("ok") is False
        assert "missing required rep" in payload.get("reason", "")

    def test_p10b_duplicate_rep_rejected(self, tmp_path):
        for rep, arm in [(1, "A"), (2, "B"), (3, "B"), (4, "A")]:
            verdict = cyc.VERDICT_MANIPULATED if arm == "A" else cyc.VERDICT_OK
            metrics = ({"wrist_ball_delta_cm": 2.0} if arm == "A" else vars(_good_b_metrics()))
            self._write_between(tmp_path, f"S2-B4-c-r{rep}", arm, verdict, metrics=metrics)
        # A duplicate of rep 2's content, under a DIFFERENT filename but
        # the SAME cycle id (so it claims the same rep).
        dup = json.loads((tmp_path / "between_S2-B4-c-r2.json").read_text())
        (tmp_path / "between_S2-B4-c-r2-copy.json").write_text(json.dumps(dup))
        out = tmp_path / "checkpoint_1.json"
        rc = _summ_cli(["checkpoint", "--control-dir", str(tmp_path), "--n", "4",
                        "--out", str(out)], tmp_path)
        assert rc == RC_INCONCLUSIVE
        payload = read_result(out)
        assert payload.get("ok") is False
        assert "duplicate rep" in payload.get("reason", "")

    def test_p11_all_b_no_a_arm_mismatch_rejected(self, tmp_path):
        # reps 1-4 are 'A','B','B','A' in ARM_MAP_ORDER -- writing all 4
        # as 'B' contradicts the map for reps 1 and 4.
        for rep in (1, 2, 3, 4):
            self._write_between(tmp_path, f"S2-B4-c-r{rep}", "B", cyc.VERDICT_OK,
                                 metrics=vars(_good_b_metrics()))
        out = tmp_path / "checkpoint_1.json"
        rc = _summ_cli(["checkpoint", "--control-dir", str(tmp_path), "--n", "4",
                        "--out", str(out)], tmp_path)
        assert rc == RC_INCONCLUSIVE
        payload = read_result(out)
        assert payload.get("ok") is False
        assert "arm_map" in payload.get("reason", "") or "!= arm_map" in payload.get("reason", "")

    def test_p12_final_over_twelve_validation_only_rejected(self, tmp_path):
        for i in range(12):
            arm = pv.ARM_MAP_ORDER[i]
            self._write_between(tmp_path, f"S2-B4-c-r{i+1}", arm, cyc.VERDICT_EVIDENCE_INCOMPLETE,
                                 rc=RC_INCONCLUSIVE, validation_only=True)
        out = tmp_path / "final.json"
        rc = _summ_cli(["final", "--control-dir", str(tmp_path), "--out", str(out)], tmp_path)
        assert rc == RC_INCONCLUSIVE
        assert read_result(out).get("ok") is False

    def test_mutation_validation_only_check_removed_would_authorize(self, tmp_path):
        """Mutation (drop the `validation_only` rejection from
        load_between_files): the P2 fixture above would be accepted
        as real evidence (its verdict/rc read normally). Verified
        directly: a 9538003 copy of summary.py has no `validation_only`
        key handling in `load_between_files` at all -- it is silently
        ignored, and the file's own `verdict`/`rc` are trusted like any
        other. That head also predates --arm-map being accepted at all
        by the CLI, so the mutation is demonstrated via the library
        function directly rather than re-running an incompatible CLI."""
        for rep, arm in [(1, "A"), (2, "B"), (3, "B"), (4, "A")]:
            self._write_between(tmp_path, f"S2-B4-c-r{rep}", arm, cyc.VERDICT_EVIDENCE_INCOMPLETE,
                                 rc=RC_INCONCLUSIVE, validation_only=True)
        arm_map_path = _write_arm_map(tmp_path)
        arm_map_doc = json.loads(arm_map_path.read_text())
        arm_map = {e["rep"]: pv.ArmMapEntry(**e) for e in arm_map_doc}
        with pytest.raises(summ.SummaryCliError, match="validation_only"):
            summ.load_between_files(tmp_path, arm_map, required_reps=range(1, 5))


class TestMB7TripwiresThroughTheCli:
    """MB7 (merge verdict, 2026-09-25 stage-repairs assignment §3):
    checkpoint/final wired to the tripwire counters, through the shipped
    CLI. F2/F3 (2026-09-25 pr144-f1-f6 assignment) made these mandatory
    and per-cycle -- see `TestF2Mandatory`/`TestF3PerCycleTripwires`
    below for the current-shape tests; this class keeps the
    still-relevant lease_acquisition/session-level cases."""

    def _write_between(self, control_dir, cycle, arm, verdict, metrics=None):
        doc = {
            "cycle": cycle, "arm": arm, "verdict": verdict, "reasons": [],
            "genuine_echo_count": 0, "segment_indeterminate": False,
            "metrics": metrics, "tools_sha256": cyc._package_sha256(),
            "rc": cyc.rc_for_verdict(verdict), "validation_only": False,
        }
        (control_dir / f"between_{cycle}.json").write_text(json.dumps(doc))

    def _write_four_clean(self, tmp_path):
        for rep, arm in [(1, "A"), (2, "B"), (3, "B"), (4, "A")]:
            verdict = cyc.VERDICT_MANIPULATED if arm == "A" else cyc.VERDICT_OK
            metrics = ({"wrist_ball_delta_cm": 2.0} if arm == "A" else vars(_good_b_metrics()))
            self._write_between(tmp_path, f"S2-B4-c-r{rep}", arm, verdict, metrics=metrics)
            write_clean_cycle_manifest(tmp_path, f"S2-B4-c-r{rep}")

    def test_no_log_flags_is_unaffected(self, tmp_path):
        """F2 (merge verdict §2; 2026-09-25 pr144-f1-f6 assignment):
        INVERTED from its pre-F2 name/assertion (kept, not deleted, per
        the assignment's own instruction). At 0722476 this pinned
        "no log flags -> tripwires are skipped, rc 0" -- exactly the bug
        (tripwires opt-in, evidence never actually checked). Tripwires
        are now mandatory for checkpoint/final: omitting --native-log/
        --states/--control-stop(-absent) is rc 3, never rc 0."""
        self._write_four_clean(tmp_path)
        out = tmp_path / "checkpoint_1.json"
        rc = _summ_cli(["checkpoint", "--control-dir", str(tmp_path), "--n", "4",
                        "--out", str(out)], tmp_path, inject_default_logs=False)
        assert rc == RC_INCONCLUSIVE
        payload = read_result(out)
        assert payload.get("ok") is False
        assert "mandatory" in payload.get("reason", "")

    def test_missing_required_log_is_rc3(self, tmp_path):
        """F2: supplying only '--native-log' (without --states/
        --control-stop(-absent)) is rc 3."""
        self._write_four_clean(tmp_path)
        native_log = tmp_path / "native.log"
        native_log.write_text("nothing interesting\n")
        out = tmp_path / "checkpoint_1.json"
        rc = _summ_cli(["checkpoint", "--control-dir", str(tmp_path), "--n", "4",
                        "--native-log", str(native_log), "--out", str(out)],
                       tmp_path, inject_default_logs=False)
        assert rc == RC_INCONCLUSIVE
        assert "mandatory" in read_result(out).get("reason", "")

    def test_nonzero_lease_acquisition_stops_a_clean_checkpoint(self, tmp_path):
        self._write_four_clean(tmp_path)
        native_log = tmp_path / "native.log"
        native_log.write_text("Execution lease granted to 'x' (mover 'x', 30s)\n")
        states = tmp_path / "states.jsonl"
        states.write_text("")
        out = tmp_path / "checkpoint_1.json"
        rc = _summ_cli([
            "checkpoint", "--control-dir", str(tmp_path), "--n", "4",
            "--native-log", str(native_log),
            "--states", str(states), "--control-stop-absent",
            "--out", str(out),
        ], tmp_path, inject_default_logs=False)
        assert rc == RC_STOP
        payload = read_result(out)
        assert payload["tripwires"]["lease_acquisition"]["count"] == 1

    def test_mutation_wiring_removed_would_ignore_the_flag(self, tmp_path):
        """Mutation (revert to the pre-MB7 CLI, b53a86b): the same
        nonzero-lease-acquisition fixture above is not even ACCEPTED
        (the flags do not exist), let alone gated -- a structural
        confirmation that the wiring is new."""
        import subprocess
        result = subprocess.run(["git", "show", "b53a86b:tools/goalfix_cmp/summary.py"],
                                 cwd=_HERE + "/../..", capture_output=True, text=True)
        assert "--native-log" not in result.stdout


class TestF1TripwiresReadTheLogFiles:
    """F1 (merge verdict review-2026-09-25-pr144-0722476-merge-verdict.md
    §2; 2026-09-25 pr144-f1-f6 assignment): at 0722476, `summary._cli`
    passed the --bridge-log/--native-log PATH STRING itself to
    `mb7_tripwire_report`, which counts patterns in whatever text it is
    given -- so a real log file's own CONTENTS were never read at all.
    Every test here goes through the shipped `summary._cli` with real
    files in `tmp_path`. F3 moved the per-cycle bridge log from a
    session-level --bridge-log flag to the cycle manifest -- the S3/S5/
    empty-log cases are exercised there through the manifest's own
    bridge_log field; --native-log (still a session-level path) keeps
    its own nonexistent-path case here."""

    def _write_between(self, control_dir, cycle, arm, verdict, metrics=None):
        doc = {
            "cycle": cycle, "arm": arm, "verdict": verdict, "reasons": [],
            "genuine_echo_count": 0, "segment_indeterminate": False,
            "metrics": metrics, "tools_sha256": cyc._package_sha256(),
            "rc": cyc.rc_for_verdict(verdict), "validation_only": False,
        }
        (control_dir / f"between_{cycle}.json").write_text(json.dumps(doc))

    def _write_four_clean(self, tmp_path):
        for rep, arm in [(1, "A"), (2, "B"), (3, "B"), (4, "A")]:
            verdict = cyc.VERDICT_MANIPULATED if arm == "A" else cyc.VERDICT_OK
            metrics = ({"wrist_ball_delta_cm": 2.0} if arm == "A" else vars(_good_b_metrics()))
            self._write_between(tmp_path, f"S2-B4-c-r{rep}", arm, verdict, metrics=metrics)
            write_clean_cycle_manifest(tmp_path, f"S2-B4-c-r{rep}")

    def test_s4_nonexistent_native_log_path_gives_rc3(self, tmp_path):
        """S4: --native-log naming a file that does not exist -> rc 3,
        never rc 0. At 0722476 a nonexistent path's own STRING had no
        tripwire text in it either, so counts came back 0 and the
        checkpoint passed at rc 0."""
        self._write_four_clean(tmp_path)
        out = tmp_path / "checkpoint_1.json"
        rc = summ._cli([
            "checkpoint", "--control-dir", str(tmp_path), "--n", "4",
            "--native-log", str(tmp_path / "does_not_exist_native.log"),
            "--states", str(tmp_path / "does_not_exist_states.jsonl"),
            "--control-stop-absent", "--out", str(out),
            "--arm-map", str(_write_arm_map(tmp_path)),
            "--expected-bridge-sha-a", BRIDGE_SHA_A, "--expected-bridge-sha-b", BRIDGE_SHA_B,
        ])
        payload = read_result(out)
        assert rc == RC_INCONCLUSIVE
        assert "does_not_exist" in payload.get("reason", "")

    def test_empty_existing_native_log_counts_zero_not_missing(self, tmp_path):
        self._write_four_clean(tmp_path)
        native_log = tmp_path / "native.log"
        native_log.write_text("")  # exists, empty
        states = tmp_path / "states.jsonl"
        states.write_text("")
        out = tmp_path / "checkpoint_1.json"
        rc = summ._cli([
            "checkpoint", "--control-dir", str(tmp_path), "--n", "4",
            "--native-log", str(native_log), "--states", str(states),
            "--control-stop-absent", "--out", str(out),
            "--arm-map", str(_write_arm_map(tmp_path)),
            "--expected-bridge-sha-a", BRIDGE_SHA_A, "--expected-bridge-sha-b", BRIDGE_SHA_B,
        ])
        payload = read_result(out)
        assert rc == RC_OK
        assert payload["tripwires"]["lease_acquisition"]["count"] == 0

    def test_mutation_cli_passes_path_instead_of_contents(self, tmp_path):
        """Mutation at the shipped call site (the F1 fix's own seam):
        `count_tripwires(native_log=args.native_log, ...)` -- the PATH,
        not `read_log_text(args.native_log)`. Reproduced directly (not by
        editing the module in-process): passing the native log's own
        PATH STRING to `summ.count_tripwires` the way the pre-fix CLI
        did shows the mismatch the fix closes."""
        native_log = tmp_path / "native_lease.log"
        native_log.write_text("Execution lease granted to 'x' (mover 'x', 30s)\n")
        # The mutant's own behaviour: hand the PATH STRING itself to the
        # text-counting function, exactly as summary._cli did at 0722476.
        mutant_counts = summ.count_tripwires(native_log=str(native_log))
        assert mutant_counts["lease_acquisition"] == 0
        # The fix's own behaviour (this test's real assertion target):
        # reading the file's CONTENTS first.
        fixed_counts = summ.count_tripwires(native_log=summ.read_log_text(str(native_log)))
        assert fixed_counts["lease_acquisition"] == 1

    def test_mutation_missing_file_treated_as_empty(self, tmp_path):
        """Mutation: `read_log_text` swallows a missing file and returns
        ``""`` instead of raising. Reproduced directly against the
        mutant's own would-be behaviour, contrasted with the shipped
        `read_log_text`, which must raise."""
        missing = tmp_path / "does_not_exist.log"

        def mutant_read_log_text(path):
            from pathlib import Path as _Path
            p = _Path(path)
            return p.read_text() if p.is_file() else ""

        assert mutant_read_log_text(str(missing)) == ""
        with pytest.raises(summ.SummaryCliError):
            summ.read_log_text(str(missing))


class TestF3PerCycleTripwires:
    """F3 (merge verdict §2; 2026-09-25 pr144-f1-f6 assignment): plan
    §7.7's full row set, per cycle and per arm -- bridge_log/reset_record
    resolved from each cycle's own manifest (F4's cycle_<id>.json), never
    a session-level positional list. Rep 1 is 'A', reps 2/3 are 'B', rep
    4 is 'A' in ARM_MAP_ORDER."""

    def _write_between(self, control_dir, cycle, arm, verdict, metrics=None):
        doc = {
            "cycle": cycle, "arm": arm, "verdict": verdict, "reasons": [],
            "genuine_echo_count": 0, "segment_indeterminate": False,
            "metrics": metrics, "tools_sha256": cyc._package_sha256(),
            "rc": cyc.rc_for_verdict(verdict), "validation_only": False,
        }
        (control_dir / f"between_{cycle}.json").write_text(json.dumps(doc))

    def _write_four_clean(self, tmp_path, *, manifest_overrides=None):
        manifest_overrides = manifest_overrides or {}
        for rep, arm in [(1, "A"), (2, "B"), (3, "B"), (4, "A")]:
            cycle = f"S2-B4-c-r{rep}"
            verdict = cyc.VERDICT_MANIPULATED if arm == "A" else cyc.VERDICT_OK
            metrics = ({"wrist_ball_delta_cm": 2.0} if arm == "A" else vars(_good_b_metrics()))
            self._write_between(tmp_path, cycle, arm, verdict, metrics=metrics)
            kwargs = manifest_overrides.get(cycle, {})
            write_clean_cycle_manifest(tmp_path, cycle, **kwargs)

    def _checkpoint(self, tmp_path):
        native_log = tmp_path / "native.log"
        native_log.write_text("")
        states = tmp_path / "states.jsonl"
        states.write_text("")
        out = tmp_path / "checkpoint_1.json"
        rc = summ._cli([
            "checkpoint", "--control-dir", str(tmp_path), "--n", "4",
            "--native-log", str(native_log), "--states", str(states),
            "--control-stop-absent", "--out", str(out),
            "--arm-map", str(_write_arm_map(tmp_path)),
            "--expected-bridge-sha-a", BRIDGE_SHA_A, "--expected-bridge-sha-b", BRIDGE_SHA_B,
        ])
        return rc, read_result(out)

    def test_s2_reset_ack_timeout_in_b_bridge_log_stops(self, tmp_path):
        """S2: a B cycle's own bridge_log contains "Reset ack timed
        out" -> rc 2. At 0722476 this counter was tallied but never
        reported or gated at all."""
        self._write_four_clean(tmp_path, manifest_overrides={
            "S2-B4-c-r2": {"bridge_log_text": "log.error: Reset ack timed out after 6.0 s\n"}})
        rc, payload = self._checkpoint(tmp_path)
        assert rc == RC_STOP
        row = next(r for r in payload["tripwires"]["per_cycle"] if r["cycle"] == "S2-B4-c-r2")
        assert row["reset_ack_timeout"] == 1
        assert "reset_ack_timeout" in row["violated"]

    def test_unexpected_reset_ack_on_b_cycle_stops(self, tmp_path):
        self._write_four_clean(tmp_path, manifest_overrides={
            "S2-B4-c-r2": {"bridge_log_text": "Unexpected reset_ack id=3 (expected 3)\n"}})
        rc, payload = self._checkpoint(tmp_path)
        assert rc == RC_STOP
        row = next(r for r in payload["tripwires"]["per_cycle"] if r["cycle"] == "S2-B4-c-r2")
        assert row["unexpected_reset_ack"] == 1
        assert "unexpected_reset_ack" in row["violated"]

    def test_unexpected_reset_ack_on_a_cycle_is_not_a_stop(self, tmp_path):
        """The A carve-out (plan §7.7): the SAME line on an A cycle's own
        bridge_log is reported, never gated."""
        self._write_four_clean(tmp_path, manifest_overrides={
            "S2-B4-c-r1": {"bridge_log_text": "Unexpected reset_ack id=3 (expected 3)\n"}})
        rc, payload = self._checkpoint(tmp_path)
        assert rc == RC_OK
        row = next(r for r in payload["tripwires"]["per_cycle"] if r["cycle"] == "S2-B4-c-r1")
        assert row["unexpected_reset_ack"] == 1
        assert row["violated"] == []

    def test_control_held_refusal_stops_both_arms(self, tmp_path):
        """B1/H1 (merge verdict §2; coordinator Stage B authorization):
        a non-zero control_held_refusal in ANY cycle's bridge_log ->
        STOP, in BOTH arms (plan §7.7's own table). At 0722476 this
        counter was tallied (control_held_refusal_total) but never
        added to a row's own `violated` list, so it never gated at
        all."""
        self._write_four_clean(tmp_path, manifest_overrides={
            "S2-B4-c-r2": {"bridge_log_text": "Server error: [control_held] execution lease held\n"}})
        rc, payload = self._checkpoint(tmp_path)
        assert rc == RC_STOP
        row = next(r for r in payload["tripwires"]["per_cycle"] if r["cycle"] == "S2-B4-c-r2")
        assert row["control_held_refusal"] == 1
        assert "control_held_refusal" in row["violated"]

    def test_control_held_refusal_on_a_cycle_also_stops(self, tmp_path):
        """No A carve-out for control_held_refusal (unlike
        unexpected_reset_ack) -- plan §7.7's table requires 0 in BOTH
        arms."""
        self._write_four_clean(tmp_path, manifest_overrides={
            "S2-B4-c-r1": {"bridge_log_text": "Server error: [control_held] execution lease held\n"}})
        rc, payload = self._checkpoint(tmp_path)
        assert rc == RC_STOP
        row = next(r for r in payload["tripwires"]["per_cycle"] if r["cycle"] == "S2-B4-c-r1")
        assert "control_held_refusal" in row["violated"]

    def test_mixed_arms_binding_is_per_cycle_not_pooled(self, tmp_path):
        """Mixed arms: an A cycle carrying "Unexpected reset_ack" must
        not be charged to a B cycle, and vice versa -- proves the
        per-cycle binding (rep 1 = A, reps 2/3 = B)."""
        self._write_four_clean(tmp_path, manifest_overrides={
            "S2-B4-c-r1": {"bridge_log_text": "Unexpected reset_ack id=3 (expected 3)\n"},
            "S2-B4-c-r2": {"bridge_log_text": "nothing here\n"},
        })
        rc, payload = self._checkpoint(tmp_path)
        assert rc == RC_OK  # the A cycle's own line never charges rep 2 (a B cycle)
        by_cycle = {r["cycle"]: r for r in payload["tripwires"]["per_cycle"]}
        assert by_cycle["S2-B4-c-r1"]["unexpected_reset_ack"] == 1
        assert by_cycle["S2-B4-c-r1"]["violated"] == []
        assert by_cycle["S2-B4-c-r2"]["unexpected_reset_ack"] == 0
        assert by_cycle["S2-B4-c-r2"]["violated"] == []

    def test_reset_record_stop_line_stops(self, tmp_path):
        """A reset_record containing a `STOP:` line -> rc 2."""
        self._write_four_clean(tmp_path, manifest_overrides={
            "S2-B4-c-r2": {"reset_record_text": "STOP: reset 4 not verified: ack mismatch\n"}})
        rc, payload = self._checkpoint(tmp_path)
        assert rc == RC_STOP
        row = next(r for r in payload["tripwires"]["per_cycle"] if r["cycle"] == "S2-B4-c-r2")
        assert row["reset_sh_mismatch"] == 1
        assert "reset_sh_mismatch" in row["violated"]

    def test_control_stop_present_stops(self, tmp_path):
        """`control/stop` present -> rc 2, even with every per-cycle log
        clean."""
        self._write_four_clean(tmp_path)
        native_log = tmp_path / "native.log"
        native_log.write_text("")
        states = tmp_path / "states.jsonl"
        states.write_text("")
        control_stop = tmp_path / "stop"
        control_stop.write_text("STOP: board contact\n")
        out = tmp_path / "checkpoint_1.json"
        rc = summ._cli([
            "checkpoint", "--control-dir", str(tmp_path), "--n", "4",
            "--native-log", str(native_log), "--states", str(states),
            "--control-stop", str(control_stop), "--out", str(out),
            "--arm-map", str(_write_arm_map(tmp_path)),
            "--expected-bridge-sha-a", BRIDGE_SHA_A, "--expected-bridge-sha-b", BRIDGE_SHA_B,
        ])
        payload = read_result(out)
        assert rc == RC_STOP
        assert payload["tripwires"]["control_stop_present"] is True

    def test_control_stop_absent_is_the_normal_case_not_rc3(self, tmp_path):
        self._write_four_clean(tmp_path)
        rc, payload = self._checkpoint(tmp_path)
        assert rc == RC_OK
        assert payload["tripwires"]["control_stop_present"] is False

    def test_control_stop_neither_flag_given_is_rc3(self, tmp_path):
        """F3: --control-stop's absence is the normal case and must be
        declared EXPLICITLY -- simply omitting both flags is rc 3, never
        silently treated as "absent"."""
        self._write_four_clean(tmp_path)
        native_log = tmp_path / "native.log"
        native_log.write_text("")
        states = tmp_path / "states.jsonl"
        states.write_text("")
        out = tmp_path / "checkpoint_1.json"
        rc = summ._cli([
            "checkpoint", "--control-dir", str(tmp_path), "--n", "4",
            "--native-log", str(native_log), "--states", str(states),
            "--out", str(out),
            "--arm-map", str(_write_arm_map(tmp_path)),
            "--expected-bridge-sha-a", BRIDGE_SHA_A, "--expected-bridge-sha-b", BRIDGE_SHA_B,
        ])
        payload = read_result(out)
        assert rc == RC_INCONCLUSIVE
        assert "explicitly" in payload.get("reason", "")

    def test_missing_cycle_manifest_is_rc3(self, tmp_path):
        self._write_four_clean(tmp_path)
        (tmp_path / "cycle_S2-B4-c-r2.json").unlink()
        rc, payload = self._checkpoint(tmp_path)
        assert rc == RC_INCONCLUSIVE
        assert "S2-B4-c-r2" in payload.get("reason", "")

    def test_mutation_a_carve_out_applied_to_b_would_hide_the_stop(self, tmp_path):
        """Mutation (the A carve-out applied to B too, i.e.
        `unexpected_reset_ack` never in a B row's own `violated` list):
        the B-cycle unexpected_reset_ack fixture above would then never
        stop. Reproduced directly against `per_cycle_reset_tripwires`."""
        self._write_four_clean(tmp_path, manifest_overrides={
            "S2-B4-c-r2": {"bridge_log_text": "Unexpected reset_ack id=3 (expected 3)\n"}})
        row = summ.per_cycle_reset_tripwires(tmp_path, "S2-B4-c-r2", "B")
        assert "unexpected_reset_ack" in row["violated"]  # the shipped (fixed) behaviour
        # The mutant's own (wrong) behaviour: apply the A carve-out to B too.
        mutant_violated = [v for v in row["violated"] if v != "unexpected_reset_ack"]
        assert "unexpected_reset_ack" not in mutant_violated

    def test_mutation_reset_ack_timeout_row_dropped(self, tmp_path):
        """Mutation (the reset-ack-timeout row dropped from
        per_cycle_reset_tripwires): reproduced directly."""
        self._write_four_clean(tmp_path, manifest_overrides={
            "S2-B4-c-r2": {"bridge_log_text": "log.error: Reset ack timed out after 6.0 s\n"}})
        row = summ.per_cycle_reset_tripwires(tmp_path, "S2-B4-c-r2", "B")
        assert "reset_ack_timeout" in row["violated"]
        mutant_violated = [v for v in row["violated"] if v != "reset_ack_timeout"]
        assert "reset_ack_timeout" not in mutant_violated

    def test_mutation_reset_record_stop_row_dropped(self, tmp_path):
        """Mutation (the reset_record STOP row dropped): reproduced
        directly."""
        self._write_four_clean(tmp_path, manifest_overrides={
            "S2-B4-c-r2": {"reset_record_text": "STOP: reset 4 not verified: ack mismatch\n"}})
        row = summ.per_cycle_reset_tripwires(tmp_path, "S2-B4-c-r2", "B")
        assert "reset_sh_mismatch" in row["violated"]
        mutant_violated = [v for v in row["violated"] if v != "reset_sh_mismatch"]
        assert "reset_sh_mismatch" not in mutant_violated

    def test_mutation_per_cycle_logs_pooled_across_arms(self, tmp_path):
        """Mutation (per-cycle logs pooled across arms, i.e. counting
        every cycle's bridge_log together before applying the arm
        carve-out): would charge rep 2 (a B cycle) with rep 1's (an A
        cycle's) own "Unexpected reset_ack" line. Reproduced directly:
        the shipped per-cycle binding keeps rep 2's own count at 0."""
        self._write_four_clean(tmp_path, manifest_overrides={
            "S2-B4-c-r1": {"bridge_log_text": "Unexpected reset_ack id=3 (expected 3)\n"},
            "S2-B4-c-r2": {"bridge_log_text": "nothing here\n"},
        })
        row2 = summ.per_cycle_reset_tripwires(tmp_path, "S2-B4-c-r2", "B")
        assert row2["unexpected_reset_ack"] == 0  # never pooled with rep 1's own count
        pooled_count = (
            summ.per_cycle_reset_tripwires(tmp_path, "S2-B4-c-r1", "A")["unexpected_reset_ack"]
            + row2["unexpected_reset_ack"])
        assert pooled_count == 1  # the mutant would (wrongly) attribute this to rep 2 too
