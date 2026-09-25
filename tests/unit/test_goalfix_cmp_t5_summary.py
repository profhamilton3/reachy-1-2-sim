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
    """F2: checkpoint/final now require the full tripwire input set.
    Tests in this module that are not themselves about tripwire
    behaviour (MB3 rejection, arm-map validation, ordering, ...) get
    clean/empty default logs here so they can reach the logic they
    actually exercise; a test that wants specific tripwire content
    passes its own --bridge-log/--native-log/--states, which wins (see
    `_summ_cli`'s own `inject_default_logs`)."""
    bridge = control_dir / "_default_bridge_log.txt"
    native = control_dir / "_default_native_log.txt"
    states = control_dir / "_default_states.jsonl"
    for p in (bridge, native, states):
        if not p.exists():
            p.write_text("")
    return str(bridge), str(native), str(states)


def _summ_cli(argv, control_dir, inject_default_logs=True):
    extra = [
        "--arm-map", str(_write_arm_map(control_dir)),
        "--expected-bridge-sha-a", BRIDGE_SHA_A,
        "--expected-bridge-sha-b", BRIDGE_SHA_B,
    ]
    if inject_default_logs and "--bridge-log" not in argv:
        bridge, native, states = _default_log_paths(control_dir)
        extra += ["--bridge-log", bridge, "--native-log", native, "--states", states]
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
    CLI."""

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

    def test_no_log_flags_is_unaffected(self, tmp_path):
        """F2 (merge verdict §2; 2026-09-25 pr144-f1-f6 assignment):
        INVERTED from its pre-F2 name/assertion (kept, not deleted, per
        the assignment's own instruction). At 0722476 this pinned
        "no log flags -> tripwires are skipped, rc 0" -- exactly the bug
        (tripwires opt-in, evidence never actually checked). Tripwires
        are now mandatory for checkpoint/final: omitting all three
        (--bridge-log/--native-log/--states) is rc 3, never rc 0."""
        self._write_four_clean(tmp_path)
        out = tmp_path / "checkpoint_1.json"
        rc = _summ_cli(["checkpoint", "--control-dir", str(tmp_path), "--n", "4",
                        "--out", str(out)], tmp_path, inject_default_logs=False)
        assert rc == RC_INCONCLUSIVE
        payload = read_result(out)
        assert payload.get("ok") is False
        assert "mandatory" in payload.get("reason", "")

    def test_missing_required_log_is_rc3(self, tmp_path):
        """F2: supplying only '--bridge-log' (without --native-log/
        --states) is rc 3 -- the F2 mandatory-input check fires before
        any per-counter missing_log check is even reached, since the
        full set is required up front."""
        self._write_four_clean(tmp_path)
        bridge_log = tmp_path / "bridge_log.txt"
        bridge_log.write_text("nothing interesting\n")
        out = tmp_path / "checkpoint_1.json"
        rc = _summ_cli(["checkpoint", "--control-dir", str(tmp_path), "--n", "4",
                        "--bridge-log", str(bridge_log), "--out", str(out)], tmp_path)
        assert rc == RC_INCONCLUSIVE
        assert "mandatory" in read_result(out).get("reason", "")

    def test_nonzero_lease_acquisition_stops_a_clean_checkpoint(self, tmp_path):
        self._write_four_clean(tmp_path)
        bridge_log = tmp_path / "bridge_log.txt"
        bridge_log.write_text("nothing here\n")
        native_log = tmp_path / "native.log"
        native_log.write_text("Execution lease granted to 'x' (mover 'x', 30s)\n")
        states = tmp_path / "states.jsonl"
        states.write_text("")
        out = tmp_path / "checkpoint_1.json"
        rc = _summ_cli([
            "checkpoint", "--control-dir", str(tmp_path), "--n", "4",
            "--bridge-log", str(bridge_log),
            "--native-log", str(native_log),
            "--states", str(states),
            "--out", str(out),
        ], tmp_path)
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
    files in `tmp_path`."""

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

    def _checkpoint(self, tmp_path, **logs):
        out = tmp_path / "checkpoint_1.json"
        argv = ["checkpoint", "--control-dir", str(tmp_path), "--n", "4", "--out", str(out)]
        for flag, value in logs.items():
            argv += [flag, value]
        rc = _summ_cli(argv, tmp_path)
        return rc, read_result(out)

    def test_s3_real_bridge_log_content_is_read_and_stops(self, tmp_path):
        """S3: a bridge log FILE containing the control_held string, for
        a session with a B cycle -> rc 2, control_held_refusal count 1.
        At 0722476 this counted 0 (the path string has no such text)."""
        self._write_four_clean(tmp_path)
        bridge_log = tmp_path / "bridge_log_r2.txt"
        bridge_log.write_text("log.warning: Server error: [control_held] execution lease held\n")
        native_log = tmp_path / "native.log"
        native_log.write_text("")
        states = tmp_path / "states.jsonl"
        states.write_text("")
        rc, payload = self._checkpoint(
            tmp_path, **{"--bridge-log": str(bridge_log), "--native-log": str(native_log),
                         "--states": str(states)})
        assert rc == RC_STOP
        assert payload["tripwires"]["control_held_refusal"]["count"] == 1

    def test_s4_nonexistent_log_paths_give_rc3(self, tmp_path):
        """S4: --bridge-log/--native-log naming files that do not exist
        -> rc 3, never rc 0. At 0722476 a nonexistent path's own STRING
        had no tripwire text in it either, so counts came back 0 and the
        checkpoint passed at rc 0."""
        self._write_four_clean(tmp_path)
        rc, payload = self._checkpoint(
            tmp_path,
            **{"--bridge-log": str(tmp_path / "does_not_exist_bridge.log"),
               "--native-log": str(tmp_path / "does_not_exist_native.log"),
               "--states": str(tmp_path / "does_not_exist_states.jsonl")})
        assert rc == RC_INCONCLUSIVE
        assert "does_not_exist" in payload.get("reason", "")

    def test_s5_path_name_matches_but_contents_are_clean(self, tmp_path):
        """S5: the log PATH's own name contains the control_held pattern,
        but its contents are clean -> count 0, rc 0. At 0722476 this
        counted 1 (matching the path string), STOPping a clean session."""
        self._write_four_clean(tmp_path)
        bridge_log = tmp_path / "server error [control_held] but contents are clean.log"
        bridge_log.write_text("nothing interesting here\n")
        native_log = tmp_path / "native.log"
        native_log.write_text("")
        states = tmp_path / "states.jsonl"
        states.write_text("")
        rc, payload = self._checkpoint(
            tmp_path, **{"--bridge-log": str(bridge_log), "--native-log": str(native_log),
                         "--states": str(states)})
        assert rc == RC_OK
        assert payload["tripwires"]["control_held_refusal"]["count"] == 0

    def test_empty_existing_log_counts_zero_not_missing(self, tmp_path):
        self._write_four_clean(tmp_path)
        bridge_log = tmp_path / "bridge_log.txt"
        bridge_log.write_text("")  # exists, empty
        native_log = tmp_path / "native.log"
        native_log.write_text("")
        states = tmp_path / "states.jsonl"
        states.write_text("")
        rc, payload = self._checkpoint(
            tmp_path, **{"--bridge-log": str(bridge_log), "--native-log": str(native_log),
                         "--states": str(states)})
        assert rc == RC_OK
        assert payload["tripwires"]["control_held_refusal"]["count"] == 0
        assert payload["tripwires"]["lease_acquisition"]["count"] == 0

    def test_mutation_cli_passes_path_instead_of_contents(self, tmp_path):
        """Mutation at the shipped call site (the F1 fix's own seam):
        `mb7_tripwire_report(bridge_log=args.bridge_log, ...)` -- the
        PATH, not `read_log_text(args.bridge_log)`. Reproduced directly
        (not by editing the module in-process): passing the bridge log's
        own PATH STRING to `summ.count_tripwires` the way the pre-fix CLI
        did shows the mismatch the fix closes."""
        bridge_log = tmp_path / "bridge_log_r2.txt"
        bridge_log.write_text("log.warning: Server error: [control_held] execution lease held\n")
        # The mutant's own behaviour: hand the PATH STRING itself to the
        # text-counting function, exactly as summary._cli did at 0722476.
        mutant_counts = summ.count_tripwires(bridge_log=str(bridge_log))
        assert mutant_counts["control_held_refusal"] == 0
        # The fix's own behaviour (this test's real assertion target):
        # reading the file's CONTENTS first.
        fixed_counts = summ.count_tripwires(bridge_log=summ.read_log_text(str(bridge_log)))
        assert fixed_counts["control_held_refusal"] == 1

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
