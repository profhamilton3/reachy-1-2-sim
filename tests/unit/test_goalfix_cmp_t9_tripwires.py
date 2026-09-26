"""T9 acceptance tests (assignment §2, T9): reset tripwires anchored to
the real source lines (review §3.7) -- "lease acquisition" never matched
anything the native server logs (it logs "Execution lease granted to
...", server.py:1059) and "pause" was a bare substring. Includes a
source-drift guard: each cited line is grepped from the real file, so a
future edit that changes the wording fails this test loudly rather than
silently reopening the gap."""
import json
import os
import re
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../.."))

from tools.goalfix_cmp import summary as summ  # noqa: E402

_REPO = os.path.abspath(os.path.join(_HERE, "../.."))


class TestSourceLinesHaveNotDrifted:
    """Greps the ACTUAL repo files for each cited literal -- catches drift,
    per the assignment's own instruction, rather than trusting the
    citation forever."""

    @pytest.mark.parametrize("name,sources", list(summ.TRIPWIRE_SOURCE_LINES.items()))
    def test_source_line_still_contains_the_cited_text(self, name, sources):
        for relpath, lineno, literal in sources:
            path = os.path.join(_REPO, relpath)
            with open(path) as f:
                lines = f.readlines()
            # The literal may be a %-format string with placeholders; only
            # check the text up to the first "%" is present verbatim.
            fixed_part = literal.split("%")[0]
            window = "".join(lines[max(0, lineno - 3):lineno + 2])
            assert fixed_part in window, (
                f"{relpath}:{lineno}: {fixed_part!r} no longer found near the cited line "
                "-- update TRIPWIRE_SOURCE_LINES and _TRIPWIRE_PATTERN_TEXT together")


class TestGroundedPatterns:
    def test_reset_ack_timeout_matches_bridge_and_watcher_real_strings(self):
        bridge_text = "log.error: Reset ack timed out after 6.0 s — entering ABORTED state"
        watcher_text = "reset_watcher: reset ack timed out or was refused; not publishing"
        counts = summ.count_tripwires(bridge_log=bridge_text, watcher_log=watcher_text)
        assert counts["reset_ack_timeout"] == 2

    def test_unexpected_reset_ack_matches_bridge_real_string(self):
        counts = summ.count_tripwires(bridge_log="Unexpected reset_ack id=3 (expected 3)")
        assert counts["unexpected_reset_ack"] == 1

    def test_lease_acquisition_matches_native_real_string_not_old_pattern(self):
        native_text = "Execution lease granted to 'client-1' (mover 'client-1', 30s)"
        counts = summ.count_tripwires(native_log=native_text)
        assert counts["lease_acquisition"] == 1
        # The OLD pattern ("lease acquisition") never appears in the real
        # string -- confirms this test would have failed against the bug.
        assert "lease acquisition" not in native_text.lower()

    def test_reset_sh_mismatch_matches_stop_prefix_not_only_ack_mismatch(self):
        """"ack mismatch or other STOP" (plan §7.7) -- the old pattern
        ("ack mismatch") missed every OTHER reset.sh STOP reason."""
        other_stop = "STOP: reset 4 not verified: request publish failed (rc=1)"
        counts = summ.count_tripwires(reset_sh_log=other_stop)
        assert counts["reset_sh_mismatch"] == 1
        assert "ack mismatch" not in other_stop.lower()

    def test_missing_log_is_none_not_zero(self):
        counts = summ.count_tripwires()  # nothing supplied at all
        assert counts["reset_ack_timeout"] is None
        assert counts["lease_acquisition"] is None

    def test_pause_message_stays_ungroundable_via_log_text(self):
        counts = summ.count_tripwires(
            bridge_log="anything", watcher_log="anything", native_log="anything",
            reset_sh_log="anything")
        assert counts["pause_message"] is None

    def test_control_held_refusal_now_groundable_via_bridge_log(self):
        """MB7 (merge verdict, 2026-09-25 stage-repairs assignment §3;
        owner decision (a), readiness review §4 Q2): the earlier
        "ungroundable" claim was wrong for the bridge's OWN connection --
        the bridge logs every server error it receives
        (mujoco_remote_backend.py:486-488), including a native
        control_held refusal sent to it."""
        real_text = 'log.warning: Server error: [control_held] execution lease held'
        counts = summ.count_tripwires(bridge_log=real_text)
        assert counts["control_held_refusal"] == 1
        # An unrelated bridge log still correctly counts 0, not None.
        assert summ.count_tripwires(bridge_log="anything")["control_held_refusal"] == 0
        # Still None only when NO bridge log is supplied at all.
        assert summ.count_tripwires()["control_held_refusal"] is None


class TestCheckTripwires:
    def test_b_expected_all_zero_and_complete(self):
        counts = {k: 0 for k in summ.TRIPWIRE_SOURCES}
        r = summ.check_tripwires(counts, "B")
        assert r.violated == []
        # control_held_refusal/pause_message are ungroundable, so even an
        # explicit 0 here (not None) is accepted -- this fixture supplies
        # 0, not None, to isolate the "violated" path from "incomplete".
        assert r.incomplete == []

    def test_a_allows_unexpected_reset_ack_only(self):
        counts = {k: 0 for k in summ.TRIPWIRE_SOURCES}
        counts["unexpected_reset_ack"] = 2
        assert summ.check_tripwires(counts, "A").violated == []
        assert summ.check_tripwires(counts, "B").violated == ["unexpected_reset_ack"]

    def test_reset_ack_timeout_always_violates(self):
        counts = {k: 0 for k in summ.TRIPWIRE_SOURCES}
        counts["reset_ack_timeout"] = 1
        assert summ.check_tripwires(counts, "A").violated == ["reset_ack_timeout"]
        assert summ.check_tripwires(counts, "B").violated == ["reset_ack_timeout"]

    def test_a_missing_log_is_incomplete_not_a_silent_pass(self):
        counts = summ.count_tripwires()  # every source missing
        r = summ.check_tripwires(counts, "B")
        assert "reset_ack_timeout" in r.incomplete
        assert "lease_acquisition" in r.incomplete
        assert r.violated == []  # never silently passed as "0 violations"


class TestMutationOldPatternsRemoved:
    def test_old_lease_acquisition_pattern_would_have_missed_the_real_log(self):
        native_text = "Execution lease granted to 'client-1' (mover 'client-1', 30s)"
        old_pattern_matches = "lease acquisition" in native_text.lower()
        assert not old_pattern_matches
        assert summ.count_tripwires(native_log=native_text)["lease_acquisition"] == 1

    def test_old_bare_pause_pattern_would_have_false_positived(self):
        unrelated_text = "the recorder is on pause between legs, nothing wrong here"
        old_pattern_matches = "pause" in unrelated_text.lower()
        assert old_pattern_matches  # the bare-substring bug
        # The current implementation never matches "pause" at all (it is
        # ungroundable, see TRIPWIRE_SOURCES) -- so this text can never
        # cause a false positive under the fix.
        assert summ.count_tripwires()["pause_message"] is None


class TestMB7PauseTripwire:
    """MB7 (merge verdict, 2026-09-25 stage-repairs assignment §3): the
    "pause" counter is states.jsonl-based, never log text."""

    def _state_row(self, sim_step, paused=False):
        return {"type": "state", "sim_step": sim_step, "paused": paused}

    def _write(self, tmp_path, name, rows):
        path = tmp_path / name
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return str(path)

    def test_missing_path_is_none_not_zero(self):
        assert summ.count_pause_tripwire(None) is None
        assert summ.count_pause_tripwire([]) is None

    def test_clean_run_counts_zero(self, tmp_path):
        rows = [self._state_row(i) for i in range(5)]
        p = self._write(tmp_path, "states.jsonl", rows)
        assert summ.count_pause_tripwire([p]) == 0

    def test_paused_true_row_counts(self, tmp_path):
        rows = [self._state_row(0), self._state_row(1, paused=True), self._state_row(2)]
        p = self._write(tmp_path, "states.jsonl", rows)
        assert summ.count_pause_tripwire([p]) == 1

    def test_duplicate_sim_step_within_epoch_counts(self, tmp_path):
        # A state pushed twice at the same sim_step (paused, not
        # advancing) -- counted even when `paused` itself was never set.
        rows = [self._state_row(0), self._state_row(1), self._state_row(1), self._state_row(2)]
        p = self._write(tmp_path, "states.jsonl", rows)
        assert summ.count_pause_tripwire([p]) == 1

    def test_blind_spot_is_always_documented_never_folded_into_zero(self):
        report = summ.mb7_tripwire_report(bridge_log="x", native_log="x", states_paths=None)
        assert report["pause"]["count"] is None
        assert report["pause"]["blind_spots"] == [summ.MB7_BLIND_SPOTS["pause"]]
        # Even when groundable and 0, the blind spot is STILL reported.
        report2 = summ.mb7_tripwire_report(bridge_log="x", native_log="x", states_paths=["/dev/null"])
        assert report2["lease_acquisition"]["count"] == 0
        assert report2["lease_acquisition"]["blind_spots"] == [summ.MB7_BLIND_SPOTS["lease_acquisition"]]

    def test_check_mb7_tripwires_violated_vs_missing(self):
        report = {
            "lease_acquisition": {"count": 0, "observable_scope": "", "blind_spots": []},
            "control_held_refusal": {"count": 2, "observable_scope": "", "blind_spots": []},
            "pause": {"count": None, "observable_scope": "", "blind_spots": []},
        }
        r = summ.check_mb7_tripwires(report)
        assert r.violated == ["control_held_refusal"]
        assert r.missing_log == ["pause"]

    def test_mutation_never_wired_in_would_ignore_tripwires(self, tmp_path):
        """Mutation (never call mb7_tripwire_report/check_mb7_tripwires
        from the checkpoint/final CLI, as at b53a86b): supplying
        --bridge-log/--native-log/--states has NO effect on rc at all.
        Verified directly against a b53a86b copy of summary.py, whose
        CLI does not even accept these flags."""
        import subprocess
        result = subprocess.run(
            ["git", "show", "b53a86b:tools/goalfix_cmp/summary.py"],
            cwd=_REPO, capture_output=True, text=True)
        assert "--bridge-log" not in result.stdout
        assert "mb7_tripwire_report" not in result.stdout
