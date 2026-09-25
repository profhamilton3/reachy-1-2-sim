"""T9 acceptance tests (assignment §2, T9): reset tripwires anchored to
the real source lines (review §3.7) -- "lease acquisition" never matched
anything the native server logs (it logs "Execution lease granted to
...", server.py:1059) and "pause" was a bare substring. Includes a
source-drift guard: each cited line is grepped from the real file, so a
future edit that changes the wording fails this test loudly rather than
silently reopening the gap."""
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

    def test_ungroundable_counters_are_always_none(self):
        counts = summ.count_tripwires(
            bridge_log="anything", watcher_log="anything", native_log="anything",
            reset_sh_log="anything")
        assert counts["control_held_refusal"] is None
        assert counts["pause_message"] is None


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
