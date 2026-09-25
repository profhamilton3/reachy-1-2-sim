"""D8 narrow-grant historical validation (opt-in ONLY -- never run as part
of an ordinary ``pytest`` sweep).

Scope, exactly as recorded in
IITG-Reachy-Project/outputs/b4-goalfix-comparison-plan-2026-09-24.md §11
(D8, approved 2026-09-24): a read-only run of the echo classifier over the
three sessions already analysed on 2026-09-23 (B4 s1, B1 s1, B2 s2),
limited to ``commands.jsonl``/``states.jsonl`` checked against their
``SHA256SUMS`` -- no other file in those session directories is opened.
This module never reads ``ledger.csv``, ``recorder_logs/``, or anything
under the Stage 2 worktree.

Two separate numbers are computed and compared, never merged or tuned
toward each other:

1. The ORIGINAL ``verify_goal_feedback.classify`` definition's echo count
   (wall-time lookback, no coincidence subclassing) -- must reproduce the
   published 85,372 / 83,750 / 87,294.
2. A large-file-optimised extension of the same arrays that adds
   simulation-time lookback, start-coincidence and timing-ambiguous
   handling (the same rules as ``tools/goalfix_cmp/echo.py``) -- reported
   separately, as a delta, never used to adjust (1).

Path-coincidence is NOT computed here: it requires the route/waypoint
context normally read from ``ledger.csv``, which is outside this grant's
two-file scope. Noted as a limitation in the handoff, not worked around.

Gating: skipped unless ``GOALFIX_CMP_ALLOW_HISTORICAL=1`` is set AND the
three session directories are present on disk (they will not be on a
laptop other than the one this comparison's evidence lives on).
"""
import ast
import hashlib
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../..",):
    sys.path.insert(0, os.path.join(_HERE, _p))

pytestmark = pytest.mark.skipif(
    os.environ.get("GOALFIX_CMP_ALLOW_HISTORICAL") != "1",
    reason="opt-in only (D8 narrow grant) -- set GOALFIX_CMP_ALLOW_HISTORICAL=1 to run; "
           "reads real Stage 2 evidence (commands.jsonl/states.jsonl only, checksum-verified)")

_REFERENCE_DIR = os.path.join(_HERE, "../fixtures/goalfix_cmp/reference")
_VERIFY_GOAL_FEEDBACK_PATH = os.path.join(_REFERENCE_DIR, "verify_goal_feedback.py")
_VERIFY_GOAL_FEEDBACK_SHA256 = (
    "9862d8f15244e423ffe55b79e73418719c2af90c460c5c199ffed8cd2b2cdd5b")

#: session tag -> (evidence dir, published echo count from
#: outputs/analysis-2026-09-23-goal-feedback-verification-report.md).
SESSIONS = {
    "B4": ("~/e1-stage2-B4-s1-2026-09-18", 85372),
    "B1": ("~/e1-stage2-B1-s1-2026-09-19", 83750),
    "B2": ("~/e1-stage2-B2-s2-2026-09-22", 87294),
}

LOOKBACK_S = 0.5


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_from_vendored(*names):
    actual = _sha256(_VERIFY_GOAL_FEEDBACK_PATH)
    assert actual == _VERIFY_GOAL_FEEDBACK_SHA256, "vendored verify_goal_feedback.py has drifted"
    source = open(_VERIFY_GOAL_FEEDBACK_PATH).read()
    tree = ast.parse(source)
    wanted = set(names)
    nodes = [n for n in tree.body
             if (isinstance(n, ast.FunctionDef) and n.name in wanted)
             or (isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets))]
    ns = {"np": np, "hashlib": hashlib, "os": os, "re": __import__("re"), "json": __import__("json")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), _VERIFY_GOAL_FEEDBACK_PATH, "exec"), ns)
    for name in names:
        assert name in ns, f"{name!r} not found in vendored verify_goal_feedback.py"
    return ns


def _session_dirs_present():
    return all(os.path.isdir(os.path.expanduser(path)) for path, _ in SESSIONS.values())


pytestmark = [pytestmark, pytest.mark.skipif(
    not _session_dirs_present(), reason="Stage 2 session directories not present on this host")]


import re as _re  # noqa: E402
import json as _json  # noqa: E402

from tools.goalfix_cmp._simtime import (  # noqa: E402
    assign_state_epochs, bracket_commands,
)

#: Extends the vendored RX_HEAD to also capture sim_time_s, from the exact
#: State.asdict() field order (sim_step, sim_time_s, wall_time_ns, cmd_seq)
#: -- native_mujoco/protocol.py -- so the fast byte-regex path used for the
#: original reproduction can also feed the simulation-time extension
#: without a second, full JSON parse of a multi-gigabyte file.
_RX_HEAD_SIMTIME = _re.compile(
    rb'"sim_step":(\d+),"sim_time_s":([-0-9.eE+]+),"wall_time_ns":(\d+),"cmd_seq":(\d+)')
_RX_POS = _re.compile(rb'"position_rad":([-0-9.eE+]+)')


def _load_states_with_simtime(path):
    """(sim_time_s, cmd_seq, epoch, position[:8]) arrays, same byte-regex
    approach as the vendored ``load_states`` (never a full JSON parse --
    this file is multi-gigabyte). ``epoch`` comes from ``sim_step`` drops,
    exactly ``evidence.py``'s own rule."""
    sim_step, sim_t, seq, pos = [], [], [], []
    with open(path, "rb") as f:
        for line in f:
            head = line[:2700]
            m = _RX_HEAD_SIMTIME.search(head)
            if not m:
                continue
            p = _RX_POS.findall(head)[:8]
            if len(p) < 8:
                continue
            sim_step.append(int(m.group(1)))
            sim_t.append(float(m.group(2)))
            seq.append(int(m.group(4)))
            pos.append([float(x) for x in p])
    sim_step = np.array(sim_step, dtype=np.int64)
    epoch = assign_state_epochs(sim_step)
    return (np.array(sim_t, dtype=np.float64), np.array(seq), epoch, np.array(pos))


def _load_command_epochs(path):
    """(seq, epoch) for every joint_command line, in file order -- a full
    parse, but commands.jsonl is tens of MB, not gigabytes. The Nth reset
    line ends epoch N-1, exactly ``_simtime.assign_command_epochs``'s rule;
    computed inline (not via that helper) because it also needs ``seq``
    from the same pass, and this file is cheap enough not to need the
    ``load_cmds``-then-separate-epoch-pass indirection."""
    seqs, epochs = [], []
    epoch = 0
    with open(path) as f:
        for line in f:
            d = _json.loads(line)
            t = d.get("type")
            if t == "reset":
                epoch += 1
            elif t == "joint_command":
                seqs.append(d["seq"])
                epochs.append(epoch)
    return np.array(seqs, dtype=np.int64), np.array(epochs, dtype=np.int64)


def _reclassify_with_sim_time_and_coincidence(targets8, cmd_seq, cmd_epoch, sim_t, sim_seq,
                                               state_epoch, spos32):
    """Start-coincidence and timing-ambiguous, layered onto the original's
    exact-match search -- simulation-time lookback (per reset epoch)
    instead of wall-time, and the t_hi-state exclusion. Path-coincidence is
    out of scope here (no route context available under the D8 grant) -- an
    exact match that is neither a start- nor timing-ambiguous coincidence
    is left as "genuine_or_path", the single number this delta-report can
    honestly give without route data."""
    n = len(targets8)
    label = np.full((n, 8), "carry", dtype=object)
    brackets = bracket_commands(cmd_seq, cmd_epoch, sim_t, sim_seq, state_epoch)

    epoch_first_index = {}
    for i, e in enumerate(cmd_epoch):
        epoch_first_index.setdefault(int(e), i)

    prev_by_epoch = {}
    for i in range(n):
        epoch = int(cmd_epoch[i])
        tgt = np.asarray(targets8[i])
        prev = prev_by_epoch.get(epoch)
        b = brackets[i]
        if b.unplaceable:
            for j in range(8):
                label[i, j] = "carry" if prev is not None and tgt[j] == prev[j] else "fresh"
            prev_by_epoch[epoch] = tgt
            continue

        e_idx = np.nonzero(state_epoch == epoch)[0]
        e_sim_t = sim_t[e_idx]
        t_hi = b.t_hi
        a = int(np.searchsorted(e_sim_t, t_hi - LOOKBACK_S))
        hi_local = int(np.searchsorted(e_sim_t, t_hi))
        win = spos32[e_idx[a:hi_local]] if hi_local > a else spos32[0:0]
        hi_match_row = (spos32[e_idx[hi_local]] if hi_local < len(e_idx) else None)

        for j in range(8):
            if prev is not None and tgt[j] == prev[j]:
                continue
            hit = np.nonzero(win[:, j] == tgt[j])[0] if len(win) else np.array([], dtype=int)
            hi_match = hi_match_row is not None and hi_match_row[j] == tgt[j]
            if len(hit) == 0 and not hi_match:
                label[i, j] = "fresh"
            elif len(hit) == 0 and hi_match:
                label[i, j] = "timing_ambiguous"
            elif i == epoch_first_index[epoch]:
                label[i, j] = "start_coincidence"
            else:
                label[i, j] = "genuine_or_path"
        prev_by_epoch[epoch] = tgt
    return label


class TestHistoricalD8:
    @pytest.mark.parametrize("tag", sorted(SESSIONS))
    def test_reproduces_published_count_and_reports_new_classifier_delta(self, tag):
        ev_dir, published_count = SESSIONS[tag]
        ev_dir = os.path.expanduser(ev_dir)
        sums_path = os.path.join(ev_dir, "SHA256SUMS")
        assert os.path.isfile(sums_path), f"{tag}: no SHA256SUMS at session root"

        ns = _extract_from_vendored("sha", "sums_of", "verified", "load_states", "load_cmds",
                                     "classify", "LOOKBACK_NS", "RX_HEAD", "RX_POS", "RX_CMP")
        sums = ns["sums_of"](ev_dir)
        run = os.listdir(os.path.join(ev_dir, "e1_server_runs"))[0]
        base = f"e1_server_runs/{run}"

        states_path = ns["verified"](ev_dir, f"{base}/states.jsonl", sums)
        cmds_path = ns["verified"](ev_dir, f"{base}/commands.jsonl", sums)

        st, sseq, spos, _scmp = ns["load_states"](states_path)
        spos32 = spos.astype(np.float32).astype(np.float64)
        cmds = ns["load_cmds"](cmds_path)
        cseq = np.array([c[0] for c in cmds])
        first_ge = np.searchsorted(np.maximum.accumulate(sseq), cseq, side="left")
        ok = first_ge < len(st)
        placed_cmds = [c for c, k in zip(cmds, ok) if k]
        t_hi = st[first_ge[ok]]

        cls, _age = ns["classify"](placed_cmds, st, spos32, t_hi)
        original_echo_count = int((cls == 1).sum())

        assert original_echo_count == published_count, (
            f"{tag}: original definition gives {original_echo_count}, "
            f"published is {published_count}")

        # -- new classifier delta (report only, sim-time + coincidence) --
        sim_t, sim_sseq, state_epoch, sim_spos = _load_states_with_simtime(states_path)
        sim_spos32 = sim_spos.astype(np.float32).astype(np.float64)
        n_state_epochs = int(state_epoch.max()) + 1

        cmd_seq_all, cmd_epoch_all = _load_command_epochs(cmds_path)
        n_cmd_epochs = int(cmd_epoch_all.max()) + 1
        assert len(cmd_seq_all) == len(cmds), (
            f"{tag}: command_epoch pass found {len(cmd_seq_all)} joint_command lines, "
            f"vendored load_cmds found {len(cmds)}")
        assert np.array_equal(cmd_seq_all, cseq), f"{tag}: seq order mismatch between the two parses"

        targets8 = [c[1] for c in cmds]
        new_labels = _reclassify_with_sim_time_and_coincidence(
            targets8, cmd_seq_all, cmd_epoch_all, sim_t, sim_sseq, state_epoch, sim_spos32)
        new_counts = {lbl: int((new_labels == lbl).sum())
                      for lbl in ("carry", "fresh", "start_coincidence",
                                  "timing_ambiguous", "genuine_or_path")}

        print(f"\n{tag}: original(wall-time, single continuous window, no subclass) "
              f"echo={original_echo_count} (matches published {published_count})")
        print(f"{tag}: {n_state_epochs} state epoch(s) (sim_step drops), "
              f"{n_cmd_epochs} command epoch(s) (reset entries) -- "
              f"{'consistent' if n_state_epochs == n_cmd_epochs else 'MISMATCH'}")
        print(f"{tag}: new (sim-time, per-epoch, coincidence, no route context) genuine_or_path="
              f"{new_counts['genuine_or_path']} start_coincidence={new_counts['start_coincidence']} "
              f"timing_ambiguous={new_counts['timing_ambiguous']} fresh={new_counts['fresh']} "
              f"carry={new_counts['carry']}")
        print(f"{tag}: delta (new genuine_or_path - original echo) = "
              f"{new_counts['genuine_or_path'] - original_echo_count}")

        if n_state_epochs != n_cmd_epochs:
            # Report-only section: a trailing reset with no further
            # commands before the recording ends produces exactly one more
            # state epoch than command epoch and is benign (evidence.py's
            # own strict cross-check would flag this as evidence_incomplete
            # for a REAL cycle -- noted in the handoff, not asserted here
            # since this delta report is diagnostic, not a gate).
            print(f"{tag}: NOTE epoch count mismatch ({n_state_epochs} state vs "
                  f"{n_cmd_epochs} command) -- consistent with a trailing reset "
                  "and no further commands before the recording ends")
