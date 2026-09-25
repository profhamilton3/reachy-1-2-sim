"""Simulation-time keying and command bracketing (plan §7.0).

Every measured interval is simulation time, keyed by (reset_epoch,
sim_time_s), because sim_step/sim_time_s restart at every reset. Wall or
monotonic time (``wall_time_ns``, which IS ``time.monotonic_ns()`` despite
the name) is used only to order records within a file and to locate leg
windows through the linker sidecars -- never as a measured interval.

Epoch assignment: ``commands.jsonl`` carries "reset" entries inline, in the
order they were issued -- the Nth reset entry ends epoch N-1 and starts
epoch N. Every ``joint_command`` entry is assigned the epoch of the most
recent reset seen before it in file order (epoch 0 before any reset).
``states.jsonl`` has no reset entries, so a state's epoch is inferred from
its own ``sim_step``: a drop (``sim_step[i] < sim_step[i-1]``) starts a new
epoch. ``evidence.py`` cross-checks the two counts against each other rather
than trusting either alone.

Commands carry no timestamp. A command's application time is the bracket
``[t_lo, t_hi]`` between the last state (in the SAME epoch) that has not yet
reported it and the first that has, by ``cmd_seq``, exactly as
``verify_goal_feedback.py``'s ``t_hi`` bracketing does for the single-epoch
case -- extended here with the ``t_lo`` companion and per-epoch scoping the
plan's §7.0 requires. A command that is never reported, or whose bracket
would reach into the previous epoch (it is the epoch's first reported
command), is unplaceable -- counted, never dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Bracket:
    epoch: int
    t_lo: Optional[float]          # sim_time_s, or None if unplaceable
    t_hi: Optional[float]          # sim_time_s, or None if unplaceable
    hi_state_index: Optional[int]  # index into the full state arrays
    seq_ambiguous: bool = False    # T2: a mid-epoch restart made this seq ambiguous

    @property
    def unplaceable(self) -> bool:
        return self.t_hi is None or self.t_lo is None or self.seq_ambiguous


def assign_state_epochs(sim_step: Sequence[int]) -> np.ndarray:
    """Epoch index per state row, from ``sim_step`` drops."""
    sim_step = np.asarray(sim_step)
    epoch = np.zeros(len(sim_step), dtype=np.int64)
    e = 0
    for i in range(1, len(sim_step)):
        if sim_step[i] < sim_step[i - 1]:
            e += 1
        epoch[i] = e
    return epoch


def assign_command_epochs(command_kinds: Sequence[str]) -> np.ndarray:
    """Epoch index per ``commands.jsonl`` row (any kind). A ``"reset"`` row
    is itself stamped with the epoch it ENDS; callers exclude reset rows
    before bracketing (only ``joint_command`` rows are bracketed)."""
    epoch = np.zeros(len(command_kinds), dtype=np.int64)
    e = 0
    for i, kind in enumerate(command_kinds):
        epoch[i] = e
        if kind == "reset":
            e += 1
    return epoch


@dataclass(frozen=True)
class BridgeSessionInfo:
    """Per epoch (T2/review §3.1/M2): whether a bridge restart happened at
    the epoch's own start (the container-recreate regime, where native's
    carried ``cmd_seq`` outranks the new bridge's own low-numbered
    ``seq``), where in the state array that carried value was first
    replaced, and whether the epoch is seq-ambiguous."""
    restart: bool = False
    replaced_state_index: Optional[int] = None
    ambiguous: bool = False


def detect_bridge_sessions(
    cmd_seq: np.ndarray, cmd_epoch: np.ndarray,
    state_cmd_seq: np.ndarray, state_epoch: np.ndarray,
) -> Dict[int, BridgeSessionInfo]:
    """One ``BridgeSessionInfo`` per epoch present in ``cmd_epoch``.

    A restart is detected exactly as the assignment specifies: the epoch's
    first ``joint_command`` has ``seq`` <= the ``cmd_seq`` carried by that
    epoch's first state (native's own counter, untouched by the reset that
    started this epoch -- ``server.py:338-351``/``:377``). Where it is
    detected, ``replaced_state_index`` is the first state (global index)
    whose reported ``cmd_seq`` differs from that carried value -- the point
    at which the NEW bridge's own numbering starts being reflected.

    A MID-epoch restart -- the command stream's own ``seq`` failing to
    strictly increase within one epoch, e.g. a bridge reconnect with no
    accompanying native reset -- is reported as ``ambiguous`` rather than
    resolved into further sessions: disambiguating which of two
    same-numbered sessions a given state's ``cmd_seq`` belongs to needs
    more than ``seq`` alone, and guessing is exactly what the assignment
    forbids ("Seq ambiguity is evidence incomplete... Do not guess").
    """
    out: Dict[int, BridgeSessionInfo] = {}
    for epoch in np.unique(cmd_epoch):
        epoch = int(epoch)
        cidx = np.nonzero(cmd_epoch == epoch)[0]
        sidx = np.nonzero(state_epoch == epoch)[0]
        if len(cidx) == 0 or len(sidx) == 0:
            out[epoch] = BridgeSessionInfo()
            continue

        c_seqs = cmd_seq[cidx]
        if len(c_seqs) > 1 and np.any(np.diff(c_seqs) <= 0):
            out[epoch] = BridgeSessionInfo(ambiguous=True)
            continue

        carried = int(state_cmd_seq[sidx[0]])
        first_cmd_seq = int(c_seqs[0])
        if first_cmd_seq > carried:
            out[epoch] = BridgeSessionInfo()
            continue

        s_seqs = state_cmd_seq[sidx]
        replaced = np.nonzero(s_seqs != carried)[0]
        replaced_global = int(sidx[replaced[0]]) if len(replaced) else None
        out[epoch] = BridgeSessionInfo(restart=True, replaced_state_index=replaced_global)
    return out


def bracket_commands(
    cmd_seq: Sequence[int],
    cmd_epoch: Sequence[int],
    state_sim_time_s: Sequence[float],
    state_cmd_seq: Sequence[int],
    state_epoch: Sequence[int],
    bridge_sessions: Optional[Dict[int, BridgeSessionInfo]] = None,
) -> List[Bracket]:
    """``[t_lo, t_hi]`` for every command, one call for the whole evidence
    set (grouped and vectorised per epoch internally, so this stays well
    under the §2.8 60 s/12-minute-cycle budget).

    ``bridge_sessions`` (``detect_bridge_sessions``'s own output; computed
    here if omitted) restricts the accumulate/searchsorted to the states
    AFTER a detected restart's carried value was first replaced (T2): the
    fix for the container-recreate regime, where including the stale
    leading run in ``np.maximum.accumulate`` swallows every subsequent
    LOW, freshly-restarted seq under the old HIGH carried one, making
    373-of-374 commands unplaceable (review §3.1, E2). An epoch flagged
    ``ambiguous`` marks every one of its commands unplaceable AND
    ``seq_ambiguous`` -- evidence incomplete, never guessed at.
    """
    cmd_seq_a = np.asarray(cmd_seq)
    cmd_epoch_a = np.asarray(cmd_epoch)
    state_sim_time_s_a = np.asarray(state_sim_time_s)
    state_cmd_seq_a = np.asarray(state_cmd_seq)
    state_epoch_a = np.asarray(state_epoch)

    if bridge_sessions is None:
        bridge_sessions = detect_bridge_sessions(
            cmd_seq_a, cmd_epoch_a, state_cmd_seq_a, state_epoch_a)

    out: List[Optional[Bracket]] = [None] * len(cmd_seq_a)
    for epoch in np.unique(cmd_epoch_a):
        epoch_i = int(epoch)
        cidx = np.nonzero(cmd_epoch_a == epoch)[0]
        sidx_full = np.nonzero(state_epoch_a == epoch)[0]
        info = bridge_sessions.get(epoch_i, BridgeSessionInfo())

        if info.ambiguous:
            for ci in cidx:
                out[ci] = Bracket(epoch_i, None, None, None, seq_ambiguous=True)
            continue

        if len(sidx_full) == 0:
            for ci in cidx:
                out[ci] = Bracket(epoch_i, None, None, None)
            continue

        # A restart's own boundary state (the LAST state still reporting the
        # stale carried value) is a legitimate t_lo for the new session's
        # very first command, even though it must be excluded from the
        # accumulate/searchsorted below (including it there is the
        # original bug: its high carried value poisons
        # ``np.maximum.accumulate`` for every subsequent, freshly-restarted
        # LOW seq). Every leg command must be placeable across a recreate
        # (assignment T2 acceptance) -- this is what makes that true for
        # the new session's first command specifically, not just the rest.
        boundary_t_lo: Optional[float] = None
        if info.restart:
            if info.replaced_state_index is None:
                for ci in cidx:
                    out[ci] = Bracket(epoch_i, None, None, None)
                continue
            start_pos = int(np.searchsorted(sidx_full, info.replaced_state_index))
            sidx = sidx_full[start_pos:]
            if start_pos > 0:
                boundary_t_lo = float(state_sim_time_s_a[sidx_full[start_pos - 1]])
        else:
            sidx = sidx_full

        if len(sidx) == 0:
            for ci in cidx:
                out[ci] = Bracket(epoch_i, None, None, None)
            continue

        st = state_sim_time_s_a[sidx]
        sseq = state_cmd_seq_a[sidx]
        acc = np.maximum.accumulate(sseq)
        first_ge = np.searchsorted(acc, cmd_seq_a[cidx], side="left")
        for ci, fg in zip(cidx, first_ge):
            if fg >= len(st):
                out[ci] = Bracket(epoch_i, None, None, None)
            elif fg == 0 and boundary_t_lo is not None:
                out[ci] = Bracket(epoch_i, boundary_t_lo, float(st[0]), int(sidx[0]))
            elif fg == 0:
                out[ci] = Bracket(epoch_i, None, float(st[0]), int(sidx[0]))
            else:
                out[ci] = Bracket(
                    epoch_i, float(st[fg - 1]), float(st[fg]), int(sidx[fg]))
    assert all(b is not None for b in out)
    return out  # type: ignore[return-value]
