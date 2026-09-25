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
from typing import List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Bracket:
    epoch: int
    t_lo: Optional[float]          # sim_time_s, or None if unplaceable
    t_hi: Optional[float]          # sim_time_s, or None if unplaceable
    hi_state_index: Optional[int]  # index into the full state arrays

    @property
    def unplaceable(self) -> bool:
        return self.t_hi is None or self.t_lo is None


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


def bracket_commands(
    cmd_seq: Sequence[int],
    cmd_epoch: Sequence[int],
    state_sim_time_s: Sequence[float],
    state_cmd_seq: Sequence[int],
    state_epoch: Sequence[int],
) -> List[Bracket]:
    """``[t_lo, t_hi]`` for every command, one call for the whole evidence
    set (grouped and vectorised per epoch internally, so this stays well
    under the §2.8 60 s/12-minute-cycle budget)."""
    cmd_seq_a = np.asarray(cmd_seq)
    cmd_epoch_a = np.asarray(cmd_epoch)
    state_sim_time_s_a = np.asarray(state_sim_time_s)
    state_cmd_seq_a = np.asarray(state_cmd_seq)
    state_epoch_a = np.asarray(state_epoch)

    out: List[Optional[Bracket]] = [None] * len(cmd_seq_a)
    for epoch in np.unique(cmd_epoch_a):
        cidx = np.nonzero(cmd_epoch_a == epoch)[0]
        sidx = np.nonzero(state_epoch_a == epoch)[0]
        if len(sidx) == 0:
            for ci in cidx:
                out[ci] = Bracket(int(epoch), None, None, None)
            continue
        st = state_sim_time_s_a[sidx]
        sseq = state_cmd_seq_a[sidx]
        acc = np.maximum.accumulate(sseq)
        first_ge = np.searchsorted(acc, cmd_seq_a[cidx], side="left")
        for ci, fg in zip(cidx, first_ge):
            if fg >= len(st):
                out[ci] = Bracket(int(epoch), None, None, None)
            elif fg == 0:
                out[ci] = Bracket(int(epoch), None, float(st[0]), int(sidx[0]))
            else:
                out[ci] = Bracket(
                    int(epoch), float(st[fg - 1]), float(st[fg]), int(sidx[fg]))
    assert all(b is not None for b in out)
    return out  # type: ignore[return-value]
