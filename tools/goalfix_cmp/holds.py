"""Hold drift (plan §5 P4/P6, §7.4).

A hold window is bounded by two commands: the last one assigned to a goal
and the first assigned to the NEXT one (segments.py's own goal assignment
-- the same boundary check_c6 in pathcheck.py inspects for a spurious
command). Every quantity here is simulation time. Reusable directly against
any other named window (the 3 s parked window, the 3 s ``LEAD_IN_S``) by
passing its own ``(t_lo_s, t_hi_s)`` bounds instead.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from reachy_ai.motion.rig_routes import R_JOINTS  # noqa: E402

from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import segments as seg  # noqa: E402


@dataclass
class HoldWindow:
    goal_index: int          # the goal being held (the one commands stopped at)
    t_lo_s: float
    t_hi_s: float
    first_state_index: int
    last_state_index: int


@dataclass
class HoldStats:
    window: HoldWindow
    target: Dict[str, float]                 # the target in force through the hold
    target_drift: Dict[str, float]            # must be 0 in B (C6 already gates this)
    realised_drift: Dict[str, float]          # per-joint: last state - first state
    static_offset_first: Dict[str, float]     # first state's position - target
    static_offset_last: Dict[str, float]      # last state's position - target
    window_command_indices: List[int] = None  # any joint_command found inside the window


def find_hold_windows(
    assignment: seg.GoalAssignment, brackets, command_indices: Sequence[int],
) -> List[Tuple[int, float, float]]:
    """``(goal_index_being_held, t_lo_s, t_hi_s)`` for every gap between a
    goal's last command and the next DIFFERENT goal's first command --
    exactly the boundary ``pathcheck.check_c6`` also inspects."""
    windows = []
    n = len(assignment.goal_index)
    for i in range(n - 1):
        g, nxt = assignment.goal_index[i], assignment.goal_index[i + 1]
        if g is None or nxt is None or nxt == g:
            continue
        t_lo = brackets[command_indices[i]].t_hi
        t_hi = brackets[command_indices[i + 1]].t_lo
        if t_lo is None or t_hi is None:
            continue
        windows.append((g, t_lo, t_hi))
    return windows


def commands_in_window(
    all_command_indices: Sequence[int], brackets, epoch: int, t_lo_s: float, t_hi_s: float,
) -> List[int]:
    """Global indices of every ``joint_command`` (any goal assignment,
    including indeterminate/carry ones) whose bracket's ``t_hi`` falls
    within ``[t_lo_s, t_hi_s]`` of ``epoch`` -- reusable against ANY named
    window (the HOVER hold, the 3 s parked window, ``LEAD_IN_S``), per T4
    item 5."""
    out = []
    for i in all_command_indices:
        b = brackets[i]
        if b.unplaceable or b.epoch != epoch:
            continue
        # Strict on the lower bound: a window's own t_lo_s is the PRECEDING
        # goal's last command's own t_hi, which must not count as "inside"
        # the gap it only marks the start of.
        if t_lo_s < b.t_hi <= t_hi_s:
            out.append(i)
    return out


def target_drift_over_window(
    commands: ev.Commands, window_command_indices: Sequence[int],
) -> Dict[str, float]:
    """Change in commanded target over the window, taken from any
    ``joint_command`` inside it (T4 item 5) -- must be 0 in B, per C6.
    Computed from evidence, never asserted: an empty window (no commands
    of its own, the expected case for a real hold) drifts 0 by
    construction, not by assumption -- there is nothing for the target to
    have changed AGAINST."""
    if not window_command_indices:
        return {j: 0.0 for j in R_JOINTS}
    first_t = commands.target_rad[window_command_indices[0], :8]
    last_t = commands.target_rad[window_command_indices[-1], :8]
    return {name: float(last_t[k] - first_t[k]) for k, name in enumerate(R_JOINTS)}


def hold_window_stats(
    window: Tuple[int, float, float], epoch: int, target8: Dict[str, float],
    evidence: ev.Evidence, all_command_indices: Sequence[int],
) -> Optional[HoldStats]:
    goal_index, t_lo_s, t_hi_s = window
    states = evidence.states
    mask = (states.epoch == epoch) & (states.sim_time_s >= t_lo_s) & (states.sim_time_s <= t_hi_s)
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return None
    first_i, last_i = int(idx[0]), int(idx[-1])
    first_pos = {name: float(states.position_rad[first_i, k]) for k, name in enumerate(R_JOINTS)}
    last_pos = {name: float(states.position_rad[last_i, k]) for k, name in enumerate(R_JOINTS)}
    realised_drift = {j: last_pos[j] - first_pos[j] for j in R_JOINTS}
    static_first = {j: first_pos[j] - target8.get(j, 0.0) for j in R_JOINTS}
    static_last = {j: last_pos[j] - target8.get(j, 0.0) for j in R_JOINTS}
    win_idx = commands_in_window(all_command_indices, evidence.brackets, epoch, t_lo_s, t_hi_s)
    target_drift = target_drift_over_window(evidence.commands, win_idx)
    return HoldStats(
        HoldWindow(goal_index, t_lo_s, t_hi_s, first_i, last_i),
        target8, target_drift, realised_drift, static_first, static_last, win_idx)
