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


def hold_window_stats(
    window: Tuple[int, float, float], epoch: int, target8: Dict[str, float],
    states: ev.States,
) -> Optional[HoldStats]:
    goal_index, t_lo_s, t_hi_s = window
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
    target_drift = {j: 0.0 for j in R_JOINTS}  # no commands in the window, by construction
    return HoldStats(
        HoldWindow(goal_index, t_lo_s, t_hi_s, first_i, last_i),
        target8, target_drift, realised_drift, static_first, static_last)
