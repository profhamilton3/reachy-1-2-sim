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


#: Coordinator ruling (2026-09-25 stage-1 rulings, Q-hold, §2): the
#: lead-in and parked-tail windows are not goal transitions, so they have
#: no real ``goal_index`` -- these sentinels stand in for them wherever a
#: ``HoldWindow.goal_index`` is reported (never fed to ``route[...]``).
LEAD_IN_GOAL_INDEX = -1
PARKED_TAIL_GOAL_INDEX = -2


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
) -> List[Tuple[int, float, float, int]]:
    """``(goal_index_being_held, t_lo_s, t_hi_s, last_k_command_index)``
    for every gap between a goal's last command and the next DIFFERENT
    goal's first command -- exactly the boundary ``pathcheck.check_c6``
    also inspects, and (Q-hold ruling §2) located BY ASSIGNMENT, made
    under R-tie/R-const, never by the mere absence of commands.
    ``last_k_command_index`` is the GLOBAL index of the last command
    assigned to the goal being held -- the window's own reference target
    (never the window's own first/last sample)."""
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
        windows.append((g, t_lo, t_hi, command_indices[i]))
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
    last_k_target8: Dict[str, float],
) -> Dict[str, float]:
    """Q-hold ruling (§2): per-joint MAXIMUM ``|target - last_k_target8|``
    over every ``joint_command`` found inside the window (0 if there are
    none) -- ``last_k_target8`` is the reference (the target of the LAST
    command assigned to the goal/boundary this window follows), never the
    window's own first/last sample. Computed from evidence, never
    asserted: an empty window (no commands of its own, the expected case
    for a real hold) drifts 0 by construction, not by assumption -- there
    is nothing for the target to have changed AGAINST."""
    out = {j: 0.0 for j in R_JOINTS}
    if not window_command_indices:
        return out
    for i in window_command_indices:
        tgt8 = commands.target_rad[i, :8]
        for k, name in enumerate(R_JOINTS):
            d = abs(float(tgt8[k]) - float(last_k_target8.get(name, 0.0)))
            if d > out[name]:
                out[name] = d
    return out


def hold_window_stats(
    window: Tuple[int, float, float, int], epoch: int, target8: Dict[str, float],
    evidence: ev.Evidence, all_command_indices: Sequence[int],
) -> Optional[HoldStats]:
    goal_index, t_lo_s, t_hi_s, last_k_command_index = window
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
    last_k_target8 = {name: float(evidence.commands.target_rad[last_k_command_index, k])
                       for k, name in enumerate(R_JOINTS)}
    target_drift = target_drift_over_window(evidence.commands, win_idx, last_k_target8)
    return HoldStats(
        HoldWindow(goal_index, t_lo_s, t_hi_s, first_i, last_i),
        target8, target_drift, realised_drift, static_first, static_last, win_idx)


def edge_window_stats(
    *, goal_index: int, t_lo_s: float, t_hi_s: float, epoch: int,
    last_k_target8: Dict[str, float], evidence: ev.Evidence, all_command_indices: Sequence[int],
) -> Optional[HoldStats]:
    """The lead-in / parked-tail windows (Q-hold ruling §2): not a goal
    transition -- bounded by the SIDECAR's own aligned span at one end
    (the leg's first/last aligned state) and the leg's own first/last
    command at the other, with ``last_k_target8`` supplied directly
    (there is no preceding "goal's last command" to read it from, unlike
    a settle hold). Raises ``ValueError`` (mapped to ``EvidenceError`` by
    the caller) if the window is empty or inverted -- the sidecar's
    aligned span not containing this window is evidence incomplete, per
    the ruling, never silently "no drift"."""
    if t_hi_s < t_lo_s:
        raise ValueError(
            f"edge window inverted: t_lo_s={t_lo_s} > t_hi_s={t_hi_s} -- the sidecar's "
            "aligned span does not contain this window")
    states = evidence.states
    mask = (states.epoch == epoch) & (states.sim_time_s >= t_lo_s) & (states.sim_time_s <= t_hi_s)
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return None
    first_i, last_i = int(idx[0]), int(idx[-1])
    first_pos = {name: float(states.position_rad[first_i, k]) for k, name in enumerate(R_JOINTS)}
    last_pos = {name: float(states.position_rad[last_i, k]) for k, name in enumerate(R_JOINTS)}
    realised_drift = {j: last_pos[j] - first_pos[j] for j in R_JOINTS}
    static_first = {j: first_pos[j] - last_k_target8.get(j, 0.0) for j in R_JOINTS}
    static_last = {j: last_pos[j] - last_k_target8.get(j, 0.0) for j in R_JOINTS}
    win_idx = commands_in_window(all_command_indices, evidence.brackets, epoch, t_lo_s, t_hi_s)
    target_drift = target_drift_over_window(evidence.commands, win_idx, last_k_target8)
    return HoldStats(
        HoldWindow(goal_index, t_lo_s, t_hi_s, first_i, last_i),
        last_k_target8, target_drift, realised_drift, static_first, static_last, win_idx)
