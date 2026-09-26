"""W-blk: a settle-gap measurement window that does not depend on goal
assignment (D-1, owner ruling 2026-09-26-owner-option-c; decision report
``outputs/decision-2026-09-26-pr144-option-c-window-proof.md`` §2).

``segments.assign_goals``' own C0 goal assignment is corrupted by an echo
just as easily on a genuinely-improved (B) recording as on a manipulated
(A) one -- W4's old rule ("an indeterminate/missing affected segment ->
inconclusive_baseline") threw away the great majority of real A cycles for
exactly this reason (91-94% of a real setup leg's own commands went
unassigned; see the decision report §3.1). This module locates the SAME
HOVER -> REST_SHUT measurement window a different way: by the bridge's own
send behaviour, not by which goal a setpoint's VALUE happens to match.

Basis, entirely from code (decision report §2): ``rig_motion.fly_route``
flies each waypoint as a goto, 0-5 re-stream passes, then
``time.sleep(settle_s)``. The bridge sends a ``joint_command`` only when
SDK writes are pending (``mujoco_remote_backend.py``, both arms), polling
every ``MERGE_PERIOD_S``. So every settle sleep is a COMMAND-FREE interval
in the recording, whatever values the surrounding commands carry -- an
echo changes a value, it cannot create or remove a command-free interval.
Partitioning on that interval, instead of on goal assignment, is therefore
robust to exactly the corruption that broke W2/W4's old boundary.

Pure; no I/O. Sim time only (plan §7.0) -- the wall-clock partition is
computed too, but report-only (``partition_identical_wall_sim``), never
gated.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from reachy_ai.motion.rig_routes import R_JOINTS  # noqa: E402

#: rig_motion.fly_route's own settle_s default (pinned; W-AC6 checks this
#: against the live signature). This module may not import
#: reachy_ai.tasks.rig_motion at runtime (assignment W1: "import none of
#: them from production code at runtime") -- only a test may cross-check it.
SETTLE_S = 0.3
#: The bridge's own _send poll period: mujoco_remote_backend.py
#: 8c0dad2:387 and b59404c:552.
MERGE_PERIOD_S = 0.02
#: One merge period below settle_s: the smallest a genuine settle sleep's
#: OWN lower bracket bound (``lb``) can read, after the one 20ms merge
#: period the bracket math can shave off either end.
SETTLE_GAP_S = SETTLE_S - MERGE_PERIOD_S  # 0.28


@dataclass
class BlockInfo:
    first: int             # GLOBAL command index of the block's first command
    last: int               # GLOBAL command index of the block's last command
    n: int
    ordinal: Optional[str]   # the waypoint name this block SHOULD be (route[k].name), None if k >= len(route)
    nearest: str             # the waypoint name actually nearest the block's last command (argmin L-inf)
    d_nearest_deg: float
    d_second_deg: float


@dataclass
class WindowResult:
    valid: bool
    reasons: List[str] = field(default_factory=list)
    lead_in: int = 0
    blocks: List[BlockInfo] = field(default_factory=list)
    #: Report-only (never gated): the largest within-block gap (ub) and
    #: the smallest settle gap (lb) actually seen, in sim seconds.
    largest_within_gap_s: Optional[float] = None
    smallest_settle_gap_s: Optional[float] = None
    #: Report-only: the same settle/within/ambiguous partition,
    #: recomputed on the brackets' own wall_time_ns instead of
    #: sim_time_s. None if it could not be computed (e.g. a bracket
    #: without a hi_state_index to anchor a wall reading on).
    partition_identical_wall_sim: Optional[bool] = None
    segment_start: Optional[int] = None      # GLOBAL command index: last command of the HOVER block
    first_rest: Optional[int] = None          # GLOBAL command index: first command of the REST block
    segment_commands: List[int] = field(default_factory=list)  # GLOBAL indices, [segment_start, first_rest)
    t_lo_start: Optional[float] = None
    t_hi_start: Optional[float] = None
    t_lo_end: Optional[float] = None
    t_hi_end: Optional[float] = None
    epoch: Optional[int] = None
    state_indices: List[int] = field(default_factory=list)


def _target8(evidence, global_i: int) -> Dict[str, float]:
    row = evidence.commands.target_rad[global_i, :8]
    return {name: float(row[k]) for k, name in enumerate(R_JOINTS)}


def _linf_deg(a: Dict[str, float], b: Dict[str, float]) -> float:
    return float(np.degrees(max(abs(a.get(j, 0.0) - b.get(j, 0.0)) for j in R_JOINTS)))


def _nearest_waypoints(target8: Dict[str, float], route: Sequence) -> List[tuple]:
    """``[(name, distance_deg), ...]`` over every route waypoint, sorted
    ascending by distance -- the argmin (and the runner-up) V-c needs."""
    dists = [(wp.name, _linf_deg(target8, wp.pose)) for wp in route]
    dists.sort(key=lambda t: t[1])
    return dists


def _wall_lo_hi(evidence, global_i: int) -> Optional[tuple]:
    """The wall-clock analogue of a command's own sim-time bracket:
    ``(wall_time_ns[hi-1], wall_time_ns[hi])`` -- the same
    hi_state_index-relative pair ``_simtime.bracket_commands`` uses for
    the sim-time bracket in the ordinary (non-restart) case. Report-only
    (``partition_identical_wall_sim``): a restart-boundary edge case that
    the sim-time bracket handles via ``bridge_sessions`` is approximated
    here, never gated on."""
    b = evidence.brackets[global_i]
    hi = b.hi_state_index
    if hi is None or hi <= 0:
        return None
    return (float(evidence.states.wall_time_ns[hi - 1]), float(evidence.states.wall_time_ns[hi]))


def identify_window(
    evidence, leg, *, hover: str = "HOVER", rest_shut: str = "REST_SHUT", rest: str = "REST",
) -> WindowResult:
    """Decision report §2, steps 1-6. ``leg`` is any object exposing
    ``.command_indices`` (GLOBAL indices into ``evidence.commands``, in
    file order) and ``.route_rad`` (a ``.name``/``.pose`` waypoint
    sequence) -- ``cycle.LegSpec`` or an equivalent test mock."""
    route = list(leg.route_rad)
    idx = list(leg.command_indices)
    reasons: List[str] = []

    for name in (hover, rest_shut, rest):
        if not any(wp.name == name for wp in route):
            reasons.append(f"wblk:missing_waypoint={name}")
    if reasons:
        return WindowResult(valid=False, reasons=reasons)

    # V-a: every command in the leg must be placeable. Bracket math below
    # is meaningless for an unplaceable command, so this is checked, and
    # returned on, before anything else.
    unplaceable = [i for i in idx if evidence.brackets[i].unplaceable]
    if unplaceable:
        return WindowResult(valid=False, reasons=[f"wblk:unplaceable {unplaceable}"])

    # Step 1: lead-in -- the maximal leading run of commands whose
    # compliant flag has any non-None entry among the 8 right-arm joints
    # (the SDK turn_on, which the bridge may split into 1-2 commands).
    # The gap after it is not classified (removed before partitioning).
    lead_in = 0
    for i in idx:
        c = evidence.commands.compliant[i]
        if c is not None and any(v is not None for v in c[:8]):
            lead_in += 1
        else:
            break
    rem = idx[lead_in:]
    if len(rem) < 2:
        reasons.append(f"wblk:block_count={1 if rem else 0}")
        return WindowResult(valid=False, reasons=reasons, lead_in=lead_in)

    # Steps 2-3: gap bounds and classification, sim time (plan §7.0).
    kinds: List[str] = []
    largest_within = None
    smallest_settle = None
    for a, b in zip(rem, rem[1:]):
        ba, bb = evidence.brackets[a], evidence.brackets[b]
        lb = bb.t_lo - ba.t_hi
        ub = bb.t_hi - ba.t_lo
        if lb >= SETTLE_GAP_S:
            kinds.append("settle")
            if smallest_settle is None or lb < smallest_settle:
                smallest_settle = lb
        elif ub < SETTLE_GAP_S:
            kinds.append("within")
            if largest_within is None or ub > largest_within:
                largest_within = ub
        else:
            kinds.append("ambiguous")
            reasons.append(f"wblk:ambiguous_gap after={a} lb={lb} ub={ub}")

    # Step 4: blocks -- the runs between settle gaps (an ambiguous gap
    # does not split a block either; V-e already flags it above, and
    # merging keeps the block table meaningful for the OTHER validity
    # checks below rather than aborting before they can also be reported).
    block_bounds: List[tuple] = []
    start = 0
    for k, kind in enumerate(kinds):
        if kind == "settle":
            block_bounds.append((start, k))
            start = k + 1
    block_bounds.append((start, len(rem) - 1))

    n_blocks = len(block_bounds)
    if n_blocks != len(route):
        reasons.append(f"wblk:block_count={n_blocks}")

    blocks: List[BlockInfo] = []
    for k, (lo, hi) in enumerate(block_bounds):
        last_global = rem[hi]
        tgt8 = _target8(evidence, last_global)
        ranked = _nearest_waypoints(tgt8, route)
        nearest_name, d_nearest = ranked[0]
        d_second = ranked[1][1] if len(ranked) > 1 else float("inf")
        ordinal = route[k].name if k < len(route) else None
        if ordinal is not None and nearest_name != ordinal:
            reasons.append(
                f"wblk:label_mismatch block={k} ordinal={ordinal} nearest={nearest_name}")
        blocks.append(BlockInfo(
            first=rem[lo], last=last_global, n=hi - lo + 1,
            ordinal=ordinal, nearest=nearest_name,
            d_nearest_deg=d_nearest, d_second_deg=d_second))

    # Report-only: the same partition, recomputed on wall_time_ns.
    wall_kinds: List[Optional[str]] = []
    wall_ok = True
    for a, b in zip(rem, rem[1:]):
        wa, wb = _wall_lo_hi(evidence, a), _wall_lo_hi(evidence, b)
        if wa is None or wb is None:
            wall_ok = False
            break
        lb_w = (wb[0] - wa[1]) / 1e9
        ub_w = (wb[1] - wa[0]) / 1e9
        if lb_w >= SETTLE_GAP_S:
            wall_kinds.append("settle")
        elif ub_w < SETTLE_GAP_S:
            wall_kinds.append("within")
        else:
            wall_kinds.append("ambiguous")
    partition_identical_wall_sim = (wall_kinds == kinds) if wall_ok else None

    result = WindowResult(
        valid=(len(reasons) == 0), reasons=reasons, lead_in=lead_in, blocks=blocks,
        largest_within_gap_s=largest_within, smallest_settle_gap_s=smallest_settle,
        partition_identical_wall_sim=partition_identical_wall_sim)
    if not result.valid:
        return result

    # Step 5: segment, only computed once every validity condition above
    # (V-a/b/c/e) has passed -- block k's ordinal is only trustworthy
    # (guaranteed 1:1 with route order) once n_blocks == len(route) and
    # every label matched.
    hover_k = next(k for k, wp in enumerate(route) if wp.name == hover)
    rest_k = next(k for k, wp in enumerate(route) if wp.name == rest)

    segment_start = blocks[hover_k].last
    first_rest = blocks[rest_k].first
    b_start = evidence.brackets[segment_start]
    b_rest = evidence.brackets[first_rest]

    start_pos = idx.index(segment_start)
    rest_pos = idx.index(first_rest)
    segment_commands = idx[start_pos:rest_pos]

    epoch = int(evidence.commands.epoch[segment_start])
    states = evidence.states
    mask = (states.epoch == epoch) & (states.sim_time_s >= b_start.t_lo) & (states.sim_time_s < b_rest.t_lo)
    state_idx = np.nonzero(mask)[0]

    # V-d: non-empty, one epoch (guaranteed by the mask itself), contiguous seq.
    if len(state_idx) == 0:
        reasons.append("wblk:window_states empty")
    else:
        seqs = states.seq[state_idx]
        if not np.array_equal(np.diff(seqs), np.ones(len(seqs) - 1, dtype=seqs.dtype)):
            reasons.append("wblk:window_states seq not contiguous")

    result.valid = len(reasons) == 0
    result.reasons = reasons
    if not result.valid:
        return result

    result.segment_start = segment_start
    result.first_rest = first_rest
    result.segment_commands = segment_commands
    result.t_lo_start = b_start.t_lo
    result.t_hi_start = b_start.t_hi
    result.t_lo_end = b_rest.t_lo
    result.t_hi_end = b_rest.t_hi
    result.epoch = epoch
    result.state_indices = [int(i) for i in state_idx]
    return result
