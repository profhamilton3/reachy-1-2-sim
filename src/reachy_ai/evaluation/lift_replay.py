"""Grading a RECORDED lift offline (#172): states + panel evidence -> verdict.

A live lift leaves two records behind:

  * the simulator's state stream (`states.jsonl`): every joint, every object
    pose, at ~30 Hz, with the host's wall clock (`host_time`);
  * the panel task (`panel-task-final.json`): the crane's kept events, each
    stamped with the container's wall clock (`t`), and its own measurements.

This module turns the pair into the `EpisodeResult` + `PanelRouteTaskSpec`
that `panel_routes.evaluate_lift_object` judges.  The lift's numbers are
re-measured FROM THE STATES — simulator truth — rather than copied from what
the crane reported, so the replay is a check on the run and not a restatement
of it; the crane's own figures are kept beside them for comparison.

Contacts are not in the recorded states (the server streams them only when it
records), so the caller supplies them: `contacts_at(state)` returns the bodies
in contact in that state, computed by posing the offline world kinematically
(`scripts/replay_lift_evaluation.py`).  This module stays offline-safe — no
MuJoCo, no native imports — so it can be tested on constructed records.

PHASES come from the crane's own segments (`contact_rules.CRANE_PHASES`).
Records made before #172 kept only the crane's measurement events, not its
phase events, so for those the windows are rebuilt from the events that bound
them: STRADDLE (the close begins), HELD (the hold ends; it began `hold_s`
earlier), SUPPORTED (the hand opens), RELEASED (the withdrawal begins).

THE APPROACH TO THE START POSTURE IS NOT THE LIFT (owner ruling, 2026-10-10).
From HOME the worker first flies the measured route to PRESENT
(`rig_routes.path`: RAISE_TO_SIDE), and only then does the crane's
ATTEMPT_START begin the lift.  States before ATTEMPT_START are tagged with the
leg's route, so their contacts are judged under that route's rules and
exceptions and reported apart from the lift.  ATTEMPT_START is the crane's
phase event when the record has one; before #172 it is the first state at
which the arm stands at the start posture, the legs flown.
"""

from __future__ import annotations

import dataclasses
import math
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

from reachy_ai.evaluation import contact_rules as CR
from reachy_ai.evaluation.panel_routes import (
    COLLIDABLE_KEY, CONTACT_SAMPLES_KEY, LIFT_PUT_BACK_KEY, LIFT_RISE_KEY,
    LIFT_SLIP_KEY, WITHDRAWAL_KEY,
)
from reachy_ai.experience.models import (EpisodeResult, EpisodeStatus,
                                         PanelRouteTaskSpec)

#: The crane's hold, in seconds (`crane_pick_live.CraneSpec.hold_s`).  Used
#: only to rebuild the hold window of a record that kept no phase events.
DEFAULT_HOLD_S = 2.0

#: The right arm's seven joints, in the order `grasp_alignment.hand_frames`
#: takes them.
ARM_JOINTS = ("r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw",
              "r_elbow_pitch", "r_forearm_yaw", "r_wrist_pitch", "r_wrist_roll")

#: Matches to an event's position spread wider than this are ambiguous.
AMBIGUOUS_S = 0.5

Contacts = Callable[[Mapping[str, Any]], Iterable[Tuple[str, str, float]]]


@dataclasses.dataclass(frozen=True)
class PhaseWindow:
    phase: CR.Phase
    start: float
    end: float


# ---------------------------------------------------------------------------
# Reading one recorded state
# ---------------------------------------------------------------------------

def joints_deg(state: Mapping[str, Any]) -> Dict[str, float]:
    """Joint name -> SDK degrees.  The server reports qpos, and every joint's
    SDK sign is +1 (`joint_map`), so the radians are SDK radians."""
    return {j["name"]: math.degrees(float(j["position_rad"]))
            for j in state.get("joints") or ()}


def object_pos(state: Mapping[str, Any], oid: str) -> Optional[List[float]]:
    for o in state.get("objects") or ():
        if o.get("object_id") == oid:
            return [float(v) for v in o["pos_xyz"][:3]]
    return None


def object_quat(state: Mapping[str, Any], oid: str) -> Optional[List[float]]:
    for o in state.get("objects") or ():
        if o.get("object_id") == oid:
            return [float(v) for v in o.get("quat_wxyz") or (1, 0, 0, 0)]
    return None


def in_hand(state: Mapping[str, Any], oid: str):
    """The object's centre in the thumb frame (m), as the crane measures it
    (`crane_pick_live._in_hand`), from the recorded joints."""
    import numpy as np

    from reachy_ai.motion import grasp_alignment as G

    deg = joints_deg(state)
    h = G.hand_frames([deg[n] for n in ARM_JOINTS], deg["r_gripper"])
    return h.thumb_rot.T @ (np.asarray(object_pos(state, oid)) - h.thumb.center)


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def clock_offset(events: Sequence[Mapping[str, Any]],
                 states: Sequence[Mapping[str, Any]], oid: str) -> float:
    """Seconds to SUBTRACT from an event's `t` to put it on the states' clock.

    The two clocks are the container's and the host's.  Each crane event
    that carries the object's measured position is matched to the state in
    which the object stood exactly there (the observation came from that
    state), and the median difference is the offset.  An event whose
    position matches states spread over more than `AMBIGUOUS_S` (an object
    standing perfectly still) says nothing about the clock and is skipped.
    0.0 when nothing matches.
    """
    diffs = []
    for e in events:
        pos = e.get("object")
        if not (isinstance(pos, list) and len(pos) >= 3):
            continue
        hits = []
        for s in states:
            p = object_pos(s, oid)
            if p is not None and math.dist(p, pos[:3]) < 1e-13:
                hits.append(float(s["host_time"]))
        if hits and max(hits) - min(hits) <= AMBIGUOUS_S:
            t = float(e["t"])
            diffs.append(t - min(hits, key=lambda h: abs(t - h)))
    if not diffs:
        return 0.0
    diffs.sort()
    return diffs[len(diffs) // 2]


def crane_phase_windows(events: Sequence[Mapping[str, Any]], *, start: float,
                        end: float, hold_s: float = DEFAULT_HOLD_S,
                        offset: float = 0.0) -> List[PhaseWindow]:
    """The lift's phases as time windows on the states' clock.

    ``start``/``end`` bound the job (already on the states' clock); event
    times have ``offset`` subtracted.  Before the crane's first segment the
    worker was flying the approach to PRESENT, which is APPROACH too.
    """
    def at(e: Mapping[str, Any]) -> float:
        return float(e["t"]) - offset

    marks: List[Tuple[float, CR.Phase]] = [(start, CR.Phase.APPROACH)]
    phase_events = [e for e in events if e.get("kind") == "phase"]
    for e in phase_events:
        ph = CR.crane_phase(str(e.get("name", "")))
        if ph is not None:
            marks.append((at(e), ph))
    if not phase_events:
        # A record from before #172: rebuild from the bounding events.
        for e in events:
            kind = e.get("kind")
            if kind == "STRADDLE" and e.get("ok"):
                marks.append((at(e), CR.Phase.GRASP))
            elif kind == "HELD":
                marks.append((at(e) - hold_s, CR.Phase.HOLD))
                marks.append((at(e), CR.Phase.SET_DOWN))
            elif kind == "HALT" and e.get("stage") != "withdrawal" and \
                    not any(x.get("kind") == "CLOSED" for x in events):
                marks.append((at(e), CR.Phase.WITHDRAW))
    # The crane names no release segment: the hand opens inside "replace",
    # once the object is supported, and the withdrawal starts at RELEASED.
    for e in events:
        if e.get("kind") == "SUPPORTED":
            marks.append((at(e), CR.Phase.RELEASE))
        elif e.get("kind") == "RELEASED":
            marks.append((at(e), CR.Phase.WITHDRAW))

    marks.sort(key=lambda m: m[0])
    out: List[PhaseWindow] = []
    for i, (t0, ph) in enumerate(marks):
        t1 = marks[i + 1][0] if i + 1 < len(marks) else end
        t0, t1 = max(t0, start), min(t1, end)
        if t1 > t0:
            if out and out[-1].phase is ph:
                out[-1] = PhaseWindow(ph, out[-1].start, t1)
            else:
                out.append(PhaseWindow(ph, t0, t1))
    return out


def approach_legs(states: Sequence[Mapping[str, Any]], start_posture: str,
                  wanted: str) -> Tuple[List[str], List[str], Optional[float]]:
    """The measured legs flown to reach ``wanted``, the leg each state was
    flown on, and when the arm first stood at ``wanted`` (None if never).

    A state belongs to the leg that is carrying the arm: a leg ends when the
    arm has reached its end posture and then left it, so a dwell at an
    intermediate posture belongs to the leg that arrived there.
    """
    from reachy_ai.motion import rig_routes as R

    legs = (R.path(start_posture, wanted) or []) \
        if start_posture and wanted and start_posture != wanted else []
    ends = [b for leg in legs for (a, b), r in R.POSTURE_TRANSITIONS.items()
            if r == leg][:len(legs)]
    if not legs:
        t0 = float(states[0]["host_time"]) if states else None
        return [], ["" for _ in states], t0
    out: List[str] = []
    i, reached, arrived = 0, False, None
    for s in states:
        pose = R.posture_of(joints_deg(s))
        if i < len(legs) - 1:
            if pose == ends[i]:
                reached = True
            elif reached:
                i, reached = i + 1, False
        out.append(legs[i])
        if arrived is None and i == len(legs) - 1 and pose == wanted:
            arrived = float(s["host_time"])
    return legs, out, arrived


def phase_at(windows: Sequence[PhaseWindow], t: float) -> Optional[CR.Phase]:
    for w in windows:
        if w.start <= t < w.end:
            return w.phase
    return windows[-1].phase if windows and t >= windows[-1].end else None


def _window(windows: Sequence[PhaseWindow], phase: CR.Phase
            ) -> Optional[PhaseWindow]:
    hit = [w for w in windows if w.phase is phase]
    if not hit:
        return None
    return PhaseWindow(phase, hit[0].start, hit[-1].end)


# ---------------------------------------------------------------------------
# The episode
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ReplayedLift:
    result: EpisodeResult
    spec: PanelRouteTaskSpec
    plan: Optional[Dict[str, Any]]
    windows: List[PhaseWindow]
    clock_offset_s: float
    #: The measured legs flown to the start posture, and when the lift began.
    approach_legs: List[str]
    attempt_start: float
    #: What the crane itself reported, for comparison with the re-measure.
    crane_reported: Dict[str, Any]


def _job_bounds(task: Mapping[str, Any]) -> Tuple[float, float]:
    """The job: from "Executing in simulation." to the task's final answer
    (both on the container's clock, like the crane's events)."""
    events = [e for e in task.get("events") or () if e.get("role") == "reachy"]
    begin = next((float(e["at"]) for e in events
                  if str(e.get("text", "")).startswith("Executing")), None)
    if begin is None or not events:
        raise ValueError("the panel task records no execution")
    return begin, float(events[-1]["at"])


def replay_lift(task: Mapping[str, Any], states: Sequence[Mapping[str, Any]],
                *, contacts_at: Optional[Contacts] = None,
                collidable: Sequence[str] = (), episode_id: str = "",
                hold_s: float = DEFAULT_HOLD_S) -> ReplayedLift:
    """Rebuild one lift's episode from its panel task and its states."""
    from reachy_ai.motion import rig_routes as R

    proposal = task.get("proposal") or {}
    evidence = task.get("execution_evidence") or {}
    crane = evidence.get("crane") or {}
    events = crane.get("events") or []
    oid = str(proposal.get("object_id") or "")
    if not oid:
        raise ValueError("the task names no object")

    offset = clock_offset(events, states, oid)
    begin, finish = _job_bounds(task)
    begin, finish = begin - offset, finish - offset
    job = [s for s in states if begin <= float(s["host_time"]) <= finish]
    if not job:
        raise ValueError("no recorded state falls inside the job")

    legs, leg_of, arrived = approach_legs(
        job, str(evidence.get("start_posture") or ""),
        str(proposal.get("expected_start_posture") or ""))
    marked = [float(e["t"]) - offset for e in events
              if e.get("kind") == "phase" and e.get("name") == "ATTEMPT_START"]
    attempt = marked[0] if marked else (arrived if arrived is not None
                                        else begin)
    windows = crane_phase_windows(events, start=attempt, end=finish,
                                  hold_s=hold_s, offset=offset)

    def during(w: Optional[PhaseWindow]) -> List[Mapping[str, Any]]:
        if w is None:
            return []
        return [s for s in job if w.start <= float(s["host_time"]) < w.end]

    metrics: Dict[str, float] = {"total_steps": float(job[-1]["sim_step"]
                                                      - job[0]["sim_step"])}

    # PICKUP: where the object stood when the close began.
    grasp = _window(windows, CR.Phase.GRASP)
    before = [s for s in job if grasp is None
              or float(s["host_time"]) < grasp.start]
    pickup = object_pos(before[-1], oid) if before else None

    hold = during(_window(windows, CR.Phase.HOLD))
    if pickup is not None and hold:
        metrics[LIFT_RISE_KEY] = min(object_pos(s, oid)[2] for s in hold) \
            - pickup[2]
        ref = in_hand(hold[0], oid)
        metrics[LIFT_SLIP_KEY] = max(
            float(math.dist(in_hand(s, oid), ref)) for s in hold)

    released = _window(windows, CR.Phase.WITHDRAW)
    after = [s for s in job if released is not None
             and float(s["host_time"]) >= released.start]
    if pickup is not None and after and crane.get("replaced"):
        put = object_pos(after[0], oid)
        metrics[LIFT_PUT_BACK_KEY] = math.dist(put[:2], pickup[:2])

    wd = crane.get("withdrawal")
    if isinstance(wd, dict) and "clean" in wd:
        metrics[WITHDRAWAL_KEY] = 1.0 if wd.get("clean") else 0.0

    raw = []
    if contacts_at is not None:
        for s, leg in zip(job, leg_of):
            found = list(contacts_at(s))
            if not found:
                continue
            t = float(s["host_time"])
            if t < attempt:
                ph, route = CR.Phase.APPROACH, leg
            else:
                ph = phase_at(windows, t) or CR.Phase.APPROACH
                route = ""
            pose = R.posture_of(joints_deg(s))
            raw.extend((b1, b2, ph, pose, d, route) for b1, b2, d in found)

    first, last = job[0], job[-1]
    result = EpisodeResult(
        episode_id=episode_id or str(task.get("task_id", "")),
        trial_id=episode_id or str(task.get("task_id", "")),
        status=(EpisodeStatus.SUCCEEDED if task.get("state") == "completed"
                else EpisodeStatus.FAILED),
        termination_reason=str(task.get("detail", "")),
        success=task.get("state") == "completed",
        start_sim_step=int(first["sim_step"]),
        end_sim_step=int(last["sim_step"]),
        sim_duration_s=float(last["sim_time_s"]) - float(first["sim_time_s"]),
        wall_duration_s=float(last["host_time"]) - float(first["host_time"]),
        metrics=metrics,
        contact_summary=({CONTACT_SAMPLES_KEY: [c.to_dict() for c in
                                                CR.aggregate(raw)],
                          COLLIDABLE_KEY: list(collidable)}
                         if contacts_at is not None else {}),
        final_object_states={o["object_id"]: {"object_id": o["object_id"],
                                              "pos_xyz": list(o["pos_xyz"])}
                             for o in last.get("objects") or ()},
        final_joint_positions_deg=joints_deg(last),
    )
    spec = PanelRouteTaskSpec(
        task_id=f"panel:lift_object:{result.episode_id}",
        task_type="lift_object", ability="lift_object",
        route=str(proposal.get("route") or "CRANE_LIFT"),
        route_version=int(proposal.get("route_version") or 1),
        expected_start_posture=str(proposal.get("expected_start_posture") or ""),
        expected_end_posture=R.POSTURE_PRESENT,
        initial_object_positions={o["object_id"]: list(o["pos_xyz"])
                                  for o in first.get("objects") or ()},
        target_object_id=oid, arm_policy="right",
        forbidden_contact_policy="fail")
    held = crane.get("lifted_and_held") or {}
    rep = crane.get("replaced") or {}
    return ReplayedLift(
        result=result, spec=spec, plan=crane.get("plan"), windows=windows,
        clock_offset_s=offset, approach_legs=legs, attempt_start=attempt,
        crane_reported={
            "rise_start_mm": held.get("rise_start_mm"),
            "rise_end_mm": held.get("rise_end_mm"),
            "in_hand_slip_mm": held.get("in_hand_slip_mm"),
            "offset_from_start_mm": rep.get("offset_from_start_mm"),
            "withdrawal_clean": (wd or {}).get("clean"),
            "final_posture": evidence.get("final_posture"),
            "criteria": crane.get("criteria"),
            "panel_state": task.get("state"),
        })
