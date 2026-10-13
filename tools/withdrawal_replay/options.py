"""The four ways to release and withdraw, flown in the rebuilt offline world.

  base   the code before #173: open to -65 over 0.8 s, hold 0.8 s; the way
         straight up, else the bounded sideways correction, else refuse.
  O1     the same release; then straight up, sideways, or a SEPARATING
         retreat whose first leg may start under the allowance.
  O2     CHECK BEFORE OPENING: open only until the grip force reads zero,
         judge the way out for the fully opened hand (-65, then -68), open no
         further if neither is clear; after opening, open to -68 if a pad is
         still under the allowance.  Straight up or sideways only.
  O1+O2  O2's release and check, with O1's separating retreat as well.  This
         is what `crane_pick_live.execute_crane_pick` now does; the decisions
         below are its own functions (`plan_way_out`), and the sequence and
         timings are its sequence and timings.

The way out is flown like the live code flies it: correcting or separating
legs with the monitor off, a 0.5 s hold, then the climb with the monitor on
(any pad force, or the object moving more than `CraneLimits.object_moved_m`,
halts it -- and a halt is a refusal).

Measured per lift and option, from the offline physics:
  * refused (and why, or where it halted);
  * min_clearance_mm: the least pad-to-cube separation while the hand
    withdraws (collision boxes, measured hand and cube);
  * residual_mm: the deepest pad-on-cube contact from the end of the release
    to the end of the withdrawal (0.0: the pads never touch the cube again);
  * drag_mm: how far the release moved the cube, horizontally, from the
    state before the hand opened to the state after it opened.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from reachy_ai.motion import grasp_alignment as G
from reachy_ai.motion.kinematics import joint_path
from reachy_ai.tasks import crane_pick_live as CP

from .rebuild import LIM, OID, SPEC, Rebuilt

OPTIONS = ("base", "O1", "O2", "O1+O2")
_KINDS = {"base": ("direct", "sideways"), "O1": CP.WAY_OUT_KINDS,
          "O2": ("direct", "sideways"), "O1+O2": CP.WAY_OUT_KINDS}
_STEP_S = 1.0 / 25.0        # primitives._INTERP_HZ and transit.STEP_HZ


def _ramp_gripper(rb: Rebuilt, start: float, end: float, seconds: float,
                  stop_when=None) -> float:
    """`primitives.smooth_move` / `release_grip` on the gripper."""
    w = rb.world
    steps = max(1, int(seconds / _STEP_S))
    goal = start
    for i in range(1, steps + 1):
        goal = start + i / steps * (end - start)
        w.target("r_gripper", goal)
        w.run(seconds / steps)
        if stop_when is not None and stop_when():
            break
    return goal


def _way_out(rb: Rebuilt, cmd, kinds, gripper_deg: Optional[float] = None):
    q, gm, live, off, _pos = rb.measured(cmd)
    return CP.plan_way_out(rb.planner, rb.scene, rb.plan, cmd, rb.above,
                           gm if gripper_deg is None else gripper_deg, live, off,
                           kinds=kinds)


def _fly(rb: Rebuilt, way: CP.WayOut, rec: Dict[str, Any]) -> None:
    """Stream the way out as `execute_crane_pick.stream` does: 0.05 deg
    steps at 25 Hz from the last commanded pose; the monitor off for the
    first ``unmonitored`` legs, then a 0.5 s hold, then on."""
    w = rb.world
    mon = {"on": False, "ref": None}

    def watch():
        h = G.hand_frames(w.arm(), w.gripper())
        b = G.object_box(rb.scene.get(OID), *w.cube())
        rec["min_clearance_mm"] = min(rec["min_clearance_mm"],
                                      1000 * G.obb_separation(h.thumb, b),
                                      1000 * G.obb_separation(h.finger, b))
        rec["residual_mm"] = max(rec["residual_mm"], 1000 * w.pad_contact_m())
        if mon["on"] and not rec.get("halt"):
            force = w.grip_force()
            moved = float(np.linalg.norm(w.cube()[0] - mon["ref"]))
            if force > LIM.contact_force_n:
                rec["halt"] = f"grip force {force:.2f} N"
            elif moved > LIM.object_moved_m:
                rec["halt"] = f"object moved {1000 * moved:.2f} mm"

    cur = list(way.path[0])
    for idx, q in enumerate(way.path[1:], start=1):
        if idx > way.unmonitored and not mon["on"]:
            if way.unmonitored:
                w.run(0.5, watch)
            mon["on"], mon["ref"] = True, w.cube()[0].copy()
        n = max(2, math.ceil(max(abs(b - a) for a, b in zip(cur, q)) / 0.05) + 1)
        for p in joint_path(cur, q, n)[1:]:
            w.command_arm(p)
            w.run(_STEP_S, watch)
            if rec.get("halt"):
                return
        cur = list(q)


def run_option(rb: Rebuilt, option: str) -> Dict[str, Any]:
    """Release and withdraw one rebuilt lift by ``option``."""
    if option not in OPTIONS:
        raise KeyError(option)
    kinds = _KINDS[option]
    w = rb.world
    cmd = list(rb.cmd)
    rb.hold_at(cmd)
    held = w.cube()[0]
    rec: Dict[str, Any] = {"lift": rb.name, "option": option, "refused": False,
                           "min_clearance_mm": math.inf, "residual_mm": 0.0}

    if option in ("O2", "O1+O2"):
        let_go = _ramp_gripper(rb, w.gripper(), SPEC.opening_deg, 0.8,
                               stop_when=lambda: w.grip_force() <= LIM.contact_force_n)
        w.run(0.5)
        rec["let_go_deg"] = round(let_go, 1)
        opening = None
        tiers = [k for k in (CP.PRE_OPEN_PREFERRED, kinds) if set(k) <= set(kinds)]
        for tier in tiers:                     # as `check_before_opening`
            for cand in CP.release_openings(SPEC):
                found, _ = _way_out(rb, cmd, tier, gripper_deg=cand)
                if found is not None:
                    opening = cand
                    break
            if opening is not None:
                break
        if opening is None:
            rec.update(refused=True, why="refused before opening (hand kept at let-go)",
                       drag_mm=1000 * float(np.linalg.norm((w.cube()[0] - held)[:2])),
                       residual_mm=1000 * w.pad_contact_m())
            return rec
        _ramp_gripper(rb, w.gripper(), opening, 0.4)
        w.run(0.8)
    else:
        _ramp_gripper(rb, w.gripper(), SPEC.opening_deg, 0.8)
        w.run(0.8)

    rec["drag_mm"] = 1000 * float(np.linalg.norm((w.cube()[0] - held)[:2]))
    rec["residual_mm"] = 1000 * w.pad_contact_m()
    found, _ = _way_out(rb, cmd, kinds)
    rec["opened_wider"] = False
    if option in ("O2", "O1+O2") and (found is None or found.kind != "direct") \
            and w.gripper() > SPEC.wide_opening_deg + 0.5:
        _ramp_gripper(rb, w.gripper(), SPEC.wide_opening_deg, 0.3)
        w.run(0.5)
        rec["opened_wider"] = True
        rec["residual_mm"] = max(rec["residual_mm"], 1000 * w.pad_contact_m())
        found, _ = _way_out(rb, cmd, kinds)
    rec["final_opening_deg"] = round(w.gripper(), 1)
    if found is None:
        rec.update(refused=True, why="withdrawal judged not clear after release")
        return rec
    rec["kind"] = found.kind
    rec["start_mm"] = found.report.get("start_mm")
    _fly(rb, found, rec)
    if rec.get("halt"):
        rec.update(refused=True, why=f"halted: {rec['halt']}")
    return rec


def withdraw_from_recorded(rb: Rebuilt) -> Dict[str, Any]:
    """The NEW withdrawal step alone, from the exact recorded post-release
    state (the live release, not a rebuilt one): judge, open wider if a pad
    is under the allowance, fly."""
    w = rb.world
    cmd = list(rb.cmd)
    rb.hold_at(cmd, state=rb.record["post_release"])
    w.target("r_gripper", w.gripper())
    w.run(0.3)
    rec: Dict[str, Any] = {"lift": rb.name, "option": "O1+O2 withdrawal from the recorded release",
                           "refused": False, "min_clearance_mm": math.inf,
                           "residual_mm": 1000 * w.pad_contact_m()}
    found, _ = _way_out(rb, cmd, CP.WAY_OUT_KINDS)
    rec["opened_wider"] = False
    if (found is None or found.kind != "direct") and w.gripper() > SPEC.wide_opening_deg + 0.5:
        _ramp_gripper(rb, w.gripper(), SPEC.wide_opening_deg, 0.3)
        w.run(0.5)
        rec["opened_wider"] = True
        found, _ = _way_out(rb, cmd, CP.WAY_OUT_KINDS)
    if found is None:
        rec.update(refused=True, why="withdrawal judged not clear")
        return rec
    rec["kind"] = found.kind
    _fly(rb, found, rec)
    if rec.get("halt"):
        rec.update(refused=True, why=f"halted: {rec['halt']}")
    return rec


def summarise(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for option in OPTIONS:
        mine = [r for r in rows if r["option"] == option]
        if not mine:
            continue
        flown = [r for r in mine if not r["refused"]]
        out[option] = {
            "refusals": sum(r["refused"] for r in mine), "lifts": len(mine),
            "residual_lifts": sum(r["residual_mm"] > 0.0 for r in mine),
            "residual_max_mm": max(r["residual_mm"] for r in mine),
            "drag_max_mm": max(r["drag_mm"] for r in mine),
            "drag_mean_mm": sum(r["drag_mm"] for r in mine) / len(mine),
            "min_clearance_mm": min((r["min_clearance_mm"] for r in flown),
                                    default=float("nan")),
        }
    return out


def choose(summary: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """The owner's rule: 0 refusals and zero residual pad contact first, then
    the smallest drag (mean).  None when no option has 0 refusals."""
    zero = [o for o, s in summary.items() if s["refusals"] == 0]
    if not zero:
        return None
    return min(zero, key=lambda o: (summary[o]["residual_lifts"],
                                    summary[o]["residual_max_mm"],
                                    summary[o]["drag_mean_mm"]))
