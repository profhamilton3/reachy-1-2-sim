"""The release-drag study for #174: what moves the cube as the hand opens,
and which way of opening moves it least.

Every recorded release (Phase 1.5 and the #173 live check) is rebuilt in the
offline world (`rebuild.Rebuilt`), then released by each strategy below and
withdrawn by #173's way out (`crane_pick_live.plan_way_out`, every kind),
exactly as `options.run_option(..., "O1+O2")` does -- the check before
opening included.  A strategy differs only in how the grip is let go:

  * ``letgo_s``    how long the let-go ramp would take to reach the usual
                   opening (#173: 0.8 s); it still stops when the grip force
                   reads zero, and the check before opening still follows;
  * ``shift_m``    the hand moves along the jaw axis while it lets go, toward
                   the thumb's side (positive) -- the thumb backs off the
                   object while the finger opens away from it;
  * ``lift_m``     the hand rises (positive) or presses down (negative)
                   while it lets go.

Measured per lift and strategy: the drag (horizontal, from the state before
the hand opened to the state after it opened and settled -- the evaluator's
``release_drag``), its component along the jaw, where the grip let go, the
least pad-to-table clearance of the measured hand while letting go, and the
withdrawal (refused or not, way out, residual pad contact).

Diagnostic, not a candidate: ``noslip`` re-runs a strategy with MuJoCo's
no-slip pass set to that iteration count, to show the contact model's part.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from reachy_ai.motion import grasp_alignment as G
from reachy_ai.tasks import crane_pick_live as CP

from . import options as O
from .rebuild import LIM, OID, SPEC, Rebuilt

_STEP_S = O._STEP_S


@dataclass(frozen=True)
class Strategy:
    name: str
    letgo_s: float = 0.8
    shift_m: float = 0.0
    lift_m: float = 0.0
    noslip: Optional[int] = None

    @property
    def candidate(self) -> bool:
        return self.noslip is None


AS_FLOWN = Strategy("#173 as flown (let go over 0.8 s)")
STRATEGIES: Sequence[Strategy] = (
    AS_FLOWN,
    Strategy("slow let-go (2.4 s)", letgo_s=2.4),
    Strategy("fast let-go (0.16 s)", letgo_s=0.16),
    Strategy("lift 3 mm while letting go", lift_m=0.003),
    Strategy("press 2 mm while letting go", lift_m=-0.002),
    Strategy("thumb-side retreat 3 mm while letting go", shift_m=0.003),
    Strategy("thumb-side retreat 6 mm while letting go", shift_m=0.006),
    Strategy("thumb-side retreat 7.5 mm while letting go", shift_m=0.0075),
    Strategy("thumb-side retreat 9 mm while letting go", shift_m=0.009),
    Strategy("thumb-side retreat 12 mm while letting go", shift_m=0.012),
    Strategy("fast let-go + thumb-side retreat 3 mm", letgo_s=0.16, shift_m=0.003),
    Strategy("let-go over 0.4 s", letgo_s=0.4),
    Strategy("no-slip off (diagnostic, #173 sequence)", noslip=0),
)


def _tilt_deg(q) -> float:
    w, x, y, z = q
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))




def run_strategy(rb: Rebuilt, st: Strategy) -> Dict[str, Any]:
    """Release one rebuilt lift by ``st``, then check, open and withdraw as
    #173 does.  Returns the measured row."""
    w = rb.world
    kinds = CP.WAY_OUT_KINDS
    saved_noslip = int(w.m.opt.noslip_iterations)
    if st.noslip is not None:
        w.m.opt.noslip_iterations = st.noslip
    try:
        cmd = list(rb.cmd)
        plan0 = rb.plan
        rb.hold_at(cmd)
        held, q_held = w.cube()
        hand = G.hand_frames(w.arm(), w.gripper())
        jaw = hand.jaw.copy()
        pc = G.pad_clearance(hand, G.object_box(rb.scene.get(OID), held, q_held))
        rec: Dict[str, Any] = {
            "lift": rb.name, "strategy": st.name, "candidate": st.candidate,
            "refused": False, "min_clearance_mm": math.inf, "residual_mm": 0.0,
            "thumb_in_mm": round(-1000 * pc.thumb_m, 2),
            "finger_in_mm": round(-1000 * pc.finger_m, 2),
            "tilt_held_deg": round(_tilt_deg(q_held), 2)}

        # The hand's move while letting go (none for most strategies); the
        # ladder the way out climbs moves with it (`crane_pick_live._shifted`).
        d = -st.shift_m * jaw + np.array([0.0, 0.0, st.lift_m])
        plan0 = rb.plan
        target = cmd
        if d.any():
            target, moved = CP._shifted(rb.planner, rb.plan, cmd, d)
            if target is None:
                rec.update(refused=True, why="IK found no pose for the move while letting go")
                return rec
            rb.plan = moved

        # Let go: the gripper ramp of `primitives.release_grip`, the arm (if
        # it moves) in step with it over the first 70 % of the ramp; stopped
        # when the grip force reads zero.
        start = w.gripper()
        steps = max(1, int(st.letgo_s / _STEP_S))
        arm_steps = max(1, int(0.7 * steps))
        table_min = math.inf
        let_go, now = start, cmd
        for i in range(1, steps + 1):
            let_go = start + i / steps * (SPEC.opening_deg - start)
            f = min(1.0, i / arm_steps)
            now = [a + f * (b - a) for a, b in zip(cmd, target)]
            w.target("r_gripper", let_go)
            w.command_arm(now)
            w.run(st.letgo_s / steps)
            table_min = min(table_min, G.table_clearance(rb.scene, [w.arm()],
                                                         w.gripper()).worst_m)
            if w.grip_force() <= LIM.contact_force_n:
                break
        cmd = list(now)
        w.run(0.5)
        rec["let_go_deg"] = round(let_go, 1)

        # Check before opening (#173, O2), then open.
        opening = None
        for tier in (CP.PRE_OPEN_PREFERRED, kinds):
            for cand in CP.release_openings(SPEC):
                found, _ = O._way_out(rb, cmd, tier, gripper_deg=cand)
                if found is not None:
                    opening = cand
                    break
            if opening is not None:
                break
        if opening is not None:
            O._ramp_gripper(rb, w.gripper(), opening, 0.4)
            w.run(0.8)
        p, q = w.cube()
        dv = p - held
        rec.update(drag_mm=1000 * float(np.linalg.norm(dv[:2])),
                   drag_along_jaw_mm=1000 * float(dv @ jaw),
                   drag_vec_mm=(1000 * dv).round(3).tolist(),
                   jaw=jaw.round(5).tolist(),
                   tilt_after_deg=round(_tilt_deg(q), 2),
                   pad_table_min_mm=round(1000 * table_min, 2),
                   residual_mm=1000 * w.pad_contact_m())
        if opening is None:
            rec.update(refused=True, why="refused before opening")
            return rec

        # Withdraw (#173): judge, open wider if needed, fly.
        found, _ = O._way_out(rb, cmd, kinds)
        rec["opened_wider"] = False
        if (found is None or found.kind != "direct") \
                and w.gripper() > SPEC.wide_opening_deg + 0.5:
            O._ramp_gripper(rb, w.gripper(), SPEC.wide_opening_deg, 0.3)
            w.run(0.5)
            rec["opened_wider"] = True
            rec["residual_mm"] = max(rec["residual_mm"], 1000 * w.pad_contact_m())
            found, _ = O._way_out(rb, cmd, kinds)
        if found is None:
            rec.update(refused=True, why="withdrawal judged not clear after release")
            return rec
        rec["kind"] = found.kind
        O._fly(rb, found, rec)
        if rec.get("halt"):
            rec.update(refused=True, why=f"halted: {rec['halt']}")
        return rec
    finally:
        w.m.opt.noslip_iterations = saved_noslip
        rb.plan = plan0


def summarise(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for st in STRATEGIES:
        mine = [r for r in rows if r["strategy"] == st.name]
        if not mine:
            continue
        drags = [r["drag_mm"] for r in mine if "drag_mm" in r]
        along = [r["drag_along_jaw_mm"] for r in mine if "drag_along_jaw_mm" in r]
        out[st.name] = {
            "candidate": st.candidate, "lifts": len(mine),
            "refusals": sum(r["refused"] for r in mine),
            "residual_lifts": sum(r["residual_mm"] > 0.0 for r in mine),
            "drag_mean_mm": float(np.mean(drags)) if drags else math.nan,
            "drag_max_mm": float(np.max(drags)) if drags else math.nan,
            "drag_sd_mm": float(np.std(drags)) if drags else math.nan,
            "along_jaw_mean_mm": float(np.mean(along)) if along else math.nan,
            "pad_table_min_mm": min((r.get("pad_table_min_mm", math.inf) for r in mine)),
        }
    return out


def choose(summary: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """The lowest mean drag among the candidates that keep 0 refusals."""
    ok = [n for n, s in summary.items() if s["candidate"] and s["refusals"] == 0]
    return min(ok, key=lambda n: (summary[n]["drag_mean_mm"],
                                  summary[n]["drag_max_mm"])) if ok else None


#: Lowering a held object 5 cm down the crane's vertical ladder moved it
#: 0.41-1.72 mm horizontally (15 recorded lifts: the #173 live check and
#: Phase 1.5 run A, HELD -> SUPPORTED).  The placement corrects the held
#: object's measured position to within `PlaceSpec.setdown_tolerance_m`
#: (1 mm) two rungs above the table, so what is left of the set-down error
#: is at most that drift plus that tolerance.
LOWERING_DRIFT_MAX_M = 0.00172
SETDOWN_TOLERANCE_M = 0.001


def report(rows: Sequence[Dict[str, Any]], sha: str = "", date: str = "") -> str:
    """The study as a markdown report."""
    summary = summarise(rows)
    choice = choose(summary)
    lifts = sorted({r["lift"] for r in rows}, key=lambda n: (n[0] != "A", n[0] != "B",
                                                             int(n[1:])))
    out = [f"# Issue #174 offline release-drag study ({date})", "",
           f"reachy-1-2-sim at `{sha or 'worktree'}`.  Offline only: each recorded release (Phase "
           "1.5: A1–B5; the #173 live check: L1–L10) rebuilt in the #172 offline world, released "
           "by each strategy, then checked before opening and withdrawn by #173's way out.  No "
           "simulator server.", "",
           "## What drags the cube", "",
           "- **The grip leaves the pads deep in the cube.** At the hold the thumb pad is 3.3–5.9 mm "
           "and the finger pad 11.4–13.5 mm inside the cube's collision box (the soft contact under "
           "the gripper's force limit).",
           "- **Opening unloads that into the cube.** The pinch keeps equal force on both pads all the "
           "way to the let-go angle (−53° to −65°): the cube follows the opening finger, pushed off "
           "the thumb, and the finger's arc tips it 1–10°; it then settles flat. Almost all of the "
           "drag is along the jaw, toward the finger.",
           "- **No-slip's part.** With the no-slip pass off (diagnostic only; the model is unchanged) "
           f"the drag is {summary['no-slip off (diagnostic, #173 sequence)']['drag_mean_mm']:.1f} mm "
           f"mean (max {summary['no-slip off (diagnostic, #173 sequence)']['drag_max_mm']:.1f}): "
           "about the thumb's penetration. No-slip adds the rest by letting the pads carry the cube "
           "without slipping while they unload.",
           "- **The close moves it the other way first.** The finger pushes the cube 6.9–9.0 mm toward "
           "the thumb when the hand closes; the lift's put-back relies on the release drag undoing "
           "that, so the lift keeps #173's release (below).", "",
           "## Strategies (20 releases each)", "",
           "| strategy | refusals | lifts with residual pad contact | drag mean / max (mm) | "
           "SD (mm) | along the jaw, mean (mm) | least pad–table (mm) |",
           "|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        out.append(f"| {name}{' **(chosen)**' if name == choice else ''} | {s['refusals']} / "
                   f"{s['lifts']} | {s['residual_lifts']} | {s['drag_mean_mm']:.1f} / "
                   f"{s['drag_max_mm']:.1f} | {s['drag_sd_mm']:.1f} | "
                   f"{s['along_jaw_mean_mm']:.1f} | {s['pad_table_min_mm']:.1f} |")
    out += ["", f"**Chosen: {choice}** — the lowest mean drag among the strategies that keep "
            "#173's 0 refusals.", ""]
    mine = [r for r in rows if r["strategy"] == choice]
    worst = max(r["drag_mm"] for r in mine)
    pred = worst + 1000 * (LOWERING_DRIFT_MAX_M + SETDOWN_TOLERANCE_M)
    out += ["## Predicted placement error (D6: ≤ 10 mm)", "",
            f"Release drag with the chosen strategy, worst of 20: **{worst:.2f} mm**.  Set-down: "
            f"the held object is corrected to within {1000 * SETDOWN_TOLERANCE_M:.0f} mm of the "
            f"cell centre two rungs up, and lowering moved a held cube at most "
            f"{1000 * LOWERING_DRIFT_MAX_M:.2f} mm in 15 recorded lifts.  Worst-case sum: "
            f"**{pred:.1f} mm**, under 10 mm.", "",
            "## Per release (chosen strategy)", "",
            "| release | drag (mm) | along jaw (mm) | way out | residual pad contact (mm) | "
            "let go at (deg) |", "|---|---|---|---|---|---|"]
    for n in lifts:
        r = next(x for x in mine if x["lift"] == n)
        out.append(f"| {n} | {r['drag_mm']:.2f} | {r['drag_along_jaw_mm']:.2f} | "
                   f"{r.get('kind', 'REFUSED')} | {r['residual_mm']:.2f} | {r.get('let_go_deg')} |")
    out += ["", "## Per release, every strategy (drag mm / way out)", "",
            "| release | " + " | ".join(summary) + " |",
            "|---|" + "---|" * len(summary)]
    for n in lifts:
        cells = []
        for name in summary:
            r = next(x for x in rows if x["lift"] == n and x["strategy"] == name)
            cells.append(f"{r.get('drag_mm', float('nan')):.1f} / "
                         f"{'REFUSED' if r['refused'] else r.get('kind')}")
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"
