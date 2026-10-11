#!/usr/bin/env python3
"""Grade recorded lifts offline, and the measured panel routes, under #172.

NO SIMULATOR IS RUN LIVE.  Two things happen, both offline:

  1. RECORDED LIFTS.  Each argument names one recorded lift (a directory with
     `live/states.jsonl` and `live/panel/panel-task-final.json`).  Every
     recorded state inside the job is posed KINEMATICALLY in the offline world
     (`SimulationCore._load_world`, which now has the table and the rails):
     joints and object poses written into qpos, `mj_forward`, and the contacts
     read off.  Nothing is stepped.  The lift is then rebuilt from the states
     (`evaluation.lift_replay`) and judged by `evaluate_lift_object`.

  2. MEASURED ROUTES (`--routes`).  rest_forearm, stow_arm and wave are flown
     in the headless offline world by `OfflineRouteRunner` and judged with the
     shared contact rule table.  Pointing cannot be flown offline (it needs a
     planner and a live board) and is listed as such.

Usage:
  scripts/replay_lift_evaluation.py --out report.md \\
      A1=path/to/runA/lift-1:refused  A2=path/to/runA/lift-2:full ... --routes

The expectation after the colon is what was OBSERVED in the run: `full` (a
completed full cycle) or `refused` (lifted and put back; withdrawal refused).
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
from typing import List, Tuple

_REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (_REPO / "src", _REPO / "native_mujoco"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reachy_ai.evaluation import contact_rules as CR  # noqa: E402
from reachy_ai.evaluation.base import ViolationKind  # noqa: E402
from reachy_ai.evaluation.lift_replay import replay_lift  # noqa: E402
from reachy_ai.evaluation.panel_routes import (  # noqa: E402
    LIFT_PUT_BACK_KEY, LIFT_RISE_KEY, LIFT_SLIP_KEY, contact_samples, evaluate,
)

MODEL = _REPO / "native_mujoco" / "model" / "reachy_1_2.xml"
SCENE = _REPO / "scenes" / "FWDCenterLabSivaPool.yaml"
EXPECTED = {"full": "full-cycle pass",
            "refused": "lift pass, withdrawal refused"}


class KinematicWorld:
    """The offline world, posed from a recorded state.  Never stepped."""

    def __init__(self, model_path: str, scene_path: str) -> None:
        import mujoco
        from simulation_core import load_world

        self._mj = mujoco
        self.model, self.objects, _specs, _xml = load_world(model_path,
                                                            scene_path)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self._joint = {}
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,
                                          mujoco.mjtJoint.mjJNT_SLIDE):
                self._joint[name] = int(m.jnt_qposadr[j])
        self._free = {}
        for oid in self.objects:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, oid)
            for j in range(m.njnt):
                if m.jnt_bodyid[j] == b and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    self._free[oid] = int(m.jnt_qposadr[j])
        self.collidable = sorted({
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
            for g in range(m.ngeom)
            if (m.geom_contype[g] or m.geom_conaffinity[g])
            and not CR.role_of(mujoco.mj_id2name(
                m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
            ) in CR.ROBOT_ROLES} - {None})

    def contacts(self, state) -> List[Tuple[str, str, float]]:
        mj, m, d = self._mj, self.model, self.data
        for j in state.get("joints") or ():
            adr = self._joint.get(j["name"])
            if adr is not None:
                d.qpos[adr] = float(j["position_rad"])
        for o in state.get("objects") or ():
            adr = self._free.get(o["object_id"])
            if adr is not None:
                d.qpos[adr:adr + 3] = o["pos_xyz"][:3]
                d.qpos[adr + 3:adr + 7] = o.get("quat_wxyz") or (1, 0, 0, 0)
        mj.mj_forward(m, d)
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > 0.0:
                continue        # inside the margin, not touching
            b1 = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom1]))
            b2 = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom2]))
            out.append((b1 or "", b2 or "", float(c.dist)))
        return out


def _read_states(path: pathlib.Path) -> List[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _verdict_label(v) -> str:
    if not v.is_successful:
        return "lift FAIL"
    if v.reported.get("full_cycle"):
        return "full-cycle pass"
    return f"lift pass, withdrawal {v.reported.get('withdrawal')}"


def replay_lifts(items, world) -> Tuple[List[dict], List[str]]:
    rows, details = [], []
    for name, folder, expect in items:
        folder = pathlib.Path(folder)
        task = json.loads((folder / "live/panel/panel-task-final.json").read_text())
        states = _read_states(folder / "live/states.jsonl")
        rep = replay_lift(task, states, contacts_at=world.contacts,
                          collidable=world.collidable, episode_id=name)
        v = evaluate("lift_object", rep.result, rep.spec, plan=rep.plan)
        judged = CR.judge_all(contact_samples(rep.result) or [],
                              route=rep.spec.route,
                              target_id=rep.spec.target_object_id)
        unintended = [x.description for x in v.violations
                      if x.kind is ViolationKind.FORBIDDEN_CONTACT]
        m = rep.result.metrics
        got = _verdict_label(v)
        rows.append({
            "lift": name, "observed": EXPECTED[expect], "verdict": got,
            "match": got == EXPECTED[expect],
            "rise_cm": m.get(LIFT_RISE_KEY, float("nan")) * 100,
            "crane_rise_cm": (rep.crane_reported["rise_end_mm"] or 0) / 10,
            "slip_mm": m.get(LIFT_SLIP_KEY, float("nan")) * 1000,
            "crane_slip_mm": rep.crane_reported["in_hand_slip_mm"],
            "put_back_mm": m.get(LIFT_PUT_BACK_KEY, float("nan")) * 1000,
            "crane_offset_mm": rep.crane_reported["offset_from_start_mm"],
            "ends_present": v.reported.get("ends_at_present"),
            "withdrawal": v.reported.get("withdrawal_detail"),
            "withdrawal_contacts": v.reported.get("withdrawal_contacts") or [],
            "approach": v.reported.get("approach") or {},
            "legs": rep.approach_legs,
            "attempt_start": rep.attempt_start,
            "contacts": len(judged), "unintended": unintended,
            "allowed": sorted({f"{j.sample.body1}–{j.sample.body2} @ "
                               f"{j.sample.phase.value}"
                               + (f" on {j.sample.route}" if j.sample.route
                                  else "") for j in judged if j.allowed}),
            "sim_steps": (rep.result.start_sim_step, rep.result.end_sim_step),
            "offset_s": rep.clock_offset_s,
            "windows": [(w.phase.value, round(w.end - w.start, 2))
                        for w in rep.windows],
        })
        details.append(f"### {name}\n\n```\n{v.explanation}\n```\n")
        print(f"{name}: {got} (observed: {EXPECTED[expect]})", flush=True)
    return rows, details


def fly_routes() -> List[dict]:
    from reachy_ai.evaluation.panel_offline import OfflineRouteRunner, executable
    from reachy_ai.motion.recipe import TrajectoryRecipe

    rows = []
    for ability in ("rest_forearm", "stow_arm", "wave", "point_cell",
                    "point_object"):
        ok, why = executable(ability)
        if not ok:
            rows.append({"ability": ability, "flown": False, "why": why})
            continue
        recipe = TrajectoryRecipe.load(
            str(_REPO / "recipes" / "panel" / f"{ability}_baseline_v1.yaml"))
        runner = OfflineRouteRunner(ability, str(MODEL), str(SCENE))
        result, spec = runner.run(recipe, seed=0)
        v = evaluate(ability, result, spec, recipe, runner.policy)
        judged = CR.judge_all(contact_samples(result) or [], route=spec.route)
        rows.append({"ability": ability, "flown": True, "route": spec.route,
                     "successful": v.is_successful,
                     "unintended": [j for j in judged if not j.allowed],
                     "allowed": [j for j in judged if j.allowed],
                     "collidable": result.contact_summary.get("collidable_bodies")})
        print(f"{ability}: successful={v.is_successful} "
              f"unintended={len(rows[-1]['unintended'])}", flush=True)
    return rows


def _md(rows, routes, details, sha: str) -> str:
    today = datetime.date.today().isoformat()
    out = [f"# Issue #172 offline replay — {today}", "",
           f"reachy-1-2-sim worktree at `{sha}`.  Offline only: recorded "
           "states posed kinematically in the offline FWDCenterLabSivaPool "
           "world (table and rails present); measured routes flown headless. "
           "No live simulator.", "",
           "## Recorded lifts", "",
           "| lift | observed | verdict | match | rise (cm) replay / crane | "
           "slip (mm) replay / crane | put back xy (mm) / crane 3D | "
           "ends at PRESENT | withdrawal (reported) | unintended in the lift "
           "(ATTEMPT_START→release) | approach leg (own route) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        legs = "; ".join(f"{leg}: {len(found)} unintended" if found else
                         f"{leg}: clean" for leg, found in r["approach"].items()
                         ) or ", ".join(r["legs"]) or "none flown"
        out.append(
            f"| {r['lift']} | {r['observed']} | {r['verdict']} | "
            f"{'✅' if r['match'] else '❌'} | {r['rise_cm']:.2f} / "
            f"{r['crane_rise_cm']:.2f} | {r['slip_mm']:.3f} / "
            f"{r['crane_slip_mm']} | {r['put_back_mm']:.2f} / "
            f"{r['crane_offset_mm']} | {'yes' if r['ends_present'] else 'no'} | "
            f"{r['withdrawal']} | {len(r['unintended']) or 'none'} | {legs} |")
    n_full = sum(r["verdict"] == "full-cycle pass" for r in rows)
    n_ref = sum(r["verdict"] == "lift pass, withdrawal refused" for r in rows)
    out += ["", f"Totals: {n_full} full-cycle pass, {n_ref} lift pass with "
            f"withdrawal refused, {sum(not r['match'] for r in rows)} "
            "mismatches.", ""]
    out += ["Contacts the rule table allowed, per lift:", ""]
    for r in rows:
        out.append(f"- {r['lift']}: " + ("; ".join(r["allowed"]) or "none"))
    groups = (
        ("in the lift verdict (ATTEMPT_START → release)",
         [(r["lift"], d) for r in rows for d in r["unintended"]]),
        ("in the withdraw phase (reported with the withdrawal)",
         [(r["lift"], d) for r in rows for d in r["withdrawal_contacts"]]),
        ("on the approach legs (judged under their own route; reported)",
         [(r["lift"], d) for r in rows for found in r["approach"].values()
          for d in found]),
    )
    for title, found in groups:
        out += ["", f"Unintended contacts {title}: "
                + ("none" if not found else ""), ""]
        for lift, d in found:
            out.append(f"- {lift}: {d}")
    out += ["", "Phase windows (s) and clock offset, per lift:", ""]
    for r in rows:
        out.append(f"- {r['lift']}: offset {r['offset_s']:+.3f} s; sim_step "
                   f"{r['sim_steps'][0]}→{r['sim_steps'][1]}; legs "
                   f"{', '.join(r['legs']) or 'none'}; " + ", ".join(
                       f"{p} {d}" for p, d in r["windows"]))
    if routes:
        out += ["", "## Measured panel routes, offline world with table and "
                "rails", "",
                "| ability | route | flown | successful | contacts allowed | "
                "unintended |", "|---|---|---|---|---|---|"]
        for r in routes:
            if not r["flown"]:
                out.append(f"| {r['ability']} | — | no: {r['why']} | — | — | — |")
                continue
            allowed = "; ".join(f"{j.describe()} ({j.why})"
                                for j in r["allowed"]) or "none"
            out.append(f"| {r['ability']} | {r['route']} | yes | "
                       f"{r['successful']} | {allowed} | "
                       f"{len(r['unintended']) or 'none'} |")
        if routes and routes[0].get("collidable"):
            out += ["", "Collidable non-robot bodies: "
                    + ", ".join(routes[0]["collidable"])]
    out += ["", "## Rule table", "", "| roles | allowed in | reason |",
            "|---|---|---|"]
    for row in CR.rules_as_rows():
        out.append(f"| {row['roles']} | {row['allowed_in']} | {row['reason']} |")
    out += ["", "## Route exceptions", "", "| route | link | role | at pose | "
            "reason |", "|---|---|---|---|---|"]
    for e in CR.ROUTE_EXCEPTIONS:
        out.append(f"| {e.route} | {e.link} | {e.role.value} | {e.pose} | "
                   f"{e.reason} |")
    out += ["", "## Verdicts in full", ""] + details
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("lifts", nargs="*", help="NAME=DIR:full|refused")
    p.add_argument("--out", required=True)
    p.add_argument("--routes", action="store_true")
    p.add_argument("--sha", default="")
    a = p.parse_args(argv)

    items = []
    for spec in a.lifts:
        name, rest = spec.split("=", 1)
        folder, expect = rest.rsplit(":", 1)
        if expect not in EXPECTED:
            p.error(f"{spec}: expectation must be one of {sorted(EXPECTED)}")
        items.append((name, folder, expect))

    world = KinematicWorld(str(MODEL), str(SCENE))
    rows, details = replay_lifts(items, world)
    routes = fly_routes() if a.routes else []
    pathlib.Path(a.out).write_text(_md(rows, routes, details, a.sha))
    print(f"wrote {a.out}")
    bad = [r for r in rows if not r["match"]] + \
        [r for r in routes if r.get("flown") and r["unintended"]]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
