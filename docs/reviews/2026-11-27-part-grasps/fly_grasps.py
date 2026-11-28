#!/usr/bin/env python3
"""Part D (refs #212, #216, #214): one pick-and-place per PART design, in physics.

    python3 fly_grasps.py --design 111 --out results/part111.json

THE PROTOCOL, FIXED BEFORE ANY FLIGHT.
  planner   the one the repo would use for the shape: the crane for blocks, the
            side-on grasp for cylinders (the crane refuses every cylinder).
  scene     FWDCenterLabSivaPool, offline world (`tools.place_sweep.fly`): the
            crane module flies unchanged against the simulator's MuJoCo model
            and contact settings, through the host mirror of its IK.  No server,
            SDK, bridge or latency.  The part stands upright at yaw 0; the rest
            of the board is empty.
  cells     the planner's own validated pair first (crane r3c2 -> r3c3, #195;
            side-on r2c2 -> r1c1, #180).  If PLANNING refuses, the pairs in
            FALLBACK are tried in order; the first that plans is flown, once.
            Every refusal before it is recorded.  A refusal about the part's
            height does not depend on the cell, so it is not retried.
  verdict   the executor's own criteria and the repo's D6 evaluator on the
            recorded run, as tests/integration/test_place_offline.py does.

One flight per design: this shows whether a design CAN be picked and what went
wrong if not.  It is not a success rate.
"""
import argparse
import json
import math
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src"), str(REPO / "native_mujoco")]

import tools.place_sweep  # noqa: E402,F401  (paths)
import tools.withdrawal_replay  # noqa: E402,F401
from tools.place_sweep.fly import as_task, fly  # noqa: E402

PRIMARY = {"crane": ("r3c2", "r3c3"), "side": ("r2c2", "r1c1")}
FALLBACK = [("r3c2", "r3c3"), ("r2c2", "r1c1"), ("r2c2", "r2c3"), ("r2c3", "r2c2"),
            ("r3c3", "r3c2"), ("r1c2", "r1c1"), ("r2c1", "r3c1")]
CELL_INDEPENDENT = "tall; the pads at"


def planner_for(design):
    return "crane" if design[0] == "1" else "side"


def clean(value):
    """JSON-safe, with numpy scalars and arrays turned into plain numbers."""
    return json.loads(json.dumps(value, default=lambda o: getattr(o, "tolist", lambda: str(o))()))


def judge(run, dest, oid):
    """The repo's D6 evaluator on the recorded run (see test_place_offline.py)."""
    from replay_lift_evaluation import MODEL, SCENE, KinematicWorld
    from reachy_ai.evaluation.lift_replay import replay_place
    from reachy_ai.evaluation.panel_routes import evaluate
    world = KinematicWorld(str(MODEL), str(SCENE))
    replay = replay_place(as_task(run, dest, oid), run["states"], contacts_at=world.contacts,
                          collidable=world.collidable, episode_id="offline")
    verdict = evaluate("place_on_cell", replay.result, replay.spec, plan=replay.plan)
    return {"successful": bool(verdict.is_successful), "explanation": str(verdict.explanation),
            "reported": clean(verdict.reported)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    oid, planner = f"part{args.design}_a", planner_for(args.design)
    pairs = [PRIMARY[planner]] + [p for p in FALLBACK if p != PRIMARY[planner]]
    # Read from the scene model itself: fly() does not hand events to the crane's
    # placement planner, so its GRASP_ENVELOPE event would not be seen here.
    from reachy_ai.scene.awareness import SceneModel
    model = SceneModel.from_yaml(str(REPO / "scenes" / "FWDCenterLabSivaPool.yaml"))
    seen_as = model.grasp_target(oid)
    record = {"design": args.design, "object": oid, "planner": planner,
              "scene_kind": model.get(oid).kind,
              "grasp_envelope": ({"of": seen_as.envelope_of, "outline": seen_as.kind,
                                  "size_mm": [round(1000 * v, 3) for v in seen_as.size]}
                                 if seen_as.envelope_of else None),
              "refused_pairs": [], "flown": None}
    started = time.time()
    for origin, dest in pairs:
        run = fly(oid=oid, origin=origin, dest=dest, grasp=planner, record=True)
        if "refused" in run:
            record["refused_pairs"].append({"origin": origin, "dest": dest, "why": run["refused"]})
            print(f"{oid} {origin}->{dest} refused: {run['refused']}", flush=True)
            if CELL_INDEPENDENT in run["refused"]:
                record["stopped"] = "the refusal does not depend on the cell"
                break
            continue
        outcome = run["outcome"]
        held = outcome.get("held") or {}
        replaced = outcome.get("replaced") or {}
        flown = {
            "origin": origin, "dest": dest, "grasp": run["grasp"],
            "criteria": run["criteria"], "held": held, "replaced": replaced,
            "placement_error_mm": run["error_mm"], "tilt_deg": run["tilt_deg"],
            "release_drag_mm": run["drag_mm"],
            "pad_contact_after_release_mm": run["pad_contact_after_release_mm"],
            "others_moved_max_mm": run["others_moved_max_mm"],
            "withdrawal": outcome.get("withdrawal"), "aborted": run["aborted"],
            "refusal_in_flight": outcome.get("refusal"), "halt": outcome.get("halt"),
            "final": run["final"], "sim_s": run["sim_s"],
            "final_object": run["final_object"],
            "event_kinds": [k for k, _ in run["events"]],
            "events_of_interest": [dict(kw, kind=k) for k, kw in run["events"]
                                   if k in ("HELD", "IN_HAND", "SUPPORTED", "RELEASE", "RELEASED",
                                            "ABORT", "REFUSED", "HALT", "CORRECTION",
                                            "STRADDLE", "SETDOWN_MEASURE", "side_opening_chosen",
                                            "side_roll_model")],
        }
        try:
            flown["d6"] = judge(run, dest, oid)
        except Exception as exc:                       # recorded, not hidden
            flown["d6"] = {"error": f"{type(exc).__name__}: {exc}"}
        record["flown"] = clean(flown)
        break
    record["wall_s"] = round(time.time() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    f = record["flown"]
    if f is None:
        print(f"{oid}: NOT FLOWN ({len(record['refused_pairs'])} pair(s) refused)")
    else:
        print(f"{oid}: flown {f['origin']}->{f['dest']} {f['grasp']}: criteria {f['criteria']} "
              f"err {f['placement_error_mm']:.1f} mm tilt {f['tilt_deg']:.1f} deg drag {f['release_drag_mm']} "
              f"aborted={f['aborted']} final={f['final']} d6={f['d6'].get('successful', f['d6'])}")


if __name__ == "__main__":
    main()
