"""Read-only route-clearance recorder (review 2026-09-12, E1:
docs/reviews/2026-09-12-repair-review-and-56-74-design.md, section 5).

NOT YET FIT TO FLY E1 -- prerequisite recorded in docs/adr/0003 ("What is
still open", E1): this recorder streams the seven ARM7 joints only, so the
realised-clearance report assumes the gripper fully open (`gripper_deg=None`)
at every sample. The aperture is the variable that decides whether the hand
is inside the tube at all (ADR-0003's discrepancy section), so a log without
`r_gripper.present_position` cannot say what clearance the hand actually had.
Add `r_gripper` to the recorded joints (same `r_arm` object, same read-only
access) and thread the per-sample aperture into `realised_clearance` before
the first operator flight; the synthetic-log tests need the same.

No through-the-move data exist for the tabletop legs (PRESENT/HOVER <->
REST) -- the rail rows are the only realised-clearance numbers on record,
and they are tube readings on a different corridor. This script closes that
gap the safe way: it CONNECTS to the SDK only to STREAM `present_position`
at 20 Hz while an operator flies a named route by hand (teach pendant,
notebook cell, or the panel) -- it never sets a `goal_position` and never
calls anything in `src/reachy_ai/motion/primitives.py`. The realised joint
trajectory is saved as JSON under `runs/`, then reported as per-link worst
clearance against every scene object, under both hand models ("tube",
today's default, and "shells", #56/#74's candidate), next to the PLANNED
clearance at the same sample count -- so a re-flight can finally say what
the realised tracking degradation actually costs, on either model.

Refuses to run unless REACHY_SIM_RECORD_CLEARANCE=1 is set: streaming joint
telemetry during a flight an operator did not mean to record is still a
surprise worth a deliberate opt-in, even though nothing here can move the
arm.

Usage (operator flies the route by hand during --duration):
    export REACHY_SIM_RECORD_CLEARANCE=1
    python3 scripts/measure_route_clearance.py --route LOWER_TO_REST \\
        --duration 15 --host <ip>
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional, Sequence

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

try:
    from reachy_sdk import ReachySDK
except ImportError:
    ReachySDK = None  # only needed by main(); the reporting half is offline

from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.motion.kinematics import hand_radius, joint_path, link_capsules  # noqa: E402
from reachy_ai.scene.awareness import SceneModel  # noqa: E402

SAMPLE_HZ = 20.0
RUNS_DIR = _HERE.parent / "runs"
HAND_MODES = ("tube", "shells")


# ── Recording (needs a live arm) ─────────────────────────────────────────────

def record_joint_log(arm, duration_s: float, hz: float = SAMPLE_HZ) -> List[Dict]:
    """Stream `present_position` for every ARM7 joint at `hz`, for
    `duration_s` seconds. Read-only: reads `.present_position` on each
    joint object and nothing else -- it never assigns `.goal_position` or
    calls `turn_on`/`turn_off`, so it cannot command the arm even by
    accident."""
    period = 1.0 / hz
    joints = {name: getattr(arm, name) for name in R.ARM7}
    samples: List[Dict] = []
    t0 = time.monotonic()
    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= duration_s:
            break
        pose = {name: float(j.present_position) for name, j in joints.items()}
        samples.append({"t": elapsed, "joints": pose})
        next_tick = t0 + period * (len(samples))
        time.sleep(max(0.0, next_tick - time.monotonic()))
    return samples


def save_log(samples: List[Dict], route: str, scene_path: str) -> pathlib.Path:
    RUNS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = RUNS_DIR / f"route_clearance_{route}_{stamp}.json"
    with open(path, "w") as f:
        json.dump({"route": route, "scene": scene_path, "sample_hz": SAMPLE_HZ,
                  "samples": samples}, f, indent=1)
    return path


# ── Reporting (pure geometry -- offline, unit-tested on a synthetic log) ────

def _worst_clearance_over_samples(
    q7_and_gripper: Sequence[tuple], scene: SceneModel,
) -> Dict[str, Dict[str, float]]:
    """Per-hand-mode, per-object worst clearance over a sequence of
    (q7, gripper_deg) samples."""
    out: Dict[str, Dict[str, float]] = {hand: {} for hand in HAND_MODES}
    for q7, gripper_deg in q7_and_gripper:
        for hand in HAND_MODES:
            caps = link_capsules(q7, "right", gripper_deg, hand=hand)
            for oid, c in scene.clearances(caps).items():
                worst = out[hand].get(oid)
                if worst is None or c.distance < worst:
                    out[hand][oid] = c.distance
    return out


def realised_clearance(samples: List[Dict], scene: SceneModel) -> Dict[str, Dict[str, float]]:
    """Worst per-link clearance actually reached, from a recorded log.

    The log has no commanded gripper aperture (only ARM7 is streamed), so
    this uses the worst case (`gripper_deg=None`, fully open) -- the safe
    assumption `link_capsules` itself documents, not a guess specific to
    this script.
    """
    return _worst_clearance_over_samples(
        (([sample["joints"][j] for j in R.ARM7], None) for sample in samples),
        scene)


def planned_clearance(
    route: str, n_samples: int, scene: SceneModel,
) -> Dict[str, Dict[str, float]]:
    """Worst per-link clearance the COMMANDED path implies, sampled
    `n_samples` times per leg of `route`'s own `FOOTPRINT_LEGS` entry -- the
    same joint-space line `fly_route` actually commands. Routes with no
    `FOOTPRINT_LEGS` entry (WAVE, POINT) return empty per-hand dicts: there
    is nothing here to compare a realised log against for them.
    """
    if route not in R.FOOTPRINT_LEGS:
        return {hand: {} for hand in HAND_MODES}
    waypoints = R.FOOTPRINT_LEGS[route]
    steps = max(2, n_samples)

    def samples():
        for a, b in zip(waypoints, waypoints[1:]):
            qa = [a[j] for j in R.ARM7]
            qb = [b[j] for j in R.ARM7]
            gripper_deg = max(a["r_gripper"], b["r_gripper"], key=hand_radius)
            for q in joint_path(qa, qb, steps=steps):
                yield q, gripper_deg

    return _worst_clearance_over_samples(samples(), scene)


def report(samples: List[Dict], route: str, scene_path: str) -> Dict:
    """The full report: realised vs planned, both hand models, for `route`
    against `scene_path`'s own object poses (as loaded -- this script has no
    live scene link, only the SDK's joint telemetry; a re-flight that needs
    live object poses should record them separately)."""
    scene = SceneModel.from_yaml(scene_path)
    return {
        "route": route,
        "scene": scene_path,
        "n_samples": len(samples),
        "realised": realised_clearance(samples, scene),
        "planned": planned_clearance(route, len(samples), scene),
    }


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    if os.environ.get("REACHY_SIM_RECORD_CLEARANCE") != "1":
        print("Refusing to run: set REACHY_SIM_RECORD_CLEARANCE=1 to record.\n"
              "This script only STREAMS present_position -- it never "
              "commands the arm -- but starting a recording an operator did "
              "not ask for is still a surprise worth an explicit opt-in.")
        sys.exit(1)
    if ReachySDK is None:
        print("FAIL: reachy-sdk not installed. Run: pip install reachy-sdk==0.7.0")
        sys.exit(1)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("REACHY_IP", "localhost"))
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--route", required=True, choices=sorted(R.FOOTPRINT_LEGS))
    parser.add_argument("--scene", default=str(
        _HERE.parent / "scenes" / "FWDCenterLabSivaPool.yaml"))
    parser.add_argument("--duration", type=float, default=15.0,
                        help="seconds to record; fly the route by hand during "
                             "this window")
    args = parser.parse_args()

    reachy = ReachySDK(host=args.host, sdk_port=args.port)
    print(f"Recording {args.route} for {args.duration:.1f}s at {SAMPLE_HZ:.0f} Hz "
         "-- fly the route now. This only reads present_position.")
    samples = record_joint_log(reachy.r_arm, args.duration)
    path = save_log(samples, args.route, args.scene)
    print(f"Saved {len(samples)} samples to {path}")

    result = report(samples, args.route, args.scene)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
