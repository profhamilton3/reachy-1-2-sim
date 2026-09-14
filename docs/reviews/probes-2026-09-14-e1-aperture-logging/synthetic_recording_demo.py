"""Demonstration for the aperture-logging PR (2026-09-14): builds a
synthetic joint log in the exact schema `record_joint_log`/`save_log` now
produce -- ARM7 plus `r_gripper`, degrees, `t` from a monotonic clock -- and
runs it through `measure_route_clearance.report()`, both the strict default
and the explicit `allow_missing_aperture=True` compatibility path, so the
calculated clearance output can be inspected without any hardware or a real
flight. Offline: no SDK import is exercised, no simulator, no motion.

Run from the repo root:

    PYTHONPATH=src:native_mujoco:scripts python3 \\
        docs/reviews/probes-2026-09-14-e1-aperture-logging/synthetic_recording_demo.py
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "../../../scripts"))
sys.path.insert(0, os.path.join(_HERE, "../../../src"))

import measure_route_clearance as mrc  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402

SCENE_PATH = os.path.relpath(
    os.path.join(_HERE, "../../../scenes/FWDCenterLabSivaPool.yaml"))


def synthetic_lower_to_rest_flight():
    """A plausible LOWER_TO_REST flight: PRESENT -> REST_SHUT -> REST over
    1.5 s at 20 Hz, gripper closing from -45 (open) at PRESENT to +20 (shut)
    by REST, roughly matching FOOTPRINT_LEGS's own commanded endpoints --
    hand-built, not recorded, but in exactly the schema a real recording
    would use."""
    waypoints = [R.PRESENT, R.REST_SHUT, R.REST]
    hz = 20.0
    samples = []
    t = 0.0
    for a, b in zip(waypoints, waypoints[1:]):
        steps = 10
        for k in range(steps):
            frac = k / (steps - 1)
            joints = {j: a[j] + frac * (b[j] - a[j]) for j in R.ARM7}
            joints["r_gripper"] = a["r_gripper"] + frac * (b["r_gripper"] - a["r_gripper"])
            samples.append({"t": round(t, 4), "joints": joints})
            t += 1.0 / hz
    return samples


def main():
    samples = synthetic_lower_to_rest_flight()

    print("=== schema sample (first and last) ===")
    print(json.dumps(samples[0], indent=2))
    print(json.dumps(samples[-1], indent=2))

    print(f"\n=== validated_samples: {len(samples)} samples, strict (default) ===")
    q7_and_gripper, assumed = mrc.validated_samples(samples)
    print(f"all valid: assumed={assumed} (expect [])")

    print("\n=== report(): realised vs planned, both hand models ===")
    result = mrc.report(samples, "LOWER_TO_REST", SCENE_PATH)
    print(json.dumps(result, indent=2))

    print("\n=== compare against the OLD (pre-aperture) behaviour: "
          "assumed-open on every sample ===")
    old_style = [{"t": s["t"], "joints": {j: s["joints"][j] for j in R.ARM7}}
                for s in samples]
    old_result = mrc.report(old_style, "LOWER_TO_REST", SCENE_PATH,
                           allow_missing_aperture=True)
    print(f"assumed_samples: {len(old_result['realised_aperture_assumed_samples'])} "
         f"of {old_result['n_samples']} (the whole log, as expected for a "
         "schema-1-shaped input)")

    print("\n=== the difference this makes: realised tube clearance, "
          "old (assumed-open) vs new (actual aperture) ===")
    for oid in result["realised"]["tube"]:
        new_c = result["realised"]["tube"][oid]
        old_c = old_result["realised"]["tube"][oid]
        print(f"  {oid:14s} new(actual aperture)={new_c*100:+7.2f} cm  "
             f"old(assumed open)={old_c*100:+7.2f} cm  "
             f"delta={((new_c - old_c) * 100):+6.2f} cm")

    print("\n=== missing-data guard: a corrupted sample is refused, not silently open ===")
    corrupted = [dict(s) for s in samples]
    corrupted[5] = {"t": corrupted[5]["t"],
                    "joints": {**corrupted[5]["joints"], "r_gripper": float("nan")}}
    try:
        mrc.report(corrupted, "LOWER_TO_REST", SCENE_PATH)
        print("FAIL: should have raised ApertureDataError")
    except mrc.ApertureDataError as exc:
        print(f"raised as expected: {exc}")


if __name__ == "__main__":
    main()
