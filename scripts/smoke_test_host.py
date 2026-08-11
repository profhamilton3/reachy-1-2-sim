"""Host-side CLI smoke test for the Reachy 1.2 simulator.

Connects to localhost:50051 via reachy-sdk, enumerates joints, exercises
compliant toggle, sends bounded joint commands, and verifies convergence.
Exits nonzero on any failure.  Does not require Jupyter.

Usage:
    python3 scripts/smoke_test_host.py
    python3 scripts/smoke_test_host.py --host 127.0.0.1 --port 50051

The simulator Docker container must be running before this script is called.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

try:
    from reachy_sdk import ReachySDK
except ImportError:
    print("FAIL: reachy-sdk not installed.  Run: pip install reachy-sdk==0.7.0")
    sys.exit(1)

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

_EXPECTED_JOINTS = {
    "r_arm": [
        "r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw",
        "r_elbow_pitch", "r_forearm_yaw", "r_wrist_pitch", "r_wrist_roll",
        "r_gripper",
    ],
    "l_arm": [
        "l_shoulder_pitch", "l_shoulder_roll", "l_arm_yaw",
        "l_elbow_pitch", "l_forearm_yaw", "l_wrist_pitch", "l_wrist_roll",
        "l_gripper",
    ],
    "head": [
        "neck_roll", "neck_pitch", "neck_yaw",
        "l_antenna", "r_antenna",
    ],
}

_SAFE_COMMAND_DEG = 5.0   # small bounded command, safe for dry-run mode
_CONVERGE_TOL_DEG = 2.0
_CONVERGE_WAIT_S  = 1.5


def _check(label: str, condition: bool, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    print(f"  {tag}  {label}" + (f": {detail}" if detail else ""))
    return condition


def run_smoke_test(host: str, port: int) -> int:
    """Run all checks.  Returns number of failures."""
    failures = 0
    print(f"\nReachy 1.2 simulator smoke test — {host}:{port}")
    print("=" * 60)

    # ── 1. Connection ─────────────────────────────────────────────────────────
    print("\n[1] Connection")
    try:
        reachy = ReachySDK(host=host, sdk_port=port)
        time.sleep(0.5)
        print(f"  {PASS}  Connected to {host}:{port}")
    except Exception as exc:
        print(f"  {FAIL}  Connection failed: {exc}")
        return 1

    # ── 2. Joint enumeration ──────────────────────────────────────────────────
    print("\n[2] Joint enumeration")
    for part_name, expected in _EXPECTED_JOINTS.items():
        part = getattr(reachy, part_name, None)
        if part is None:
            failures += 1
            print(f"  {FAIL}  {part_name}: not found on SDK object")
            continue
        present = set(part.joints.keys())
        for jname in expected:
            if not _check(f"{part_name}.{jname}", jname in present):
                failures += 1

    # ── 3. Joint reads ────────────────────────────────────────────────────────
    print("\n[3] Joint position reads")
    for part_name in ("r_arm", "l_arm", "head"):
        part = getattr(reachy, part_name, None)
        if part is None:
            continue
        for jname, joint in part.joints.items():
            try:
                pos = joint.present_position
                ok = math.isfinite(pos)
                if not _check(f"{jname} present_position={pos:.2f}°", ok):
                    failures += 1
            except Exception as exc:
                print(f"  {FAIL}  {jname}: read error: {exc}")
                failures += 1

    # ── 4. Compliant toggle ───────────────────────────────────────────────────
    print("\n[4] Compliant toggle (turn_on / turn_off)")
    for part_name, sample_joint_path in (
        ("r_arm",  lambda r: r.r_arm.r_shoulder_pitch),
        ("head",   lambda r: r.head.neck_roll),
    ):
        try:
            reachy.turn_on(part_name)
            time.sleep(0.2)
            j = sample_joint_path(reachy)
            is_stiff = not j.compliant
            reachy.turn_off(part_name)
            time.sleep(0.2)
            is_compliant = j.compliant
            ok = is_stiff and is_compliant
            if not _check(f"{part_name} toggle: stiff={is_stiff} relaxed={is_compliant}", ok):
                failures += 1
        except Exception as exc:
            print(f"  {FAIL}  {part_name} toggle error: {exc}")
            failures += 1

    # ── 5. Bounded joint command + convergence ────────────────────────────────
    print(f"\n[5] Bounded command ({_SAFE_COMMAND_DEG}°) + convergence")
    try:
        reachy.turn_on("r_arm")
        time.sleep(0.3)
        j = reachy.r_arm.r_shoulder_pitch
        start_pos = j.present_position
        target = start_pos + _SAFE_COMMAND_DEG
        j.goal_position = target
        time.sleep(_CONVERGE_WAIT_S)
        final = j.present_position
        error = abs(final - target)
        ok = error <= _CONVERGE_TOL_DEG
        if not _check(
            f"r_shoulder_pitch: target={target:.2f}° final={final:.2f}° err={error:.2f}°",
            ok
        ):
            failures += 1
        # Return to start
        j.goal_position = start_pos
        time.sleep(_CONVERGE_WAIT_S)
        reachy.turn_off("r_arm")
    except Exception as exc:
        print(f"  {FAIL}  command/convergence error: {exc}")
        failures += 1

    # ── 6. Force sensors + fans ───────────────────────────────────────────────
    print("\n[6] Force sensors + fans")
    for attr, names in (
        ("force_sensors", ("r_force_gripper", "l_force_gripper")),
    ):
        group = getattr(reachy, attr, None)
        if group is None:
            print(f"  {SKIP}  {attr}: not present")
            continue
        for name in names:
            sensor = getattr(group, name, None)
            ok = sensor is not None and math.isfinite(sensor.force)
            if not _check(f"{name} force={sensor.force if sensor else '?'}", ok):
                failures += 1

    fans = getattr(reachy, "fans", None)
    if fans is None:
        print(f"  {SKIP}  fans: not present")
    else:
        for fname in ("r_arm_fan", "l_arm_fan", "head_fan"):
            fan = getattr(fans, fname, None)
            ok = fan is not None
            if not _check(f"{fname} readable", ok):
                failures += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failures == 0:
        print("  RESULT: PASS — all checks succeeded")
    else:
        print(f"  RESULT: FAIL — {failures} check(s) failed")
    print("=" * 60 + "\n")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Reachy 1.2 simulator host smoke test")
    parser.add_argument("--host", default=os.environ.get("REACHY_IP", "localhost"))
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()
    sys.exit(run_smoke_test(args.host, args.port))


if __name__ == "__main__":
    main()
