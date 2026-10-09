"""Host-side CLI smoke test for the Reachy 1.2 simulator.

Connects to localhost:50051 via reachy-sdk, enumerates joints, exercises
compliant toggle, sends bounded joint commands, and verifies convergence.
Then checks the scene (#2): its manipulable objects load, and every one's grasp
point passes `SceneModel.check_point` -- above the tabletop and outside every
static obstacle.  That is the pad-point rule, not a reachability or clearance
check.  Exits nonzero on any failure.  Does not require Jupyter.

Usage:
    python3 scripts/smoke_test_host.py
    python3 scripts/smoke_test_host.py --host 127.0.0.1 --port 50051
    python3 scripts/smoke_test_host.py --scene scenes/FWDCenterLabSivaPool.yaml

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
    # Reported by main(); importing this module must not exit, so the
    # scene check below can be exercised without the SDK.
    ReachySDK = None

# The bounded command goes through the motion primitives and the safety gate
# like every other move (#107), never as a raw goal_position write here.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from reachy_ai.motion import primitives as P  # noqa: E402
from reachy_ai.scene.awareness import SceneModel  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

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


def _bounded_command(reachy) -> bool:
    """[5]: nudge r_shoulder_pitch by _SAFE_COMMAND_DEG and back, through
    `primitives.nudge_joint` (which checks `safety.gate_check()`).  A refused
    gate is a FAIL, never a skip.  Returns True on pass."""
    arm = reachy.r_arm
    try:
        reachy.turn_on("r_arm")
        time.sleep(0.3)
        j = arm.r_shoulder_pitch
        start_pos = j.present_position
        target = P.nudge_joint(arm, "r_shoulder_pitch", _SAFE_COMMAND_DEG)
        time.sleep(_CONVERGE_WAIT_S)
        final = j.present_position
        error = abs(final - target)
        ok = _check(
            f"r_shoulder_pitch: target={target:.2f}° final={final:.2f}° err={error:.2f}°",
            error <= _CONVERGE_TOL_DEG)
        # Return to start
        P.nudge_joint(arm, "r_shoulder_pitch", start_pos - j.present_position)
        time.sleep(_CONVERGE_WAIT_S)
        return ok
    except P.MotionRefused as exc:
        print(f"  {FAIL}  safety gate refused the bounded command: {exc}")
        return False
    except Exception as exc:
        print(f"  {FAIL}  command/convergence error: {exc}")
        return False
    finally:
        try:
            reachy.turn_off("r_arm")
        except Exception as exc:
            print(f"  {FAIL}  turn_off after the bounded command: {exc}")


def default_scene_path() -> str:
    """The scene the simulator is configured for, as a path in this checkout.

    The container names it by its in-container path (`/opt/scenes/...`), which
    does not exist on the host, so only the file name is taken from
    REACHY_SIM_SCENE_FILE.  Unset means the panel's scene.
    """
    name = os.path.basename(os.environ.get("REACHY_SIM_SCENE_FILE", "")
                            or "FWDCenterLabSivaPool.yaml")
    return os.path.join(_REPO, "scenes", name)


def check_scene(scene_path: str) -> int:
    """[7]: the scene's manipulable objects load, and each grasp point passes
    `SceneModel.check_point`.  Returns the number of failures."""
    try:
        scene = SceneModel.from_yaml(scene_path)
    except Exception as exc:
        print(f"  {FAIL}  {scene_path}: scene did not load: {exc}")
        return 1
    ids = scene.manipulable_ids()
    print(f"  {PASS}  {os.path.basename(scene_path)} loaded: "
          f"{len(ids)} manipulable object(s)")
    if not ids:
        print(f"  {SKIP}  grasp points: the scene declares no manipulable objects")
        return 0
    failures = 0
    for oid in ids:
        point = scene.grasp_point(oid)
        violation = scene.check_point(point)
        where = "(" + ", ".join(f"{v:.3f}" for v in point) + ")"
        if not _check(f"{oid} grasp point {where}", violation is None,
                      "" if violation is None else str(violation)):
            failures += 1
    return failures


def run_smoke_test(host: str, port: int, scene_path: str = "") -> int:
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
    if not _bounded_command(reachy):
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

    # ── 7. Scene objects + grasp points ──────────────────────────────────────
    print("\n[7] Scene objects + grasp points (SceneModel.check_point)")
    failures += check_scene(scene_path or default_scene_path())

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
    parser.add_argument("--scene", default="",
                        help="scene YAML for check [7] (default: the file named "
                             "by REACHY_SIM_SCENE_FILE, else FWDCenterLabSivaPool)")
    args = parser.parse_args()
    if ReachySDK is None:
        print("FAIL: reachy-sdk not installed.  Run: pip install reachy-sdk==0.7.0")
        sys.exit(1)
    sys.exit(run_smoke_test(args.host, args.port, args.scene))


if __name__ == "__main__":
    main()
