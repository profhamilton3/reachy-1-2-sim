"""Offline stand-in for `scripts/measure_route_clearance.py`'s recorder
step, used only by `tests/unit/test_e1_leg_chain.py` to drive the REAL
`leg.sh`/`parked.sh` end to end (PR #124 re-review, R3) without
`reachy_sdk` or a live native server.

Reproduces exactly what `measure_route_clearance.main()` does immediately
after recording -- `save_log` -> `link_e1_flight.build_base_sidecar` ->
`link_e1_flight.write_sidecar` (`scripts/measure_route_clearance.py:
776-789`) -- using synthetic samples and a synthetic native-server run
directory in place of a live recording and a live server connection.
Everything downstream of this (the linker, the experiment gate, the tail
check, `start_variant`) is the real, unmodified code, run by the real
`leg.sh`/`parked.sh` exactly as an operator would invoke them.

Selected by `E1_PYTHON` pointing to a small wrapper (built by the test)
that intercepts only the `measure_route_clearance.py` invocation and
delegates every other call to the real `python3`.

Test-controlled via environment variables (all optional):
  E1_TEST_POSE           "armon" (settling gripper/wrist_roll, the R1
                          INIT shape) or anything else for an all-zero
                          stiff-zero-shaped pose (parked.sh's shape).
  E1_TEST_GRIPPER_END    armon pose's settled r_gripper, degrees.
  E1_TEST_OBJECT_ID      tracked board object id (must match the scene).
  E1_TEST_CONTACT_AT_S   if set, a contact is placed in the synthetic
                          state stream at this offset (seconds).
  E1_TEST_DISPLACE_M     the board object's total displacement (m) over
                          the recording, linear in wall time.
  E1_TEST_CORRUPT_LINE   if set, states.jsonl line at this 0-based index
                          (not the last line) is overwritten with
                          non-JSON text -- malformed evidence.
  E1_TEST_NO_RUN_DIR     if "1", no run directory/states.jsonl is written
                          at all -- missing evidence.
  E1_TEST_RESET_MID_FLIGHT  if "1", `sim_step` is made to drop back to 0
                          partway through states.jsonl -- a scene reset
                          mid-recording (#126's `reset_in_window`).
  E1_TEST_OMIT_OBJECT    if "1", the tracked board object is left out of
                          every state's `objects` list -- it is never
                          tracked, so it has no displacement_m entry
                          (#126's board-coverage requirement).
"""
import argparse
import json
import math
import os
import pathlib
import sys
import types

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "native_mujoco")

import measure_route_clearance as mrc  # noqa: E402
import link_e1_flight as lf  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402


def synth_samples(seconds, *, hz=20.0, t0_wall_ns=1_000_000_000_000,
                  pose="armon", gripper_end=-38.1):
    out = []
    n = int(seconds * hz)
    for k in range(n):
        t = k / hz
        p = {j: 0.0 for j in R.R_JOINTS}
        if pose == "armon":
            frac = min(t / 8.0, 1.0)
            p["r_gripper"] = gripper_end * (1 - math.exp(-4 * frac)) / (1 - math.exp(-4))
            p["r_wrist_roll"] = 39.0 * frac
        out.append({"t": t, "wall_time_ns": t0_wall_ns + int(t * 1e9), "joints": p})
    return out


def build_run_dir(root, samples, *, object_id, contact_at_s, displace_m,
                  corrupt_line=None, reset_mid_flight=False, omit_object=False):
    run = pathlib.Path(root) / "run_test"
    run.mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(json.dumps({"contacts_tracked": True}))
    t0 = samples[0]["wall_time_ns"]
    t_end = samples[-1]["wall_time_ns"]
    lines = []
    seq = 0
    step = 0
    w = t0 - 200_000_000
    hi = t_end + 200_000_000
    n_states = int((hi - w) / 20_000_000) + 1
    reset_at_index = n_states // 2
    i = 0
    while w <= hi:
        t_rel = (w - t0) / 1e9
        k = min(max(int(round(t_rel * 20)), 0), len(samples) - 1)
        joints = [{"name": j, "position_rad": math.radians(samples[k]["joints"][j]),
                   "compliant": False, "effort": 0.0} for j in R.R_JOINTS]
        frac = min(max((w - t0) / (t_end - t0), 0.0), 1.0) if t_end != t0 else 0.0
        pos = [0.4318 + displace_m * frac, -0.1524, 0.76]
        contacts = []
        if contact_at_s is not None and abs(t_rel - contact_at_s) < 0.011:
            contacts = [{"arm_geom": "r_hand_tube", "object_id": object_id}]
        objects = [] if omit_object else [
            {"object_id": object_id, "pos_xyz": pos, "quat_wxyz": [1, 0, 0, 0]}]
        # A real mid-recording scene reset restarts `sim_step` at 0 (the
        # same counter `contacts_in_flight_window` inspects for
        # `reset_in_window` -- see its own docstring); `seq`/`wall_time_ns`
        # stay monotonic regardless, exactly like the real server.
        step_field = 0 if (reset_mid_flight and i >= reset_at_index) else step
        lines.append(json.dumps({
            "seq": seq, "sim_step": step_field, "cmd_seq": 1, "wall_time_ns": w,
            "joints": joints,
            "objects": objects,
            "contacts": contacts,
        }))
        seq += 1
        step += 1
        i += 1
        w += 20_000_000
    if corrupt_line is not None and 0 <= corrupt_line < len(lines):
        lines[corrupt_line] = "{not json"
    (run / "states.jsonl").write_text("\n".join(lines) + "\n")
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True)
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--record-root", required=True)
    ap.add_argument("--port", default=None)
    ap.add_argument("--status-url", default=None)
    args, _ = ap.parse_known_args()

    pose = os.environ.get("E1_TEST_POSE", "armon")
    object_id = os.environ.get("E1_TEST_OBJECT_ID", "pool_box_1")
    contact_at_s = os.environ.get("E1_TEST_CONTACT_AT_S")
    contact_at_s = float(contact_at_s) if contact_at_s else None
    displace_m = float(os.environ.get("E1_TEST_DISPLACE_M", "0.0"))
    gripper_end = float(os.environ.get("E1_TEST_GRIPPER_END", "-38.1"))
    no_run_dir = os.environ.get("E1_TEST_NO_RUN_DIR") == "1"
    corrupt_line = os.environ.get("E1_TEST_CORRUPT_LINE")
    corrupt_line = int(corrupt_line) if corrupt_line else None
    reset_mid_flight = os.environ.get("E1_TEST_RESET_MID_FLIGHT") == "1"
    omit_object = os.environ.get("E1_TEST_OMIT_OBJECT") == "1"

    root = pathlib.Path(args.record_root)
    root.mkdir(parents=True, exist_ok=True)

    # Same convention test_link_e1_flight.py uses (monkeypatch.setattr
    # RUNS_DIR) -- here via direct assignment, since this runs as a
    # separate subprocess pytest's monkeypatch fixture cannot reach: keep
    # every log this stub writes under the test's own tmp_path rather
    # than the real (gitignored, but shared) repo `runs/` directory.
    mrc.RUNS_DIR = root / "runs"
    samples = synth_samples(args.duration, pose=pose, gripper_end=gripper_end)
    path = mrc.save_log(samples, args.route, args.scene)
    print(f"Saved {len(samples)} samples to {path}")
    if no_run_dir:
        run_dir = root / "run_missing"
    else:
        run_dir = build_run_dir(root, samples, object_id=object_id,
                                contact_at_s=contact_at_s, displace_m=displace_m,
                                corrupt_line=corrupt_line,
                                reset_mid_flight=reset_mid_flight,
                                omit_object=omit_object)

    fake_ident = types.SimpleNamespace(
        as_dict=lambda: {"ok": True}, manifest={"contacts_tracked": True},
        scene_chain_sha256={})
    sidecar = lf.build_base_sidecar(
        log_path=str(path), samples=samples, scene_path=args.scene,
        identity_check=fake_ident, run_dir=str(run_dir))
    sidecar_path = lf.write_sidecar(str(path), sidecar)
    print(f"Saved link sidecar to {sidecar_path}")


if __name__ == "__main__":
    main()
