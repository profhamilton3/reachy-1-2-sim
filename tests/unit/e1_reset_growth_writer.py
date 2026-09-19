#!/usr/bin/env python3
"""Background helper spawned by `e1_reset_fake_docker.py`'s `torn_growing`
mode (`tests/unit/test_e1_reset_verify.py`'s `TestResetShEndToEnd`, §4.F):
appends a short burst of complete, realistically-sized state records to
`states.jsonl` at short intervals -- independent of, and outliving, the
fake `docker` process that spawns it -- ending with one deliberately torn
record (no trailing newline, cut mid-write). Reproduces a native server
that keeps appending at pace while `reset_verify.py verify` polls
underneath it, the same growing-file shape as the B1 `reset_18` defect.

Usage: e1_reset_growth_writer.py <states_path> <start_seq> <start_sim_step>
                                  <count> <interval_s>
"""
import json
import sys
import time


def _joints(n=21):
    return [{"name": f"j{i}", "position_rad": 0.001 * i,
              "velocity_rad_s": 0.0, "effort": 0.01 * i,
              "compliant": False, "target_rad": 0.001 * i}
            for i in range(n)]


def _objects(n=6):
    return [{"object_id": f"obj_{i}", "pos_xyz": [0.1 * i, 0.2 * i, 0.3 * i],
              "quat_wxyz": [1.0, 0.0, 0.0, 0.0]} for i in range(n)]


def _state_line(seq, sim_step, wall_time_ns):
    doc = {"type": "state", "seq": seq, "sim_step": sim_step,
           "sim_time_s": sim_step * 0.05, "wall_time_ns": wall_time_ns,
           "cmd_seq": 0, "scene_revision": "r1", "paused": False,
           "joints": _joints(), "objects": _objects(),
           "grippers": [{"side": "right", "grasp": False, "force_n": 0.0}],
           "force_sensors": [{"name": "r_wrist", "force_n": [0.0, 0.0, 0.0]}],
           "interactive": [], "warnings": [], "contacts": []}
    return json.dumps(doc, separators=(",", ":")) + "\n"


def main() -> int:
    states_path, start_seq, start_step, count, interval_s = sys.argv[1:6]
    seq, sim_step = int(start_seq), int(start_step)
    with open(states_path, "a") as f:
        for _ in range(int(count)):
            f.write(_state_line(seq, sim_step, time.time_ns()))
            f.flush()
            seq += 1
            sim_step += 1
            time.sleep(float(interval_s))
        torn = _state_line(seq, sim_step, time.time_ns())
        f.write(torn[: len(torn) // 2])
        f.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
