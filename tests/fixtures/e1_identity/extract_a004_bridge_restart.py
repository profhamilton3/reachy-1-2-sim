"""Extraction script for a004_bridge_restart.json. NEVER run by the tests.

Usage (read-only on the sealed a004 session; writes only --out):
    python extract_a004_bridge_restart.py --session <sealed a004 session dir> --out a004_bridge_restart.json

Reads, from the sealed session:
  e1_server_runs/run_20261002_035557/commands.jsonl   (all 4 rows)
  e1_server_runs/run_20261002_035557/states.jsonl     (streamed; keeps states seq 3723 and 3724 only)
  plan_S2-B4-c-r1.json                                (setup leg entry)
and records each file's sha256. States are a PROJECTION of the sealed lines
(type, seq, sim_step, wall_time_ns, cmd_seq plus the 8 right-arm joints'
name/position_rad/effort/compliant; values equal the sealed lines). The
command rows are verbatim. The test rewrites only `wall_time_ns`.
"""
import argparse, hashlib, json
from pathlib import Path

RUN = "e1_server_runs/run_20261002_035557"
R_JOINTS = ["r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw", "r_elbow_pitch",
            "r_forearm_yaw", "r_wrist_pitch", "r_wrist_roll", "r_gripper"]
BASELINE_SEQ, POST_SEQ = 3723, 3724


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reduce_state(s):
    keep = {k: s[k] for k in ("type", "seq", "sim_step", "wall_time_ns", "cmd_seq")}
    by = {j["name"]: j for j in s["joints"]}
    keep["joints"] = [{k: by[n][k] for k in ("name", "position_rad", "effort", "compliant")}
                      for n in R_JOINTS]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = Path(a.session)
    cmds_p, states_p, plan_p = s / RUN / "commands.jsonl", s / RUN / "states.jsonl", s / "plan_S2-B4-c-r1.json"
    commands = [json.loads(l) for l in cmds_p.read_text().splitlines() if l.strip()]
    assert [c.get("seq", c["type"]) for c in commands] == [1, 2, "reset", 1], commands
    found = {}
    with open(states_p, "rb") as f:
        for raw in f:
            if raw.startswith(b'{"type":"state","seq":%d,' % BASELINE_SEQ) or \
               raw.startswith(b'{"type":"state","seq":%d,' % POST_SEQ):
                d = json.loads(raw)
                found[d["seq"]] = reduce_state(d)
            if len(found) == 2:
                break
    base, post = found[BASELINE_SEQ], found[POST_SEQ]
    assert base["cmd_seq"] == 2 and post["cmd_seq"] == 1
    plan = json.loads(plan_p.read_text())
    out = {
        "source": {
            "session": s.name,
            "files": {str(p.relative_to(s)): {"sha256": sha(p)} for p in (cmds_p, states_p, plan_p)},
            "note": "a004 (2026-10-02): r1 setup turn_on after a bridge restart; native cmd_seq 2 -> 1",
        },
        "plan_leg": {"name": "S2-B4-c-r1-setup", "plan_entry": plan["legs"]["S2-B4-c-r1-setup"]},
        "commands": commands,
        "baseline_state": base,
        "post_state": post,
    }
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
