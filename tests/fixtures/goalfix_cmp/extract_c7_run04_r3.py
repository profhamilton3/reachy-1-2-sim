"""Extraction script for c7_run04_r3_setup.json. NEVER run by the tests.

Usage (read-only on the sealed batch; writes only --out):
    PYTHONPATH=<M checkout/export>/src:<...>/native_mujoco:<...>:<...>/scripts \
      python extract_c7_run04_r3.py --repo <M export> --batch <sealed batch dir> --out <fixture.json>

It reads only run-04's SHA256SUMS-listed files: run_manifest.json,
cli/validation_S2-B4-c-r3.json, capture.jsonl, crosscheck.json, and
ev/e1_server_runs/run_stage_a_slice/{commands,states}.jsonl (+ derived sums).
The setup leg is rebuilt from evidence alone: epoch-3 joint_commands from the
epoch's first command through the last command of the final route waypoint block
recorded in the validation JSON's window (global 2923..4252); the sidecar under
control/ is NOT read. Reproduction is checked against the recorded JSON.
"""
import argparse, hashlib, json, sys
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    repo = Path(a.repo)
    for sub in ("src", "native_mujoco", ".", "scripts"):
        sys.path.insert(0, str(repo / sub))
    import numpy as np
    from tools.goalfix_cmp import evidence as ev, cycle as cy, pathcheck as pc, echo
    from tools.goalfix_cmp._units import route_rad
    from reachy_ai.motion import rig_routes as R

    b = Path(a.batch)
    r4 = b / "run-04"
    evdir = r4 / "ev" / "e1_server_runs" / "run_stage_a_slice"
    read = {
        "run-04/run_manifest.json": r4 / "run_manifest.json",
        "run-04/cli/validation_S2-B4-c-r3.json": r4 / "cli" / "validation_S2-B4-c-r3.json",
        "run-04/capture.jsonl": r4 / "capture.jsonl",
        "run-04/crosscheck.json": r4 / "crosscheck.json",
        "run-04/ev/e1_server_runs/run_stage_a_slice/commands.jsonl": evdir / "commands.jsonl",
        "run-04/ev/e1_server_runs/run_stage_a_slice/states.jsonl": evdir / "states.jsonl",
        "run-04/ev/e1_server_runs/run_stage_a_slice/derived-SHA256SUMS": evdir / "derived-SHA256SUMS",
    }
    rec = json.loads(read["run-04/cli/validation_S2-B4-c-r3.json"].read_text())
    e = ev.verify_and_load(str(evdir), "states.jsonl", "commands.jsonl", "derived-SHA256SUMS")
    LO, HI = 2923, 4252
    idx = [i for i in range(LO, HI + 1)
           if e.commands.kind[i] == "joint_command" and not e.brackets[i].unplaceable]
    t_on = ev.leg_turn_on_state_index(e, idx)
    sp = {n: float(e.states.position_rad[t_on, k]) for k, n in enumerate(pc.R_JOINTS)}
    route = route_rad(R.PLACE_ROUTE)
    leg = cy.LegSpec("setup", route, R.CRITICAL_JOINTS, idx, sp, t_on)
    lr = cy.evaluate_leg(e, leg)
    res = {k: v for k, v in lr.pathcheck.items() if isinstance(v, pc.CheckResult)}
    man = json.loads(read["run-04/run_manifest.json"].read_text())
    cyc = next(c for c in man["cycles"] if c["name"] == "S2-B4-c-r3")
    rec_setup = cyc["legs"]["setup"]
    for cid, r in res.items():
        assert rec_setup[cid]["passed"] == r.passed, cid
        assert rec_setup[cid]["index"] == r.first_violation_index, cid
        assert rec_setup[cid]["detail"] == r.detail, cid
    assert json.loads(json.dumps(lr.per_waypoint)) == rec["per_waypoint"]["setup"]

    # shipped classifier at M over this leg (setup goto context only)
    ctx = lr.goto_context
    full = [None] * len(e.commands)
    for li, gi in enumerate(idx):
        full[gi] = ctx[li]
    echo_res = echo.classify_commands(e, full, leg_turn_on_state_index={idx[0]: t_on})
    labels = {}
    for jn in pc.R_JOINTS:
        runs = []
        for li, gi in enumerate(idx):
            lab = echo_res[gi].joints[jn].label
            if runs and runs[-1][2] == lab and runs[-1][1] == li - 1:
                runs[-1][1] = li
            else:
                runs.append([li, li, lab])
        labels[jn] = runs

    t21 = e.commands.target_rad[np.asarray(idx)]
    other = t21[:, 8:21]
    assert (other == other[0]).all(), "non-right-arm joints not constant"
    rows = [{"g": int(gi), "l": li, "epoch": int(e.commands.epoch[gi]),
             "t_hi": float(e.brackets[gi].t_hi),
             "arm8": [float(x) for x in t21[li, :8]]} for li, gi in enumerate(idx)]

    # carry provenance from capture.jsonl (gen == epoch 3)
    builds, subs, batches = {}, {}, {}
    for line in read["run-04/capture.jsonl"].open():
        d = json.loads(line)
        if d.get("gen") != 3:
            continue
        if d["type"] == "build":
            builds[d["seq"]] = d
        elif d["type"] == "submit":
            subs.setdefault(d["batch_id"], []).append(d)
    GJ = pc.R_JOINTS.index("r_gripper")
    chain = []  # commands local 154..157: origin, carry, first exact S, next
    for li, gi in enumerate(idx):
        seq = li + 1
        bd = builds[seq]
        assert bd["target"][:8] == [float(x) for x in t21[li, :8]]
        if li >= 154 and li <= 157:
            src = bd["source"]
            msgs = []
            for bid in sorted({s["batch_id"] for s in bd["drained"]}):
                ss = subs.get(bid, [])
                msgs.append({"batch_id": bid,
                             "joints_carried": sorted(s["index"] for s in ss),
                             "gripper_value": next((s["goal_position"] for s in ss if s["index"] == GJ), None)})
            chain.append({"local": li, "global": gi, "build_seq": seq,
                          "gripper_target": bd["target"][GJ], "gripper_source_batch": src[GJ],
                          "messages": msgs})
    out = {
        "source": {
            "batch": str(b), "run": "run-04", "cycle": "S2-B4-c-r3", "leg": "setup", "arm": "B",
            "M": "155fc1549812146a8579080c72560c2c1875262c",
            "P": "f97dc59f1c567289b4aea58a8615163421dd3e85",
            "root_SHA256SUMS_sha256": sha(b / "SHA256SUMS"),
            "sealed_files_read_sha256": {k: sha(v) for k, v in read.items()},
            "extraction_script": "tests/fixtures/goalfix_cmp/extract_c7_run04_r3.py",
            "extraction_script_sha256": sha(__file__),
            "leg_reconstruction": ("no sidecar read: epoch-3 joint_command rows global 2923..4252 "
                                   "(epoch first command .. last command of the REST block recorded in "
                                   "the validation JSON window); reproduces recorded per-check table "
                                   "and per_waypoint exactly"),
        },
        "route": "PLACE_ROUTE",
        "start_pose8": sp,
        "recorded_setup_checks": rec_setup,
        "other13_constant": [float(x) for x in other[0]],
        "rows": rows,
        "echo_labels_runs": labels,
        "gripper_chain": chain,
        "indices": {"chain_origin_local": 154, "carry_local": 155, "first_exact_S_local": 156,
                    "c7_recorded_first_violation_index": rec_setup["C7"]["index"]},
    }
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", a.out, Path(a.out).stat().st_size)


if __name__ == "__main__":
    main()
