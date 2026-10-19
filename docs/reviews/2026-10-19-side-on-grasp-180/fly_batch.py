"""Offline physics flights of the side-on grasp, the crane and the planner
choice (#180), through `tools/place_sweep/fly.py` -- the executor itself
(`crane_pick_live.execute_crane_pick`) on the offline world.

Jobs are a JSON list of `fly.fly` keyword sets (oid, origin, dest, grasp,
others, cube_xy_offset, yaw_deg); each row written is the job, the #174
criteria and the numbers the README tables use.  One process per job.

Run: python3 fly_batch.py JOBS.json OUT.jsonl [--procs N]
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO, REPO / "src", REPO / "native_mujoco", REPO / "tests" / "unit"):
    sys.path.insert(0, str(p))

#: The events a row keeps (the rest stays out of the table).
_KEEP = {"GRASP_CHOSEN", "side_pick_chosen", "side_place_chosen", "STRADDLE", "CLOSED",
         "HELD", "SUPPORTED", "RELEASED", "WITHDRAWN", "HALT", "ABORT", "CORRECTION",
         "SETDOWN_CORRECTION", "ALIGN_REFUSED", "DESCENT_CHECK_FAILED"}


def one(job):
    from tools.place_sweep import fly as F
    t0 = time.time()
    try:
        r = F.fly(**job)
    except Exception as exc:                 # a crash is a row, with its trace
        return dict(job=job, error=f"{type(exc).__name__}: {exc}",
                    trace=traceback.format_exc()[-2000:])
    if "refused" in r:
        return dict(job=job, refused=r["refused"], detail=r.get("detail"),
                    wall_s=round(time.time() - t0, 1))
    out = r["outcome"]
    held = out.get("lifted_and_held") or {}
    rep = out.get("replaced") or {}
    pl = out.get("place") or {}
    ev = [dict(kind=k, **{kk: vv for kk, vv in e.items() if kk not in ("candidates",)})
          for k, e in r["events"] if k in _KEEP]
    closed = next((e for e in ev if e["kind"] == "CLOSED"), {})
    chosen = next((e for e in ev if e["kind"] in ("side_place_chosen", "side_pick_chosen")), {})
    return dict(
        job=job, grasp=r["grasp"], choice=r.get("choice"),
        azimuth_deg=chosen.get("azimuth_deg"), pitch_deg=chosen.get("pitch_deg"),
        criteria=r["criteria"], all_criteria=bool(r["criteria"]) and all(r["criteria"].values()),
        halt=out.get("halt"), refusal=out.get("refusal"), final=r["final"],
        error_mm=round(r["error_mm"], 2), tilt_deg=round(r["tilt_deg"], 3),
        others_moved_max_mm=round(r["others_moved_max_mm"], 4),
        drag_mm=r["drag_mm"], way_out=r["way_out"], aborted=r["aborted"],
        pad_contact_after_release_mm=round(r["pad_contact_after_release_mm"], 3),
        rise_mm=held.get("rise_end_mm"), slip_mm=held.get("in_hand_slip_mm"),
        grip_force_n=held.get("grip_force_n"), closed_grip_deg=closed.get("gripper_measured_deg"),
        grip_force_after_open_n=rep.get("grip_force_after_open_n"),
        setdown_margin=pl.get("setdown_margin_deg"),
        sim_s=round(r["sim_s"], 1), wall_s=round(time.time() - t0, 1), events=ev)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs")
    p.add_argument("out")
    p.add_argument("--procs", type=int, default=max(1, mp.cpu_count() - 2))
    a = p.parse_args()
    jobs = json.loads(pathlib.Path(a.jobs).read_text())
    with mp.Pool(a.procs, maxtasksperchild=1) as pool, open(a.out, "w") as fh:
        for row in pool.imap(one, jobs):
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            j = row["job"]
            print(j.get("oid"), j.get("origin"), "->", j.get("dest"), j.get("grasp"), "|",
                  row.get("error") or row.get("refused") or
                  f"{row['grasp']} all={row['all_criteria']} halt={row['halt']} "
                  f"err={row['error_mm']} final={row['final']}", flush=True)


if __name__ == "__main__":
    main()
