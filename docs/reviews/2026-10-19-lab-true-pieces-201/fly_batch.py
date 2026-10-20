"""Offline physics flights for #201 (from #180's fly_batch.py), through
`tools/place_sweep/fly.py` -- the executor itself
(`crane_pick_live.execute_crane_pick`) on the offline world.

Jobs are a JSON list of `fly.fly` keyword sets (oid, origin, dest, grasp,
others, cube_xy_offset, yaw_deg); each row written is the job, the #174
criteria and the numbers the README tables use.  One process per job.

#201: a job may also carry ``contact``, a pad contact setting applied to the
compiled model before the flight (so candidates are compared without editing
the MJCF): {"condim": 3|4|6, "torsion": m, "rolling": m, "priority": int,
"multiccd": bool}.  With ``priority`` above the other geom's, a pad contact
takes the pad's own condim and friction instead of the larger of the two.
``condim``/``torsion``/``rolling`` go on the four pad geoms only; ``multiccd``
is MuJoCo's global flag, which MuJoCo 3.11 has ON by default (a disable bit).  Absent, the model file's own setting flies.

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


PADS = ("r_thumb_col", "r_finger_col", "l_thumb_col", "l_finger_col")


def apply_contact(m, c):
    """Apply a candidate pad contact setting to a compiled model."""
    import mujoco
    for name in PADS:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if "condim" in c:
            m.geom_condim[g] = c["condim"]
        if c.get("torsion") is not None:
            m.geom_friction[g, 1] = c["torsion"]
        if c.get("rolling") is not None:
            m.geom_friction[g, 2] = c["rolling"]
        if "priority" in c:
            m.geom_priority[g] = c["priority"]
    if "multiccd" in c:                      # MuJoCo 3.11: on unless disabled
        flag = int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        m.opt.disableflags = (m.opt.disableflags & ~flag) if c["multiccd"] else (m.opt.disableflags | flag)


def patch_contact(c):
    from tools.withdrawal_replay import rebuild
    init = rebuild.World.__init__

    def __init__(self):
        init(self)
        apply_contact(self.m, c)
    rebuild.World.__init__ = __init__


def one(job):
    from tools.place_sweep import fly as F
    t0 = time.time()
    job = dict(job)
    contact = job.pop("contact", None)
    if contact:
        patch_contact(contact)
    try:
        r = F.fly(**job)
    except Exception as exc:                 # a crash is a row, with its trace
        job = dict(job, contact=contact) if contact else job
        return dict(job=job, error=f"{type(exc).__name__}: {exc}",
                    trace=traceback.format_exc()[-2000:])
    job = dict(job, contact=contact) if contact else job
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
