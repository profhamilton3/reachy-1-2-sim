"""Where a LEVEL hand reaches on the sim board (#180).  IK only; no checks.

For each grid cell and approach azimuth (every 15 deg, plus the taught one):
the jaw-gap centre at the cell centre, `--height` above the table, moved
`--short` along the approach (short of the centre, so the palm stays out of
a 6 cm cube), the hand level (thumb-frame x up).  Solved with
`crane_pick_live.solve_gap` (gap-centred IK, attitude held <= 1 deg) on the
host IK mirror, from the taught seeds first (taught_seeds.json), then
`GRASP_STARTS`.  Reports the least joint margin (deg) and its joint.

Run: python3 reach_level.py [--height 0.034 --short 0.012 --opening -65] > reach_level.txt
"""
import argparse
import json
import math
import multiprocessing as mp
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO / "src", REPO / "tests" / "unit"):
    sys.path.insert(0, str(p))

SCENE = str(REPO / "scenes" / "FWDCenterLabSivaPool.yaml")


def level_rot(az, pitch=0.0):
    """Hand rotation for approach azimuth ``az``, the approach pitched
    ``pitch`` deg down from level (0 = level); thumb-frame x stays in the
    vertical plane of the approach (no roll about the hand axis)."""
    a, b = math.radians(az), math.radians(pitch)
    app = np.array([math.cos(a) * math.cos(b), math.sin(a) * math.cos(b), -math.sin(b)])
    z = -app
    side = np.cross(np.array([0.0, 0.0, 1.0]), app)
    side /= np.linalg.norm(side)
    x = np.cross(app, side)
    return np.column_stack([x, np.cross(z, x), z]), app


def one(args):
    cell, az, h, short, opening, pitch = args
    from host_ik_arm import HostIKArm
    from reachy_ai.motion.kinematics import R_ARM_JOINTS, joint_limits
    from reachy_ai.motion import grasp_alignment as G
    from reachy_ai.scene.awareness import SceneModel
    from reachy_ai.tasks import crane_pick_live as CP

    scene = SceneModel.from_yaml(SCENE)
    x, y, _z = scene.cell_center(cell)
    rot, app = level_rot(az, pitch)
    g = np.array([x, y, scene.table_surface_z + h]) - short * app
    seeds = [s["joints_deg"] for s in json.loads((HERE / "taught_seeds.json").read_text())
             ["squares"].values()] + CP.GRASP_STARTS
    arm = HostIKArm()
    best = None
    for s in seeds:
        hh = G.hand_frames(s, opening)
        q = CP._ik(arm, rot, g + (hh.wrist - hh.gap_mid), s)
        if q is None:
            continue
        got = CP.solve_gap(arm, rot, g, opening, q, 0.0005)
        if got is None or CP._rot_err_deg(G.hand_rotation(got[0]), rot) > 1.0:
            continue
        m = {n: min(v - a, b - v) for n, v, (a, b) in
             zip(R_ARM_JOINTS, got[0], joint_limits("right"))}
        w = min(m, key=m.get)
        if best is None or m[w] > best[0]:
            best = (m[w], w, [round(v, 2) for v in got[0]])
    return {"cell": cell, "az": az, "ok": best is not None,
            "margin_deg": None if best is None else round(best[0], 2),
            "joint": None if best is None else best[1],
            "q": None if best is None else best[2]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=float, default=0.034)
    p.add_argument("--short", type=float, default=0.012)
    p.add_argument("--opening", type=float, default=-65.0)
    p.add_argument("--pitch", type=float, default=0.0)
    p.add_argument("--cells", default="")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    from reachy_ai.scene.awareness import SceneModel
    cells = SceneModel.from_yaml(SCENE).grid_cells()
    if a.cells:
        cells = [c for c in cells if c[5:] in a.cells.split(',')]
    jobs = [(c, float(az), a.height, a.short, a.opening, a.pitch) for c in cells for az in range(-90, 181, 15)]
    with mp.Pool(max(1, mp.cpu_count() - 2)) as pool:
        rows = pool.map(one, jobs)
    print(f"# level hand, gap centre {1000 * a.height:.0f} mm above the table, "
          f"{1000 * a.short:.0f} mm short of the cell centre, opening {a.opening}, pitch {a.pitch}")
    print("cell     " + " ".join(f"{az:>5d}" for az in range(-90, 181, 15)))
    for c in cells:
        mine = [r for r in rows if r["cell"] == c]
        print(f"{c[5:]:8s} " + " ".join(f"{r['margin_deg']:5.1f}" if r["ok"] else "    ." for r in mine))
    (HERE / f"reach_level{a.tag}.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
