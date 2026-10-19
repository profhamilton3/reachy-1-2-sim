"""Is a crane attitude REACHABLE at a cell at all, whatever IK start set?  (#195)

For each (cell, face, tilt, lean) the grasp pose -- jaw-gap centre on the object's centre,
hand at the attitude `crane_pick_live.grasp_rotation` builds -- is solved with the joint
stops as hard bounds (scipy least_squares, many random starts inside the travel), and the
best residual reported: position error (mm) and attitude error (deg).  Independent of the
IK service and of `GRASP_STARTS`; it answers the geometry question only.

PYTHONPATH=.:src:native_mujoco python docs/reviews/2026-10-19-crane-retune-195/reach_attitude.py OUT.jsonl
"""
import json, math, multiprocessing as mp, os, sys
import numpy as np
from dataclasses import replace
import tools.place_sweep  # noqa: F401

SCENE = 'scenes/FWDCenterLabSivaPool.yaml'
CELLS = [f'r{r}c{c}' for r in (1, 2, 3) for c in (1, 2, 3)]
#: The tilts the palm leaves graspable (`palm_geometry.py`: both pad centres below the top
#: with the palm clear only up to ~20-25 deg), every 30 deg of lean.  Face x only: for a
#: square object face y is face x with the lean turned 90 deg.
TILTS = [10.0, 15.0, 20.0, 25.0, 30.0]
LEANS = [-180.0 + 30.0 * i for i in range(12)]


def one(args):
    cell, face, tilt, lean = args
    from scipy.optimize import least_squares
    from reachy_ai.motion import grasp_alignment as G
    from reachy_ai.motion.kinematics import joint_limits, _rotvec
    from reachy_ai.scene.awareness import SceneModel
    from reachy_ai.tasks import crane_pick_live as CP
    scene = SceneModel.from_yaml(SCENE)
    x, y, z = scene.cell_center('cell_' + cell)
    obj = G.object_box(scene.get('red_cube'), (x, y, z + 0.03), (1, 0, 0, 0))
    rot = CP.grasp_rotation(obj, replace(CP.CraneSpec(), face_axis=face, tilt_deg=tilt,
                                         lean_bearing_deg=lean))
    lims = np.array(joint_limits('right'))
    rng = np.random.default_rng(0)

    def res(q):
        h = G.hand_frames(list(q), -65.0)
        return np.concatenate([1000 * (obj.center - h.gap_mid),
                               math.degrees(1) * _rotvec(rot @ G.hand_rotation(list(q)).T)])
    best = None
    for _ in range(15):
        q0 = lims[:, 0] + rng.random(7) * (lims[:, 1] - lims[:, 0])
        r = least_squares(res, q0, bounds=(lims[:, 0], lims[:, 1]), x_scale=10.0, max_nfev=400)
        e = res(r.x)
        pe, ae = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
        if best is None or pe + ae < best[0] + best[1]:
            best = (pe, ae, [round(float(v), 1) for v in r.x])
        if pe < 0.5 and ae < 0.5:
            break
    return dict(cell=cell, face=face, tilt=tilt, lean=lean, pos_mm=round(best[0], 2),
                att_deg=round(best[1], 2), q=best[2])


if __name__ == '__main__':
    out = sys.argv[1]
    grid = [(c, f, t, l) for c in CELLS for f in ('x',) for t in TILTS for l in LEANS]
    done = set()
    if os.path.exists(out):
        done = {(r['cell'], r['face'], r['tilt'], r['lean']) for r in map(json.loads, open(out))}
    todo = [g for g in grid if g not in done]
    with mp.Pool(max(1, mp.cpu_count() - 2), maxtasksperchild=50) as pool, open(out, 'a') as fh:
        for i, r in enumerate(pool.imap_unordered(one, todo, chunksize=4)):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if i % 200 == 0:
                print(f'{i}/{len(todo)}', flush=True)
