"""Stage 1 of the #195 attitude search: the LIFT (`plan_crane_pick`, every crane check) for one
object on every reachable cell, over a (face, tilt, lean) grid.  No motion; host IK mirror.

Run: PYTHONPATH=.:src:native_mujoco python docs/reviews/2026-10-19-crane-retune-195/lift_grid.py OUT.json [--starts default|wide]
"""
import json, multiprocessing as mp, sys, time
from dataclasses import replace
import numpy as np
import tools.place_sweep  # noqa: F401  (paths)

SCENE = 'scenes/FWDCenterLabSivaPool.yaml'
CELLS = [f'r{r}c{c}' for r in (1, 2, 3) for c in (1, 2, 3)]
TILTS = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
LEANS = [-180.0 + 15.0 * i for i in range(24)]


def wide_starts():
    from reachy_ai.tasks import crane_pick_live as CP
    out = list(CP.GRASP_STARTS)
    for sp in (-130.0, -100.0, -70.0, -40.0, -10.0):
        for sr in (-40.0, -10.0, 15.0):
            for ay in (-60.0, 0.0, 45.0, 90.0):
                for el in (-130.0, -100.0, -60.0, -20.0):
                    for fy in (-60.0, 0.0, 60.0):
                        out.append([sp, sr, ay, el, fy, 0.0, 0.0])
    return out


def one(args):
    oid, cell, face, tilt, lean, starts = args
    from host_ik_arm import HostIKArm
    from reachy_ai.motion.kinematics import CartesianPlanner
    from reachy_ai.scene.awareness import SceneModel
    from reachy_ai.tasks import crane_pick_live as CP
    scene = SceneModel.from_yaml(SCENE)
    x, y, z = scene.cell_center('cell_' + cell)
    pos = (x, y, z + scene.get(oid).size[2] / 2)
    scene.update_poses({oid: pos})
    pl = CartesianPlanner(HostIKArm(), scene=scene)
    spec = replace(CP.CraneSpec(), face_axis=face, tilt_deg=tilt, lean_bearing_deg=lean)
    row = dict(cell=cell, face=face, tilt=tilt, lean=lean)
    t0 = time.monotonic()
    try:
        p = CP.plan_crane_pick(pl, scene, oid, CP.Observation(pos, (1, 0, 0, 0), 0, False), spec,
                               starts=(wide_starts() if starts == 'wide' else None))
        pr = p.checks['pad_route']
        row.update(ok=True, route_mm=round(min(pr['route_thumb_mm'], pr['route_finger_mm']), 2),
                   insertion=p.insertion_rung, q0=[round(v, 1) for v in p.rungs[0]],
                   margin=CP.joint_margins(p.rungs[0]),
                   ladder_margin=min(CP.joint_margins(q)['min_deg'] for q in p.rungs))
    except CP.CraneRefused as e:
        row.update(ok=False, stage=e.stage, reason=e.reason[:120])
    except Exception as e:
        row.update(ok=False, stage=type(e).__name__, reason=str(e)[:120])
    row['s'] = round(time.monotonic() - t0, 2)
    return row


if __name__ == '__main__':
    import os
    out = sys.argv[1]                          # JSON lines; an interrupted run resumes
    starts = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == '--starts' else 'default'
    oid = 'red_cube'
    grid = [(oid, c, f, t, l, starts) for c in CELLS for f in ('x', 'y') for t in TILTS for l in LEANS]
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r['cell'], r['face'], r['tilt'], r['lean']))
    todo = [g for g in grid if (g[1], g[2], g[3], g[4]) not in done]
    with mp.Pool(max(1, mp.cpu_count() - 2), maxtasksperchild=50) as pool, open(out, 'a') as fh:
        for i, r in enumerate(pool.imap_unordered(one, todo, chunksize=4)):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if i % 200 == 0:
                print(f'{i}/{len(todo)}', flush=True)
    rows = [json.loads(line) for line in open(out)]
    print('ok', sum(r['ok'] for r in rows), 'of', len(rows))
    for c in CELLS:
        print(c, sum(r['ok'] for r in rows if r['cell'] == c))
