"""Stage 2 of the #195 attitude search: the PLACE attitude (`tools.place_sweep.attitude.one`,
the whole `plan_crane_place` preflight) from one origin to every other reachable cell, over
the tilts the palm leaves graspable (`palm_geometry.py`).  No motion; host IK mirror.

PYTHONPATH=.:src:native_mujoco python docs/reviews/2026-10-19-crane-retune-195/place_grid.py OUT.jsonl [--from r3c2] [--object red_cube]
"""
import json, multiprocessing as mp, os, sys
import tools.place_sweep  # noqa: F401
from tools.place_sweep.attitude import one

CELLS = [f'r{r}c{c}' for r in (1, 2, 3) for c in (1, 2, 3)]
TILTS = [10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0]
LEANS = [-180.0 + 15.0 * i for i in range(24)]


def job(a):
    r = one(a)
    r.update(origin=a[1], dest=a[2], object=a[0])
    return r


if __name__ == '__main__':
    out = sys.argv[1]
    origin = sys.argv[sys.argv.index('--from') + 1] if '--from' in sys.argv else 'r3c2'
    oid = sys.argv[sys.argv.index('--object') + 1] if '--object' in sys.argv else 'red_cube'
    grid = [(oid, origin, d, 'x', t, l) for d in CELLS if d != origin for t in TILTS for l in LEANS]
    done = set()
    if os.path.exists(out):
        done = {(r['object'], r['origin'], r['dest'], r['face'], r['tilt'], r['lean'])
                for r in map(json.loads, open(out))}
    todo = [g for g in grid if g not in done]
    with mp.Pool(max(1, mp.cpu_count() - 2), maxtasksperchild=50) as pool, open(out, 'a') as fh:
        for i, r in enumerate(pool.imap_unordered(job, todo, chunksize=2)):
            fh.write(json.dumps(r, default=str) + '\n'); fh.flush()
            if r.get('feasible'):
                print('FEASIBLE', r['dest'], r['tilt'], r['lean'], flush=True)
            if i % 100 == 0:
                print(f'{i}/{len(todo)}', flush=True)
