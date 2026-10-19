"""A batch of offline crane flights (`fly_probe`), in parallel, one JSON line per flight.

Jobs file: a JSON list of objects with the `fly_probe` options as keys (origin, dest, obj,
face, tilt, lean, place_face, place_tilt, place_lean, cap_lat, cap_vert, dx, dy, yaw, tag).

PYTHONPATH=.:src:native_mujoco python docs/reviews/2026-10-19-crane-retune-195/fly_batch.py JOBS.json OUT.jsonl [--procs 4]
"""
import json, multiprocessing as mp, sys, time, traceback


def run(job):
    import fly_probe as FP
    t0 = time.monotonic()
    try:
        out = FP.run_job(job)
    except Exception as e:                      # a flight that breaks is reported, not dropped
        out = {'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-800:]}
    out['job'] = job
    out['wall_s'] = round(time.monotonic() - t0, 1)
    return out


if __name__ == '__main__':
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    jobs = json.load(open(sys.argv[1]))
    procs = int(sys.argv[sys.argv.index('--procs') + 1]) if '--procs' in sys.argv else 4
    with mp.Pool(procs) as pool, open(sys.argv[2], 'w') as fh:
        for i, r in enumerate(pool.imap_unordered(run, jobs)):
            fh.write(json.dumps(r, default=str) + '\n'); fh.flush()
            j = r['job']
            print(f"{i + 1}/{len(jobs)} {j.get('tag', '')} {j.get('origin')}->{j.get('dest')}: "
                  f"{r.get('refused') or r.get('error') or r.get('halt') or r.get('criteria')}", flush=True)
