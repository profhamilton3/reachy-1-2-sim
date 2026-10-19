"""Stage 3 of the #195 search: around the band stages 1-2 found (r3c2, lean ~90, tilt 20-25),
on a finer grid, the lift from r3c2 and the placement from r3c2 to every other cell, with the
crane's own IK start set.  Records the joint margins of every plan.

PYTHONPATH=.:src:native_mujoco python docs/reviews/2026-10-19-crane-retune-195/fine_grid.py OUT.jsonl
"""
import json, multiprocessing as mp, sys
from dataclasses import replace
import tools.place_sweep  # noqa: F401

SCENE = 'scenes/FWDCenterLabSivaPool.yaml'
CELLS = [f'r{r}c{c}' for r in (1, 2, 3) for c in (1, 2, 3)]
TILTS = [20.0, 21.25, 22.5, 23.75, 25.0, 26.25]
LEANS = [80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0]


def one(a):
    origin, dest, tilt, lean = a
    from host_ik_arm import HostIKArm
    from reachy_ai.motion.kinematics import CartesianPlanner
    from reachy_ai.scene.awareness import SceneModel
    from reachy_ai.tasks import crane_pick_live as CP
    from reachy_ai.tasks import crane_place_live as PL
    scene = SceneModel.from_yaml(SCENE)
    x, y, z = scene.cell_center('cell_' + origin)
    pos = (x, y, z + 0.03)
    scene.update_poses({'red_cube': pos})
    pl = CartesianPlanner(HostIKArm(), scene=scene)
    spec = replace(CP.CraneSpec(), tilt_deg=tilt, lean_bearing_deg=lean)
    seen = CP.Observation(pos, (1, 0, 0, 0), 0, False)
    row = dict(origin=origin, dest=dest, tilt=tilt, lean=lean)
    try:
        if dest is None:
            p = CP.plan_crane_pick(pl, scene, 'red_cube', seen, spec)
            rungs, pr = p.rungs, p.checks['pad_route']
        else:
            pp = PL.plan_crane_place(pl, scene, 'red_cube', dest, seen, spec=spec)
            rungs, pr = pp.pick.rungs + pp.dest.rungs, pp.pick.checks['pad_route']
            row.update(carry=pp.carry_rung, psi=round(pp.psi_deg, 1),
                       dest_margin=CP.joint_margins(pp.dest.rungs[0]),
                       out_route_mm=pp.checks['pad_route_out']['route_thumb_mm'],
                       out_route_finger_mm=pp.checks['pad_route_out']['route_finger_mm'],
                       out_route_palm_mm=pp.checks['pad_route_out']['route_palm_mm'])
        row.update(ok=True, route_mm=min(pr['route_thumb_mm'], pr['route_finger_mm']),
                   palm_mm=pr['route_palm_mm'], pick_margin=CP.joint_margins(rungs[0]),
                   finger_below_top_mm=pr['end']['finger_center_below_top_mm'],
                   thumb_below_top_mm=pr['end']['thumb_center_below_top_mm'])
    except CP.CraneRefused as e:
        row.update(ok=False, stage=e.stage, reason=e.reason[:150])
    return row


if __name__ == '__main__':
    jobs = [('r3c2', d, t, l) for d in [None] + [c for c in CELLS if c != 'r3c2']
            for t in TILTS for l in LEANS]
    with mp.Pool(6) as pool, open(sys.argv[1], 'w') as fh:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=2)):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if i % 50 == 0:
                print(f'{i}/{len(jobs)}', flush=True)
