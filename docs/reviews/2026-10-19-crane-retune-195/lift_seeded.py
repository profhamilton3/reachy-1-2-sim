"""The lift plan (`plan_crane_pick`, every check) for each attitude `reach_attitude.py` found
reachable, with its bounded-least-squares pose added to the IK start set (#195): separates
"the start set never finds it" from "a check refuses it".

PYTHONPATH=.:src:native_mujoco python docs/reviews/2026-10-19-crane-retune-195/lift_seeded.py REACH.jsonl OUT.jsonl
"""
import json, multiprocessing as mp, sys
from dataclasses import replace
import tools.place_sweep  # noqa: F401

SCENE = 'scenes/FWDCenterLabSivaPool.yaml'


def one(r):
    from host_ik_arm import HostIKArm
    from reachy_ai.motion.kinematics import CartesianPlanner
    from reachy_ai.scene.awareness import SceneModel
    from reachy_ai.tasks import crane_pick_live as CP
    scene = SceneModel.from_yaml(SCENE)
    x, y, z = scene.cell_center('cell_' + r['cell'])
    pos = (x, y, z + 0.03)
    scene.update_poses({'red_cube': pos})
    pl = CartesianPlanner(HostIKArm(), scene=scene)
    spec = replace(CP.CraneSpec(), face_axis=r['face'], tilt_deg=r['tilt'], lean_bearing_deg=r['lean'])
    out = dict(cell=r['cell'], tilt=r['tilt'], lean=r['lean'], seed=r['q'])
    try:
        p = CP.plan_crane_pick(pl, scene, 'red_cube', CP.Observation(pos, (1, 0, 0, 0), 0, False), spec,
                               starts=[r['q']] + list(CP.GRASP_STARTS))
        pr = p.checks['pad_route']
        out.update(ok=True, route_mm=min(pr['route_thumb_mm'], pr['route_finger_mm']),
                   palm_mm=pr['route_palm_mm'], end=pr['end'], q0=[round(v, 2) for v in p.rungs[0]],
                   margin=CP.joint_margins(p.rungs[0]),
                   ladder_min_margin=min(CP.joint_margins(q)['min_deg'] for q in p.rungs))
    except CP.CraneRefused as e:
        out.update(ok=False, stage=e.stage, reason=e.reason[:150])
    return out


if __name__ == '__main__':
    rows = [r for r in map(json.loads, open(sys.argv[1])) if r['pos_mm'] < 0.5 and r['att_deg'] < 1.0]
    with mp.Pool(4) as pool, open(sys.argv[2], 'w') as fh:
        for r in pool.imap_unordered(one, rows):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            print(r['cell'], r['tilt'], r['lean'], r['ok'], r.get('route_mm'), r.get('palm_mm'),
                  r.get('stage', ''), r.get('reason', '')[:80], flush=True)
