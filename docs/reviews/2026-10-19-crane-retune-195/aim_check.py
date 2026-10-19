"""The set-down aim at the October mount (#195, scope 3): the destination's joint margins
(least margin and its joint, at the set-down pose) for each aim, at the chosen attitude.

PYTHONPATH=.:src:native_mujoco python docs/reviews/2026-10-19-crane-retune-195/aim_check.py
"""
from dataclasses import replace
import tools.place_sweep  # noqa: F401
from host_ik_arm import HostIKArm
from reachy_ai.motion.kinematics import CartesianPlanner, _BASE
from reachy_ai.scene.awareness import SceneModel
from reachy_ai.tasks import crane_pick_live as CP
from reachy_ai.tasks import crane_place_live as PL

SCENE = 'scenes/FWDCenterLabSivaPool.yaml'
SPEC = PL.PLACE_GRASP
AIMS = [('centre', 0.0, (0.0, 0.0)), ('1 mm to (0,0)', 0.001, (0.0, 0.0)),
        ('1 mm to the base', 0.001, (float(_BASE[0]), float(_BASE[1]))),
        ('2 mm to the base', 0.002, (float(_BASE[0]), float(_BASE[1])))]

for dest in ('r2c1', 'r2c2', 'r3c3'):
    for name, m, xy in AIMS:
        scene = SceneModel.from_yaml(SCENE)
        x, y, z = scene.cell_center('cell_r3c2')
        pos = (x, y, z + 0.03)
        scene.update_poses({'red_cube': pos})
        pl = CartesianPlanner(HostIKArm(), scene=scene)
        try:
            pp = PL.plan_crane_place(pl, scene, 'red_cube', dest, CP.Observation(pos, (1, 0, 0, 0), 0, False),
                                     spec=SPEC, place=replace(PL.PlaceSpec(), aim_m=m, aim_toward_xy=xy))
            jm = CP.joint_margins(pp.dest.rungs[0])
            low = min(CP.joint_margins(q)['min_deg'] for q in pp.dest.rungs[:4])
            print(f'{dest} {name:18} ok  set-down margin {jm["min_deg"]:6.3f} ({jm["joint"]}), '
                  f'shoulder roll {jm["shoulder_roll_deg"]:6.2f}, lowest 4 rungs {low:6.3f}')
        except CP.CraneRefused as e:
            print(f'{dest} {name:18} REFUSED {e.stage}: {e.reason[:90]}')
