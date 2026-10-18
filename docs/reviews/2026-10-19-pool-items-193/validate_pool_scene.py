"""Focused checks for #193: schema, compile, settle of the parked slots, one placement each, render.

Run from the repo root:  PYTHONPATH=.:src:native_mujoco python docs/reviews/2026-10-19-pool-items-193/validate_pool_scene.py
"""
import json, os, sys
import mujoco, numpy as np
from scene_io import load_scene
from scene_loader import load_scene as validate
from placement import ObjectPlacer, pool_ids
from objects import build_scene_model_xml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..', '..')
SCENE = os.path.join(ROOT, 'scenes', 'FWDCenterLabSivaPool.yaml')
NEW = ['pool_box_4', 'pool_box_5', 'pool_cyl_4', 'pool_cyl_5']

doc = load_scene(SCENE)
sc = validate(SCENE, document=doc)                     # validates against scenes/scene.schema.json
ids = [o['id'] for o in doc['objects']]
print('objects', len(ids), 'unique', len(set(ids)) == len(ids), 'placeable', len(pool_ids(doc)))
assert len(set(ids)) == len(ids) and len(pool_ids(doc)) == 14
byid = {o['id']: o for o in doc['objects']}
for i in ('red_cube', 'blue_cylinder', 'soda_can', 'foam_block'):
    print(' ', i, byid[i]['material']['rgba'])
for a, b in (('pool_box_1', 'pool_box_4'), ('pool_box_1', 'pool_box_5'), ('pool_cyl_1', 'pool_cyl_4'), ('pool_cyl_1', 'pool_cyl_5')):
    assert byid[a]['geometry'] == byid[b]['geometry'] and byid[a]['material'] == byid[b]['material']

model = mujoco.MjModel.from_xml_string(build_scene_model_xml(doc))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
for j in range(model.njnt):
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
        a = model.jnt_qposadr[j]; data.qpos[a:a + 7] = model.qpos0[a:a + 7]
mujoco.mj_forward(model, data)
bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
p0 = {n: data.xpos[bid(n)].copy() for n in NEW}
for _ in range(2000): mujoco.mj_step(model, data)
for n in NEW:
    d = np.linalg.norm(data.xpos[bid(n)] - p0[n]); v = np.linalg.norm(data.qvel[model.body_dofadr[bid(n)]:model.body_dofadr[bid(n)] + 6])
    print(f'  {n}: parked at {np.round(p0[n], 3)}  moved {d * 1000:.2f} mm in 2000 steps  speed {v:.1e}')
    assert d < 0.002 and v < 1e-3

placer = ObjectPlacer(model, data, doc)
for n, cell in zip(NEW, ('r1c1', 'r2c2', 'r1c2', 'r2c1')):
    placer.place(n, cell)
    for _ in range(600): mujoco.mj_step(model, data)
    c = placer.cells[cell]; x, y, z = data.xpos[bid(n)]
    print(f'  place {n} -> {cell}: ({x:.3f}, {y:.3f}, {z:.3f})  centre ({c.x:.3f}, {c.y:.3f})')
    assert abs(x - c.x) < c.half_extent and abs(y - c.y) < c.half_extent

# render: one overview of the parked pool and the board with two of the new pieces placed
for n in NEW[2:]: placer.place(n, 'r3c3' if n.endswith('4') else 'r3c1')
for _ in range(600): mujoco.mj_step(model, data)
import cv2
r = mujoco.Renderer(model, 720, 1280); cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:] = (0.45, 0.0, 0.5); cam.distance = 2.6; cam.azimuth = 150; cam.elevation = -28
r.update_scene(data, cam); img = r.render(); r.close()
cv2.imwrite(os.path.join(HERE, 'default_scene_pool.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print('ok')
