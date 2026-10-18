"""Physics: motors-off settle at REST; then fly PRESENT->...->REST->...->PRESENT with objects on column 3."""
import sys, math, pathlib
import numpy as np, mujoco
S = pathlib.Path(__file__).parent; sys.path.insert(0, str(S))
# usage: PYTHONPATH=src python3 docs/reviews/2026-10-17-october-layout-192/fly_rest.py . [args]
from settle import World, NAMES
REPO = pathlib.Path(sys.argv[1]); w = World(REPO)
from reachy_ai.motion import rig_routes as R
m, d, core = w.m, w.d, w.core
REST, REST_OUT, PRESENT_SHUT = R.REST, R.REST_OUT, R.PRESENT_SHUT
qr = [REST[n] for n in NAMES]
q1, cons, tip, sp = w.settle(qr[:7], grip=R.SHUT, seconds=10.0)
print("motors-off 10 s from REST: drift", np.round(np.array(q1) - np.array(qr), 1).tolist(), "tip", np.round(tip, 3).tolist(), "speed", round(sp, 4))
q1, cons, tip, sp = w.settle(qr[:7], grip=R.SHUT, seconds=5.0, stiff=True)
print("motors-on 5 s at REST: error", np.round(np.array(q1) - np.array(qr), 1).tolist(), "contacts", cons)

def body_free_adr(name):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    j = m.body_jntadr[b]; return m.jnt_qposadr[j]
sys.path.insert(0, str(REPO / "src"))
from reachy_ai.scene.awareness import SceneModel
sm = SceneModel.from_yaml(str(REPO / "scenes/FWDCenterLabSivaPool.yaml"))
def place(layout):
    for oid, cell in layout.items():
        x, y, z = sm.cell_center(cell); a = body_free_adr(oid)
        d.qpos[a:a+3] = [x, y, z + sm.get(oid).size[2] / 2 + 0.001]; d.qpos[a+3:a+7] = [1, 0, 0, 0]
def fly(route, layout, seg_s=3.0):
    core.reset(); place(layout)
    for n, v in zip(NAMES, [route[0][n] for n in NAMES]): d.qpos[w.adr[n]] = math.radians(v)
    d.qvel[:] = 0; mujoco.mj_forward(m, d); core.controller.sync_targets_to_current(d)
    for n in NAMES: core.controller.set_compliant("act_" + n, False)
    for _ in range(500): core.advance()
    start = {oid: d.qpos[body_free_adr(oid):body_free_adr(oid)+3].copy() for oid in layout}
    robot = [g for g in range(m.ngeom) if m.geom_contype[g] == 2]
    objs = {oid: [g for g in range(m.ngeom) if m.geom_bodyid[g] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, oid)] for oid in layout}
    worst = {oid: 9.0 for oid in layout}; touch = set()
    for a, b in zip(route, route[1:]):
        steps = int(seg_s / m.opt.timestep)
        for k in range(steps):
            t = (k + 1) / steps
            tg = [0.0] * 21
            for e in __import__("joint_map").JOINT_TABLE:
                tg[e.mjcf_index] = core.controller.state[e.mjcf_index].goal_position if hasattr(core.controller.state[e.mjcf_index], "goal_position") else 0.0
            for n in NAMES:
                core.controller.set_goal_position("act_" + n, math.radians(a[n] + t * (b[n] - a[n])))
            core.advance()
            if k % 25 == 0:
                for oid, gs in objs.items():
                    for g in gs:
                        for r in robot:
                            # box-box: a spurious 0 at large distmax; exact under 3 cm
                            cap = 0.03 if m.geom_type[r] == m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else 0.2
                            dd = mujoco.mj_geomDistance(m, d, r, g, cap, np.zeros(6))
                            worst[oid] = min(worst[oid], dd)
                for c in d.contact[:d.ncon]:
                    n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""; n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
                    b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1]); b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2])
                    if n1.startswith("r_") or n2.startswith("r_"): touch.add(tuple(sorted((n1 or b1, n2 or b2))))
    for _ in range(1000): core.advance()
    moved = {oid: round(float(np.linalg.norm(d.qpos[body_free_adr(oid):body_free_adr(oid)+3] - start[oid])) * 1000, 2) for oid in layout}
    final = {n: round(math.degrees(d.qpos[w.adr[n]]) - route[-1][n], 1) for n in NAMES[:4]}
    print("  worst MJCF distance (cm):", {k: round(v * 100, 1) for k, v in worst.items()}, "moved mm:", moved, "final gross err:", final)
    print("  robot contacts:", sorted(touch))
LOWER = [R.PRESENT, PRESENT_SHUT, REST_OUT, REST]
LIFT = [REST, REST_OUT, PRESENT_SHUT, R.PRESENT]
for layout in ({"soda_can": "cell_r1c3", "blue_cylinder": "cell_r2c3", "red_cube": "cell_r3c3"},
               {"blue_cylinder": "cell_r1c3", "soda_can": "cell_r2c3", "foam_block": "cell_r3c3"}):
    print("layout", layout)
    print(" LOWER_TO_REST"); fly(LOWER, layout)
    print(" LIFT_TO_PRESENT"); fly(LIFT, layout)
