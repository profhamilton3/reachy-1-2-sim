"""Per-cell right-arm reach (the FWDCenterLabMCC.yaml table): MuJoCo, 40-restart DLS IK on r_tip, 5 cm above the
surface, orientation unconstrained, joints clamped to MJCF ranges.  Robot model
only (no scene), torso placed as given."""
import sys, math, pathlib
import numpy as np, mujoco
REPO = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
august = len(sys.argv) > 2 and sys.argv[2] == "august"   # the model before #192
torso = "0 0 1" if august else None
xml = (REPO / "native_mujoco/model/reachy_1_2.xml").read_text()
if august:   # torso at (0, 0, 1.0) and the URDF's +10 deg shoulder-roll stop
    xml = xml.replace('<body name="torso" pos="0.206 0 1.15">', '<body name="torso" pos="0 0 1">')
    xml = xml.replace('range="-3.14 0.354"', 'range="-3.14 0.174"')
xml = xml.replace('meshdir="meshes"', f'meshdir="{REPO}/native_mujoco/model/meshes"')
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "r_tip")
names = ["r_shoulder_pitch","r_shoulder_roll","r_arm_yaw","r_elbow_pitch","r_forearm_yaw","r_wrist_pitch","r_wrist_roll"]
jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
qadr = [m.jnt_qposadr[j] for j in jid]; dadr = [m.jnt_dofadr[j] for j in jid]
lo = np.array([m.jnt_range[j][0] for j in jid]); hi = np.array([m.jnt_range[j][1] for j in jid])
sh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_shoulder")
def solve(target, rng, restarts=40):
    best = (1e9, None)
    for _ in range(restarts):
        q = lo + rng.random(7) * (hi - lo)
        for it in range(300):
            d.qpos[:] = 0; d.qpos[qadr] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            e = target - d.site_xpos[site]
            if np.linalg.norm(e) < 1e-4: break
            jp = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, jp, None, site)
            J = jp[:, dadr]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), e)
            q = np.clip(q + dq, lo, hi)
        err = np.linalg.norm(e)
        if err < best[0]: best = (err, q.copy())
    return best
mujoco.mj_kinematics(m, d); shoulder = d.xpos[sh].copy()
print(f"torso at ({'0 0 1, roll stop +10' if august else 'as in the MJCF'}); right shoulder world {np.round(shoulder,3)}")
rng = np.random.default_rng(42)
rows = {1: 0.2794, 2: 0.4318, 3: 0.5842}; cols = {1: 0.1524, 2: 0.0, 3: -0.1524}
for r in (1, 2, 3):
    out = []
    for c in (1, 2, 3):
        t = np.array([rows[r], cols[c], 0.7405 + 0.05])
        err, q = solve(t, rng)
        dist = np.linalg.norm(t - shoulder)
        stops = [names[i].replace("r_","") for i in range(7) if q is not None and (q[i]-lo[i] < 0.02 or hi[i]-q[i] < 0.02)]
        out.append(f"r{r}c{c} {dist:.3f}m {'reach' if err < 1e-3 else 'NO'} err {err*1000:5.1f}mm stops:{','.join(stops) or '-'}")
    print("   " + " | ".join(out))
