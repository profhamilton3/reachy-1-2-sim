"""Motors-off settle of the right arm from a start pose, in the SivaPool world."""
import sys, math, pathlib
REPO = pathlib.Path(sys.argv[1]) if __name__ == "__main__" else None
import numpy as np, mujoco
NAMES = ["r_shoulder_pitch","r_shoulder_roll","r_arm_yaw","r_elbow_pitch","r_forearm_yaw","r_wrist_pitch","r_wrist_roll","r_gripper"]
class World:
    def __init__(self, repo):
        sys.path[:0] = [str(repo), str(repo / "native_mujoco"), str(repo / "src")]
        from simulation_core import SimulationCore
        self.core = SimulationCore.from_paths(str(repo / "native_mujoco/model/reachy_1_2.xml"), str(repo / "scenes/FWDCenterLabSivaPool.yaml"))
        self.m, self.d = self.core.model, self.core.data
        self.adr = {n: self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in NAMES}
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "r_tip")
    def settle(self, q, grip=-45.0, seconds=4.0, stiff=False):
        core, m, d = self.core, self.m, self.d
        core.reset()
        for n, v in zip(NAMES, list(q) + [grip]): d.qpos[self.adr[n]] = math.radians(v)
        d.qvel[:] = 0; mujoco.mj_forward(m, d); core.controller.sync_targets_to_current(d)
        for n in NAMES: core.controller.set_compliant("act_" + n, not stiff)
        for _ in range(int(seconds / m.opt.timestep)): core.advance()
        out = [math.degrees(d.qpos[self.adr[n]]) for n in NAMES]
        cons = set()
        for c in d.contact[:d.ncon]:
            for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                if gn.startswith("r_"):
                    cons.add((gn, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[o])))
        speed = float(np.abs(d.qvel[[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in NAMES[:7]]]).max())
        return out, sorted(cons), d.site_xpos[self.site].copy(), speed
