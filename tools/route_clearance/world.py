"""The October model for a clearance run: the MJCF compiled with a scene, and the distance reads Step 1 used
(reachy-tabletop-ai docs/design/arm-motion-step1/, replay.py and board_check.py).

Board distance is analytic: the lowest box corner or capsule end of each right-arm collision geom, over the board's
top face. `mj_geomDistance` is not used for the board, because it returns a spurious 0 for the hand's boxes against
the slab (board_check.txt). Rig, left-arm, self and piece distances are exact for box and capsule pairs
(geometry.py); Step 1's guard on `mj_geomDistance` for box pairs is not enough (geometry.py says why). Cylinder pairs
use `mj_geomDistance`. Every read is capped at CAP_M: "+9.0" means at least 9 cm.
"""
import math

import mujoco
import numpy as np

from . import _REPO, geometry
from .route import ARM7, GRIPPER

MODEL_PATH = _REPO / "native_mujoco" / "model" / "reachy_1_2.xml"
JOINTS = ARM7 + (GRIPPER,)
NECK = ("neck_roll", "neck_pitch", "neck_yaw")
ARM_GEOMS = ("r_upper_arm_col", "r_forearm_col", "r_thumb_col", "r_hand_back_col", "r_palm_col", "r_finger_col")
SELF_GEOMS = ("torso_col", "head_col")
LEFT_GEOMS = ("l_upper_arm_col", "l_forearm_col", "l_thumb_col", "l_finger_col")
BOARD_GEOM = "table_top"
CAP_M = 0.09
#: The tracked lab pieces a clearance run places on cells: (body id, kind). Scene FWDCenterLabSivaPool.
PIECES = {"cube": ("pool_box_1", "pool_box_2", "pool_box_3", "pool_box_4", "pool_box_5"),
          "cylinder": ("pool_cyl_1", "pool_cyl_2", "pool_cyl_3", "pool_cyl_4", "pool_cyl_5")}


class World:
    def __init__(self, scene_path):
        from simulation_core import SimulationCore

        self.scene_path = scene_path
        self.core = SimulationCore.from_paths(str(MODEL_PATH), str(scene_path))
        m, d = self.model, self.data = self.core.model, self.core.data
        self.jid = {n: self._id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS + NECK}
        self.adr = {n: int(m.jnt_qposadr[j]) for n, j in self.jid.items()}
        self.arm = [self._id(mujoco.mjtObj.mjOBJ_GEOM, n) for n in ARM_GEOMS]
        self.self_ = [self._id(mujoco.mjtObj.mjOBJ_GEOM, n) for n in SELF_GEOMS]
        self.left = [self._id(mujoco.mjtObj.mjOBJ_GEOM, n) for n in LEFT_GEOMS]
        boards = [g for g in range(m.ngeom) if self.gname(g) == BOARD_GEOM]   # an unnamed geom of body table_top
        if len(boards) != 1:
            raise KeyError("model must have one %s geom, has %d" % (BOARD_GEOM, len(boards)))
        self.board = boards[0]
        self.rig = [g for g in range(m.ngeom) if self.gname(g).startswith("rig_rail")]
        self.tip = self._id(mujoco.mjtObj.mjOBJ_SITE, "r_tip")
        self.cams = {side: self._id(mujoco.mjtObj.mjOBJ_CAMERA, side + "_camera") for side in ("left", "right")}
        mujoco.mj_forward(m, d)
        self.board_top = float(d.geom_xpos[self.board][2] + m.geom_size[self.board][2])
        self.cells = {}
        for g in range(m.ngeom):
            name = self.gname(g)
            if name.startswith("cell_r"):
                self.cells[name[len("cell_"):]] = (float(d.geom_xpos[g][0]), float(d.geom_xpos[g][1]),
                                                    float(m.geom_size[g][0]))
        self.home_qpos = d.qpos.copy()

    def _id(self, kind, name):
        i = mujoco.mj_name2id(self.model, kind, name)
        if i < 0:
            raise KeyError("model has no %s %r" % (kind, name))
        return i

    def gname(self, g):
        m = self.model
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or mujoco.mj_id2name(
            m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or str(g)

    # ---- kinematic state -------------------------------------------------------------------------------------
    def set_arm(self, q_deg, neck_deg=None):
        """Right arm (and optionally the neck) to these angles, everything else at the keyframe; mj_forward."""
        d = self.data
        d.qpos[:] = self.home_qpos
        for n in JOINTS:
            d.qpos[self.adr[n]] = math.radians(q_deg[n])
        for n, v in (neck_deg or {}).items():
            d.qpos[self.adr[n]] = math.radians(v)
        d.qvel[:] = 0
        mujoco.mj_forward(self.model, d)

    # ---- distances -------------------------------------------------------------------------------------------
    def _shape(self, g):
        m, d = self.model, self.data
        kind = m.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            return "box", geometry.Box(d.geom_xpos[g], d.geom_xmat[g], m.geom_size[g])
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            axis = d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g][1]
            return "capsule", (d.geom_xpos[g] - axis, d.geom_xpos[g] + axis, float(m.geom_size[g][0]))
        return None, None

    def distance(self, g1, g2):
        """Signed distance in metres, capped at CAP_M. Box-box and capsule-box pairs are exact (geometry.py, not
        mj_geomDistance's box-box artefact); any other pair (the cylinder piece) is mj_geomDistance."""
        k1, s1 = self._shape(g1)
        k2, s2 = self._shape(g2)
        if k1 == "box" and k2 == "box":
            v = geometry.box_box(s1, s2)
        elif {k1, k2} == {"box", "capsule"}:
            (p, q, r), box = (s1, s2) if k1 == "capsule" else (s2, s1)
            v = geometry.capsule_box(p, q, r, box)
        else:
            v = mujoco.mj_geomDistance(self.model, self.data, g1, g2, 0.3, None)
        return min(float(v), CAP_M)

    def board_heights(self):
        """{geom: height of its lowest point over the board's top face, metres}, analytic (board_check.py)."""
        m, d = self.model, self.data
        out = {}
        for g in self.arm:
            pos, rot, size = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                z = min((pos + rot @ (np.array([sx, sy, sz]) * size))[2]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1))
            else:  # capsule: half-length along local z, radius size[0]
                z = min((pos + rot @ np.array([0.0, 0.0, h * size[1]]))[2] for h in (-1, 1)) - size[0]
            out[self.gname(g)] = float(z - self.board_top)
        return out

    def board_clearance(self):
        """(metres, geom) of the lowest right-arm point over the board, capped at CAP_M."""
        heights = self.board_heights()
        geom = min(heights, key=heights.get)
        return min(heights[geom], CAP_M), geom

    def group_clearance(self, others, arm=None):
        best = (CAP_M, None)
        for a in (self.arm if arm is None else arm):
            for b in others:
                v = self.distance(a, b)
                if v < best[0]:
                    best = (v, self.gname(a))
        return best

    def clearances(self):
        """{group: (metres, arm geom)} for board (analytic), rig, left arm and self (forearm and hand only)."""
        return {"board": self.board_clearance(),
                "rig": self.group_clearance(self.rig),
                "left_arm": self.group_clearance(self.left),
                "self": self.group_clearance(self.self_, arm=self.arm[1:])}

    # ---- pieces ----------------------------------------------------------------------------------------------
    def body_geoms(self, body_name):
        b = self._id(mujoco.mjtObj.mjOBJ_BODY, body_name)
        return [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == b]

    def body_qadr(self, body_name):
        b = self._id(mujoco.mjtObj.mjOBJ_BODY, body_name)
        return int(self.model.jnt_qposadr[self.model.body_jntadr[b]])

    def piece_half_height(self, body_name):
        g = self.body_geoms(body_name)[0]
        m = self.model
        return float(m.geom_size[g][1] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER else m.geom_size[g][2])

    def piece_half_footprint(self, body_name):
        """Half of the piece's extent along the board's x and y, axis-aligned (radius for a cylinder)."""
        g = self.body_geoms(body_name)[0]
        return float(self.model.geom_size[g][0])

    def cell_point(self, cell, offset_xy=(0.0, 0.0)):
        x, y, _half = self.cells[cell]
        return x + offset_xy[0], y + offset_xy[1]

    def piece_distance_at(self, body_name, xy):
        """Distance from the right arm (as it stands now) to the piece standing upright at xy on the board, without
        moving the model's state: the piece's geom pose is written for the read only."""
        d = self.data
        g = self.body_geoms(body_name)[0]
        saved_pos, saved_mat = d.geom_xpos[g].copy(), d.geom_xmat[g].copy()
        d.geom_xpos[g] = (xy[0], xy[1], self.board_top + self.piece_half_height(body_name))
        d.geom_xmat[g] = np.eye(3).ravel()
        try:
            return self.group_clearance([g])
        finally:
            d.geom_xpos[g], d.geom_xmat[g] = saved_pos, saved_mat

    def arm_points(self):
        """World positions of the right-arm geoms with a bounding radius each (a cheap prefilter for pieces)."""
        m, d = self.model, self.data
        return [(d.geom_xpos[g].copy(), float(m.geom_rbound[g])) for g in self.arm]

    # ---- frames ----------------------------------------------------------------------------------------------
    def tip_world(self):
        return self.data.site_xpos[self.tip].copy()
