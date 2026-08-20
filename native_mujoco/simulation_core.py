"""R12-802: Shared simulation core for the WebSocket server and episode runner.

SimulationCore encapsulates model/scene loading, the actuator controller,
gripper model, object tracker, interactive controller, deterministic reset,
and per-step advancement.  Both the live server and the episode runner import
from here so they share identical physics behaviour.

Loading
-------
Use SimulationCore.from_paths() to build a core from file paths (the normal
case).  Pass a pre-built MjModel for testing.

Thread safety
-------------
SimulationCore is not thread-safe.  The WebSocket server owns one instance on
the sim thread; the episode runner owns one on the runner thread.  No locking
is provided or needed.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import mujoco
import numpy as np

from actuator import ActuatorController
from gripper import GripperModel
from joint_map import JOINT_TABLE, NUM_JOINTS
from objects import ObjectTracker, build_scene_model_xml
from evaluation_snapshot import ContactRecord, EvaluationSnapshot

log = logging.getLogger("reachy12.simulation_core")

# Contype bitmasks (R12-500 collision model)
_ROBOT_CONTYPE   = 2
_FIXTURE_CONTYPE = 8


def load_world(
    model_path: str,
    scene_path: Optional[str] = None,
) -> Tuple[mujoco.MjModel, List[str], List[Dict[str, Any]], Optional[str]]:
    """Load robot model + optional scene YAML into a compiled MjModel.

    Returns:
        model           — compiled MjModel ready for MjData
        tracked_ids     — list of object IDs that have dynamic free-joints
        interactive_specs — list of interactive control spec dicts
        compiled_xml    — the MJCF string used for hashing (None if model-only)
    """
    import yaml
    from scene_compiler import (
        interactive_specs as _interactive_specs,
        tracked_object_ids,
    )

    if scene_path:
        log.info("Loading scene %s into model %s", scene_path, model_path)
        try:
            from scene_loader import load_scene
            load_scene(scene_path)
        except ImportError:
            log.debug("scene_loader unavailable; skipping validation")
        with open(scene_path) as f:
            scene_doc = yaml.safe_load(f)
        compiled_xml = build_scene_model_xml(scene_doc, model_path)
        model = mujoco.MjModel.from_xml_string(compiled_xml)
        tracked = tracked_object_ids(scene_doc)
        specs = _interactive_specs(scene_doc)
        log.info("Scene: %d tracked objects, %d interactive controls",
                 len(tracked), len(specs))
        return model, tracked, specs, compiled_xml
    else:
        log.info("Loading model %s (no scene)", model_path)
        model = mujoco.MjModel.from_xml_path(model_path)
        return model, [], [], None


class SimulationCore:
    """Physics engine core shared by the server and episode runner.

    The public interface intentionally mirrors the portions of server.SimState
    that the episode runner needs, so the two code paths are parallel:

        control_step()   — apply actuator/compliance + interactive models
        advance()        — control_step + mj_step + step counter increment
        reset()          — deterministic reset to home keyframe
        set_targets()    — set joint goal positions
        snapshot()       — return a typed EvaluationSnapshot
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        tracked_ids: Optional[Sequence[str]] = None,
        interactive_specs: Optional[Sequence[Mapping[str, Any]]] = None,
        scene_revision: str = "initial",
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.step = 0
        self.scene_revision = scene_revision

        self._reset_physics()
        self.controller = ActuatorController(model)
        self.controller.sync_targets_to_current(self.data)
        self.gripper = GripperModel(model)
        self.objects = ObjectTracker(
            model, self.data, tracked_ids=list(tracked_ids or [])
        )
        self.objects.capture_initial(self.data)
        from interactive import InteractiveController
        self.interactive = InteractiveController(
            model, self.data, list(interactive_specs or [])
        )

    @classmethod
    def from_paths(
        cls,
        model_path: str,
        scene_path: Optional[str] = None,
    ) -> "SimulationCore":
        """Load model and optional scene from file paths."""
        model, tracked_ids, specs, _ = load_world(model_path, scene_path)
        return cls(model, tracked_ids=tracked_ids, interactive_specs=specs)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_physics(self) -> None:
        """Reset to the home keyframe, preserving free-joint scene poses."""
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        # Restore scene-object free-joint positions from model defaults.
        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                adr = self.model.jnt_qposadr[jid]
                self.data.qpos[adr:adr + 7] = self.model.qpos0[adr:adr + 7]
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed: Optional[int] = None, jitter_m: float = 0.0) -> None:
        """Deterministic reset: home keyframe, seeded object placement."""
        self._reset_physics()
        self.step = 0
        self.controller.sync_targets_to_current(self.data)
        self.objects.reset(self.data, seed=seed, jitter_m=jitter_m)
        self.interactive.reset(self.data)
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def set_targets(self, targets_rad: Sequence[float]) -> None:
        """Set joint goal positions for all joints that have a target in the list."""
        for entry in JOINT_TABLE:
            idx = entry.mjcf_index
            if idx < len(targets_rad):
                self.controller.set_goal_position(idx, targets_rad[idx])

    def control_step(self) -> None:
        """Apply actuator compliance model and interactive toggle logic."""
        self.controller.apply(self.data, self.model.opt.timestep)
        self.interactive.update(self.data)

    def advance(self) -> None:
        """One complete physics advance: control_step + mj_step + counter."""
        self.control_step()
        mujoco.mj_step(self.model, self.data)
        self.step += 1

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def snapshot_joints(self) -> List[Dict[str, Any]]:
        joints = []
        for entry in JOINT_TABLE:
            i = entry.mjcf_index
            st = self.controller.state[i]
            joints.append({
                "name": entry.sdk_name,
                "uid": entry.uid,
                "position_rad": float(self.data.qpos[i]),
                "velocity_rad_s": float(self.data.qvel[i]),
                "effort": float(self.data.actuator_force[i]),
                "compliant": bool(st.compliant),
                "saturated": self.controller.is_saturated(self.data, i),
            })
        return joints

    def snapshot_contacts(self) -> List[ContactRecord]:
        """All active contacts this step as typed ContactRecord objects."""
        records = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1 = int(self.model.geom_bodyid[g1])
            b2 = int(self.model.geom_bodyid[g2])

            def _gname(gid: int) -> str:
                n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                return n or ""

            def _bname(bid: int) -> str:
                n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
                return n or ""

            # MuJoCo stores 6-DOF contact frame; normal force = contactforce[0].
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            records.append(ContactRecord(
                geom1=_gname(g1),
                geom2=_gname(g2),
                body1=_bname(b1),
                body2=_bname(b2),
                contype1=int(self.model.geom_contype[g1]),
                contype2=int(self.model.geom_contype[g2]),
                conaffinity1=int(self.model.geom_conaffinity[g1]),
                conaffinity2=int(self.model.geom_conaffinity[g2]),
                pos=(float(c.pos[0]), float(c.pos[1]), float(c.pos[2])),
                normal_force=float(force[0]),
                dist=float(c.dist),
            ))
        return records

    def snapshot_grippers(self) -> Tuple[List[Dict], List[Dict]]:
        states = self.gripper.update(self.data)
        grippers, sensors = [], []
        for side, st in states.items():
            grippers.append({
                "side": side,
                "grasping": st.grasping,
                "grip_force_n": st.grip_force_n,
                "grasped_geoms": st.grasped_geoms,
            })
            sensors.append({
                "uid": st.sensor_uid,
                "force": st.grip_force_n,
            })
        return grippers, sensors

    def snapshot(self) -> EvaluationSnapshot:
        """Build a complete EvaluationSnapshot for the current physics state."""
        joints = self.snapshot_joints()
        contacts = self.snapshot_contacts()
        grippers, force_sensors = self.snapshot_grippers()

        # Count forbidden robot↔fixture contacts
        forbidden = sum(
            1 for c in contacts
            if (c.contype1 == _ROBOT_CONTYPE and c.contype2 == _FIXTURE_CONTYPE) or
               (c.contype1 == _FIXTURE_CONTYPE and c.contype2 == _ROBOT_CONTYPE)
        )

        # Joint limit violations (joints at or beyond their hard limits)
        violations = []
        saturated = []
        for j in joints:
            if j["saturated"]:
                saturated.append(j["name"])
        for entry in JOINT_TABLE:
            i = entry.mjcf_index
            lo = self.model.jnt_range[i][0]
            hi = self.model.jnt_range[i][1]
            if self.model.jnt_limited[i]:
                q = float(self.data.qpos[i])
                if q <= lo or q >= hi:
                    violations.append(entry.sdk_name)

        return EvaluationSnapshot(
            sim_step=self.step,
            sim_time_s=float(self.data.time),
            scene_revision=self.scene_revision,
            joints=joints,
            contacts=contacts,
            interactive=self.interactive.states(self.data),
            objects=self.objects.poses_as_dicts(self.data),
            grippers=grippers,
            force_sensors=force_sensors,
            forbidden_contact_count=forbidden,
            joint_limit_violations=violations,
            saturated_joints=saturated,
        )
