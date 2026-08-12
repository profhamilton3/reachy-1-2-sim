"""
R12-503: Dynamic object tracking for the native MuJoCo backend.

* Discovers dynamic scene objects (bodies with a free joint) in a compiled
  model and streams their pose (position xyz + quaternion wxyz).
* Supports deterministic reset to captured initial poses, with optional
  seeded jitter for domain randomisation.
* build_scene_model() combines the robot MJCF with compiled scene objects so
  the physics backend and pose stream stay coherent with the scene document.

Only free-joint bodies whose id is in the tracked set are streamed; poses come
from the same MjData as joints and camera frames, so MuJoCo/RViz/camera views
share one coherent sim step.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import mujoco
import numpy as np

from scene_compiler import (
    compile_scene_body_fragment,
    dynamic_object_ids,
    tracked_object_ids,
)

_MODEL_DIR = pathlib.Path(__file__).parent / "model"
_ROBOT_MODEL = _MODEL_DIR / "reachy_1_2.xml"


@dataclass
class ObjectPose:
    object_id: str
    pos_xyz: tuple
    quat_wxyz: tuple


@dataclass
class TrackedObject:
    object_id: str
    body_id: int
    qpos_adr: int          # freejoint qpos address (7 values: xyz + wxyz)
    qvel_adr: int          # freejoint qvel address (6 values)
    initial_qpos: np.ndarray = field(default_factory=lambda: np.zeros(7))


class ObjectTracker:
    """Tracks free-joint scene objects in a compiled model."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tracked_ids: Optional[Sequence[str]] = None,
    ) -> None:
        self.model = model
        self._objects: List[TrackedObject] = []

        want = set(tracked_ids) if tracked_ids is not None else None
        for jid in range(model.njnt):
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            body_id = int(model.jnt_bodyid[jid])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if want is not None and name not in want:
                continue
            qadr = int(model.jnt_qposadr[jid])
            vadr = int(model.jnt_dofadr[jid])
            self._objects.append(TrackedObject(
                object_id=name,
                body_id=body_id,
                qpos_adr=qadr,
                qvel_adr=vadr,
                initial_qpos=data.qpos[qadr:qadr + 7].copy(),
            ))

    @property
    def object_ids(self) -> List[str]:
        return [o.object_id for o in self._objects]

    def capture_initial(self, data: mujoco.MjData) -> None:
        """Snapshot current object poses as the reset baseline."""
        for o in self._objects:
            o.initial_qpos = data.qpos[o.qpos_adr:o.qpos_adr + 7].copy()

    def poses(self, data: mujoco.MjData) -> List[ObjectPose]:
        """Current pose of each tracked object (coherent with `data`)."""
        out = []
        for o in self._objects:
            q = data.qpos[o.qpos_adr:o.qpos_adr + 7]
            out.append(ObjectPose(
                object_id=o.object_id,
                pos_xyz=(float(q[0]), float(q[1]), float(q[2])),
                quat_wxyz=(float(q[3]), float(q[4]), float(q[5]), float(q[6])),
            ))
        return out

    def poses_as_dicts(self, data: mujoco.MjData) -> List[dict]:
        return [
            {"object_id": p.object_id,
             "pos_xyz": list(p.pos_xyz),
             "quat_wxyz": list(p.quat_wxyz)}
            for p in self.poses(data)
        ]

    def reset(
        self,
        data: mujoco.MjData,
        seed: Optional[int] = None,
        jitter_m: float = 0.0,
    ) -> None:
        """Restore objects to initial poses.

        With a seed and jitter_m > 0, apply deterministic uniform position
        jitter in ±jitter_m (m) — same seed => same placement.
        """
        rng = np.random.default_rng(seed) if seed is not None else None
        for o in self._objects:
            q = o.initial_qpos.copy()
            if rng is not None and jitter_m > 0.0:
                q[0:3] += rng.uniform(-jitter_m, jitter_m, size=3)
            data.qpos[o.qpos_adr:o.qpos_adr + 7] = q
            data.qvel[o.qvel_adr:o.qvel_adr + 6] = 0.0


# ---------------------------------------------------------------------------
# Model composition
# ---------------------------------------------------------------------------

def build_scene_model_xml(
    scene_doc: Mapping[str, object],
    robot_model_path: Optional[str] = None,
) -> str:
    """Return an MJCF string = robot model + compiled scene objects.

    Scene object <body> elements are inserted before the robot's closing
    </worldbody>; scene <asset> (meshes) are merged into the robot's <asset>.
    """
    path = pathlib.Path(robot_model_path) if robot_model_path else _ROBOT_MODEL
    xml = path.read_text()

    asset_xml, body_xml = compile_scene_body_fragment(scene_doc)

    if body_xml.strip():
        xml = xml.replace(
            "  </worldbody>",
            body_xml + "\n  </worldbody>",
            1,
        )
    if asset_xml.strip():
        # Insert scene meshes just after the robot's <asset> open tag.
        xml = re.sub(r"(<asset>)", r"\1\n    " + asset_xml, xml, count=1)
    return xml


def build_scene_model(
    scene_doc: Mapping[str, object],
    robot_model_path: Optional[str] = None,
) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        build_scene_model_xml(scene_doc, robot_model_path)
    )
