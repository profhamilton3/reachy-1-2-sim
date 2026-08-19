"""
R12-504: Interactive control model for the native MuJoCo backend.

Drives the operable controls of a scene (buttons, switches, levers) from their
articulation-joint state each sim step:

  * button — a spring-loaded slide joint.  Each full press (displacement past
    ``on_threshold``, then released back past ``off_threshold``) TOGGLES a
    latched on/off output.  Debounced so one press = one toggle.
  * switch / lever — a hinge joint.  ``on`` iff the handle angle is past
    ``on_threshold``.  With ``bistable`` a smooth double-well torque snaps the
    handle to the nearer end so it holds on/off after the gripper lets go.

On every update the control's geom colour is set to ``lit_rgba`` (on) or its
base colour (off), so the change is visible in the rendered camera.  ``states()``
reports each control for streaming; ``reset()`` clears latches and colours.

Imports only ``mujoco`` + ``numpy``; runs on the native host side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence

import mujoco
import numpy as np

# Double-well snap torque (N·m) pushing the handle toward whichever end it is
# past; ``width`` (rad) smooths the transition through the midpoint.  Kept modest
# and paired with joint armature/damping so light handles snap without blowing up.
_BISTABLE_TORQUE = 0.08
_BISTABLE_WIDTH = 0.15


@dataclass
class _Element:
    obj_id: str
    kind: str                    # button | switch | lever
    qpos_adr: int
    dof_adr: int
    geom_id: int                 # -1 if the geom was not found
    on_threshold: float
    off_threshold: float
    bistable: bool
    base_rgba: np.ndarray
    lit_rgba: np.ndarray
    lo: float                    # joint range limits (for bistable midpoint)
    hi: float
    on: bool = False             # latched output
    armed: bool = True           # button: ready to accept the next press


class InteractiveController:
    """Owns the operable-control state for one (model, data) pair."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        specs: Sequence[Mapping],
    ) -> None:
        self.model = model
        self._elements: List[_Element] = []
        for s in specs:
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, s.get("joint_name", "")
            )
            if jid < 0:
                continue
            gid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, s.get("geom_name", "")
            )
            base = s.get("base_rgba")
            if base is None and gid >= 0:
                base = list(model.geom_rgba[gid])
            base_arr = _rgba(base, default=(0.7, 0.7, 0.7, 1.0))
            lit_arr = _rgba(s.get("lit_rgba"), default=tuple(base_arr))
            rng = s.get("range") or [0.0, 0.0]
            kind = s.get("type", "button")
            self._elements.append(_Element(
                obj_id=str(s.get("id")),
                kind=kind,
                qpos_adr=int(model.jnt_qposadr[jid]),
                dof_adr=int(model.jnt_dofadr[jid]),
                geom_id=gid,
                on_threshold=_default_on_threshold(s, kind, rng),
                off_threshold=_default_off_threshold(s, kind, rng),
                bistable=bool(s.get("bistable", False)),
                base_rgba=base_arr,
                lit_rgba=lit_arr,
                lo=float(rng[0]),
                hi=float(rng[1]),
            ))
        # Start bistable hinges at their OFF end (the midpoint is unstable) and
        # paint the initial (off) colours.
        for e in self._elements:
            self._seat_off(data, e)
            self._paint(e)

    def _seat_off(self, data: mujoco.MjData, e: _Element) -> None:
        """Place a bistable hinge at whichever end reads OFF so it starts off."""
        if e.bistable and e.kind != "button":
            off_end = e.lo if e.lo < e.on_threshold else e.hi
            data.qpos[e.qpos_adr] = off_end
            data.qvel[e.dof_adr] = 0.0
            e.on = False

    # ── Per-step update ───────────────────────────────────────────────────
    def update(self, data: mujoco.MjData) -> None:
        for e in self._elements:
            v = float(data.qpos[e.qpos_adr])
            if e.kind == "button":
                self._update_button(e, v)
            else:
                self._update_hinge(e, v, data)
            self._paint(e)

    def _update_button(self, e: _Element, v: float) -> None:
        # Slide range is [lo, 0]; pressed = more negative than on_threshold.
        if e.armed and v <= e.on_threshold:
            e.on = not e.on          # each completed press flips the latch
            e.armed = False
        elif not e.armed and v >= e.off_threshold:
            e.armed = True           # released far enough to re-arm

    def _update_hinge(self, e: _Element, v: float, data: mujoco.MjData) -> None:
        e.on = v >= e.on_threshold
        # Bistable detent: a smooth double-well torque pushes the handle toward
        # whichever end it is past, so a gripper flip through the midpoint snaps
        # to on/off and holds against the range limit.  Joint armature/damping
        # (set in the scene) keeps the light handle stable under this torque.
        if e.bistable:
            mid = 0.5 * (e.lo + e.hi)
            data.qfrc_applied[e.dof_adr] = (
                _BISTABLE_TORQUE * float(np.tanh((v - mid) / _BISTABLE_WIDTH))
            )

    def _paint(self, e: _Element) -> None:
        if e.geom_id < 0:
            return
        self.model.geom_rgba[e.geom_id] = e.lit_rgba if e.on else e.base_rgba

    # ── Introspection / reset ─────────────────────────────────────────────
    def states(self, data: Optional[mujoco.MjData] = None) -> List[dict]:
        out = []
        for e in self._elements:
            entry = {"id": e.obj_id, "type": e.kind, "on": bool(e.on)}
            if data is not None:
                entry["value"] = float(data.qpos[e.qpos_adr])
            out.append(entry)
        return out

    @property
    def element_ids(self) -> List[str]:
        return [e.obj_id for e in self._elements]

    def reset(self, data: mujoco.MjData) -> None:
        for e in self._elements:
            e.on = False
            e.armed = True
            data.qfrc_applied[e.dof_adr] = 0.0
            self._seat_off(data, e)
            self._paint(e)


# ── Helpers ───────────────────────────────────────────────────────────────
def _rgba(value, default) -> np.ndarray:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return np.array([float(v) for v in value], dtype=float)
    return np.array([float(v) for v in default], dtype=float)


def _default_on_threshold(spec: Mapping, kind: str, rng: Sequence) -> float:
    v = spec.get("on_threshold")
    if v is not None:
        return float(v)
    lo, hi = float(rng[0]), float(rng[1])
    if kind == "button":
        return 0.6 * lo            # 60% of the inward travel
    return 0.5 * (lo + hi)          # hinge midpoint


def _default_off_threshold(spec: Mapping, kind: str, rng: Sequence) -> float:
    v = spec.get("off_threshold")
    if v is not None:
        return float(v)
    lo, hi = float(rng[0]), float(rng[1])
    if kind == "button":
        return 0.2 * lo            # near the out (rest) position
    return 0.5 * (lo + hi)
