"""R12-802: Typed state/contact snapshots for the episode runner.

EvaluationSnapshot is the authoritative per-step record produced by
SimulationCore.snapshot().  Evaluators (PR 8.4) consume these to classify
episodes; the store (PR 8.1) serialises their aggregate into EpisodeResult.

All positions are in metres, angles in radians, forces in Newtons.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple


@dataclasses.dataclass(frozen=True)
class ContactRecord:
    """One active contact pair from MuJoCo's contact list.

    MuJoCo reports contact i when
        (geom_a.contype & geom_b.conaffinity) | (geom_b.contype & geom_a.conaffinity) != 0.

    geom1/geom2 ordering follows MuJoCo (geom1 < geom2 index by convention).
    """
    geom1: str          # geom name (may be "" for unnamed geoms)
    geom2: str
    body1: str          # body that owns geom1
    body2: str
    contype1: int       # geom1 contype bitmask
    contype2: int       # geom2 contype bitmask
    conaffinity1: int
    conaffinity2: int
    pos: Tuple[float, float, float]     # contact point in world frame (m)
    normal_force: float                 # magnitude of normal force (N); 0 outside solver
    dist: float                         # signed separation distance (m, negative = penetrating)


@dataclasses.dataclass
class EvaluationSnapshot:
    """Complete per-step snapshot used by evaluators and the experience system.

    Produced by SimulationCore.snapshot() after each call to advance().
    The snapshot is a value object: callers should treat it as read-only.
    """
    sim_step: int
    sim_time_s: float
    scene_revision: str

    joints: List[Dict[str, Any]]            # same schema as protocol.State.joints
    contacts: List[ContactRecord]            # all active contacts this step
    interactive: List[Dict[str, Any]]        # same schema as protocol.State.interactive
    objects: List[Dict[str, Any]]            # same schema as protocol.State.objects
    grippers: List[Dict[str, Any]]           # same schema as protocol.State.grippers
    force_sensors: List[Dict[str, Any]]      # same schema as protocol.State.force_sensors

    # Derived quick-access fields
    forbidden_contact_count: int = 0        # robot (ct=2) ↔ fixture (ct=8) contacts
    joint_limit_violations: List[str] = dataclasses.field(default_factory=list)
    saturated_joints: List[str] = dataclasses.field(default_factory=list)

    def control_state(self, control_id: str) -> Optional[Dict[str, Any]]:
        """Return the interactive state dict for a control, or None."""
        for s in self.interactive:
            if s.get("id") == control_id:
                return s
        return None

    def is_on(self, control_id: str) -> bool:
        s = self.control_state(control_id)
        return bool(s and s.get("on"))

    def has_forbidden_contact(self) -> bool:
        return self.forbidden_contact_count > 0

    def contacts_by_contype(self, ct_a: int, ct_b: int) -> List[ContactRecord]:
        """Return contacts where one geom has contype ct_a and the other ct_b."""
        result = []
        for c in self.contacts:
            if (c.contype1 == ct_a and c.contype2 == ct_b) or \
               (c.contype1 == ct_b and c.contype2 == ct_a):
                result.append(c)
        return result
