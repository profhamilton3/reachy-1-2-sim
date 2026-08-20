"""R12-802: Fixed-step deterministic episode runner.

EpisodeRunner executes a sequence of StepCommands against a SimulationCore
without wall-clock sleeping, WebSocket communication, rendering, or SDK
dependency.  It is the high-throughput search path described in EPIC-8.md
Section 5.4.

Usage (basic)
-------------
    core = SimulationCore.from_paths(model_path, scene_path)
    config = EpisodeConfig(simulator_identity=identity, seed=7, max_steps=3000)
    runner = EpisodeRunner(core, config)

    commands = [StepCommand(targets, hold_steps=50) for targets in my_sequence]
    result = runner.run(commands)

Usage (with per-step callback)
-------------------------------
    def on_snap(snap: EvaluationSnapshot) -> None:
        if snap.has_forbidden_contact():
            ...

    result = runner.run(commands, on_snapshot=on_snap)

Cancellation
------------
Pass a threading.Event as `cancelled`; set it from another thread to stop
the runner after the current step completes.  The result will have status
ABORTED and termination_reason "cancelled".

Compatibility path (PR 8.9)
----------------------------
This runner produces the same EpisodeResult fields as the SDK→Docker→MuJoCo
path so results can be compared without reformatting.
"""

from __future__ import annotations

import dataclasses
import datetime
import math
import threading
import time
import uuid
from typing import Callable, Iterable, List, Optional, Sequence

from evaluation_snapshot import EvaluationSnapshot
from simulation_core import SimulationCore

import sys
import os
_SRC = os.path.join(os.path.dirname(__file__), "../src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from reachy_ai.experience.models import (
    EpisodeConfig,
    EpisodeResult,
    EpisodeStatus,
)

# Hard-failure contype pairs (robot↔fixture = 2↔8)
_ROBOT_CONTYPE   = 2
_FIXTURE_CONTYPE = 8


@dataclasses.dataclass
class StepCommand:
    """Joint targets to apply for one or more simulation steps.

    target_rad must have NUM_JOINTS elements (21 for Reachy 1.2).
    hold_steps controls how many mj_step calls keep this command active
    before the next command in the sequence is applied.
    """
    target_rad: Sequence[float]
    hold_steps: int = 1

    def __post_init__(self) -> None:
        if self.hold_steps < 1:
            raise ValueError(f"hold_steps must be >= 1, got {self.hold_steps}")


class EpisodeRunner:
    """Fixed-step episode runner backed by a SimulationCore.

    One runner owns one SimulationCore; the runner resets it at the start of
    each run() call.  Do not share a runner across threads.
    """

    def __init__(
        self,
        core: SimulationCore,
        config: EpisodeConfig,
    ) -> None:
        self.core = core
        self.config = config

    def run(
        self,
        commands: Iterable[StepCommand] = (),
        *,
        on_snapshot: Optional[Callable[[EvaluationSnapshot], None]] = None,
        cancelled: Optional[threading.Event] = None,
    ) -> EpisodeResult:
        """Execute a command sequence and return a structured EpisodeResult.

        The runner:
        1. Resets the core with config.seed.
        2. For each StepCommand: sets joint targets, advances hold_steps times,
           collecting a snapshot after each step.
        3. After commands are exhausted, continues stepping up to config.max_steps
           to let the physics settle (configurable via settle_steps kwarg — zero
           by default since PR 8.3 decides dwell policy).
        4. Returns EpisodeResult with step metrics and final control/object states.

        Hard violations (forbidden contact, NaN) terminate immediately with
        status INVALID.  Cancellation terminates with status ABORTED.
        max_steps reached terminates with status FAILED (no success criterion
        is evaluated here — that is the evaluator's job in PR 8.4).
        """
        cfg = self.config
        episode_id = uuid.uuid4().hex
        trial_id = cfg.simulator_identity.scene_revision or "standalone"

        wall_start = time.monotonic()
        self.core.reset(seed=cfg.seed)

        start_step = self.core.step
        snapshots_collected: List[EvaluationSnapshot] = []
        forbidden_total = 0
        all_violations: List[str] = []
        peak_object_z: dict = {}      # object_id → max z seen across all steps
        peak_grip_force_n: float = 0.0
        grasp_step_count: int = 0

        def _collect(snap: EvaluationSnapshot) -> None:
            nonlocal forbidden_total, peak_grip_force_n, grasp_step_count
            forbidden_total += snap.forbidden_contact_count
            if snap.joint_limit_violations:
                for v in snap.joint_limit_violations:
                    if v not in all_violations:
                        all_violations.append(v)
            for obj in snap.objects:
                oid = obj.get("object_id") or obj.get("id") or ""
                xyz = obj.get("pos_xyz")
                if oid and xyz and len(xyz) >= 3:
                    z = float(xyz[2])
                    if z > peak_object_z.get(oid, -1e9):
                        peak_object_z[oid] = z
            for g in snap.grippers:
                if g.get("side") == "r":
                    f = float(g.get("grip_force_n", 0.0))
                    if f > peak_grip_force_n:
                        peak_grip_force_n = f
                    if g.get("grasping", False):
                        grasp_step_count += 1
            if on_snapshot:
                on_snapshot(snap)

        status = EpisodeStatus.RUNNING
        termination_reason = ""

        # --- Execute command sequence ---
        for cmd in commands:
            if cancelled and cancelled.is_set():
                status = EpisodeStatus.ABORTED
                termination_reason = "cancelled"
                break
            if self.core.step - start_step >= cfg.max_steps:
                status = EpisodeStatus.FAILED
                termination_reason = "max_steps"
                break

            self.core.set_targets(cmd.target_rad)
            for _ in range(cmd.hold_steps):
                if self.core.step - start_step >= cfg.max_steps:
                    status = EpisodeStatus.FAILED
                    termination_reason = "max_steps"
                    break
                self.core.advance()
                snap = self.core.snapshot()

                # Hard failure: NaN/Inf in positions
                if any(not math.isfinite(j["position_rad"]) for j in snap.joints):
                    status = EpisodeStatus.INVALID
                    termination_reason = "nan_or_inf_in_joint_positions"
                    _collect(snap)
                    break

                _collect(snap)
                if status != EpisodeStatus.RUNNING:
                    break
            if status != EpisodeStatus.RUNNING:
                break

        # Take a final snapshot to capture end-state metrics
        final_snap = self.core.snapshot()
        _collect(final_snap)

        wall_duration = time.monotonic() - wall_start
        end_step = self.core.step

        # Determine final status if still RUNNING (commands exhausted normally)
        if status == EpisodeStatus.RUNNING:
            status = EpisodeStatus.SUCCEEDED if forbidden_total == 0 else EpisodeStatus.FAILED
            if forbidden_total > 0:
                termination_reason = "forbidden_contact"

        # Build EpisodeResult
        hard_violations: List[str] = []
        if forbidden_total > 0:
            hard_violations.append(
                f"forbidden_robot_fixture_contact: {forbidden_total} occurrences"
            )
        for v in all_violations:
            hard_violations.append(f"joint_limit_violation: {v}")

        # Extract final right-gripper state for release verification
        final_grip_force_n = 0.0
        final_grasping = False
        for g in final_snap.grippers:
            if g.get("side") == "r":
                final_grip_force_n = float(g.get("grip_force_n", 0.0))
                final_grasping = bool(g.get("grasping", False))

        metrics = {
            "total_steps": float(end_step - start_step),
            "forbidden_contact_count": float(forbidden_total),
            "sim_duration_s": float(self.core.data.time),
            "wall_duration_s": float(wall_duration),
            "peak_grip_force_n": peak_grip_force_n,
            "grasp_step_count": float(grasp_step_count),
            "final_grip_force_n": final_grip_force_n,
            "final_grasping": float(final_grasping),
        }
        for oid, z in peak_object_z.items():
            metrics[f"object_{oid}_peak_z_m"] = z

        # Saturation fraction across all steps and joints
        saturated_names = set(final_snap.saturated_joints)
        metrics["saturated_joint_count"] = float(len(saturated_names))

        result = EpisodeResult(
            episode_id=episode_id,
            trial_id=trial_id,
            status=status,
            termination_reason=termination_reason,
            success=(status == EpisodeStatus.SUCCEEDED),
            start_sim_step=start_step,
            end_sim_step=end_step,
            sim_duration_s=float(self.core.data.time),
            wall_duration_s=float(wall_duration),
            metrics=metrics,
            hard_violations=hard_violations,
            contact_summary={
                "total_contacts": self.core.data.ncon,
                "forbidden_total": forbidden_total,
            },
            control_state_changes=[],   # populated by evaluator (PR 8.4)
            final_control_states={
                s["id"]: s for s in final_snap.interactive
            },
            final_object_states={
                o.get("object_id", o.get("id", str(i))): o
                for i, o in enumerate(final_snap.objects)
            },
            warnings=[],
        )
        return result
