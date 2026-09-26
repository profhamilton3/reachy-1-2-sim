"""Synthetic, seeded fixture builders for tools/goalfix_cmp's tests.

Nothing here reads recorded evidence. All timing is simulation time:
``sim_step`` increments by one per 20 ms bridge tick, ``sim_time_s =
sim_step * TICK_S``. ``wall_time_ns`` is a monotonic counter kept
deliberately offset from ``sim_time_s`` (never assumed equal -- it is
``time.monotonic_ns()``, per the plan's own §7.0).

Two levels:

* Row builders (``state_row``/``command_row_joint``/``command_row_reset``/
  ``write_evidence``) -- exact schema match to ``native_mujoco/protocol.py``
  and ``native_mujoco/recorder.py``, for hand-built micro-fixtures that need
  precise control (echo-age boundaries, coincidence subclasses, malformed
  evidence).
* ``FlightSim`` -- a minimum-jerk multi-waypoint flight simulator (clean or
  echo-corrupted), for route-shaped fixtures (segment/path-check tests).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
for _p in (_REPO / "src", _REPO / "native_mujoco", _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reachy_ai.motion.rig_routes import R_JOINTS  # noqa: E402
from joint_map import JOINT_TABLE  # noqa: E402
from tools.goalfix_cmp._minjerk import s_of_tau  # noqa: E402
from tools.goalfix_cmp._units import deg_to_rad_f32  # noqa: E402

TICK_S = 0.020
JOINT_ORDER: Tuple[str, ...] = tuple(
    e.sdk_name for e in sorted(JOINT_TABLE, key=lambda e: e.mjcf_index))
assert JOINT_ORDER[:8] == R_JOINTS


def full21(pose: Dict[str, float], base: Optional[Sequence[float]] = None) -> List[float]:
    """A full 21-vector in JOINT_ORDER, `pose` (right-arm names) overlaid on
    `base` (default: all zero)."""
    out = list(base) if base is not None else [0.0] * 21
    for i, name in enumerate(JOINT_ORDER):
        if name in pose:
            out[i] = float(pose[name])
    return out


# ---------------------------------------------------------------------------
# Row builders (exact schema)
# ---------------------------------------------------------------------------

def _joint_entry(name: str, position_rad: float, compliant: bool = False) -> dict:
    return {"name": name, "uid": 0, "position_rad": position_rad,
            "velocity_rad_s": 0.0, "effort": 0.0, "compliant": compliant,
            "saturated": False}


def state_row(*, seq: int, sim_step: int, sim_time_s: float, cmd_seq: int,
              wall_time_ns: int, position_rad21: Sequence[float],
              compliant21: Optional[Sequence[bool]] = None) -> dict:
    compliant21 = list(compliant21) if compliant21 is not None else [False] * 21
    return {
        "type": "state", "seq": seq, "sim_step": sim_step,
        "sim_time_s": sim_time_s, "wall_time_ns": wall_time_ns,
        "cmd_seq": cmd_seq, "scene_revision": "", "paused": False,
        "joints": [_joint_entry(JOINT_ORDER[i], position_rad21[i], compliant21[i])
                   for i in range(21)],
        "objects": [], "grippers": [], "force_sensors": [], "interactive": [],
        "warnings": [], "contacts": [],
    }


def command_row_joint(*, seq: int, target_rad21: Sequence[float],
                       compliant: Optional[Sequence[Optional[bool]]] = None) -> dict:
    return {"type": "joint_command", "seq": seq, "target_rad": list(target_rad21),
            "mask": None, "compliant": list(compliant) if compliant is not None else None,
            "speed_limit_rad_s": None, "torque_limit_percent": None}


def command_row_reset(*, seed: Optional[int], sim_step: int, wall_time_s: float) -> dict:
    return {"type": "reset", "seed": seed, "sim_step": sim_step, "wall_time_s": wall_time_s}


def write_evidence(
    dir_path, state_rows: Sequence[dict], command_rows: Sequence[dict],
    extra_files: Optional[Dict[str, bytes]] = None,
    truncate_last_state_line: bool = False,
) -> Tuple[Path, Path, Path]:
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)
    states_path = d / "states.jsonl"
    commands_path = d / "commands.jsonl"
    states_text = "\n".join(json.dumps(r) for r in state_rows)
    if state_rows:
        states_text += "\n"
    if truncate_last_state_line and state_rows:
        # Drop the final row's closing brace -- an unparseable last line,
        # never a non-final one.
        states_text = states_text.rstrip("\n")
        states_text = states_text[: -len('"}') ] if states_text.endswith('"}') else states_text[:-1]
    states_path.write_text(states_text)
    commands_text = "\n".join(json.dumps(r) for r in command_rows)
    if command_rows:
        commands_text += "\n"
    commands_path.write_text(commands_text)

    sums: Dict[str, str] = {}
    for p in (states_path, commands_path):
        sums[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    if extra_files:
        for name, data in extra_files.items():
            (d / name).write_bytes(data)
            sums[name] = hashlib.sha256(data).hexdigest()
    sha_path = d / "SHA256SUMS"
    sha_path.write_text("\n".join(f"{h}  {name}" for name, h in sums.items()) + "\n")
    return states_path, commands_path, sha_path


def corrupt_sha256sums(sha_path: Path, rel_name: str) -> None:
    """Flip one hex digit of `rel_name`'s recorded hash, in place."""
    lines = sha_path.read_text().splitlines()
    out = []
    for line in lines:
        h, name = line.split(None, 1)
        if name == rel_name:
            flipped = ("1" if h[0] != "1" else "2") + h[1:]
            out.append(f"{flipped}  {name}")
        else:
            out.append(line)
    sha_path.write_text("\n".join(out) + "\n")


# ---------------------------------------------------------------------------
# Minimum-jerk flight simulator
# ---------------------------------------------------------------------------

#: Per-right-arm-joint state lag (rad) -- keeps every synthetic "realised"
#: position off the exact minimum-jerk curve the commands sample, so a
#: clean flight never produces an accidental bit-exact target/position
#: match (see the module's echo-fixture design notes in the handoff).
_LAG_RAD = [0.0021 + 0.0001 * j for j in range(8)]


@dataclass
class Waypoint:
    name: str
    pose: Dict[str, float]     # right-arm joint name -> radians
    seconds: float
    guard: Sequence[str] = ()


@dataclass
class ForcedEcho:
    command_index: int  # index into `command_rows` (== `evidence.commands` order)
    joint_index: int     # 0..7, right-arm index
    source_row: int      # `state_rows` index whose realised position was copied


@dataclass
class FlightResult:
    state_rows: List[dict]
    command_rows: List[dict]
    forced_echoes: List[ForcedEcho]
    goto_goal_names: List[str]          # one entry per goto actually flown (C0 sequence)
    segment_ticks: Tuple[int, int]       # (first, last) command index of the last goto->hold->restream span


class FlightSim:
    """Simulates one command+state row per 20 ms tick for a sequence of
    ``Waypoint`` gotos, minimum-jerk per joint, one merged command per tick
    (so "carry" vs "fresh" falls out of the classifier's own comparison,
    exactly as the real bridge's per-tick merge does).
    """

    def __init__(self, start_pose: Dict[str, float], *, seed: int = 0,
                 restream_passes: int = 0, restream_base_s: float = 0.8,
                 settle_s: float = 0.3, skew_ms: Dict[str, float] = None,
                 pose_units: str = "rad"):
        """``pose_units="deg"`` drives the simulator with real, degree-valued
        poses (``rig_routes`` waypoints, unconverted) -- the minimum-jerk
        interpolation itself runs in degrees, exactly as
        ``reachy_sdk.trajectory``'s does, and every command target is
        quantised to radians (``_units.deg_to_rad_f32``, the SAME formula
        ``route_rad`` uses for the waypoint endpoints) at the one point it is
        assembled into a 21-vector -- never re-derived, so a re-stream pass's
        commanded value is bit-exact with ``route_rad(route)[k].pose_rad8``
        (T1's C5 acceptance test). The default, ``"rad"``, is the pre-existing
        behaviour (poses already radians, no conversion) for the local mock
        routes existing single-mechanism tests still use."""
        self.pose = dict(start_pose)
        self.seed = seed
        self.restream_passes = restream_passes
        self.restream_base_s = restream_base_s
        self.settle_s = settle_s
        self.skew_s = {k: v / 1000.0 for k, v in (skew_ms or {}).items()}
        if pose_units not in ("rad", "deg"):
            raise ValueError(f"pose_units must be 'rad' or 'deg', got {pose_units!r}")
        self._deg = pose_units == "deg"

        # `_tick`/sim_step/sim_time_s restart at every reset(). `_row` and
        # `_state_seq` never do: a real state seq is monotonic "for the life
        # of the server process", never reset -- see
        # scripts/link_e1_flight.py's own docstring. `_cmd_seq` (the BRIDGE's
        # own outgoing numbering) restarts on `recreate()` (a new bridge
        # process), never on `reset()` alone. `_native_cmd_seq` (what states
        # actually REPORT) is the separate, persistent counter neither
        # `reset()` nor `recreate()` touches -- only a command being applied
        # does (T2/review §3.1: native's own `_cmd_seq` survives a reset,
        # `server.py:338-351`, and is set only when a pending update is
        # actually applied, `server.py:377`).
        self._tick = 0
        self._row = 0
        self._cmd_seq = 0
        self._native_cmd_seq = -1
        self._state_seq = 0
        self.state_rows: List[dict] = []
        self.command_rows: List[dict] = []
        self.goto_goal_names: List[str] = []
        self._last_segment_start: Optional[int] = None
        self._segment_bounds: Tuple[int, int] = (0, 0)
        self._emit_epoch_start_state()

    # -- internals ----------------------------------------------------
    def _to_rad(self, value) -> float:
        return deg_to_rad_f32(value) if self._deg else value

    def _sim_time(self) -> float:
        return self._tick * TICK_S

    def _wall_ns(self) -> int:
        return int(1_000_000_000 + self._row * TICK_S * 1e9)

    def _emit_command(self, target8: Dict[str, float],
                       forced: Optional[Dict[int, int]] = None) -> None:
        """`target8`: right-arm name -> value this tick, in THIS instance's
        `pose_units` (mode-native; converted to radians here, the one place
        a command target is assembled). `forced`: right-arm index ->
        `state_rows` index to copy the (lagged) realised position from,
        overriding the min-jerk value for that joint this tick."""
        target8_rad = {name: self._to_rad(v) for name, v in target8.items()}
        row_target = full21(target8_rad)
        if forced:
            for jidx, src_row in forced.items():
                src_state = self.state_rows[src_row]
                # Truncate to float32 -- the real echo path round-trips the
                # position through a float32 protobuf field before it comes
                # back as a target; a "fresh" min-jerk value is left at full
                # float64 precision, which is what makes a genuine echo's
                # bit-exact match detectable and a coincidental one
                # essentially impossible (see verify_goal_feedback.py's own
                # docstring).
                row_target[jidx] = float(np.float32(
                    src_state["joints"][jidx]["position_rad"]))
        self.command_rows.append(
            command_row_joint(seq=self._cmd_seq, target_rad21=row_target))
        self._cmd_seq += 1
        self._native_cmd_seq = self._cmd_seq - 1  # applied the same tick

    def _emit_epoch_start_state(self) -> None:
        """One state pushed before any command is issued this epoch --
        matches a real server, which pushes state continuously from reset
        -- so the epoch's very first command always has a genuine `t_lo`
        (its own bracket never needs to reach into the previous epoch).
        Reports `_native_cmd_seq` AS IT STANDS (carried across `reset()`
        and `recreate()` alike, per their own docstrings) -- never a fresh
        `-1` -- so a fixture that calls `recreate()`+`reset()` between
        legs correctly reproduces the stale-carried-value regime T2 fixes."""
        realised8 = {name: self._to_rad(self.pose.get(name, 0.0)) - _LAG_RAD[i]
                     for i, name in enumerate(R_JOINTS[:7])}
        realised8["r_gripper"] = self._to_rad(self.pose.get("r_gripper", 0.0)) - _LAG_RAD[7]
        self._append_state(realised8, cmd_seq=self._native_cmd_seq)

    def _append_state(self, realised8: Dict[str, float], cmd_seq: int) -> None:
        pos21 = full21(realised8)
        self.state_rows.append(state_row(
            seq=self._state_seq, sim_step=self._tick, sim_time_s=self._sim_time(),
            cmd_seq=cmd_seq, wall_time_ns=self._wall_ns(), position_rad21=pos21))
        self._state_seq += 1
        self._row += 1
        self._tick += 1

    def _emit_state(self, realised8: Dict[str, float]) -> None:
        self._append_state(realised8, self._native_cmd_seq)

    def _tick_command_then_state(self, target8: Dict[str, float],
                                  forced: Optional[Dict[int, int]] = None) -> int:
        """One tick: emit the command, then the state that reports it
        applied (same tick, per this module's bracket convention). Returns
        the emitted command's own `command_rows` index."""
        cmd_idx = len(self.command_rows)
        self._emit_command(target8, forced)
        realised8 = {name: self._to_rad(target8[name]) - _LAG_RAD[i]
                     for i, name in enumerate(R_JOINTS[:7])}
        realised8["r_gripper"] = self._to_rad(
            target8.get("r_gripper", self.pose.get("r_gripper", 0.0))) - _LAG_RAD[7]
        self._emit_state(realised8)
        return cmd_idx

    def _hold_ticks(self, n: int, held_pose: Dict[str, float]) -> None:
        for _ in range(n):
            realised8 = {name: self._to_rad(held_pose[name]) - _LAG_RAD[i]
                         for i, name in enumerate(R_JOINTS[:7])}
            realised8["r_gripper"] = self._to_rad(held_pose.get("r_gripper", 0.0)) - _LAG_RAD[7]
            self._emit_state(realised8)

    # -- public ---------------------------------------------------------
    def fly(self, route: Sequence[Waypoint],
            echo_rate: float = 0.0, echo_rng=None,
            echo_lookback_ticks: int = 20) -> "FlightSim":
        import random
        rng = echo_rng if echo_rng is not None else random.Random(self.seed)
        forced_echoes: List[ForcedEcho] = getattr(self, "forced_echoes", [])

        for wp in route:
            self.goto_goal_names.append(wp.name)
            start = dict(self.pose)
            goal = dict(self.pose)
            goal.update(wp.pose)
            n_ticks = max(1, round(wp.seconds / TICK_S))
            leg_start_tick = self._tick

            # i=0 is the goto's own anchor sample (tau=0, exactly the
            # previous target) -- real gotos start from the SDK's cached
            # goal, so the first setpoint of a new goto is (within float32
            # quantisation) identical to the last one, which is what C1
            # checks. Skipping it would make every goto transition an
            # artificial C1 failure.
            for i in range(0, n_ticks + 1):
                target8 = {}
                for jname in R_JOINTS:
                    a, b = start.get(jname, 0.0), goal.get(jname, 0.0)
                    if a == b:
                        target8[jname] = a
                        continue
                    skew = self.skew_s.get(jname, 0.0) if i > 0 else 0.0
                    tau = min(1.0, max(0.0, (i * TICK_S + skew) / wp.seconds))
                    target8[jname] = a + s_of_tau(tau) * (b - a)

                forced: Dict[int, int] = {}
                # Ticks near either end of a minimum-jerk goto are nearly
                # flat (zero first/second derivative), so a source drawn
                # from there can legitimately float32-truncate to the same
                # value as the immediately preceding command -- a real
                # "carry", not a detection miss. Keeping both the injection
                # point and the echo source at least 6 ticks (120 ms) from
                # either end avoids manufacturing that ambiguity by
                # construction.
                if echo_rate > 0 and 6 <= i <= n_ticks - 6:
                    for jidx, jname in enumerate(R_JOINTS):
                        if target8[jname] == start.get(jname, 0.0):
                            continue  # constant joint: would just look like a carry
                        if rng.random() < echo_rate:
                            max_back = min(echo_lookback_ticks, self._row - 6)
                            if max_back < 1:
                                continue
                            back = rng.randint(1, max_back)
                            src_row = self._row - back
                            candidate = float(np.float32(
                                self.state_rows[src_row]["joints"][jidx]["position_rad"]))
                            prev_val = (self.command_rows[-1]["target_rad"][jidx]
                                        if self.command_rows else None)
                            if prev_val is not None and candidate == prev_val:
                                # Would be indistinguishable from a carry (a
                                # second consecutive injection landing on
                                # the same source, or a plateau collision)
                                # -- skip rather than manufacture a case the
                                # classifier is right to call a carry.
                                continue
                            forced[jidx] = src_row

                cmd_idx = self._tick_command_then_state(target8, forced)
                for jidx, src_row in forced.items():
                    forced_echoes.append(ForcedEcho(cmd_idx, jidx, src_row))

            self.pose = goal
            last_goto_end_tick = self._tick - 1

            # Re-stream passes: constant holds at wp.pose (fly_route-style).
            restream_start_tick = self._tick
            for k in range(self.restream_passes):
                pass_s = self.restream_base_s * (1 + k)
                n_pass_ticks = max(1, round(pass_s / TICK_S))
                for _ in range(n_pass_ticks):
                    self._tick_command_then_state(dict(goal))
            restream_end_tick = self._tick - 1 if self.restream_passes else last_goto_end_tick

            # Settle hold: no commands, states only.
            n_hold_ticks = max(0, round(self.settle_s / TICK_S))
            self._hold_ticks(n_hold_ticks, goal)

            self._segment_bounds = (leg_start_tick, self._tick - 1)

        self.forced_echoes = forced_echoes
        return self

    def reset(self, seed: Optional[int] = None) -> None:
        """A native reset: `sim_step`/`sim_time_s` restart (a new epoch).
        Neither `_cmd_seq` (the bridge's own numbering) nor
        `_native_cmd_seq` (what states report) is touched -- `server.py`'s
        own reset does not touch its `_cmd_seq` either (T2/review §3.1)."""
        self.command_rows.append(command_row_reset(
            seed=seed, sim_step=self._tick, wall_time_s=self._sim_time()))
        self._tick = 0
        self._emit_epoch_start_state()

    def recreate(self) -> None:
        """A container recreate: a brand-new bridge process, so ITS OWN
        outgoing `seq` numbering restarts -- but native's own `cmd_seq`
        (`_native_cmd_seq` here) is untouched until the new bridge's first
        command is applied (T2/review §3.1). Always followed by `reset()`
        before any new command, per the plan's P2 (recreate) -> P3 (reset)
        order; nothing is sent in between (plan §4) -- this method itself
        sends nothing, matching that."""
        self._cmd_seq = 0

    def result(self) -> FlightResult:
        return FlightResult(self.state_rows, self.command_rows,
                             getattr(self, "forced_echoes", []),
                             self.goto_goal_names, self._segment_bounds)
