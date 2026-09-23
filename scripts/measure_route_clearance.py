"""Read-only route-clearance recorder (review 2026-09-12, E1:
docs/reviews/2026-09-12-repair-review-and-56-74-design.md, section 5).

No through-the-move data exist for the tabletop legs (PRESENT/HOVER <->
REST) -- the rail rows are the only realised-clearance numbers on record,
and they are tube readings on a different corridor. This script closes that
gap the safe way: it CONNECTS to the SDK only to STREAM `present_position`
at 20 Hz while an operator flies a named route by hand (teach pendant,
notebook cell, or the panel) -- it never sets a `goal_position` and never
calls anything in `src/reachy_ai/motion/primitives.py`. The realised joint
trajectory, INCLUDING the realised gripper aperture, is saved as JSON under
`runs/`, then reported as per-link worst clearance against every scene
object, under both hand models ("tube", today's default, and "shells",
#56/#74's candidate), next to the PLANNED clearance at the same sample
count -- so a re-flight can finally say what the realised tracking
degradation actually costs, on either model, AT THE APERTURE THE HAND
ACTUALLY HAD.

Refuses to run unless REACHY_SIM_RECORD_CLEARANCE=1 is set: streaming joint
telemetry during a flight an operator did not mean to record is still a
surprise worth a deliberate opt-in, even though nothing here can move the
arm.

Host Python may not be able to install `reachy-sdk` directly (e.g. no
CPython 3.14 wheel for a transitive dependency) -- see
`docs/e1-host-recorder-env.md` for the exact venv recipe this script was
verified against, and run from a host shell, never the container
(`e1_identity.py` refuses container runs; see that doc's last section).

## Log schema (schema_version 3)

    {
      "route": str,
      "scene": str,
      "sample_hz": float,        # NOMINAL recording rate in Hz (samples/s).
                                  # Actual spacing is best-effort -- read
                                  # each sample's own "t", never assume
                                  # exactly 1/sample_hz between samples.
      "schema_version": 3,       # 2 added aperture logging; 1 (or absent)
                                  # predates it -- "joints" has no
                                  # "r_gripper" key. See `allow_missing_aperture`
                                  # below. 3 (this one) adds "t0_wall_ns" here
                                  # and "wall_time_ns" on every sample (E1
                                  # readiness, assignment 2026-09-14, work
                                  # item 3) -- see "Sample wall-clock, and
                                  # why it is monotonic_ns" below.
      "t0_wall_ns": int,          # time.monotonic_ns() when recording started
                                  # -- schema 3+ only.
      "samples": [
        {
          "t": float,            # seconds elapsed since recording started,
                                  # from time.monotonic() -- NOT wall-clock,
                                  # NOT guaranteed evenly spaced.
          "wall_time_ns": int,   # time.monotonic_ns() at this sample --
                                  # schema 3+ only; REQUIRED for schema 3
                                  # (validated_samples raises if absent -- a
                                  # gap here is a bad recording, the same
                                  # stance as a missing r_gripper on the
                                  # current schema). See below for why this
                                  # is monotonic_ns, not time.time_ns().
          "joints": {             # every value is DEGREES, matching every
                                  # other joint reading in this codebase
                                  # (kinematics.py, rig_routes.py) --
                                  # never radians.
            "r_shoulder_pitch": float, "r_shoulder_roll": float,
            "r_arm_yaw": float, "r_elbow_pitch": float,
            "r_forearm_yaw": float, "r_wrist_pitch": float,
            "r_wrist_roll": float,
            "r_gripper": float,   # the REALISED aperture at this instant --
                                  # what this module exists to add. See
                                  # "Missing or invalid aperture" below.
          },
        },
        ...
      ],
    }

## Sample wall-clock, and why it is monotonic_ns

`wall_time_ns` is named to match `native_mujoco.protocol.State.wall_time_ns`
-- the field the native server stamps on every state it streams and every
line of `<run_dir>/states.jsonl` -- so `scripts/link_e1_flight.py` (E1
readiness work item 3c) can align a sample here to the state nearest it in
time. Despite the name, THAT field is `time.monotonic_ns()`
(`protocol.py`'s `_now_ns`, never overridden with real wall-clock time in
`server.py`'s `_build_state`), not `time.time_ns()`: confirmed by reading
`mujoco_remote_backend.py` (diffs it against its own `time.monotonic_ns()`
to compute staleness, line ~101/559) and `scripts/benchmark.py` (same, line
~94) -- both already treat it as monotonic, not wall-clock, elsewhere in
this codebase.

So this module stamps its own samples with `time.monotonic_ns()` too, never
`time.time_ns()`: a wall-clock reading and a monotonic reading share no
fixed offset, so mixing them would make every alignment computation in the
linker meaningless. This is safe BECAUSE this module and
`native_mujoco/server.py` are both host-native processes (this module's own
usage instructions run it directly on the host, connecting outward to the
SDK's gRPC port; the native server is launched the same way, `mjpython
server.py`, never inside the Docker container) -- both read literally the
same OS monotonic clock. That is a stronger guarantee than the
cross-container comparisons already elsewhere in this codebase
(`mujoco_remote_backend.py` runs inside the Docker container and diffs its
own clock against the host-native server's `wall_time_ns` over the
websocket -- comparable only to the extent the container runtime's
monotonic clock tracks the host's); this module never needs that weaker
guarantee because it never compares against anything running inside the
container.

## Missing or invalid aperture

The aperture is the variable that decides whether the hand is inside the
tube at all (docs/adr/0003's discrepancy section and "Correcting the
tube"). A sample whose `r_gripper` reading cannot be trusted is not a
sample with a known-safe default -- it is a sample with NO clearance
answer, and this module never manufactures one silently:

  * A sample missing the `r_gripper` key raises `ApertureDataError` by
    default. `allow_missing_aperture=True` only ever rescues this for a
    log DECLARED as a schema in `_MISSING_APERTURE_ELIGIBLE_SCHEMA_VERSIONS`
    (today: `schema_version == 1`, or absent, which meant 1 before this
    schema existed) -- passed explicitly to
    `validated_samples`/`realised_clearance`/`report` via their
    `schema_version` argument. **This set is deliberately NARROWER than
    `_SUPPORTED_LEGACY_SCHEMA_VERSIONS`** (E1 readiness, assignment
    2026-09-14, work item 3 -- resolved before implementation, 2026-09-14):
    schema 2 became a RECOGNISED legacy schema the day schema 3 shipped
    (`_SUPPORTED_LEGACY_SCHEMA_VERSIONS = (1, 2)`, so a schema-2 log no
    longer raises `UnsupportedSchemaVersionError`), but it was never a
    schema that predates aperture logging -- schema 2 logs HAVE `r_gripper`
    -- so it never joined the missing-aperture rescue set. **A
    `schema_version` of 2 OR `LOG_SCHEMA_VERSION` (the current schema) with
    a missing `r_gripper` key always raises, `allow_missing_aperture`
    notwithstanding**: only schema 1 predates aperture logging, so a gap in
    a 2-or-current log is a dropped field or a bad recording, never an old
    log format, and the flag exists for schema 1 only. An unrecognised
    `schema_version` (anything but `LOG_SCHEMA_VERSION` or a version in
    `_SUPPORTED_LEGACY_SCHEMA_VERSIONS`) raises `UnsupportedSchemaVersionError`
    immediately, before any per-sample check -- this module never guesses
    what an unfamiliar shape means. When the flag does apply, the fallback
    (wide-open, `gripper_deg=None`) is reported, not hidden: `report()`'s
    result names every sample index it happened to.
  * A sample whose `r_gripper` is present but not a finite number in the
    MJCF's commanded range (`None`, NaN, a string, or a value outside
    roughly [-68.8, 20.05] deg) ALWAYS raises `ApertureDataError`, with or
    without `allow_missing_aperture` or which `schema_version` was
    declared -- a corrupt reading is a different problem than an old log
    format, and is never worth guessing past.
  * A schema-3 (current) sample missing `wall_time_ns`, or whose value is
    not a finite number, ALWAYS raises `SampleTimestampError` -- schema 1
    and 2 logs predate this field and are read without it (the linker
    simply cannot align them to a server run), but a gap in a *current*
    schema log is a dropped field or a bad recording, the same stance
    `r_gripper` already gets.

## Failed recordings are preserved, marked invalid

If a fresh recording fails this validation, `main()` never discards it: it
is saved via `save_invalid_log` under a name ending `_INVALID.json`, with
the samples as actually recorded (including the offending one) and the
failure reason, so a bad flight can be diagnosed instead of re-flown
blind. The file cannot be mistaken for a normal E1 log: its top level
carries `"valid": false`, an `"error"` string, and a `"schema_version"`
this module never recognises (neither current nor a supported legacy
schema) -- so reading it back the normal way (`schema_version_of(log)`
fed into `validated_samples`/`report`) raises `UnsupportedSchemaVersionError`
outright, and it can never be accepted as usable E1 data.

## Aperture provenance in `report()`

`report()`'s result names, in plain text, where each half's aperture
actually came from: `realised_aperture_source` (the measured, per-sample
`r_gripper` reading -- see `realised_aperture_assumed_samples` for any
indices that instead used the assumed-open fallback) and
`planned_aperture_policy` (see `planned_clearance` below for the two
models' different rules -- tube's own worst-case ENDPOINT rule is
conservative there, but is NOT conservative for shells, whose narrowest
link (`finger`) is closest to an object when the aperture is SHUT, the
opposite of the aperture `hand_radius` calls worst; root-caused
2026-09-22, `outputs/analysis-2026-09-22-b2s2-finger-clearance-root-cause.md`).
The shells correction changes PER-LINK values only (`planned_clearance_by_link`);
whole-hand shells `report()`/`planned_clearance` output is unchanged on
every scene shipped in this repo -- e.g. B2's LOWER_TO_REST, where
`thumb_pad` at 2.11 cm binds before `finger`'s corrected 3.37 cm ever
would (issue #137, priority 2).

## Route scope in `report()`

`FOOTPRINT_LEGS` is the live guard's own scope (`panel_executor`'s
footprint check) and is never widened here. A realised log that flew a
route's FULL sequence from its departure posture (e.g. a "setup" recording
that starts at HOME, not at `FOOTPRINT_LEGS`'s own first waypoint) is not
comparable to a `FOOTPRINT_LEGS`-scoped planned value -- the realised worst
sample can fall on a waypoint the footprint never included (see the same
analysis doc's §4). Pass `full_route=True` to `report()` (or
`waypoints=full_route_poses(route)` to `planned_clearance` directly) when
the recording being compared flew the whole named route, not just its
footprint.

`main()` accepts `--report-scope {footprint,full}` (default `footprint`,
preserving the stdout every earlier version printed) and passes
`report_scope == "full"` as `report()`'s `full_route` argument, via
`resolve_full_route_flag` (issue #139). `--report-scope full` is refused
-- before recording starts, not after a flight is thrown away -- for any
`--route` with no `full_route_poses()` sequence (WAVE, POINT_*; see
`resolve_full_route_flag`). The printed report always names its own scope
plainly: `planned_route_scope` in the JSON result, echoed as a
`Report scope: ...` line above it.

`scripts/e1_stage1/leg.sh` (the E1 campaign) does not pass
`--report-scope`, so it still gets `footprint` by default -- its
whole-route legs (PLACE_ROUTE, RAISE_TO_SIDE, both starting at HOME,
recorded through `main()`) still print a `FOOTPRINT_LEGS`-scoped report,
the same mis-scoping issue #137 root-caused, where the realised worst can
fall on a waypoint the footprint omits. Wiring `leg.sh`/`plan.py` to pass
`--report-scope full` for those legs is its own decision, out of issue
#139's scope. Until then, re-score a saved log OFFLINE with
`report(..., full_route=True)` from the Python API (issue #137, priority
2), or re-run `main()` by hand with `--report-scope full`.

Usage (operator flies the route by hand during --duration):
    export REACHY_SIM_RECORD_CLEARANCE=1
    export REACHY_SIM_RECORD=/abs/path/e1_server_runs   # the native server's own --record dir
    python3 scripts/measure_route_clearance.py --route LOWER_TO_REST \\
        --duration 15 --host localhost --record-root "$REACHY_SIM_RECORD"

`--record-root` (or REACHY_SIM_RECORD) is required: this module refuses to
record unless it can verify simulator identity against that directory
first -- see e1_identity.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

try:
    from reachy_sdk import ReachySDK
except ImportError:
    ReachySDK = None  # only needed by main(); the reporting half is offline

from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.motion.kinematics import (  # noqa: E402
    _GRIPPER_OPEN_LIMIT_DEG,
    _GRIPPER_SHUT_LIMIT_DEG,
    hand_radius,
    joint_path,
    link_capsules,
)
from reachy_ai.scene.awareness import SceneModel  # noqa: E402

import e1_identity  # noqa: E402 -- E1 readiness work item 2, same dir (scripts/)
import link_e1_flight  # noqa: E402 -- E1 readiness work item 3

SAMPLE_HZ = 20.0
RUNS_DIR = _HERE.parent / "runs"
HAND_MODES = ("tube", "shells")

#: The joint this module exists to add to the recording.  Already the 8th
#: entry of `rig_routes.R_JOINTS` (ARM7 plus this) -- `record_joint_log`
#: streams the same 8, `getattr(arm, name).present_position`, the same
#: read-only way `tasks.rig_motion` already reads every joint including
#: this one.
GRIPPER_JOINT = "r_gripper"

#: A log without a "schema_version" field predates this module's aperture
#: logging (was 1 implicitly).  Bump this if the schema changes again.
#: 3 (E1 readiness, assignment 2026-09-14, work item 3) adds "t0_wall_ns"
#: and per-sample "wall_time_ns" -- see the module docstring's "Sample
#: wall-clock" section.
LOG_SCHEMA_VERSION = 3

#: schema_version values this module RECOGNISES as a legacy shape -- i.e.
#: never raises `UnsupportedSchemaVersionError` for.  This is NOT the same
#: set as "eligible for the missing-aperture rescue" (see
#: `_MISSING_APERTURE_ELIGIBLE_SCHEMA_VERSIONS` immediately below) --
#: schema 2 is a recognised legacy schema (it has aperture logging, just not
#: wall-clock timestamps) but was never eligible for that rescue, because it
#: never lacked `r_gripper` in the first place.  Extend this set whenever a
#: new current schema makes today's `LOG_SCHEMA_VERSION` a legacy one in turn.
_SUPPORTED_LEGACY_SCHEMA_VERSIONS = (1, 2)

#: schema_version values eligible for `allow_missing_aperture=True`'s
#: missing-`r_gripper` rescue -- ONLY the one shape that ever predated
#: aperture logging.  Deliberately narrower than
#: `_SUPPORTED_LEGACY_SCHEMA_VERSIONS`: adding a schema to that set
#: (recognising it, so it is read instead of rejected) must never silently
#: widen this one (resolved before implementation, 2026-09-14, alongside the
#: schema-3 bump -- schema 2 logs always have `r_gripper`, so a gap in one is
#: a bad recording, not an old format, exactly like a gap in the current
#: schema). Extend this set only when a schema that ACTUALLY predates
#: aperture logging is retired from `LOG_SCHEMA_VERSION`/current use -- never
#: just because a schema became legacy.
_MISSING_APERTURE_ELIGIBLE_SCHEMA_VERSIONS = (1,)

#: schema_version stamped on a diagnostic log by `save_invalid_log`.
#: Deliberately not 1, not `LOG_SCHEMA_VERSION`, and never added to
#: `_SUPPORTED_LEGACY_SCHEMA_VERSIONS` -- an invalid log fed back through
#: `validated_samples` (via `schema_version_of`) must always raise
#: `UnsupportedSchemaVersionError`, never quietly qualify for the legacy
#: fallback the way an *absent* schema_version (real schema-1 logs) does.
_INVALID_LOG_SCHEMA_VERSION = 0

#: Tolerance added to the MJCF's own commanded range when validating a
#: recorded aperture, matching `kinematics.within_limits`'s own tolerance:
#: a reading sitting exactly on a joint stop is real, not corrupt.
_GRIPPER_RANGE_TOL_DEG = 0.5

#: `report()`'s explanation of where its "realised" aperture comes from --
#: see the module docstring's "Aperture provenance in report()" section.
REALISED_APERTURE_SOURCE = (
    "measured per-sample r_gripper.present_position, validated; see "
    "realised_aperture_assumed_samples for any indices that instead used "
    "the safe wide-open assumption under allow_missing_aperture"
)

#: `report()`'s explanation of where its "planned" aperture comes from --
#: the same policy `planned_clearance` implements, in one sentence.
PLANNED_APERTURE_POLICY = (
    "per leg: tube uses the worst-case (largest-hand_radius) of that leg's "
    "two commanded ENDPOINT apertures, applied to every interpolated "
    "sample -- exact, because hand_radius is monotonic in aperture. shells "
    "instead interpolates the aperture per sample, in lockstep with the arm "
    "joints, matching what the SDK's own goto(..., MINIMUM_JERK) trajectory "
    "actually commands (rig_motion.sdk_move, primitives._stream) -- every "
    "joint, gripper included, scaled by the same s(t) from the previous "
    "goal -- holding shells at an endpoint understated its worst case "
    "(finger is closest to the table fully SHUT, not at the tube rule's "
    "worst-case OPEN); see planned_clearance's own docstring and "
    "outputs/analysis-2026-09-22-b2s2-finger-clearance-root-cause.md"
)


class ApertureDataError(ValueError):
    """A recorded sample's gripper aperture cannot be trusted: missing (and
    not explicitly allowed to be, for its declared schema_version),
    non-numeric, non-finite, or outside the MJCF's commanded range. Raised
    by `realised_clearance`/`report` rather than treated as an
    assumed-open hand -- see the module docstring's "Missing or invalid
    aperture" section. The message names the offending sample's index and
    recorded time so a bad log can be traced back to where the flight (or
    the telemetry) went wrong.
    """


class SampleTimestampError(ValueError):
    """A schema-3 (current) sample is missing `wall_time_ns`, or it is not a
    finite number. Only `LOG_SCHEMA_VERSION` requires this field -- schema 1
    and 2 logs predate it and are read without it (see the module
    docstring's "Sample wall-clock" section); a gap in a *current* schema
    log is a dropped field or a bad recording, the same stance
    `ApertureDataError` already takes for a missing `r_gripper`. There is no
    rescue flag for this one: a log that cannot be aligned to a server run
    is not usable E1 data regardless of whether its aperture is fine.
    """


class UnsupportedSchemaVersionError(ValueError):
    """`schema_version` is neither `LOG_SCHEMA_VERSION` (the current
    schema) nor a value in `_SUPPORTED_LEGACY_SCHEMA_VERSIONS` (a schema
    this module has explicit, understood fallback logic for). Raised
    immediately, before any per-sample check -- an unfamiliar shape is
    never guessed at, only a recognised one is ever read.
    """


# ── Recording (needs a live arm) ─────────────────────────────────────────────

def record_joint_log(arm, duration_s: float, hz: float = SAMPLE_HZ) -> List[Dict]:
    """Stream `present_position` for every joint in `rig_routes.R_JOINTS`
    -- the seven ARM7 joints AND `r_gripper` -- at `hz`, for `duration_s`
    seconds. Read-only: reads `.present_position` on each joint object and
    nothing else -- it never assigns `.goal_position` or calls
    `turn_on`/`turn_off`, so it cannot command the arm even by accident.
    The gripper is read exactly the same way and at the same instant as the
    other seven joints, not sampled separately or interpolated.

    Every reading is in DEGREES (the SDK's own convention, matching every
    other joint value in this codebase). `t` is seconds elapsed since this
    call started, from `time.monotonic()` -- a monotonic clock, immune to
    wall-clock adjustments, but the spacing between samples is BEST EFFORT:
    the loop compensates for cumulative drift against the nominal period
    (`next_tick`), not for a single slow iteration, so consumers must read
    each sample's own `t` rather than assume uniform `1/hz` spacing.

    Every sample also carries `wall_time_ns` (`time.monotonic_ns()`, read at
    the same instant as the joint poses) -- see the module docstring's
    "Sample wall-clock" section for why this is monotonic_ns, not
    time.time_ns(), and what it lets `link_e1_flight.py` do.
    """
    period = 1.0 / hz
    joints = {name: getattr(arm, name) for name in R.R_JOINTS}
    samples: List[Dict] = []
    t0 = time.monotonic()
    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= duration_s:
            break
        pose = {name: float(j.present_position) for name, j in joints.items()}
        samples.append({"t": elapsed, "wall_time_ns": time.monotonic_ns(),
                        "joints": pose})
        next_tick = t0 + period * (len(samples))
        time.sleep(max(0.0, next_tick - time.monotonic()))
    return samples


def save_log(samples: List[Dict], route: str, scene_path: str, *,
            t0_wall_ns: Optional[int] = None) -> pathlib.Path:
    """`t0_wall_ns` (schema 3+): `time.monotonic_ns()` at the moment
    recording started -- the same clock as every sample's own
    `wall_time_ns`, for a human/audit trail (never used for alignment math;
    the linker aligns on each sample's own `wall_time_ns`). Optional and
    `None` for a log not built from a real `record_joint_log` call (a
    synthetic test log, for instance) -- `validated_samples` never reads
    this field, only per-sample `wall_time_ns`.
    """
    RUNS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = RUNS_DIR / f"route_clearance_{route}_{stamp}.json"
    with open(path, "w") as f:
        json.dump({"route": route, "scene": scene_path, "sample_hz": SAMPLE_HZ,
                  "schema_version": LOG_SCHEMA_VERSION,
                  "t0_wall_ns": t0_wall_ns, "samples": samples},
                 f, indent=1)
    return path


def load_log(path) -> Dict:
    """Read a log exactly as `save_log`/`save_invalid_log` wrote it (or an
    older schema-1 log with no "schema_version" key at all). Returns the
    raw dict -- a caller wanting validated samples should pass this dict's
    "samples" and its `schema_version_of(...)` to
    `validated_samples`/`realised_clearance`/`report`, rather than assume
    the current schema."""
    with open(path) as f:
        return json.load(f)


def schema_version_of(log: Dict) -> int:
    """The schema_version a loaded log dict declares -- absent means 1, the
    only schema that ever predated this field."""
    return log.get("schema_version", 1)


def save_invalid_log(samples: List[Dict], route: str, scene_path: str,
                     reason: str, *,
                     t0_wall_ns: Optional[int] = None) -> pathlib.Path:
    """Persist a recording that failed aperture validation -- never
    silently discarded, so a bad flight can be diagnosed rather than
    re-flown blind. Every sample actually recorded is kept, including the
    offending one, alongside the failure `reason`.

    Marked so it can never pass for a normal, valid E1 log: the filename
    ends `_INVALID.json` (`save_log`'s never does), the JSON body carries
    `"valid": false` and `"error"`, and its `"schema_version"` is
    `_INVALID_LOG_SCHEMA_VERSION` -- a value never in
    `_SUPPORTED_LEGACY_SCHEMA_VERSIONS` and never `LOG_SCHEMA_VERSION`, so
    feeding its own declared schema_version back into
    `validated_samples`/`report` (via `schema_version_of`) raises
    `UnsupportedSchemaVersionError` immediately: it is refused outright,
    not merely re-raising the original `ApertureDataError`, and can never
    be mistaken for a schema-1 log eligible for `allow_missing_aperture`.
    """
    RUNS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = RUNS_DIR / f"route_clearance_{route}_{stamp}_INVALID.json"
    with open(path, "w") as f:
        json.dump({"valid": False, "error": reason, "route": route,
                  "scene": scene_path, "sample_hz": SAMPLE_HZ,
                  "schema_version": _INVALID_LOG_SCHEMA_VERSION,
                  "t0_wall_ns": t0_wall_ns, "samples": samples},
                 f, indent=1)
    return path


# ── Aperture validation (offline -- the same rules for a live or synthetic log) ─

def _validated_gripper_deg(value, index: int, t) -> float:
    """`value` is known present; confirm it is a finite, in-range number of
    degrees and return it as a float, or raise `ApertureDataError` naming
    the sample. Never returns a guessed value -- a present-but-bad reading
    is a different problem from a missing one and is not covered by
    `allow_missing_aperture`."""
    try:
        deg = float(value)
    except (TypeError, ValueError):
        raise ApertureDataError(
            f"sample {index} (t={t}): {GRIPPER_JOINT} is not a number: "
            f"{value!r}")
    if not math.isfinite(deg):
        raise ApertureDataError(
            f"sample {index} (t={t}): {GRIPPER_JOINT}={deg} is not finite")
    lo, hi = sorted((_GRIPPER_OPEN_LIMIT_DEG, _GRIPPER_SHUT_LIMIT_DEG))
    if not (lo - _GRIPPER_RANGE_TOL_DEG <= deg <= hi + _GRIPPER_RANGE_TOL_DEG):
        raise ApertureDataError(
            f"sample {index} (t={t}): {GRIPPER_JOINT}={deg} deg is outside "
            f"the MJCF's commanded range [{lo}, {hi}] deg "
            f"(+/-{_GRIPPER_RANGE_TOL_DEG} tol)")
    return deg


def validated_samples(
    samples: Sequence[Dict], *, schema_version: int = LOG_SCHEMA_VERSION,
    allow_missing_aperture: bool = False,
) -> Tuple[List[Tuple[List[float], float]], List[int]]:
    """[(q7, gripper_deg), ...] for every sample, plus the indices where the
    aperture was ASSUMED rather than measured.

    `schema_version` is the schema `samples` is DECLARED to be shaped as
    (default: the current schema -- the right default for a fresh
    recording, which is always current). Pass the value a loaded log
    itself carries (`schema_version_of(log)`) when reading one from disk.
    Unrecognised versions -- anything but `LOG_SCHEMA_VERSION` or a value
    in `_SUPPORTED_LEGACY_SCHEMA_VERSIONS` -- raise
    `UnsupportedSchemaVersionError` immediately, before any per-sample
    check.

    Raises `ApertureDataError` naming the sample if any reading cannot be
    trusted -- missing, or present but non-numeric / non-finite / out of
    range (this last group always, regardless of `allow_missing_aperture`
    or `schema_version`). A missing reading is rescued by
    `allow_missing_aperture=True` ONLY when `schema_version` is in
    `_MISSING_APERTURE_ELIGIBLE_SCHEMA_VERSIONS` (today: 1) -- a
    NARROWER set than "recognised at all"
    (`_SUPPORTED_LEGACY_SCHEMA_VERSIONS`, today: 1 and 2): schema 2 is a
    recognised past shape but never predated aperture logging, so a
    missing key in a `schema_version == 2` OR `== LOG_SCHEMA_VERSION`
    (current) log always raises, the flag notwithstanding, because neither
    has an excuse for a gap. Never silently substitutes an assumption: the
    only path to one is the explicit flag on schema 1, and even then every
    sample it was used for is returned in the second list so the caller
    can report it rather than lose it.

    On a `schema_version == LOG_SCHEMA_VERSION` (current) log, ALSO raises
    `SampleTimestampError` naming the sample if any `wall_time_ns` is
    missing or non-finite (checked per sample, after that sample's
    aperture -- see the module docstring's "Sample wall-clock" section);
    schema 1 and 2 predate this field and are never checked for it.
    """
    if (schema_version != LOG_SCHEMA_VERSION
            and schema_version not in _SUPPORTED_LEGACY_SCHEMA_VERSIONS):
        raise UnsupportedSchemaVersionError(
            f"schema_version={schema_version!r} is not the current schema "
            f"({LOG_SCHEMA_VERSION}) or a supported legacy schema "
            f"({_SUPPORTED_LEGACY_SCHEMA_VERSIONS}) -- refusing to guess "
            "what this log's shape means.")
    missing_aperture_eligible = (
        schema_version in _MISSING_APERTURE_ELIGIBLE_SCHEMA_VERSIONS)
    requires_wall_time = schema_version == LOG_SCHEMA_VERSION

    out: List[Tuple[List[float], float]] = []
    assumed: List[int] = []
    for i, sample in enumerate(samples):
        joints = sample["joints"]
        q7 = [joints[j] for j in R.ARM7]
        if GRIPPER_JOINT not in joints:
            if not (allow_missing_aperture and missing_aperture_eligible):
                where = sample.get('t')
                if missing_aperture_eligible:
                    detail = (
                        "This sample has no usable clearance answer; pass "
                        "allow_missing_aperture=True to fall back to the "
                        "safe (wide-open) assumption for it explicitly, or "
                        "drop it from the log.")
                elif schema_version == LOG_SCHEMA_VERSION:
                    detail = (
                        f"schema_version={schema_version} is this module's "
                        "CURRENT schema, which always includes r_gripper; "
                        "a missing key here is a dropped field or a bad "
                        "recording, not an old log format -- "
                        "allow_missing_aperture does not apply to it.")
                else:
                    detail = (
                        f"schema_version={schema_version} is a recognised "
                        "past schema that already had aperture logging "
                        "(unlike schema 1); a missing key here is a "
                        "dropped field or a bad recording, not an old log "
                        "format -- allow_missing_aperture does not apply "
                        "to it.")
                raise ApertureDataError(
                    f"sample {i} (t={where}): missing '{GRIPPER_JOINT}' "
                    f"(schema_version={schema_version!r}). {detail}")
            assumed.append(i)
            out.append((q7, None))
        else:
            gripper_deg = _validated_gripper_deg(
                joints[GRIPPER_JOINT], i, sample.get("t"))
            out.append((q7, gripper_deg))

        if requires_wall_time:
            wall_time_ns = sample.get("wall_time_ns")
            if (wall_time_ns is None
                    or not isinstance(wall_time_ns, (int, float))
                    or isinstance(wall_time_ns, bool)
                    or not math.isfinite(wall_time_ns)):
                raise SampleTimestampError(
                    f"sample {i} (t={sample.get('t')}): missing or "
                    f"non-finite 'wall_time_ns' (schema_version="
                    f"{schema_version!r}, current schema requires it -- "
                    "see the module docstring's 'Sample wall-clock' "
                    f"section); got {wall_time_ns!r}")
    return out, assumed


# ── Reporting (pure geometry -- offline, unit-tested on a synthetic log) ────

def _worst_clearance_over_samples(
    q7_and_gripper: Sequence[tuple], scene: SceneModel,
) -> Dict[str, Dict[str, float]]:
    """Per-hand-mode, per-object worst clearance over a sequence of
    (q7, gripper_deg) samples."""
    out: Dict[str, Dict[str, float]] = {hand: {} for hand in HAND_MODES}
    for q7, gripper_deg in q7_and_gripper:
        for hand in HAND_MODES:
            caps = link_capsules(q7, "right", gripper_deg, hand=hand)
            for oid, c in scene.clearances(caps).items():
                worst = out[hand].get(oid)
                if worst is None or c.distance < worst:
                    out[hand][oid] = c.distance
    return out


def realised_clearance(
    samples: List[Dict], scene: SceneModel, *,
    schema_version: int = LOG_SCHEMA_VERSION,
    allow_missing_aperture: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Worst per-link clearance actually reached, from a recorded log, AT
    EACH SAMPLE'S OWN MEASURED APERTURE (`r_gripper.present_position`,
    degrees) -- not assumed-open. This is E1's whole point: the aperture is
    the variable that decides whether the hand is inside the tube at all
    (docs/adr/0003).

    `schema_version` is `samples`'s declared schema -- see
    `validated_samples` for what it gates. Raises `ApertureDataError` or
    `UnsupportedSchemaVersionError` if any sample's aperture cannot be
    trusted; see `validated_samples` and the module docstring's "Missing
    or invalid aperture" section. Use `report()` if you also want to know
    which samples (if any) fell back to the assumed-open case under
    `allow_missing_aperture=True`.
    """
    q7_and_gripper, _assumed = validated_samples(
        samples, schema_version=schema_version,
        allow_missing_aperture=allow_missing_aperture)
    return _worst_clearance_over_samples(q7_and_gripper, scene)


#: Full flown waypoint sequence for each named route, starting pose
#: included -- what a "setup"-style recording actually flies (the whole
#: named route from its own departure posture), as distinct from
#: `FOOTPRINT_LEGS`'s scoped-down subset (the live guard's own footprint,
#: `panel_executor._footprint_refusal` -- never widened here). Each route's
#: start pose is the posture the notebook and `rig_routes` already document
#: it departing from: HOME for the two routes that leave the rail pocket,
#: REST/PRESENT for the ones that start already on the board or at the side
#: pose. See `full_route_poses`.
_FULL_ROUTE_START_POSE: Dict[str, Dict[str, float]] = {
    "PLACE_ROUTE": R.HOME,
    "RAISE_TO_SIDE": R.HOME,
    "STOW_ROUTE": R.REST,
    "STOW_FROM_SIDE": R.PRESENT,
    "LOWER_TO_REST": R.PRESENT,
    "LIFT_TO_PRESENT": R.REST,
}

#: The named `Waypoint` sequence each route actually commands after its
#: start pose (`_FULL_ROUTE_START_POSE`) -- the full sequence, not
#: `FOOTPRINT_LEGS`'s subset.
_FULL_ROUTE_WAYPOINTS: Dict[str, Tuple] = {
    "PLACE_ROUTE": R.PLACE_ROUTE,
    "RAISE_TO_SIDE": R.RAISE_TO_SIDE,
    "STOW_ROUTE": R.STOW_ROUTE,
    "STOW_FROM_SIDE": R.STOW_FROM_SIDE,
    "LOWER_TO_REST": R.LOWER_TO_REST,
    "LIFT_TO_PRESENT": R.LIFT_TO_PRESENT,
}


def full_route_poses(route: str) -> Tuple[Dict[str, float], ...]:
    """The whole commanded pose sequence for `route`, start pose included --
    what a recording that flew the entire named route (not just
    `FOOTPRINT_LEGS`'s guard-scoped subset) actually commands. Raises
    `KeyError` for a route with no measured full sequence (WAVE, POINT_*),
    same as indexing `FOOTPRINT_LEGS` directly would.
    """
    return (_FULL_ROUTE_START_POSE[route],) + tuple(
        w.pose for w in _FULL_ROUTE_WAYPOINTS[route])


def resolve_full_route_flag(route: str, report_scope: str) -> bool:
    """Map `main()`'s `--report-scope` CLI choice to `report()`'s
    `full_route` bool for `route`, refusing a `"full"` request `route` has
    no `full_route_poses()` sequence for instead of letting it fail later
    with a bare `KeyError` (issue #139).

    Raises `ValueError` (never `KeyError`) naming `route` and the supported
    set if `report_scope == "full"` but `route not in _FULL_ROUTE_WAYPOINTS`
    -- today that is WAVE/POINT_* (see `full_route_poses`'s own docstring).
    `main()`'s own `--route` choices are `sorted(R.FOOTPRINT_LEGS)`, which
    happens to equal `_FULL_ROUTE_WAYPOINTS`'s keys today, so the CLI itself
    cannot currently reach this refusal -- this function checks membership
    directly rather than assume that will always hold (and is what the
    refusal's own unit tests call directly, since argparse's `choices`
    restriction on `--route` can't be exercised with an out-of-choices
    route string). Called by `main()` before recording starts, so a bad
    `--report-scope full --route <no-full-sequence>` combination is refused
    up front, not after a flight is thrown away.
    """
    if report_scope == "full" and route not in _FULL_ROUTE_WAYPOINTS:
        raise ValueError(
            f"--report-scope full is not supported for route {route!r} -- "
            "full_route_poses() has no full-route sequence for it (only "
            f"{sorted(_FULL_ROUTE_WAYPOINTS)} do). Use --report-scope "
            "footprint (the default) instead.")
    return report_scope == "full"


def _lerp(a: float, b: float, i: int, steps: int) -> float:
    """`a` to `b` at step `i` of `steps` (endpoints included, `i` in
    `range(steps)`) -- the exact fraction `joint_path` uses for the arm
    joints, so a gripper value computed this way stays in lockstep with
    them, matching what the SDK's own `goto(..., MINIMUM_JERK)` trajectory
    actually commands (`rig_motion.sdk_move`, `primitives._stream`) --
    every joint in a pose, gripper included, scaled by the same fraction
    `t` from the previous goal each tick."""
    return a + (i / (steps - 1)) * (b - a)


def _tube_leg_samples(waypoints: Sequence[Dict[str, float]], steps: int):
    """(q7, gripper_deg) samples along `waypoints`'s legs under tube's own
    worst-ENDPOINT aperture rule -- see `planned_clearance`'s docstring."""
    for a, b in zip(waypoints, waypoints[1:]):
        qa = [a[j] for j in R.ARM7]
        qb = [b[j] for j in R.ARM7]
        gripper_deg = max(a["r_gripper"], b["r_gripper"], key=hand_radius)
        for q in joint_path(qa, qb, steps=steps):
            yield q, gripper_deg


def _shells_leg_samples(waypoints: Sequence[Dict[str, float]], steps: int):
    """(q7, gripper_deg) samples along `waypoints`'s legs with the aperture
    interpolated per sample, in lockstep with the arm joints -- see
    `planned_clearance`'s docstring."""
    for a, b in zip(waypoints, waypoints[1:]):
        qa = [a[j] for j in R.ARM7]
        qb = [b[j] for j in R.ARM7]
        path = joint_path(qa, qb, steps=steps)
        for i, q in enumerate(path):
            gripper_deg = _lerp(a["r_gripper"], b["r_gripper"], i, steps)
            yield q, gripper_deg


_LEG_SAMPLES_BY_HAND = {"tube": _tube_leg_samples, "shells": _shells_leg_samples}

#: The capsule names `link_capsules(hand=...)` can ever return, for each
#: hand model -- see that function's own docstring. Used by
#: `planned_clearance_by_link` to raise on an unrecognised `link` instead
#: of silently returning `{}`, which used to be indistinguishable from "this
#: route has no FOOTPRINT_LEGS entry" (issue #137, priority 2).
_CAPSULE_NAMES_BY_HAND: Dict[str, frozenset] = {
    "tube": frozenset({"upper_arm", "forearm", "hand"}),
    "shells": frozenset({"upper_arm", "forearm", "thumb", "thumb_pad",
                         "finger", "wrist_ball"}),
}


def _resolve_waypoints(
    route: str, waypoints: Optional[Sequence[Dict[str, float]]],
) -> Optional[Sequence[Dict[str, float]]]:
    if waypoints is not None:
        return waypoints
    return R.FOOTPRINT_LEGS.get(route)


def planned_clearance_by_link(
    route: str, n_samples: int, scene: SceneModel, link: str, *,
    hand: str = "shells", waypoints: Optional[Sequence[Dict[str, float]]] = None,
) -> Dict[str, float]:
    """Like `planned_clearance`, but the worst-case is over a SINGLE named
    link's own capsule (e.g. ``"finger"``) rather than the worst over every
    capsule of `hand` together. Reusable per-link breakdown for exactly the
    kind of question E1 stage-2 sessions needed answered ad hoc, per
    session, in throwaway evidence-dir tooling -- this is the one, shared,
    tested version, meant to replace those (see
    `outputs/analysis-2026-09-22-b2s2-finger-clearance-root-cause.md`,
    section 9). `hand="tube"` is accepted but not very meaningful: tube has
    only one hand capsule (``"hand"``), so filtering to a link name there
    either returns `planned_clearance`'s own tube numbers unchanged (that
    link) or raises (any other name -- see below).

    Raises `ValueError` if `hand` is not `"tube"`/`"shells"`, or if `link`
    is not one of the capsule names `link_capsules(hand=hand)` can ever
    produce (`_CAPSULE_NAMES_BY_HAND`) -- checked up front, before
    resolving `waypoints`, so a typo'd link name (e.g. ``"fingers"``) is
    never confused with "this route has no FOOTPRINT_LEGS entry", which is
    the only remaining case `{}` means (issue #137, priority 2).
    """
    valid_links = _CAPSULE_NAMES_BY_HAND.get(hand)
    if valid_links is None:
        raise ValueError(f"hand must be 'tube' or 'shells', got {hand!r}")
    if link not in valid_links:
        raise ValueError(
            f"{link!r} is not a capsule name link_capsules(hand={hand!r}) "
            f"ever returns -- expected one of {sorted(valid_links)}")
    resolved = _resolve_waypoints(route, waypoints)
    if resolved is None:
        return {}
    steps = max(2, n_samples)
    worst: Dict[str, float] = {}
    for q7, gripper_deg in _LEG_SAMPLES_BY_HAND[hand](resolved, steps):
        caps = [c for c in link_capsules(q7, "right", gripper_deg, hand=hand)
                if c[0] == link]
        for oid, c in scene.clearances(caps).items():
            if worst.get(oid) is None or c.distance < worst[oid]:
                worst[oid] = c.distance
    return worst


def planned_clearance(
    route: str, n_samples: int, scene: SceneModel, *,
    waypoints: Optional[Sequence[Dict[str, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Worst per-link clearance the COMMANDED path implies, sampled
    `n_samples` times per leg of `waypoints` (default: `route`'s own
    `FOOTPRINT_LEGS` entry) -- the same joint-space line `fly_route`
    actually commands. Pass `waypoints=full_route_poses(route)` to compare
    against the whole named route instead of the guard's footprint-scoped
    subset (see the module docstring's "Route scope in report()"). Routes
    with no `FOOTPRINT_LEGS` entry and no explicit `waypoints` (WAVE, POINT)
    return empty per-hand dicts: there is nothing here to compare a
    realised log against for them.

    The two hand models use DIFFERENT aperture policies along each leg,
    because they disagree on which endpoint is worse:

    * ``tube``: the worst-case (largest-`hand_radius`) of the leg's two
      commanded ENDPOINT apertures, held constant across every interpolated
      sample on the leg. `hand_radius` is monotonic in how open the tube
      is, so its own worst endpoint really is the worst point anywhere on
      the leg -- holding it constant is exact, not an approximation.
    * ``shells``: the aperture is interpolated per sample, in lockstep with
      the arm joints (the same fraction the SDK's own `goto(...,
      MINIMUM_JERK)` trajectory drives every commanded joint, gripper
      included, by -- `rig_motion.sdk_move`, `primitives._stream`). Shells'
      `finger` capsule
      swings about the thumb's X axis as the gripper opens and closes, and
      is NOT monotonically closer to an object at either extreme -- on
      `LOWER_TO_REST`, `finger` is closest to the table at the fully SHUT
      aperture, the opposite of what `hand_radius` calls worst-case. Only
      `finger` depends on aperture at all (`thumb`, `thumb_pad` and
      `wrist_ball` ride the thumb frame); holding shells at the tube rule's
      endpoint silently skips every SHUT sample in between, which is
      exactly the bug root-caused 2026-09-22 (see
      `outputs/analysis-2026-09-22-b2s2-finger-clearance-root-cause.md`).
    """
    resolved = _resolve_waypoints(route, waypoints)
    if resolved is None:
        return {hand: {} for hand in HAND_MODES}
    steps = max(2, n_samples)

    out: Dict[str, Dict[str, float]] = {}
    for hand in HAND_MODES:
        worst: Dict[str, float] = {}
        for q7, gripper_deg in _LEG_SAMPLES_BY_HAND[hand](resolved, steps):
            caps = link_capsules(q7, "right", gripper_deg, hand=hand)
            for oid, c in scene.clearances(caps).items():
                if worst.get(oid) is None or c.distance < worst[oid]:
                    worst[oid] = c.distance
        out[hand] = worst
    return out


def report(
    samples: List[Dict], route: str, scene_path: str, *,
    schema_version: int = LOG_SCHEMA_VERSION,
    allow_missing_aperture: bool = False,
    full_route: bool = False,
) -> Dict:
    """The full report: realised vs planned, both hand models, for `route`
    against `scene_path`'s own object poses (as loaded -- this script has no
    live scene link, only the SDK's joint telemetry; a re-flight that needs
    live object poses should record them separately).

    `schema_version` is `samples`'s declared schema (default: current --
    right for a fresh recording; pass a loaded log's own
    `schema_version_of(log)` when reading one from disk). `realised` is
    computed at each sample's own measured aperture. Raises
    `ApertureDataError` or `UnsupportedSchemaVersionError` (see the module
    docstring) unless `allow_missing_aperture=True` is passed for a log
    declared as a supported legacy schema -- in which case
    `realised_aperture_assumed_samples` names every sample index that fell
    back to the wide-open assumption, so the fallback is visible in the
    report rather than silent. An empty list there means every sample's
    aperture was actually measured. `realised_aperture_source` and
    `planned_aperture_policy` name, in plain text, where each half's
    aperture actually comes from -- see the module docstring's "Aperture
    provenance" section.

    `full_route=True` compares `realised` against `full_route_poses(route)`
    (the whole named route from its own departure posture) instead of
    `FOOTPRINT_LEGS`'s guard-scoped subset -- pass this when `samples` is a
    recording of the entire route (e.g. a "setup" flight from HOME), not
    just its footprint-scoped leg. See the module docstring's "Route scope
    in report()" section, including why this is a Python-API-only
    parameter with no CLI flag. `planned_route_scope` in the result says
    which one was actually used.
    """
    scene = SceneModel.from_yaml(scene_path)
    q7_and_gripper, assumed = validated_samples(
        samples, schema_version=schema_version,
        allow_missing_aperture=allow_missing_aperture)
    waypoints = full_route_poses(route) if full_route else None
    return {
        "route": route,
        "scene": scene_path,
        "n_samples": len(samples),
        "realised": _worst_clearance_over_samples(q7_and_gripper, scene),
        "realised_aperture_assumed_samples": assumed,
        "realised_aperture_source": REALISED_APERTURE_SOURCE,
        "planned": planned_clearance(route, len(samples), scene,
                                      waypoints=waypoints),
        "planned_aperture_policy": PLANNED_APERTURE_POLICY,
        "planned_route_scope": (
            "full_route_poses(route) -- the whole named route from its own "
            "departure posture" if full_route else
            "FOOTPRINT_LEGS[route] -- the live guard's own footprint-scoped "
            "subset"),
    }


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    if os.environ.get("REACHY_SIM_RECORD_CLEARANCE") != "1":
        print("Refusing to run: set REACHY_SIM_RECORD_CLEARANCE=1 to record.\n"
              "This script only STREAMS present_position -- it never "
              "commands the arm -- but starting a recording an operator did "
              "not ask for is still a surprise worth an explicit opt-in.")
        sys.exit(1)
    if ReachySDK is None:
        print("FAIL: reachy-sdk not installed. Run: pip install reachy-sdk==0.7.0")
        sys.exit(1)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("REACHY_IP", "localhost"))
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--route", required=True, choices=sorted(R.FOOTPRINT_LEGS))
    parser.add_argument(
        "--report-scope", choices=("footprint", "full"), default="footprint",
        help="which planned-route comparison the printed report uses: "
             "'footprint' (default, preserves prior stdout) compares "
             "against FOOTPRINT_LEGS[route], the live guard's own "
             "scoped subset; 'full' compares against "
             "full_route_poses(route), the whole named route from its own "
             "departure posture -- refused for a route with no full "
             "sequence (WAVE, POINT_*). See report()'s full_route "
             "parameter and resolve_full_route_flag.")
    parser.add_argument("--scene", default=str(
        _HERE.parent / "scenes" / "FWDCenterLabSivaPool.yaml"))
    parser.add_argument("--duration", type=float, default=15.0,
                        help="seconds to record; fly the route by hand during "
                             "this window")
    parser.add_argument(
        "--record-root", default=os.environ.get("REACHY_SIM_RECORD"),
        help="the native server's own --record directory (E1 readiness "
             "work item 2). Defaulting from REACHY_SIM_RECORD is a path "
             "HINT ONLY -- every fact about the directory it names (that "
             "it is live, that its scene matches --scene, that its "
             "backend is mujoco-remote) is verified by the identity "
             "check below, never assumed from the env var.")
    parser.add_argument(
        "--status-url", default="http://localhost:8080/status",
        help="the container camera server's /status endpoint (issue #40), "
             "used to confirm the bridge backend is mujoco-remote")
    args = parser.parse_args()

    try:
        full_route = resolve_full_route_flag(args.route, args.report_scope)
    except ValueError as exc:
        print(f"FAIL: {exc}")
        sys.exit(2)

    if not args.record_root:
        print("FAIL: --record-root is required (or set REACHY_SIM_RECORD) "
              "-- the native server's own --record directory, so this "
              "script can verify it is talking to the live physics rather "
              "than a claimed backend.")
        sys.exit(2)

    # Constructed lazily, on first use inside _read_sdk_joints -- N1
    # (2026-09-15 matrix readiness): verify_simulator_identity only calls
    # read_sdk_joints after the loopback and run-dir checks already pass,
    # so a mistyped --host (or a stray REACHY_IP) never opens a gRPC
    # channel before those refusals fire.
    _reachy_box: Dict[str, object] = {}

    def _reachy() -> "ReachySDK":
        if "sdk" not in _reachy_box:
            _reachy_box["sdk"] = ReachySDK(host=args.host, sdk_port=args.port)
        return _reachy_box["sdk"]

    def _read_sdk_joints() -> Dict[str, float]:
        reachy = _reachy()
        return {name: float(getattr(reachy.r_arm, name).present_position)
                for name in R.R_JOINTS}

    # Verified simulator identity (E1 readiness work item 2): established
    # BEFORE the recording loop connects, and refused outright -- not
    # downgraded to a warning -- on any failure. See e1_identity.py's
    # module docstring for exactly what this does and does not prove.
    identity = e1_identity.verify_simulator_identity(
        host=args.host, port=args.port, scene_path=args.scene,
        record_root=args.record_root, status_url=args.status_url,
        read_sdk_joints=_read_sdk_joints)
    if not identity.ok:
        print("FAIL: simulator identity could not be established -- "
              "refusing to record. Reasons:")
        for reason in identity.reasons:
            print(f"  - {reason}")
        sys.exit(2)
    print(f"Simulator identity verified: run_dir={identity.run_dir}")

    reachy = _reachy()

    t0_wall_ns = time.monotonic_ns()
    print(f"Recording {args.route} for {args.duration:.1f}s at {SAMPLE_HZ:.0f} Hz "
         "-- fly the route now. This only reads present_position.")
    samples = record_joint_log(reachy.r_arm, args.duration)

    # Fail fast, before saving as a normal log or reporting, if the
    # recording itself came back with an untrustworthy aperture on any
    # sample (a disconnected joint, a stale SDK read) -- proving the log
    # actually meets E1's requirement rather than trusting that
    # record_joint_log worked. The recording is not thrown away: it is
    # preserved as a marked-invalid diagnostic artifact (see
    # save_invalid_log) so the flight can be inspected instead of re-flown
    # blind, and it can never be picked up as valid E1 data.
    try:
        validated_samples(samples, allow_missing_aperture=False)
    except ApertureDataError as exc:
        invalid_path = save_invalid_log(samples, args.route, args.scene, str(exc),
                                        t0_wall_ns=t0_wall_ns)
        print(f"FAIL: recorded log has an unusable gripper aperture: {exc}\n"
              f"Saved {len(samples)} samples as INVALID to {invalid_path} "
              "for diagnosis -- not a usable E1 log. E1 needs every "
              "sample's real aperture, not an assumption. Check the SDK "
              "connection to r_gripper and re-fly.")
        sys.exit(1)

    path = save_log(samples, args.route, args.scene, t0_wall_ns=t0_wall_ns)
    print(f"Saved {len(samples)} samples to {path}")

    # Automatic artefact linking (E1 readiness work item 3b): write the
    # sidecar while the run directory is still fresh, then refuse to call
    # this flight usable E1 data if the board object was not where the
    # YAML says it should be -- both files stay on disk either way (the
    # flight is diagnosable), but exit 3 marks it distinctly from a normal
    # success (exit 0) or an aperture failure (exit 1).
    sidecar = link_e1_flight.build_base_sidecar(
        log_path=str(path), samples=samples, scene_path=args.scene,
        identity_check=identity, run_dir=identity.run_dir)
    sidecar_path = link_e1_flight.write_sidecar(str(path), sidecar)
    print(f"Saved link sidecar to {sidecar_path}")
    settled = sidecar["settled_pose_check"]
    if not all(v.get("ok") for v in settled.values()):
        print(f"FAIL: settled_pose_check found a board object away from "
              f"its YAML pose -- see {sidecar_path}. The log and sidecar "
              "are both saved for diagnosis; this is not usable E1 data.")
        for oid, check in settled.items():
            if not check.get("ok"):
                print(f"  - {oid}: {check}")
        sys.exit(3)

    result = report(samples, args.route, args.scene, full_route=full_route)
    print(f"Report scope: {result['planned_route_scope']}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
