# Evidence for the simulator-E1 readiness PR (2026-09-14)

Supports: `docs/adr/0003-arm-geometry-and-the-hand-tube.md`'s "E1
(simulator)" entry; implements
`outputs/assignment-2026-09-14-sim-e1-readiness.md`. Offline throughout —
no server started, no SDK connection, no motion — see the handoff for the
full commit-by-commit evidence.

## What each new script is

| script | what it does |
|---|---|
| `scripts/make_e1_boards.py` | Deterministic generator for the six board scene YAMLs under `scenes/e1_boards/`. Each `extends: ../FWDCenterLabSivaPool.yaml` and overrides only the named object's `pose.position` (via `SceneModel.rest_point`), matching `tests/unit/test_footprint_boards.py`'s in-memory boards exactly. Re-running it reproduces the committed files byte-for-byte. |
| `scripts/e1_identity.py` | `verify_simulator_identity()` — called by the recorder before it records. Verifies loopback host, exactly one live physics run directory, round-by-round joint/timing correlation against that directory's own advancing stream (not a single cached snapshot), scene-hash identity (including the `extends` chain), and the container bridge's `/status`. Refuses (does not warn) on any failure. See its own module docstring for exactly what this does and does not prove — the real robot's SDK protocol carries no simulator-identity field, so this is a correlation/ambiguity-refusal check, not a cryptographic proof. |
| `scripts/link_e1_flight.py` | Builds and refreshes `<log>.link.json`: identity, manifest, scene chain, first/last-sample object poses, a 5 mm settled-pose check, displacement, and (when a run directory is available) the alignment block — per-sample server-state alignment (nearest `wall_time_ns`, refined by 8-joint residual within ±60 ms), contacts in that window, and clearance recomputed on the server's own `qpos`, both hand models. |
| `native_mujoco/contact_accumulator.py` | `ContactAccumulator` — folds arm-link/object contacts across the physics steps between two state pushes (gated behind `--record`). `add_step_from_data()` is the production path the server calls: it filters `data.contact` by geom/body ID before any name lookup or force computation (see "Contact-accumulator overhead" below for why). |

`scripts/measure_route_clearance.py` gained: the identity check ahead of
`record_joint_log` (`--record-root`/`--status-url`, exits 2 before
recording on any identity failure); schema 3 (`wall_time_ns` per sample,
`t0_wall_ns` on the header — schema 2 stays a recognised-but-strict legacy
shape, never eligible for the missing-aperture rescue, which stays
schema-1-only); and the sidecar write + settled-pose exit-3 check right
after `save_log`.

## How to run the offline tests

    PYTHONPATH=src:native_mujoco:scripts python3 -m pytest \
        tests/unit/test_e1_boards.py \
        tests/unit/test_e1_identity.py \
        tests/unit/test_link_e1_flight.py \
        tests/unit/test_contact_accumulator.py \
        tests/unit/test_protocol.py \
        tests/unit/test_mujoco_remote_backend.py \
        tests/unit/test_measure_route_clearance.py \
        tests/unit/test_footprint_boards.py -q

Each of the touched files also passes run alone (no conftest.py in this
repo; sys.path leakage between files in a full run can hide a file that
fails in isolation — see `PYTHONPATH=src:native_mujoco python3 -m pytest
tests/unit/<file>.py -q`).

## Contact-accumulator overhead (measured, offline)

`native_mujoco.contact_accumulator`'s own module docstring carries the same
numbers; repeated here for the review record. Measured on a compiled
`scenes/e1_boards/B2_foam_r3c3.yaml`, `mj_forward` only (never `mj_step`),
N=2000-3000 calls:

| pose | `ncon` | `contact_records()` + `add_step()` | `add_step_from_data()` |
|---|---|---|---|
| home keyframe (arm at rest, object on its board pose) | 125 | ~824 µs/call (**41%** of the 2000 µs / 500 Hz step budget) | ~113 µs/call (**5.6%**) |
| object moved onto `r_finger_col` (deep penetration) | 108 | ~245–258 µs/call (13%) | ~117 µs/call (5.9%) |

This robot model has substantial self-contact even at rest (125 simultaneous
MuJoCo contacts, none of them arm/object) — `contact_records()`'s
`mj_id2name`(×4)/`mj_contactForce` loop pays for every one of them
regardless of whether any involves the arm or a tracked object.
`add_step_from_data()` filters `data.contact` by geom/body **ID** (cached
once per model) before paying that cost, a measured 7.3× reduction. The
server (`native_mujoco/server.py`) calls `add_step_from_data()`, gated
behind `--record`; `tests/unit/test_contact_accumulator.py`'s
`TestAddStepFromDataMatchesAddStep` proves the two paths produce identical
results on the same model/data.

## Synchronisation limits

(From the assignment, verbatim — design the linker and any Δ_sim reading
around these.)

1. **Two clocks, one host.** Recorder samples: `t` (monotonic elapsed) and,
   after this PR, `wall_time_ns`. Server states: `wall_time_ns` and
   `sim_time_s`. Both processes run on the host, so wall time is
   comparable; expect tens of ms of skew from gRPC + websocket hops.
2. **The SDK reading lags the physics.** `present_position` is the bridge's
   copy of the last 50 Hz state it received (`_ingest_state`), so a recorder
   sample describes the physics 0–20 ms ago plus transport latency. The
   joint-vector refinement in the linker corrects the alignment; it cannot
   recover a pose the stream never carried.
3. **Rates.** Physics 500 Hz; state stream and server recorder 50 Hz;
   recorder 20 Hz best-effort. A contact shorter than 20 ms is **never
   missed** (the accumulator folds every physics step) but is placed at
   ±20 ms; the recorder's clearance is evaluated only at its own samples,
   so "clearance at the contact instant" must come from the linker's
   server-side recomputation, not from the nearest recorder sample.
4. **Realised clearance is a sampled minimum.** Between samples the arm can
   be closer than any sample shows; the server-side recomputation at 50 Hz
   tightens this to 20 ms but does not eliminate it. Report both.
5. **Object poses during a contact** are the moved poses; the recorder's
   clearance is against the YAML pose (where the object *was*), which is
   the right question for "did the path enter the object's space" — the
   linker reports displacement separately and never rewrites the YAML pose.

**Also resolved during implementation, not in the original assignment
text**: `native_mujoco.protocol.State.wall_time_ns` is `time.monotonic_ns()`
(`protocol.py`'s `_now_ns`, never overridden with real wall-clock time in
`server.py`'s `_build_state`) despite its name — confirmed by
`mujoco_remote_backend.py` and `scripts/benchmark.py`, both of which already
diff it against `time.monotonic_ns()` elsewhere in this codebase. The
recorder's own new `wall_time_ns` field is therefore also
`time.monotonic_ns()`, not `time.time_ns()`, so the two are the SAME clock
and directly comparable — this holds because both
`scripts/measure_route_clearance.py` and `native_mujoco/server.py` are
host-native processes (never inside the Docker container), unlike the
existing container-crossing comparisons elsewhere in this codebase. See
`scripts/measure_route_clearance.py`'s "Sample wall-clock" docstring
section for the full evidence trail.

## Operator checklist

`outputs/e1-operator-checklist-2026-09-14.md` — not run by this PR. The
pilot flight it describes needs a separate, written approval (D1) and is
not part of this assignment.
