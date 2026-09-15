"""E1 readiness (assignment 2026-09-14, work item 3c): offline linker.

Produces / refreshes `<log>.link.json` (work item 3b's sidecar) from a
recorder log (`scripts/measure_route_clearance.py`, schema 3) plus the
native server's own run directory -- so "reproduce this flight's clearance
calculation" needs only the log, the run directory (or its retained
`manifest.json`/`states.jsonl`), and the board YAML chain by sha. No
ledger, no running server.

Two halves:

  * `build_base_sidecar()` -- called by `measure_route_clearance.main()`
    RIGHT AFTER `save_log`, while the identity check and run directory are
    still fresh: identity, manifest, scene chain, first/last-sample object
    poses, `settled_pose_check` (5 mm xy/z against the board's own
    overridden poses, read from the LEAF scene file -- not the whole
    resolved scene), displacement, `no_reshape_asserted`.
  * `align_and_recompute()` -- the offline alignment block: for every
    recorder sample, the server state chosen by (i) nearest `wall_time_ns`
    then (ii) refined to the state within +/-60 ms whose 8-joint vector is
    nearest, and clearance recomputed on the server's own `qpos` (both
    hand models) -- so a contact instant is judged on the physics' joints,
    not the 20 Hz SDK reading. Runnable standalone, with no live server,
    given only the log and a run directory.

    Contacts (work item 4) are NOT read only from the aligned states --
    the server pushes a state (and drains its contact accumulator) at
    50 Hz while a recorder sample lands at ~20 Hz, so most states are
    never any sample's nearest match. Every state's own `contacts` field
    is the accumulator's drain since the PREVIOUS push -- a contiguous,
    non-overlapping window (see `native_mujoco/contact_accumulator.py`) --
    so the union of `contacts` from EVERY state whose `sim_step` falls
    between the first and last ALIGNED sample's state (inclusive) is the
    complete, non-duplicated contact record for the flight, independent of
    which states the 20 Hz samples happened to land on. A state missing
    the `contacts` key entirely (a run recorded before work item 4) is
    reported via `contacts_recorded: False` rather than folded into a
    silent zero; a `contacts` field that is present but malformed raises
    `ContactEvidenceError` rather than being silently dropped.

Alignment needs `wall_time_ns` on both sides to mean the same clock -- see
`measure_route_clearance.py`'s "Sample wall-clock" docstring section: both
are `time.monotonic_ns()`, and both this module and the native server are
host-native processes, so they are directly comparable.

CLI: python3 scripts/link_e1_flight.py <log_path> [--run-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE.parent / "native_mujoco"))

from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.motion.kinematics import link_capsules  # noqa: E402
from reachy_ai.scene.awareness import SceneModel  # noqa: E402

#: Alignment refinement window -- a sample's chosen state must be within
#: this of the sample's own wall_time_ns by nearest-neighbour BEFORE the
#: joint-vector refinement narrows among candidates inside it.
_ALIGN_WINDOW_S = 0.06

#: Settled-pose tolerance (E1 plan rev 2, section 4) -- 5 mm in xy and z.
_SETTLED_TOL_XY_M = 0.005
_SETTLED_TOL_Z_M = 0.005

HAND_MODES = ("tube", "shells")


class ContactEvidenceError(RuntimeError):
    """A state's `contacts` field is present but malformed -- raised
    instead of silently treating the state as contact-free, since that
    would make a real contact invisible to the checklist's STOP rule."""


def read_states(run_dir: str) -> List[dict]:
    """Every line of `<run_dir>/states.jsonl`, parsed."""
    path = pathlib.Path(run_dir) / "states.jsonl"
    states = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                states.append(json.loads(line))
    return states


def _server_joint_degrees(state: dict) -> Dict[str, float]:
    return {
        j.get("name"): math.degrees(j.get("position_rad", 0.0))
        for j in state.get("joints", [])
        if j.get("name") in R.R_JOINTS
    }


def _validated_contacts_field(state: dict) -> Optional[List[dict]]:
    """The state's own `contacts` list, or `None` if the key is absent
    entirely (a run recorded before work item 4, or with contact
    tracking off -- distinct from a state that HAS the key with an empty
    list, which means the accumulator genuinely saw nothing). Raises
    `ContactEvidenceError` if the key is present but not a list of dicts
    each naming `arm_geom`/`object_id` -- a malformed entry must fail
    loudly, never be read as "no contacts"."""
    if "contacts" not in state:
        return None
    contacts = state["contacts"]
    if not isinstance(contacts, list):
        raise ContactEvidenceError(
            f"state sim_step={state.get('sim_step')!r}: 'contacts' is "
            f"{type(contacts).__name__}, expected a list")
    for c in contacts:
        if not isinstance(c, dict) or "arm_geom" not in c or "object_id" not in c:
            raise ContactEvidenceError(
                f"state sim_step={state.get('sim_step')!r}: malformed "
                f"contact entry {c!r}")
    return contacts


def contacts_in_sim_step_range(
    states: Sequence[dict], lo: int, hi: int,
    contacts_by_sim_step: Optional[Dict[int, List[dict]]] = None,
) -> Tuple[List[dict], bool]:
    """Union of contacts for every state whose `sim_step` falls in
    `[lo, hi]` inclusive (not just the states an individual recorder
    sample happened to align to -- see `align_and_recompute`'s
    docstring). `contacts_by_sim_step`, when given, overrides each
    state's own `contacts` field (keyed by `sim_step`) rather than
    reading it from the state -- for a caller with its own window
    bookkeeping; still applied across the full range, not just aligned
    states. Returns `(contacts, contacts_recorded)`: `contacts_recorded`
    is `False` only when NOT ONE state in range carries evidence (no
    `contacts` key anywhere, and no override given) -- callers must not
    conflate that with a genuine zero-contact flight."""
    out: List[dict] = []
    contacts_recorded = False
    for state in states:
        sim_step = state.get("sim_step")
        if sim_step is None or not (lo <= sim_step <= hi):
            continue
        if contacts_by_sim_step is not None:
            out.extend(contacts_by_sim_step.get(sim_step, []))
            contacts_recorded = True
            continue
        contacts = _validated_contacts_field(state)
        if contacts is None:
            continue
        contacts_recorded = True
        out.extend(contacts)
    return out, contacts_recorded


def _joint_residual_deg(sample_joints_deg: Dict[str, float],
                        server_joints_deg: Dict[str, float]) -> float:
    return math.sqrt(sum(
        (server_joints_deg.get(n, 0.0) - sample_joints_deg.get(n, 0.0)) ** 2
        for n in R.R_JOINTS))


def align_sample(sample: dict, states: Sequence[dict],
                 window_s: float = _ALIGN_WINDOW_S) -> Tuple[dict, float, float]:
    """(chosen_state, wall_offset_s, joint_residual_deg) for one recorder
    sample: nearest by `wall_time_ns`, then refined to the closest 8-joint
    vector among every state within `window_s` of that nearest state's own
    wall_time_ns (so a wall-clock skew within the window is corrected by
    which state's JOINTS actually match, not just which is chronologically
    closest)."""
    target_wall = sample["wall_time_ns"]
    nearest = min(states, key=lambda s: abs(s.get("wall_time_ns", 0) - target_wall))
    window_ns = window_s * 1e9
    candidates = [
        s for s in states
        if abs(s.get("wall_time_ns", 0) - nearest.get("wall_time_ns", 0)) <= window_ns
    ]
    sample_joints_deg = dict(sample["joints"])
    # Joint residual is the primary key; a tie (e.g. a parked arm, where
    # several states share the same pose) is broken by nearest wall time,
    # so alignment does not collapse distinct-in-time samples onto the
    # same state just because nothing moved.
    best = min(candidates,
              key=lambda s: (
                  _joint_residual_deg(sample_joints_deg, _server_joint_degrees(s)),
                  abs(s.get("wall_time_ns", 0) - target_wall)))
    wall_offset_s = (best.get("wall_time_ns", 0) - target_wall) / 1e9
    residual = _joint_residual_deg(sample_joints_deg, _server_joint_degrees(best))
    return best, wall_offset_s, residual


def align_samples(samples: Sequence[dict], states: Sequence[dict]) -> List[dict]:
    out = []
    for i, sample in enumerate(samples):
        state, wall_offset_s, residual_deg = align_sample(sample, states)
        out.append({
            "sample_index": i,
            "server_seq": state.get("seq"),
            "server_sim_step": state.get("sim_step"),
            "wall_offset_s": wall_offset_s,
            "joint_residual_deg": residual_deg,
        })
    return out


def leaf_board_object_ids(scene_path: str) -> List[str]:
    """Object ids the BOARD FILE ITSELF names -- read from the leaf
    document only (not the resolved/inherited scene), which is exactly
    which objects a board "overrides"."""
    doc = yaml.safe_load(pathlib.Path(scene_path).read_text())
    return [o["id"] for o in (doc.get("objects") or [])
            if isinstance(o, dict) and "id" in o]


def _objects_snapshot(state: dict) -> Dict[str, dict]:
    return {
        o.get("object_id"): {
            "pos_xyz": list(o.get("pos_xyz", [0.0, 0.0, 0.0])),
            "quat_wxyz": list(o.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0])),
            "sim_step": state.get("sim_step"),
            "wall_time_ns": state.get("wall_time_ns"),
        }
        for o in state.get("objects", [])
    }


def settled_pose_check(
    board_object_ids: Sequence[str],
    expected_pose_by_id: Dict[str, Tuple[float, float, float]],
    state: dict,
    tol_xy_m: float = _SETTLED_TOL_XY_M,
    tol_z_m: float = _SETTLED_TOL_Z_M,
) -> Dict[str, dict]:
    """{object_id: {yaml_xyz, stream_xyz, dxy_m, dz_m, ok}} for every board
    object, comparing `state`'s live pose against the board's own resolved
    YAML pose. `ok=False` for an object the state does not carry (never
    silently skipped)."""
    objects_by_id = {o.get("object_id"): o for o in state.get("objects", [])}
    out = {}
    for oid in board_object_ids:
        expected = expected_pose_by_id.get(oid)
        actual = objects_by_id.get(oid)
        if expected is None:
            out[oid] = {"ok": False, "error": f"{oid} has no expected pose"}
            continue
        if actual is None:
            out[oid] = {"ok": False,
                       "error": f"{oid} not present in this server state"}
            continue
        ax, ay, az = actual.get("pos_xyz", [0.0, 0.0, 0.0])
        ex, ey, ez = expected
        dxy = math.hypot(ax - ex, ay - ey)
        dz = abs(az - ez)
        out[oid] = {
            "yaml_xyz": [ex, ey, ez], "stream_xyz": [ax, ay, az],
            "dxy_m": dxy, "dz_m": dz,
            "ok": dxy <= tol_xy_m and dz <= tol_z_m,
        }
    return out


def build_base_sidecar(
    *, log_path: str, samples: Sequence[dict], scene_path: str,
    identity_check, run_dir: str, no_reshape_asserted: bool = True,
) -> dict:
    """The sidecar's non-alignment fields -- built once, right after
    `save_log`, while the identity check and run directory are fresh."""
    states = read_states(run_dir)
    board_object_ids = leaf_board_object_ids(scene_path)
    scene_model = SceneModel.from_yaml(scene_path)
    expected_pose_by_id = {
        oid: scene_model.get(oid).center for oid in board_object_ids
        if oid in scene_model.objects
    }

    first_state, _, _ = align_sample(samples[0], states)
    last_state, _, _ = align_sample(samples[-1], states)
    objects_first = _objects_snapshot(first_state)
    objects_last = _objects_snapshot(last_state)

    displacement_m = {
        oid: math.dist(objects_first[oid]["pos_xyz"], objects_last[oid]["pos_xyz"])
        for oid in set(objects_first) & set(objects_last)
    }

    return {
        "log": pathlib.Path(log_path).name,
        "identity": identity_check.as_dict(),
        "server_run_dir": str(run_dir),
        "manifest": identity_check.manifest,
        "scene": {
            "path": str(scene_path),
            "chain_sha256": identity_check.scene_chain_sha256,
            "board_object_ids": board_object_ids,
        },
        "objects_at_first_sample": objects_first,
        "objects_at_last_sample": objects_last,
        "settled_pose_check": settled_pose_check(
            board_object_ids, expected_pose_by_id, first_state),
        "displacement_m": displacement_m,
        "no_reshape_asserted": no_reshape_asserted,
    }


def sidecar_path_for(log_path: str) -> pathlib.Path:
    return pathlib.Path(log_path).with_suffix(".link.json")


def write_sidecar(log_path: str, sidecar: dict) -> pathlib.Path:
    path = sidecar_path_for(log_path)
    path.write_text(json.dumps(sidecar, indent=2))
    return path


def read_sidecar(log_path: str) -> dict:
    with open(sidecar_path_for(log_path)) as f:
        return json.load(f)


def _clearance_for_state(state: dict, scene_model: SceneModel) -> Dict[str, dict]:
    """Server-side clearance recomputed on THIS state's own qpos, both hand
    models -- the counterpart to the recorder's own aperture-driven
    clearance, but evaluated at 50 Hz server resolution rather than the
    recorder's 20 Hz best-effort sampling."""
    server_deg = _server_joint_degrees(state)
    q7 = [server_deg.get(j, 0.0) for j in R.ARM7]
    gripper_deg = server_deg.get("r_gripper")
    out = {}
    for hand in HAND_MODES:
        caps = link_capsules(q7, "right", gripper_deg, hand=hand)
        out[hand] = {oid: c.distance for oid, c in scene_model.clearances(caps).items()}
    return out


def align_and_recompute(
    samples: Sequence[dict], run_dir: str, scene_path: str,
    contacts_by_sim_step: Optional[Dict[int, List[dict]]] = None,
) -> dict:
    """The alignment block: per-sample alignment and server-side
    clearance recomputed at each aligned state (both hand models), plus
    the FULL contact record for the flight -- every state's own
    `contacts` field (see `native_mujoco.contact_accumulator`) unioned
    across the entire `[first aligned sim_step, last aligned sim_step]`
    range, not just the states individual samples happened to align to
    (the server pushes state at 50 Hz; a recorder sample lands at ~20 Hz,
    so most states are never any sample's nearest match -- see
    `contacts_in_sim_step_range`). `contacts_by_sim_step`, if given,
    overrides each state's own field rather than being read from it, but
    is still applied over the same full range."""
    states = read_states(run_dir)
    scene_model = SceneModel.from_yaml(scene_path)
    alignment = align_samples(samples, states)
    by_sim_step = {s.get("sim_step"): s for s in states}

    clearance_by_sample = []
    for record in alignment:
        sim_step = record["server_sim_step"]
        state = by_sim_step.get(sim_step)
        if state is not None:
            clearance_by_sample.append(_clearance_for_state(state, scene_model))
        else:
            clearance_by_sample.append({})

    aligned_sim_steps = [r["server_sim_step"] for r in alignment
                         if r["server_sim_step"] is not None]
    if aligned_sim_steps:
        contacts_in_window, contacts_recorded = contacts_in_sim_step_range(
            states, min(aligned_sim_steps), max(aligned_sim_steps),
            contacts_by_sim_step=contacts_by_sim_step)
    else:
        contacts_in_window, contacts_recorded = [], False

    return {
        "alignment": alignment,
        "server_side_clearance": clearance_by_sample,
        "contacts": contacts_in_window,
        "contacts_recorded": contacts_recorded,
    }


def link_flight(log_path: str, run_dir: Optional[str] = None) -> pathlib.Path:
    """CLI/library entry: read `<log_path>` (schema 3) and its existing
    sidecar (for `scene.path` and `server_run_dir` when `run_dir` is not
    given -- so this is runnable on retained artefacts with no live
    server), compute the alignment block, and write the refreshed
    sidecar."""
    with open(log_path) as f:
        log = json.load(f)
    sidecar = read_sidecar(log_path)
    resolved_run_dir = run_dir or sidecar.get("server_run_dir")
    scene_path = sidecar.get("scene", {}).get("path") or log.get("scene")
    block = align_and_recompute(log["samples"], resolved_run_dir, scene_path)
    sidecar.update(block)
    return write_sidecar(log_path, sidecar)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path")
    parser.add_argument("--run-dir", default=None,
                        help="override the run directory named in the "
                             "existing sidecar (e.g. retained artefacts "
                             "moved to a new path)")
    args = parser.parse_args()
    path = link_flight(args.log_path, run_dir=args.run_dir)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
