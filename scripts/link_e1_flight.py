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
    so the union of `contacts` from EVERY state whose `wall_time_ns` falls
    in `[first sample, last sample] +/- the alignment window` is the
    complete, non-duplicated contact record for the flight, independent of
    which states the 20 Hz samples happened to land on AND independent of
    the joint-residual refinement that alignment applies on top of the
    wall-time match (blocker B1 fix-up: keying the window on the ALIGNED
    states' `sim_step` -- as the first cut of this file did -- trims up to
    one alignment window off the end whenever the arm is moving, because a
    moving arm's aligned state is systematically earlier than the sample's
    own wall time; see `contacts_in_flight_window`). The window is keyed
    on each state's `seq` (monotonic for the life of the server process),
    not `sim_step` (which restarts at 0 on every scene reset within the
    same recording run dir). A state missing the `contacts` key entirely
    (a run recorded before work item 4) is reported via
    `contacts_recorded: False` rather than folded into a silent zero; a
    `contacts` field that is present but malformed raises
    `ContactEvidenceError` rather than being silently dropped. So does a
    state missing `seq` or `wall_time_ns` (or carrying a non-int value for
    either) -- a run of `.get(..., -1)` defaults used to make a state like
    that silently unplaceable in the window, which is how a real contact
    went missing with no error at all (F7/F9 fix-up, 2026-09-15 review).
    The window's own coverage is also verified, not assumed: the last
    sample's covering push (the first state pushed AT OR AFTER it -- see
    `contacts_in_flight_window`) must actually exist inside the padded
    window, and every state between the window's ends must actually have
    been recorded (`seq` increasing by exactly 1) -- either gap reads as
    `contacts_recorded: False`, never as a false "no contacts" success.

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


def _require_seq(state: dict) -> int:
    """`state['seq']`, validated. Missing or non-int `seq` cannot be placed
    in seq order at all -- silently defaulting it (the pre-F9 behaviour)
    is how a state, and any contact it carries, used to fall out of the
    window with no error. Raises `ContactEvidenceError` instead."""
    if "seq" not in state:
        raise ContactEvidenceError(
            f"state sim_step={state.get('sim_step')!r}: missing 'seq'")
    seq = state["seq"]
    if not isinstance(seq, int) or isinstance(seq, bool):
        raise ContactEvidenceError(
            f"state sim_step={state.get('sim_step')!r}: 'seq' is "
            f"{type(seq).__name__}, expected int")
    return seq


def _require_wall_ns(state: dict) -> int:
    """`state['wall_time_ns']`, validated the same way as `_require_seq`.
    P4 (2026-09-15 review): the pre-fix window filter used
    `s.get('wall_time_ns', -1)`, which silently reads a state missing this
    field as far outside any window -- excluding it, and any contact it
    carries, with `contacts_recorded` still reading `True`. Raises
    instead."""
    if "wall_time_ns" not in state:
        raise ContactEvidenceError(
            f"state seq={state.get('seq')!r}: missing 'wall_time_ns'")
    wall = state["wall_time_ns"]
    if not isinstance(wall, int) or isinstance(wall, bool):
        raise ContactEvidenceError(
            f"state seq={state.get('seq')!r}: 'wall_time_ns' is "
            f"{type(wall).__name__}, expected int")
    return wall


#: Padding applied to the sample-derived wall-time window, in nanoseconds.
#: This is deliberately the SAME constant as `_ALIGN_WINDOW_S`: the
#: alignment refinement already assumes a chosen state can be up to
#: `_ALIGN_WINDOW_S` away (by wall time) from the sample it represents
#: (`align_sample`'s `window_s`), so a contact window narrower than that
#: could exclude physics that alignment itself considers "this sample's".
#: It is also, independently, >= 3x the server's own state-push period
#: (`native_mujoco/server.py`: `_STATE_HZ = 50` -> 20 ms between pushes;
#: `_ALIGN_WINDOW_S` = 60 ms = 3 push periods): every state's `contacts`
#: field is the accumulator's drain since the PREVIOUS push, so the
#: state pushed immediately after `hi` still covers the 20 ms interval
#: ending at its own wall time, which starts strictly before `hi` -- a
#: pad of one push period would already be enough to guarantee no drain
#: is split by the boundary; 60 ms is 3x that margin. Over-inclusion at
#: each end (a state just outside `[first sample, last sample]` whose
#: own drain window doesn't actually overlap it) is the fail-closed
#: direction and is recorded in `contacts_window` for the record.
_WINDOW_PAD_NS = int(_ALIGN_WINDOW_S * 1e9)


def contacts_in_flight_window(
    states: Sequence[dict], first_sample_wall_ns: int, last_sample_wall_ns: int,
) -> Tuple[List[dict], bool, dict]:
    """Union of every state's own `contacts` for states whose
    `wall_time_ns` lies in `[first_sample_wall_ns - pad, last_sample_wall_ns
    + pad]` (`pad` = `_WINDOW_PAD_NS`, see its docstring) -- independent of
    which states the recorder samples happened to align to, and of the
    joint-residual refinement `align_sample` applies on top of wall time
    (see `align_and_recompute`'s docstring for why keying on the ALIGNED
    states' `sim_step` -- the pre-fix behaviour -- drops the tail of a
    moving-arm flight).

    Keyed on `seq` (monotonic for the life of the server process), not
    `sim_step`, which restarts at 0 on every scene reset within the same
    `--record` run dir; a duplicate `seq` inside the window is impossible
    for a real `states.jsonl` and raises `ContactEvidenceError` rather
    than silently double-counting or dropping one copy. A state missing
    `seq` or `wall_time_ns`, or carrying a non-int value for either, also
    raises (`_require_seq`/`_require_wall_ns`) -- treating it as simply
    outside the window, the pre-fix behaviour, is how a real contact went
    missing with `contacts_recorded` still reading `True` (F9; P4/P5).

    F7 (2026-09-15 review): a state's `contacts` field is the
    accumulator's drain since the PREVIOUS push, so the one state that can
    cover the instant `last_sample_wall_ns` is whichever is pushed FIRST
    AT OR AFTER it -- "the covering push" -- however late the sim
    thread's render/encode stall delayed it past the ~20ms period the pad
    assumes (P1: pushed 61ms after the last sample, outside the 60ms pad).
    Rather than assume that push landed inside `[lo, hi]`, `covered` checks
    it directly: the window's earliest state must be at or before
    `first_sample_wall_ns` AND its latest state must be at or after
    `last_sample_wall_ns`. If the state stream stops before the flight's
    last sample at all (P2: the server died mid-recording, or the
    recorder outlived it), no push anywhere can cover it and `covered` is
    `False`. Separately, `seq_contiguous` checks that every push between
    the window's ends was actually recorded (`seq` increasing by exactly
    1): a real `states.jsonl` never skips one (`Recorder.record_state` is
    called synchronously on the sim thread for every push), so a gap here
    means a chunk of accumulated contact evidence is simply absent --
    unrecoverable no matter how the pad is sized (this is the "don't just
    widen the pad" fix: the pad governs over-inclusion at the edges, never
    whether the evidence in between actually exists).

    Returns `(contacts, contacts_recorded, meta)`. `contacts_recorded` is
    `True` only when the window is `covered`, `seq_contiguous`, non-empty,
    and EVERY state in it carries the `contacts` key (strict -- see
    `_validated_contacts_field`); any one of those gaps reads as
    `contacts_recorded: False` -- incomplete evidence must never report as
    a successful zero. `meta` carries the window's `seq` bounds, wall
    bounds, state counts (for the evidence-coverage line printed by the
    CLI), `covered`, `seq_contiguous`, and `reset_in_window` -- True if
    `sim_step` decreases between consecutive (by `seq`) states inside the
    window, meaning a scene reset happened mid-recording (not just
    between recordings) and any clearance/contact attribution here should
    be treated as suspect."""
    ordered = sorted(states, key=_require_seq)
    walls = [_require_wall_ns(s) for s in ordered]
    for a, b in zip(walls, walls[1:]):
        if b < a:
            raise ContactEvidenceError(
                f"states.jsonl is not in wall-clock order along seq: "
                f"wall_time_ns {b} follows {a}")

    lo = first_sample_wall_ns - _WINDOW_PAD_NS
    hi = last_sample_wall_ns + _WINDOW_PAD_NS
    window = [s for s, w in zip(ordered, walls) if lo <= w <= hi]
    window_walls = [w for w in walls if lo <= w <= hi]

    covered = (
        bool(window_walls)
        and window_walls[0] <= first_sample_wall_ns
        and window_walls[-1] >= last_sample_wall_ns
    )

    seqs = [s["seq"] for s in window]
    seq_contiguous = True
    for a, b in zip(seqs, seqs[1:]):
        if b == a:
            raise ContactEvidenceError(f"duplicate seq {b} in states.jsonl")
        if b != a + 1:
            seq_contiguous = False

    out: List[dict] = []
    with_key = 0
    prev_step = None
    reset_in_window = False
    for s in window:
        if prev_step is not None and s.get("sim_step", 0) < prev_step:
            reset_in_window = True
        prev_step = s.get("sim_step", prev_step)
        contacts = _validated_contacts_field(s)
        if contacts is None:
            continue
        with_key += 1
        out.extend(contacts)
    meta = {
        "seq_lo": window[0]["seq"] if window else None,
        "seq_hi": window[-1]["seq"] if window else None,
        "wall_lo_ns": lo,
        "wall_hi_ns": hi,
        "states_in_window": len(window),
        "states_with_contacts_key": with_key,
        "reset_in_window": reset_in_window,
        "covered": covered,
        "seq_contiguous": seq_contiguous,
    }
    contacts_recorded = (
        covered and seq_contiguous and with_key == len(window) and len(window) > 0
    )
    return out, contacts_recorded, meta


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
) -> dict:
    """The alignment block: per-sample alignment and server-side
    clearance recomputed at each aligned state (both hand models), plus
    the FULL contact record for the flight -- every state's own
    `contacts` field (see `native_mujoco.contact_accumulator`) unioned
    across the wall-time window `[first sample, last sample]` (plus the
    alignment window as padding), not just the states individual samples
    happened to align to (the server pushes state at 50 Hz; a recorder
    sample lands at ~20 Hz, so most states are never any sample's nearest
    match -- see `contacts_in_flight_window`).

    Both the clearance lookup and the contact window are keyed on `seq`,
    not `sim_step`: `sim_step` restarts at 0 on every scene reset within
    the same `--record` run dir (checklist section 4 resets BETWEEN
    recordings in one server run), so it is not unique across the whole
    `states.jsonl`, while `seq` is monotonic for the life of the server
    process."""
    states = read_states(run_dir)
    scene_model = SceneModel.from_yaml(scene_path)
    alignment = align_samples(samples, states)
    by_seq = {s.get("seq"): s for s in states}

    clearance_by_sample = []
    for record in alignment:
        state = by_seq.get(record["server_seq"])
        if state is not None:
            clearance_by_sample.append(_clearance_for_state(state, scene_model))
        else:
            clearance_by_sample.append({})

    if samples:
        contacts_in_window, contacts_recorded, contacts_window = (
            contacts_in_flight_window(
                states, samples[0]["wall_time_ns"], samples[-1]["wall_time_ns"]))
    else:
        contacts_in_window, contacts_recorded, contacts_window = [], False, {}

    return {
        "alignment": alignment,
        "server_side_clearance": clearance_by_sample,
        "contacts": contacts_in_window if contacts_recorded else None,
        "contacts_recorded": contacts_recorded,
        "contacts_window": contacts_window,
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
    try:
        path = link_flight(args.log_path, run_dir=args.run_dir)
    except ContactEvidenceError as exc:
        print(f"FAIL: contact evidence malformed -- {exc}")
        sys.exit(5)
    sidecar = read_sidecar(args.log_path)
    if not sidecar.get("contacts_recorded"):
        w = sidecar.get("contacts_window", {})
        reasons = []
        if w.get("states_in_window", 0) == 0:
            reasons.append("no recorded state falls inside the window "
                            "(wrong run dir?)")
        else:
            if not w.get("covered", True):
                reasons.append(
                    "the recorded states do not span the flight (the "
                    "stream ended early, or the covering push after the "
                    "last sample landed outside the window)")
            if not w.get("seq_contiguous", True):
                reasons.append("a gap in recorded states (seq) means some "
                                "contact evidence is missing")
            if w.get("states_with_contacts_key", 0) != w.get("states_in_window", 0):
                reasons.append("not every state in the window carries the "
                                "'contacts' key")
        reason = "; ".join(reasons) or "insufficient evidence"
        print(f"FAIL: no contact evidence in the server run dir -- this is NOT "
              f"usable E1 data ({reason}; contacts_recorded=false; see {path})")
        sys.exit(4)
    w = sidecar["contacts_window"]
    print(f"Wrote {path}: contacts={len(sidecar['contacts'])} "
          f"(evidence on {w['states_with_contacts_key']}/{w['states_in_window']} "
          f"states, reset_in_window={w['reset_in_window']})")


if __name__ == "__main__":
    main()
