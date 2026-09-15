"""E1 readiness (assignment 2026-09-14, work item 3): scripts/link_e1_flight.py
and the schema-3 changes to scripts/measure_route_clearance.py it depends on.

Offline throughout: every run directory is written with the REAL
`native_mujoco.recorder.Recorder`, and there is no live server or SDK
connection anywhere in this file.
"""

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, Tuple

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../scripts"))
sys.path.insert(0, os.path.join(_HERE, "../../src"))
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))

import link_e1_flight as lef  # noqa: E402
import measure_route_clearance as mrc  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.scene.awareness import SceneModel  # noqa: E402
from recorder import Recorder  # noqa: E402

_BOARD_SCENE = os.path.join(_HERE, "../../scenes/e1_boards/B4_pool_box_1_r2c3.yaml")
_POSE_DEG = dict(R.REST)


def _rad_joints(pose_deg: Dict[str, float]) -> list:
    return [{"name": name, "position_rad": math.radians(pose_deg[name])}
            for name in R.R_JOINTS]


def _state(seq, sim_step, wall_time_ns, pose_deg=_POSE_DEG, objects=None,
          contacts=None):
    state = {
        "type": "state", "seq": seq, "sim_step": sim_step, "sim_time_s": 0.1,
        "wall_time_ns": wall_time_ns, "scene_revision": "r1",
        "joints": _rad_joints(pose_deg), "objects": objects or [], "grippers": [],
    }
    if contacts is not None:
        state["contacts"] = contacts
    return state


def _contact(sim_step, object_id="pool_box_1", arm_geom="r_forearm_col"):
    """One `ContactAccumulator.drain()`-shaped entry, tagged by the
    state's own sim_step so a test can tell which state's window it came
    from after the union."""
    return {
        "arm_geom": arm_geom, "object_id": object_id, "steps": 3,
        "max_normal_force_n": 1.5, "min_dist_m": -0.001,
        "first_sim_step": sim_step, "last_sim_step": sim_step,
        "pos_at_max_force": [0.1, 0.2, 0.3],
    }


def _write_states(tmp_path, states):
    rec = Recorder.new(tmp_path, {"scene_path": _BOARD_SCENE})
    for s in states:
        rec.record_state(s)
    rec.finalize(total_steps=len(states), duration_s=0.1)
    return rec.run_dir


# B1 fixture: reproduces the review's repro (50 states at 50 Hz, 20 recorder
# samples at ~20 Hz, 10 states carrying a contact) -- but with the aligned
# window's first/last index NOT at the ends of the state stream (2..46 of
# 0..49), so states 0-1 and 47-49 are genuinely OUTSIDE the flight window
# and double as the "flight-boundary" exclusion case.
_CW_BASE_WALL_NS = 5_000_000_000
_CW_STEP_NS = 20_000_000  # 50 Hz server push
_CW_N_STATES = 50
_CW_FIRST_ALIGNED_IDX = 2
_CW_LAST_ALIGNED_IDX = 46
_CW_CONTACT_IDXS = (2, 5, 9, 14, 20, 26, 33, 38, 42, 46)  # 10 windows


def _cw_aligned_idxs():
    lo, hi = _CW_FIRST_ALIGNED_IDX, _CW_LAST_ALIGNED_IDX
    return [lo + round(k * (hi - lo) / 19) for k in range(20)]


def _cw_run_dir(tmp_path, contact_idxs=None):
    """Every state carries its own `contacts` key -- `[]` when nothing
    touched anything, a real drain when it did -- matching what the real
    server actually writes (`_build_state` always sets the key while
    `--record` is on; see F4 in the 2026-09-14 re-review: a fixture that
    OMITS the key on contact-free states exercises the partial-coverage
    path, not the real one)."""
    contact_idxs = set(_CW_CONTACT_IDXS if contact_idxs is None else contact_idxs)
    states = [
        _state(i, 100 + i, _CW_BASE_WALL_NS + i * _CW_STEP_NS,
              contacts=([_contact(100 + i)] if i in contact_idxs else []))
        for i in range(_CW_N_STATES)
    ]
    return _write_states(tmp_path, states)


def _cw_samples():
    return [
        {"t": k * 0.05, "wall_time_ns": _CW_BASE_WALL_NS + idx * _CW_STEP_NS,
         "joints": dict(_POSE_DEG)}
        for k, idx in enumerate(_cw_aligned_idxs())
    ]


@dataclass(frozen=True)
class _FakeIdentity:
    """Stands in for e1_identity.SimulatorIdentityCheck -- only the
    attributes build_base_sidecar actually reads."""
    manifest: dict = field(default_factory=dict)
    scene_chain_sha256: Dict[str, str] = field(default_factory=dict)

    def as_dict(self):
        return {"ok": True, "manifest": self.manifest,
               "scene_chain_sha256": self.scene_chain_sha256}


class TestAlignment:
    def test_alignment_picks_the_intended_state(self, tmp_path):
        pose_a = dict(_POSE_DEG)
        pose_b = {**_POSE_DEG, "r_wrist_roll": _POSE_DEG["r_wrist_roll"] + 20.0}
        run_dir = _write_states(tmp_path, [
            _state(1, 100, 1_000_000_000, pose_deg=pose_b),
            _state(2, 101, 1_000_050_000, pose_deg=pose_a),
        ])
        states = lef.read_states(run_dir)
        sample = {"wall_time_ns": 1_000_050_000, "joints": pose_a}
        chosen, wall_offset_s, residual_deg = lef.align_sample(sample, states)
        assert chosen["sim_step"] == 101
        assert residual_deg == pytest.approx(0.0, abs=1e-9)
        assert wall_offset_s == pytest.approx(0.0, abs=1e-9)

    def test_40ms_wall_skew_is_corrected_by_joint_refinement(self, tmp_path):
        """The chronologically-NEAREST state has the WRONG joints; a state
        40ms further away (but still inside the +/-60ms window) has the
        RIGHT joints. Alignment must pick the right one."""
        pose_a = dict(_POSE_DEG)
        pose_wrong = {**_POSE_DEG, "r_elbow_pitch": _POSE_DEG["r_elbow_pitch"] - 30.0}
        sample_wall = 5_000_000_000
        run_dir = _write_states(tmp_path, [
            _state(1, 100, sample_wall, pose_deg=pose_wrong),        # nearest in time
            _state(2, 101, sample_wall + 40_000_000, pose_deg=pose_a),  # right joints
        ])
        states = lef.read_states(run_dir)
        sample = {"wall_time_ns": sample_wall, "joints": pose_a}
        chosen, wall_offset_s, residual_deg = lef.align_sample(sample, states)
        assert chosen["sim_step"] == 101, (
            "nearest-by-wall-time alone would have picked sim_step=100 "
            "(the wrong joints) -- refinement must override it")
        assert residual_deg == pytest.approx(0.0, abs=1e-9)
        assert wall_offset_s == pytest.approx(0.040, abs=1e-9)

    def test_align_samples_produces_one_record_per_sample(self, tmp_path):
        run_dir = _write_states(tmp_path, [_state(1, 100, 1_000_000_000)])
        states = lef.read_states(run_dir)
        samples = [{"wall_time_ns": 1_000_000_000, "joints": dict(_POSE_DEG)}
                  for _ in range(3)]
        alignment = lef.align_samples(samples, states)
        assert len(alignment) == 3
        assert all(r["server_sim_step"] == 100 for r in alignment)
        assert all(r["sample_index"] == i for i, r in enumerate(alignment))


class TestContactsFullWindow:
    """B1 regression: `align_and_recompute` must report every state's
    contacts across the whole aligned window, not just the ~20 Hz states
    an individual recorder sample happened to land on."""

    def test_all_ten_contact_windows_are_reported(self, tmp_path):
        run_dir = _cw_run_dir(tmp_path)
        block = lef.align_and_recompute(_cw_samples(), run_dir, _BOARD_SCENE)
        assert block["contacts_recorded"] is True
        reported = sorted(c["first_sim_step"] for c in block["contacts"])
        expected = sorted(100 + i for i in _CW_CONTACT_IDXS)
        assert reported == expected
        assert len(block["contacts"]) == 10

    def test_contacts_between_recorder_samples_are_not_dropped(self, tmp_path):
        """The specific bug: a contact on a state that is nobody's
        alignment target (falls strictly between two 20 Hz samples) used
        to be silently dropped -- see contacts_if_only_aligned below."""
        aligned = set(_cw_aligned_idxs())
        between = [i for i in _CW_CONTACT_IDXS if i not in aligned]
        assert between, "fixture must exercise at least one non-aligned index"

        run_dir = _cw_run_dir(tmp_path)
        block = lef.align_and_recompute(_cw_samples(), run_dir, _BOARD_SCENE)
        reported = {c["first_sim_step"] for c in block["contacts"]}
        for i in between:
            assert 100 + i in reported, f"contact at index {i} (between samples) was dropped"

    def test_flight_boundary_included_and_out_of_range_excluded(self, tmp_path):
        """Contacts exactly at the first/last SAMPLE's own wall time are
        included (the window is inclusive), a contact right at the edge
        of the +/-60ms pad is included (3 states = 60ms at this fixture's
        20ms push period), and a contact one state beyond the pad is not.

        F2 (2026-09-14 re-review): this boundary is now defined by wall
        time around the SAMPLES, not by the ALIGNED states' `sim_step` --
        the pre-fix window ended at the aligned sim_step, which for an
        identical-pose fixture (no lag) coincides with the sample's own
        state, so this shape specifically pins the padded edge rather than
        re-deriving the old (buggy) sim_step-range boundary."""
        step_ns = 20_000_000
        base = 6_000_000_000
        n_states = 20
        first_sample_idx, last_sample_idx = 5, 14
        boundary_idxs = {
            "at_pad_edge_before": first_sample_idx - 3,  # exactly -60ms: included
            "outside_before": first_sample_idx - 4,       # -80ms: excluded
            "at_first_sample": first_sample_idx,
            "at_last_sample": last_sample_idx,
            "at_pad_edge_after": last_sample_idx + 3,     # exactly +60ms: included
            "outside_after": last_sample_idx + 4,          # +80ms: excluded
        }
        states = [
            _state(i, 100 + i, base + i * step_ns,
                  contacts=([_contact(100 + i)] if i in boundary_idxs.values() else []))
            for i in range(n_states)
        ]
        run_dir = _write_states(tmp_path, states)
        samples = [
            {"wall_time_ns": base + first_sample_idx * step_ns, "joints": dict(_POSE_DEG)},
            {"wall_time_ns": base + last_sample_idx * step_ns, "joints": dict(_POSE_DEG)},
        ]

        block = lef.align_and_recompute(samples, run_dir, _BOARD_SCENE)
        reported = {c["first_sim_step"] for c in block["contacts"]}
        assert 100 + boundary_idxs["at_first_sample"] in reported
        assert 100 + boundary_idxs["at_last_sample"] in reported
        assert 100 + boundary_idxs["at_pad_edge_before"] in reported
        assert 100 + boundary_idxs["at_pad_edge_after"] in reported
        assert 100 + boundary_idxs["outside_before"] not in reported
        assert 100 + boundary_idxs["outside_after"] not in reported

    def test_no_double_counting(self, tmp_path):
        run_dir = _cw_run_dir(tmp_path)
        block = lef.align_and_recompute(_cw_samples(), run_dir, _BOARD_SCENE)
        seen = [(c["arm_geom"], c["object_id"], c["first_sim_step"])
               for c in block["contacts"]]
        assert len(seen) == len(set(seen))

    def test_contacts_if_only_aligned_states_were_used_would_miss_most(self, tmp_path):
        """F6 (2026-09-14 re-review): the old test of this name asserted
        only index-set arithmetic and never called the linker, so it could
        not fail on a revert to aligned-only reading. This version drives
        the REAL alignment (`lef.align_samples`) and computes the
        aligned-only subset through it, so a revert of
        `align_and_recompute` back to reading contacts only from aligned
        states would fail this test."""
        run_dir = _cw_run_dir(tmp_path)
        states = lef.read_states(run_dir)
        alignment = lef.align_samples(_cw_samples(), states)
        aligned_sim_steps = {r["server_sim_step"] for r in alignment}
        only_aligned = [
            c for s in states if s.get("sim_step") in aligned_sim_steps
            for c in (s.get("contacts") or [])
        ]
        assert 0 < len(only_aligned) < len(_CW_CONTACT_IDXS)


class TestContactEvidenceIntegrity:
    """Missing or malformed contact evidence must never silently read as
    'no contacts' -- see `align_and_recompute`'s docstring."""

    def test_missing_contacts_key_is_flagged_not_silently_zero(self, tmp_path):
        """A run recorded before work item 4 (or with --record's contact
        tracking off) has no 'contacts' key on any state at all. F1/F4
        (2026-09-14 re-review): missing evidence must read as `None`, not
        `[]` -- an empty list is indistinguishable from a genuine
        zero-contact flight to any downstream consumer that doesn't also
        check `contacts_recorded`."""
        states = [_state(i, 100 + i, _CW_BASE_WALL_NS + i * _CW_STEP_NS)
                 for i in range(5)]
        run_dir = _write_states(tmp_path, states)
        samples = [{"wall_time_ns": _CW_BASE_WALL_NS + i * _CW_STEP_NS,
                   "joints": dict(_POSE_DEG)} for i in range(5)]
        block = lef.align_and_recompute(samples, run_dir, _BOARD_SCENE)
        assert block["contacts"] is None
        assert block["contacts_recorded"] is False
        w = block["contacts_window"]
        assert w["states_with_contacts_key"] == 0
        assert w["states_in_window"] == 5

    def test_genuine_zero_contacts_is_distinguished_from_missing(self, tmp_path):
        """The key IS present on every state (contact tracking covered
        this run) but nothing ever touched anything -- a real zero,
        distinct from the missing-evidence case above."""
        states = [_state(i, 100 + i, _CW_BASE_WALL_NS + i * _CW_STEP_NS, contacts=[])
                 for i in range(5)]
        run_dir = _write_states(tmp_path, states)
        samples = [{"wall_time_ns": _CW_BASE_WALL_NS + i * _CW_STEP_NS,
                   "joints": dict(_POSE_DEG)} for i in range(5)]
        block = lef.align_and_recompute(samples, run_dir, _BOARD_SCENE)
        assert block["contacts"] == []
        assert block["contacts_recorded"] is True

    def test_partial_coverage_is_not_recorded(self, tmp_path):
        """F4 (2026-09-14 re-review): the key is present on SOME states in
        the window but not others -- e.g. a mid-recording toggle, or (as
        the review notes) a state written by a server build that dropped
        the field partway through. `contacts_recorded` must be strictly
        `False` (not `True` on the strength of the first state that HAS
        the key, the pre-fix behaviour), `contacts` must read `None`, and
        the coverage counts must show the gap."""
        states = [
            _state(0, 100, _CW_BASE_WALL_NS, contacts=[_contact(100)]),
            _state(1, 101, _CW_BASE_WALL_NS + _CW_STEP_NS),  # no key at all
            _state(2, 102, _CW_BASE_WALL_NS + 2 * _CW_STEP_NS, contacts=[]),
        ]
        run_dir = _write_states(tmp_path, states)
        samples = [{"wall_time_ns": _CW_BASE_WALL_NS + i * _CW_STEP_NS,
                   "joints": dict(_POSE_DEG)} for i in range(3)]
        block = lef.align_and_recompute(samples, run_dir, _BOARD_SCENE)
        assert block["contacts"] is None
        assert block["contacts_recorded"] is False
        w = block["contacts_window"]
        assert w["states_with_contacts_key"] == 2
        assert w["states_in_window"] == 3

    def test_malformed_contacts_field_type_raises(self, tmp_path):
        states = [_state(0, 100, _CW_BASE_WALL_NS, contacts="not-a-list")]
        run_dir = _write_states(tmp_path, states)
        samples = [{"wall_time_ns": _CW_BASE_WALL_NS, "joints": dict(_POSE_DEG)}]
        with pytest.raises(lef.ContactEvidenceError):
            lef.align_and_recompute(samples, run_dir, _BOARD_SCENE)

    def test_malformed_contact_entry_raises(self, tmp_path):
        states = [_state(0, 100, _CW_BASE_WALL_NS, contacts=[{"bogus": True}])]
        run_dir = _write_states(tmp_path, states)
        samples = [{"wall_time_ns": _CW_BASE_WALL_NS, "joints": dict(_POSE_DEG)}]
        with pytest.raises(lef.ContactEvidenceError):
            lef.align_and_recompute(samples, run_dir, _BOARD_SCENE)


class TestCoverageGaps:
    """F7/F9 (2026-09-15 review): the pad's timing assumption ('a state is
    pushed every ~20ms, so 60ms of padding is always enough') is checked,
    not trusted, and a state that cannot be placed in the window at all
    (missing `seq`/`wall_time_ns`) fails loudly instead of silently
    dropping its evidence. In every case here, incomplete evidence must
    read as `contacts_recorded: False` -- never as a successful zero."""

    _CG_STEP_NS = 20_000_000  # 50 Hz nominal server push

    def test_delayed_covering_push_is_not_covered(self, tmp_path):
        """P1: the push that covers the flight's last sample (its
        `contacts` field drains everything since the previous push, up to
        and including its own push time) lands 61ms after the last
        sample -- 1ms past the 60ms pad -- carrying a real contact. The
        pre-fix window silently excluded that state and still reported
        `contacts_recorded=True` with full evidence on every state it DID
        include. The fix must refuse instead: there is no push anywhere
        inside the window whose own time is at or after the last sample,
        so the tail of the flight has no evidence at all."""
        base = 7_000_000_000
        states = [_state(i, 100 + i, base + i * self._CG_STEP_NS, contacts=[])
                 for i in range(5)]  # idx0..idx4, ending at base+80ms
        last_sample_wall = base + 4 * self._CG_STEP_NS + 10_000_000  # 10ms after idx4
        covering_wall = last_sample_wall + 61_000_000  # P1's exact shape
        states.append(_state(5, 105, covering_wall, contacts=[_contact(105)]))
        run_dir = _write_states(tmp_path, states)

        samples = [
            {"wall_time_ns": base, "joints": dict(_POSE_DEG)},
            {"wall_time_ns": last_sample_wall, "joints": dict(_POSE_DEG)},
        ]
        block = lef.align_and_recompute(samples, run_dir, _BOARD_SCENE)
        assert block["contacts_recorded"] is False, (
            "the covering push (with a real contact) landed outside the "
            "pad -- this must not read as a clean, fully-covered zero")
        assert block["contacts"] is None
        assert block["contacts_window"]["covered"] is False

    def test_stream_ending_early_is_not_covered(self, tmp_path):
        """P2: the server stopped writing states.jsonl 2 seconds before
        the recorder's last sample (recorder outlived the server). No
        state anywhere covers the last sample; the pre-fix code reported
        exit 0, contacts=0, 'evidence on 50/50 states' -- a clean-looking
        report built entirely on states that all predate the flight's own
        end."""
        base = 8_000_000_000
        states = [_state(i, 100 + i, base + i * self._CG_STEP_NS, contacts=[])
                 for i in range(50)]
        run_dir = _write_states(tmp_path, states)

        last_state_wall = base + 49 * self._CG_STEP_NS
        samples = [
            {"wall_time_ns": base, "joints": dict(_POSE_DEG)},
            {"wall_time_ns": last_state_wall + 2_000_000_000,
             "joints": dict(_POSE_DEG)},
        ]
        block = lef.align_and_recompute(samples, run_dir, _BOARD_SCENE)
        assert block["contacts_recorded"] is False
        assert block["contacts"] is None
        assert block["contacts_window"]["covered"] is False
        assert block["contacts_window"]["states_with_contacts_key"] == \
            block["contacts_window"]["states_in_window"], (
            "every state DOES carry the key -- the gap is coverage, not "
            "the evidence key, and must still refuse")

    def test_missing_wall_time_ns_raises_not_silently_dropped(self, tmp_path):
        """P4: a state inside the window has no `wall_time_ns` and carries
        a contact. The pre-fix filter (`s.get('wall_time_ns', -1)`) read
        that as 'far outside any window' and silently excluded it --
        `contacts_recorded=True`, evidence 9/9, the contact just gone."""
        states = [
            _state(0, 100, _CW_BASE_WALL_NS, contacts=[]),
            _state(1, 101, _CW_BASE_WALL_NS + _CW_STEP_NS, contacts=[_contact(101)]),
            _state(2, 102, _CW_BASE_WALL_NS + 2 * _CW_STEP_NS, contacts=[]),
        ]
        del states[1]["wall_time_ns"]
        run_dir = _write_states(tmp_path, states)
        samples = [{"wall_time_ns": _CW_BASE_WALL_NS + i * _CW_STEP_NS,
                   "joints": dict(_POSE_DEG)} for i in (0, 2)]
        with pytest.raises(lef.ContactEvidenceError, match="wall_time_ns"):
            lef.align_and_recompute(samples, run_dir, _BOARD_SCENE)

    def test_missing_seq_raises_not_silently_dropped(self, tmp_path):
        """P5: a state inside the window has no `seq`. The pre-fix code
        crashed with a bare `KeyError` (an undocumented exit 1, not a
        clean `FAIL:` / exit 5) -- this must raise the documented
        `ContactEvidenceError` instead."""
        states = [
            _state(0, 100, _CW_BASE_WALL_NS, contacts=[]),
            _state(1, 101, _CW_BASE_WALL_NS + _CW_STEP_NS, contacts=[_contact(101)]),
        ]
        del states[1]["seq"]
        run_dir = _write_states(tmp_path, states)
        samples = [{"wall_time_ns": _CW_BASE_WALL_NS, "joints": dict(_POSE_DEG)},
                  {"wall_time_ns": _CW_BASE_WALL_NS + _CW_STEP_NS,
                   "joints": dict(_POSE_DEG)}]
        with pytest.raises(lef.ContactEvidenceError, match="seq"):
            lef.align_and_recompute(samples, run_dir, _BOARD_SCENE)

    def test_wall_time_out_of_seq_order_raises(self, tmp_path):
        """F9 follow-up (review §5.2): `wall_time_ns` must be
        non-decreasing along `seq` -- a single-threaded server with a
        monotonic clock cannot produce the reverse, so a file that does is
        corrupted evidence, not data to sort past silently."""
        states = [
            _state(0, 100, _CW_BASE_WALL_NS, contacts=[]),
            _state(1, 101, _CW_BASE_WALL_NS - _CW_STEP_NS, contacts=[]),
        ]
        run_dir = _write_states(tmp_path, states)
        samples = [{"wall_time_ns": _CW_BASE_WALL_NS, "joints": dict(_POSE_DEG)},
                  {"wall_time_ns": _CW_BASE_WALL_NS - _CW_STEP_NS,
                   "joints": dict(_POSE_DEG)}]
        with pytest.raises(lef.ContactEvidenceError, match="wall-clock order"):
            lef.align_and_recompute(samples, run_dir, _BOARD_SCENE)


class TestMovingArmSdkLag:
    """F2 (2026-09-14 re-review, probe E2): with a moving arm, the SDK
    reading `align_sample` matches against reflects a state OLDER than the
    sample's own wall time (pipeline lag), so the aligned state is
    systematically earlier than the sample. Keying the contact window on
    the aligned states' `sim_step` (the pre-fix behaviour) therefore trims
    real physics off the END of the window. This reproduces that shape
    directly (not via the identical-pose fixtures the original tests used,
    which cannot see the bug -- Q4 in the re-review) and proves the fix by
    checking the window against the WALL-TIME-independent-of-alignment
    result, while separately confirming the alignment itself really is
    lagged."""

    _MA_N_STATES = 60
    _MA_STEP_NS = 20_000_000  # 50 Hz server push, matching _STATE_HZ
    _MA_BASE_WALL_NS = 9_000_000_000
    _MA_LAG_STATES = 2  # 40 ms SDK lag, matching probe E2
    _MA_DEG_PER_STATE = 0.5

    @classmethod
    def _pose(cls, i):
        i = max(i, 0)
        return {**_POSE_DEG,
               "r_shoulder_pitch": _POSE_DEG["r_shoulder_pitch"] + cls._MA_DEG_PER_STATE * i}

    def _run_dir(self, tmp_path, contact_idxs):
        states = [
            _state(i, 100 + i, self._MA_BASE_WALL_NS + i * self._MA_STEP_NS,
                  pose_deg=self._pose(i),
                  contacts=([_contact(100 + i)] if i in contact_idxs else []))
            for i in range(self._MA_N_STATES)
        ]
        return _write_states(tmp_path, states)

    def _lagged_samples(self):
        """A sample every 3 states (~15 Hz, close to the recorder's real
        ~20 Hz), starting at state index 2 so the lag never reaches
        before state 0; each sample's `joints` are the SDK's reading of
        the arm `_MA_LAG_STATES` states EARLIER than the state whose wall
        time the sample shares -- i.e. the SDK is stale by 40 ms."""
        idxs = list(range(2, self._MA_N_STATES, 3))
        return [
            {"wall_time_ns": self._MA_BASE_WALL_NS + i * self._MA_STEP_NS,
             "joints": self._pose(i - self._MA_LAG_STATES)}
            for i in idxs
        ]

    def test_tail_contacts_survive_alignment_lag(self, tmp_path):
        last_two = [self._MA_N_STATES - 2, self._MA_N_STATES - 1]  # 58, 59
        run_dir = self._run_dir(tmp_path, contact_idxs=set(last_two))
        samples = self._lagged_samples()

        block = lef.align_and_recompute(samples, run_dir, _BOARD_SCENE)

        # The alignment IS lagged: the last sample's aligned state is not
        # the last state in the stream.
        last_aligned_sim_step = block["alignment"][-1]["server_sim_step"]
        assert last_aligned_sim_step == 100 + (self._MA_N_STATES - 1 - self._MA_LAG_STATES), (
            "fixture assumption broken: the last sample should align "
            f"{self._MA_LAG_STATES} states before the stream's end")

        # ...yet both tail contacts -- AFTER the aligned end -- are still
        # reported, because the window is wall-time-derived, not
        # alignment-derived.
        reported = {c["first_sim_step"] for c in block["contacts"]}
        assert reported == {100 + i for i in last_two}
        assert block["contacts_recorded"] is True


class TestResetHandling:
    """F3/F5 (2026-09-14 re-review): `sim_step` restarts at 0 on every
    scene reset within one `--record` run dir; `seq` does not. These test
    the window and clearance lookup directly against that shape."""

    _RH_STEP_NS = 20_000_000

    def test_contact_from_a_prior_epoch_is_not_attributed_to_the_next(self, tmp_path):
        """Probe E1 shape: a reset happens BETWEEN two recordings in the
        same server run (checklist section 4's own procedure). Epoch A had
        a contact; epoch B (the flight actually being linked) did not.
        Because both epochs' `sim_step` restarts at 0, the OLD range-based
        window (`lo <= sim_step <= hi` over ALL states) could numerically
        re-include epoch A's contact for epoch B's flight; the wall-time
        window must not, since the epochs are far apart in wall time."""
        epoch_a_base = 1_000_000_000
        epoch_a = [
            _state(i, i, epoch_a_base + i * self._RH_STEP_NS,
                  pose_deg={**_POSE_DEG, "r_shoulder_pitch": -90.0},
                  contacts=([_contact(i)] if i == 10 else []))
            for i in range(30)
        ]
        # Epoch B starts 10 s later (wall time) -- far outside any
        # alignment/contact padding -- and its own sim_step restarts at 0,
        # numerically overlapping epoch A's.
        epoch_b_base = epoch_a_base + 10_000_000_000
        epoch_b_pose = {**_POSE_DEG, "r_shoulder_pitch": -60.0}  # distinguishable
        epoch_b = [
            _state(30 + i, i, epoch_b_base + i * self._RH_STEP_NS,
                  pose_deg=epoch_b_pose, contacts=[])
            for i in range(30)
        ]
        run_dir = _write_states(tmp_path, epoch_a + epoch_b)

        sample_idxs = list(range(0, 30, 3))
        samples = [
            {"wall_time_ns": epoch_b_base + i * self._RH_STEP_NS,
             "joints": epoch_b_pose}
            for i in sample_idxs
        ]

        block = lef.align_and_recompute(samples, run_dir, _BOARD_SCENE)

        assert block["contacts"] == []
        assert block["contacts_recorded"] is True
        assert block["contacts_window"]["reset_in_window"] is False

        # server_side_clearance is computed on epoch B's own state (keyed
        # by `seq`), not whatever `by_sim_step` last-wins happened to pick.
        scene_model = SceneModel.from_yaml(_BOARD_SCENE)
        expected_state = next(s for s in epoch_b if s["sim_step"] == sample_idxs[0])
        assert block["server_side_clearance"][0] == lef._clearance_for_state(
            expected_state, scene_model)

    def test_reset_inside_the_window_is_flagged(self, tmp_path):
        """Probe E5 shape: the reset happens DURING the recording being
        linked (not between two recordings) -- `sim_step` still restarts
        at 0 partway through, but wall time keeps advancing continuously.
        `reset_in_window` must be True so the checklist can flag the
        result as suspect rather than silently trusting it."""
        base = 2_000_000_000
        pre_reset = [_state(i, i, base + i * self._RH_STEP_NS, contacts=[])
                    for i in range(16)]
        post_reset = [
            _state(16 + i, i, base + (16 + i) * self._RH_STEP_NS, contacts=[])
            for i in range(14)
        ]
        states = pre_reset + post_reset
        run_dir = _write_states(tmp_path, states)

        samples = [
            {"wall_time_ns": states[0]["wall_time_ns"], "joints": dict(_POSE_DEG)},
            {"wall_time_ns": states[-1]["wall_time_ns"], "joints": dict(_POSE_DEG)},
        ]
        block = lef.align_and_recompute(samples, run_dir, _BOARD_SCENE)
        assert block["contacts_window"]["reset_in_window"] is True

    def test_duplicate_seq_in_window_raises(self, tmp_path):
        """A real `states.jsonl` cannot repeat `seq` (it is `self._seq +=
        1` every push, never reset) -- if it somehow does, that is
        corrupted evidence, not a state to silently pick one copy of."""
        base = 3_000_000_000
        states = [
            _state(0, 100, base, contacts=[]),
            _state(0, 101, base + self._RH_STEP_NS, contacts=[]),  # dup seq
        ]
        run_dir = _write_states(tmp_path, states)
        samples = [{"wall_time_ns": base, "joints": dict(_POSE_DEG)},
                  {"wall_time_ns": base + self._RH_STEP_NS, "joints": dict(_POSE_DEG)}]
        with pytest.raises(lef.ContactEvidenceError, match="duplicate seq"):
            lef.align_and_recompute(samples, run_dir, _BOARD_SCENE)


def _write_log_and_base_sidecar(tmp_path, monkeypatch, run_dir, samples):
    """Full setup for the end-to-end tests: a real schema-3 log via
    `save_log` and the base sidecar `measure_route_clearance.main()` would
    have written right after the flight -- everything `link_flight` needs
    on disk, for an ARBITRARY run dir + samples pair."""
    monkeypatch.setattr(mrc, "RUNS_DIR", tmp_path / "runs")
    log_path = mrc.save_log(samples, "LOWER_TO_REST", str(_BOARD_SCENE),
                            t0_wall_ns=samples[0]["wall_time_ns"])
    identity = _FakeIdentity(manifest={"scene_revision": "r1"},
                             scene_chain_sha256={"a": "sha_a"})
    base_sidecar = lef.build_base_sidecar(
        log_path=str(log_path), samples=samples, scene_path=_BOARD_SCENE,
        identity_check=identity, run_dir=str(run_dir))
    lef.write_sidecar(str(log_path), base_sidecar)
    return log_path


def _cw_write_log_and_base_sidecar(tmp_path, monkeypatch):
    """`_write_log_and_base_sidecar` for the standard 10-contact-window
    fixture (`_cw_run_dir`/`_cw_samples`)."""
    run_dir = _cw_run_dir(tmp_path)
    samples = _cw_samples()
    log_path = _write_log_and_base_sidecar(tmp_path, monkeypatch, run_dir, samples)
    return log_path, run_dir


class TestContactsThroughLinkFlight:
    """End to end: `link_flight()` (the CLI's own entry point) and the
    actual CLI subprocess, both reading only the log + its base sidecar +
    the run dir -- no live server, matching how the checklist uses it."""

    def test_link_flight_writes_all_ten_contacts(self, tmp_path, monkeypatch):
        log_path, _ = _cw_write_log_and_base_sidecar(tmp_path, monkeypatch)
        written = lef.link_flight(str(log_path))
        sidecar = json.loads(written.read_text())
        assert sidecar["contacts_recorded"] is True
        assert len(sidecar["contacts"]) == 10
        reported = {c["first_sim_step"] for c in sidecar["contacts"]}
        assert reported == {100 + i for i in _CW_CONTACT_IDXS}

    def test_link_flight_run_dir_override_also_finds_all_ten(self, tmp_path, monkeypatch):
        """`--run-dir` (retained artefacts moved to a new path) must not
        change the contact result."""
        log_path, run_dir = _cw_write_log_and_base_sidecar(tmp_path, monkeypatch)
        written = lef.link_flight(str(log_path), run_dir=str(run_dir))
        sidecar = json.loads(written.read_text())
        assert len(sidecar["contacts"]) == 10

    def test_cli_subprocess_writes_all_ten_contacts(self, tmp_path, monkeypatch):
        """The literal command the checklist runs:
        `python3 scripts/link_e1_flight.py <log_path>`. F1/F2 (2026-09-14
        re-review): a successful run must print the evidence-coverage line
        (`N/N states`, `reset_in_window=...`), not just `Wrote <path>`."""
        log_path, _ = _cw_write_log_and_base_sidecar(tmp_path, monkeypatch)
        script = os.path.join(_HERE, "../../scripts/link_e1_flight.py")
        result = subprocess.run(
            [sys.executable, script, str(log_path)],
            capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert "Wrote" in result.stdout
        assert "contacts=10" in result.stdout
        assert "reset_in_window=False" in result.stdout

        sidecar_path = lef.sidecar_path_for(str(log_path))
        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["contacts_recorded"] is True
        assert len(sidecar["contacts"]) == 10
        reported = {c["first_sim_step"] for c in sidecar["contacts"]}
        assert reported == {100 + i for i in _CW_CONTACT_IDXS}

    def test_cli_exits_4_on_missing_contact_evidence(self, tmp_path, monkeypatch):
        """F1 (2026-09-14 re-review): the pre-fix CLI exited 0 whether or
        not any state carried contact evidence, so a run against a server
        not tracking contacts read (to an operator following the
        checklist) as a clean zero-contact flight. Missing evidence must
        refuse, non-zero, with a FAIL line -- not print `Wrote` and exit 0."""
        states = [_state(i, 100 + i, _CW_BASE_WALL_NS + i * _CW_STEP_NS)
                 for i in range(5)]  # no 'contacts' key anywhere
        run_dir = _write_states(tmp_path, states)
        samples = [{"t": i * 0.05, "wall_time_ns": _CW_BASE_WALL_NS + i * _CW_STEP_NS,
                   "joints": dict(_POSE_DEG)} for i in range(5)]
        log_path = _write_log_and_base_sidecar(tmp_path, monkeypatch, run_dir, samples)

        script = os.path.join(_HERE, "../../scripts/link_e1_flight.py")
        result = subprocess.run(
            [sys.executable, script, str(log_path)],
            capture_output=True, text=True, check=False)
        assert result.returncode == 4, result.stdout + result.stderr
        assert "FAIL" in result.stdout
        assert "not usable" in result.stdout or "NOT" in result.stdout

        sidecar = json.loads(lef.sidecar_path_for(str(log_path)).read_text())
        assert sidecar["contacts_recorded"] is False
        assert sidecar["contacts"] is None

    def test_cli_exits_5_on_malformed_contact_evidence(self, tmp_path, monkeypatch):
        """Q3 (2026-09-14 re-review): a malformed `contacts` field is a
        distinct failure from missing evidence -- crash-loud (a non-zero
        exit distinct from 4), never silently dropped."""
        states = [_state(0, 100, _CW_BASE_WALL_NS, contacts=[{"bogus": True}])]
        run_dir = _write_states(tmp_path, states)
        samples = [{"t": 0.0, "wall_time_ns": _CW_BASE_WALL_NS, "joints": dict(_POSE_DEG)}]
        log_path = _write_log_and_base_sidecar(tmp_path, monkeypatch, run_dir, samples)

        script = os.path.join(_HERE, "../../scripts/link_e1_flight.py")
        result = subprocess.run(
            [sys.executable, script, str(log_path)],
            capture_output=True, text=True, check=False)
        assert result.returncode == 5, result.stdout + result.stderr
        assert "FAIL" in result.stdout


class TestCoverageGapsThroughLinkFlight:
    """F7/F9 end to end, through the actual CLI subprocess the checklist
    runs -- incomplete evidence must never exit 0."""

    _CG_STEP_NS = 20_000_000

    def test_cli_exits_4_on_delayed_covering_push(self, tmp_path, monkeypatch):
        """The review's 61ms-delayed-push repro (P1), through the CLI: a
        real contact exists on a state pushed just past the 60ms pad, and
        the run must refuse (exit 4), not print `Wrote ... contacts=0`."""
        base = 7_000_000_000
        states = [_state(i, 100 + i, base + i * self._CG_STEP_NS, contacts=[])
                 for i in range(5)]
        last_sample_wall = base + 4 * self._CG_STEP_NS + 10_000_000
        covering_wall = last_sample_wall + 61_000_000
        states.append(_state(5, 105, covering_wall, contacts=[_contact(105)]))
        run_dir = _write_states(tmp_path, states)

        samples = [{"t": 0.0, "wall_time_ns": base, "joints": dict(_POSE_DEG)},
                  {"t": 0.1, "wall_time_ns": last_sample_wall,
                   "joints": dict(_POSE_DEG)}]
        log_path = _write_log_and_base_sidecar(tmp_path, monkeypatch, run_dir, samples)

        script = os.path.join(_HERE, "../../scripts/link_e1_flight.py")
        result = subprocess.run(
            [sys.executable, script, str(log_path)],
            capture_output=True, text=True, check=False)
        assert result.returncode == 4, result.stdout + result.stderr
        assert "FAIL" in result.stdout
        assert "contacts=0" not in result.stdout, (
            "a real contact exists on the delayed covering push -- this "
            "must never print as a clean, fully-covered zero")

        sidecar = json.loads(lef.sidecar_path_for(str(log_path)).read_text())
        assert sidecar["contacts_recorded"] is False
        assert sidecar["contacts"] is None

    def test_cli_exits_4_on_stream_ending_early(self, tmp_path, monkeypatch):
        """P2 through the CLI: the recorder sampled 2 seconds past the
        last recorded state (the server died mid-recording). The pre-fix
        CLI exited 0 with `contacts=0 (evidence on 50/50 states)`."""
        base = 8_000_000_000
        states = [_state(i, 100 + i, base + i * self._CG_STEP_NS, contacts=[])
                 for i in range(50)]
        run_dir = _write_states(tmp_path, states)
        last_state_wall = base + 49 * self._CG_STEP_NS

        samples = [{"t": 0.0, "wall_time_ns": base, "joints": dict(_POSE_DEG)},
                  {"t": 2.0, "wall_time_ns": last_state_wall + 2_000_000_000,
                   "joints": dict(_POSE_DEG)}]
        log_path = _write_log_and_base_sidecar(tmp_path, monkeypatch, run_dir, samples)

        script = os.path.join(_HERE, "../../scripts/link_e1_flight.py")
        result = subprocess.run(
            [sys.executable, script, str(log_path)],
            capture_output=True, text=True, check=False)
        assert result.returncode == 4, result.stdout + result.stderr
        assert "FAIL" in result.stdout
        assert "Wrote" not in result.stdout

        sidecar = json.loads(lef.sidecar_path_for(str(log_path)).read_text())
        assert sidecar["contacts_recorded"] is False
        assert sidecar["contacts"] is None

    def test_cli_exits_5_on_missing_wall_time_ns(self, tmp_path, monkeypatch):
        """P4 through the CLI: a malformed/missing timing field on an
        in-window state must fail loudly (exit 5, a documented
        `ContactEvidenceError`), never silently drop the state (and
        whatever contact it carries) out of the window."""
        states = [
            _state(0, 100, _CW_BASE_WALL_NS, contacts=[]),
            _state(1, 101, _CW_BASE_WALL_NS + _CW_STEP_NS, contacts=[_contact(101)]),
            _state(2, 102, _CW_BASE_WALL_NS + 2 * _CW_STEP_NS, contacts=[]),
        ]
        del states[1]["wall_time_ns"]
        run_dir = _write_states(tmp_path, states)
        samples = [{"t": i * 0.05, "wall_time_ns": _CW_BASE_WALL_NS + i * _CW_STEP_NS,
                   "joints": dict(_POSE_DEG)} for i in (0, 2)]
        log_path = _write_log_and_base_sidecar(tmp_path, monkeypatch, run_dir, samples)

        script = os.path.join(_HERE, "../../scripts/link_e1_flight.py")
        result = subprocess.run(
            [sys.executable, script, str(log_path)],
            capture_output=True, text=True, check=False)
        assert result.returncode == 5, result.stdout + result.stderr
        assert "FAIL" in result.stdout

    def test_cli_exits_5_on_missing_seq(self, tmp_path, monkeypatch):
        """P5 through the CLI: the pre-fix code crashed with a bare,
        undocumented `KeyError` (exit 1) here -- this must be the same
        documented exit 5 as any other malformed-evidence case."""
        states = [
            _state(0, 100, _CW_BASE_WALL_NS, contacts=[]),
            _state(1, 101, _CW_BASE_WALL_NS + _CW_STEP_NS, contacts=[_contact(101)]),
        ]
        del states[1]["seq"]
        run_dir = _write_states(tmp_path, states)
        samples = [{"t": 0.0, "wall_time_ns": _CW_BASE_WALL_NS,
                   "joints": dict(_POSE_DEG)},
                  {"t": 0.05, "wall_time_ns": _CW_BASE_WALL_NS + _CW_STEP_NS,
                   "joints": dict(_POSE_DEG)}]
        log_path = _write_log_and_base_sidecar(tmp_path, monkeypatch, run_dir, samples)

        script = os.path.join(_HERE, "../../scripts/link_e1_flight.py")
        result = subprocess.run(
            [sys.executable, script, str(log_path)],
            capture_output=True, text=True, check=False)
        assert result.returncode == 5, result.stdout + result.stderr
        assert "FAIL" in result.stdout


class TestSettledPoseCheck:
    def test_passes_at_4mm(self):
        expected = {"pool_box_1": (0.4318, -0.1524, 0.76)}
        state = _state(1, 100, 0, objects=[
            {"object_id": "pool_box_1", "pos_xyz": [0.4318 + 0.004, -0.1524, 0.76]}])
        result = lef.settled_pose_check(["pool_box_1"], expected, state)
        assert result["pool_box_1"]["ok"] is True
        assert result["pool_box_1"]["dxy_m"] == pytest.approx(0.004, abs=1e-9)

    def test_fails_at_6mm(self):
        expected = {"pool_box_1": (0.4318, -0.1524, 0.76)}
        state = _state(1, 100, 0, objects=[
            {"object_id": "pool_box_1", "pos_xyz": [0.4318 + 0.006, -0.1524, 0.76]}])
        result = lef.settled_pose_check(["pool_box_1"], expected, state)
        assert result["pool_box_1"]["ok"] is False
        assert result["pool_box_1"]["dxy_m"] == pytest.approx(0.006, abs=1e-9)
        assert result["pool_box_1"]["yaml_xyz"] == [0.4318, -0.1524, 0.76]
        assert result["pool_box_1"]["stream_xyz"][0] == pytest.approx(0.4318 + 0.006)


class TestBuildBaseSidecar:
    def test_displacement_and_shape(self, tmp_path):
        yaml_pose = tuple(SceneModel.from_yaml(_BOARD_SCENE).get("pool_box_1").center)
        first_pose = list(yaml_pose)
        last_pose = [yaml_pose[0] + 0.01, yaml_pose[1], yaml_pose[2]]  # 1 cm drift
        run_dir = _write_states(tmp_path, [
            _state(1, 100, 1_000_000_000,
                  objects=[{"object_id": "pool_box_1", "pos_xyz": first_pose}]),
            _state(2, 101, 1_000_020_000,
                  objects=[{"object_id": "pool_box_1", "pos_xyz": last_pose}]),
        ])
        samples = [
            {"t": 0.0, "wall_time_ns": 1_000_000_000, "joints": dict(_POSE_DEG)},
            {"t": 0.02, "wall_time_ns": 1_000_020_000, "joints": dict(_POSE_DEG)},
        ]
        identity = _FakeIdentity(manifest={"scene_revision": "r1"},
                                 scene_chain_sha256={"a": "sha_a"})
        sidecar = lef.build_base_sidecar(
            log_path=str(tmp_path / "route_clearance_LOWER_TO_REST_x.json"),
            samples=samples, scene_path=_BOARD_SCENE,
            identity_check=identity, run_dir=str(run_dir))

        _assert_sidecar_shape(sidecar)
        assert sidecar["displacement_m"]["pool_box_1"] == pytest.approx(0.01, abs=1e-9)
        assert sidecar["settled_pose_check"]["pool_box_1"]["ok"] is True
        assert sidecar["scene"]["board_object_ids"] == ["pool_box_1"]
        assert sidecar["no_reshape_asserted"] is True


def _assert_sidecar_shape(sidecar: dict) -> None:
    """A small JSON-schema-like assertion set: every key the E1 readiness
    plan's sidecar example names is present, with the right shape."""
    for key in ("log", "identity", "server_run_dir", "manifest", "scene",
               "objects_at_first_sample", "objects_at_last_sample",
               "settled_pose_check", "displacement_m", "no_reshape_asserted"):
        assert key in sidecar, f"sidecar missing {key!r}"
    assert isinstance(sidecar["log"], str)
    assert isinstance(sidecar["identity"], dict)
    assert isinstance(sidecar["server_run_dir"], str)
    assert isinstance(sidecar["manifest"], dict)
    assert isinstance(sidecar["scene"], dict)
    for k in ("path", "chain_sha256", "board_object_ids"):
        assert k in sidecar["scene"]
    assert isinstance(sidecar["objects_at_first_sample"], dict)
    assert isinstance(sidecar["objects_at_last_sample"], dict)
    for oid, snap in sidecar["objects_at_first_sample"].items():
        assert set(snap) >= {"pos_xyz", "quat_wxyz", "sim_step", "wall_time_ns"}
    assert isinstance(sidecar["settled_pose_check"], dict)
    for oid, check in sidecar["settled_pose_check"].items():
        assert "ok" in check
    assert isinstance(sidecar["displacement_m"], dict)
    assert isinstance(sidecar["no_reshape_asserted"], bool)


class TestSidecarRoundTrip:
    def test_write_and_read_sidecar(self, tmp_path):
        log_path = tmp_path / "route_clearance_LOWER_TO_REST_x.json"
        log_path.write_text("{}")
        sidecar = {"log": log_path.name, "identity": {"ok": True},
                  "server_run_dir": "x", "manifest": {}, "scene": {},
                  "objects_at_first_sample": {}, "objects_at_last_sample": {},
                  "settled_pose_check": {}, "displacement_m": {},
                  "no_reshape_asserted": True}
        written = lef.write_sidecar(str(log_path), sidecar)
        assert written.name == "route_clearance_LOWER_TO_REST_x.link.json"
        loaded = lef.read_sidecar(str(log_path))
        assert loaded == sidecar


class TestSchema3RoundTrip:
    def test_save_log_load_log_validated_samples(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mrc, "RUNS_DIR", tmp_path)
        samples = [
            {"t": 0.0, "wall_time_ns": 1_000_000_000,
             "joints": {**{j: R.REST[j] for j in R.ARM7}, "r_gripper": R.REST["r_gripper"]}},
            {"t": 0.05, "wall_time_ns": 1_050_000_000,
             "joints": {**{j: R.REST[j] for j in R.ARM7}, "r_gripper": R.REST["r_gripper"]}},
        ]
        path = mrc.save_log(samples, "LOWER_TO_REST",
                           str(_BOARD_SCENE), t0_wall_ns=999_000_000)
        loaded = mrc.load_log(path)
        assert loaded["schema_version"] == mrc.LOG_SCHEMA_VERSION == 3
        assert loaded["t0_wall_ns"] == 999_000_000
        q7_and_gripper, assumed = mrc.validated_samples(
            loaded["samples"], schema_version=mrc.schema_version_of(loaded))
        assert assumed == []
        assert len(q7_and_gripper) == 2

    def test_report_unchanged_by_the_schema_bump(self):
        """report()'s NUMBERS must be identical whether the log is declared
        schema 2 (no wall_time_ns) or schema 3 (has it) -- the schema bump
        changes what is REQUIRED on disk, never the aperture/clearance
        math report() already had."""
        joints = {**{j: R.REST[j] for j in R.ARM7}, "r_gripper": R.REST["r_gripper"]}
        schema_2_log = [{"t": i / 20.0, "joints": dict(joints)} for i in range(3)]
        schema_3_log = [{"t": i / 20.0, "wall_time_ns": 1_000_000_000 + i * 50_000_000,
                        "joints": dict(joints)} for i in range(3)]

        result_2 = mrc.report(schema_2_log, "LOWER_TO_REST", _BOARD_SCENE,
                             schema_version=2)
        result_3 = mrc.report(schema_3_log, "LOWER_TO_REST", _BOARD_SCENE,
                             schema_version=3)
        assert result_2["realised"] == result_3["realised"]
        assert result_2["planned"] == result_3["planned"]
        assert result_2["n_samples"] == result_3["n_samples"]
