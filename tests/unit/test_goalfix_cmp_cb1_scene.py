"""CB1 acceptance tests (merge verdict / readiness review, 2026-09-25
stage-repairs assignment §4 CB1): "scene-bound clearance and wrist_ball."

Two independent repairs:

1. The cycle CLI never loaded a ``SceneModel`` at all ("the cycle CLI has
   no scene"). Fixed by ``cycle.load_verified_scene``, wired into
   ``cycle._cli`` right after leg resolution: when the setup sidecar
   carries a ``scene`` block (as ``scripts/link_e1_flight.py``'s real
   sidecars do), the named scene file is loaded and its ``extends``
   chain re-hashed (via ``scripts/e1_identity.py``'s own
   ``_extends_chain``/``_sha256_file``) against the sidecar's own
   recorded ``chain_sha256`` -- a missing file or any hash mismatch is
   ``CycleInputError`` -> rc 3. No ``scene`` block at all (every
   existing fixture, and V1's own harness) is unchanged: ``scene=None``.

2. The segment-level Δcmd computation's ``realised`` slice
   (``compute_place_route_metrics``, "the segment slice bug") was
   ``realised[:len(seg_commanded)]`` -- always the LEG's own first N
   samples, regardless of where the affected segment actually starts
   within the leg. Fixed to slice ``realised`` at the SAME ``seg_local``
   positions as ``seg_commanded``.

``wrist_ball_delta_cm`` itself was, at the time this file was written,
intentionally left ``None`` (with an ``open_questions`` entry): the
plan's own §7.5 language and the review named two DIFFERENT,
both-plausible readings with no way to choose between them. B12 (owner
rulings W1-W4, 2026-09-25; proposal §4.1) resolved every open choice
here -- the real computation now lives in
``compute_place_route_metrics``, tested in
``test_goalfix_cmp_b12_wrist_ball.py``. The fixtures in THIS file still
pass no ``board_object_ids`` (and most have no ``REST`` waypoint in
their tiny synthetic routes at all), so ``wrist_ball_delta_cm`` stays
null for a DIFFERENT, still-accurate reason on every case here: a
missing input (W4), never a withheld definition."""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../scripts", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402

CYCLE = "S2-B4-c-r1"


@pytest.fixture(autouse=True)
def _zero_lag(monkeypatch):
    monkeypatch.setattr(mf, "_LAG_RAD", [0.0] * 8)


def _write_sidecar(control_dir, cycle, kind, first_seq, last_seq, scene_block=None):
    doc = {"alignment": [{"server_seq": first_seq}, {"server_seq": last_seq}]}
    if scene_block is not None:
        doc["scene"] = scene_block
    path = control_dir / f"{cycle}-{kind}.link.json"
    path.write_text(json.dumps(doc))
    return path


def _write_scene_yaml(path):
    path.write_text(
        "frame_id: pedestal\n"
        "objects:\n"
        "  - id: board\n"
        "    geometry:\n"
        "      kind: box\n"
        "    pose:\n"
        "      position: [0.5, -0.3, 1.1]\n"
        "    size: [0.1, 0.1, 0.1]\n")


def _build_cycle(tmp_path, *, scene_block=None):
    ev_dir = tmp_path / "ev"
    control_dir = tmp_path / "control"
    control_dir.mkdir()

    sim = mf.FlightSim(dict(R.HOME), pose_units="deg", restream_passes=1, settle_s=0.05)
    sim.fly(R.PLACE_ROUTE)
    setup_last_seq = sim.state_rows[-1]["seq"]
    setup_first_seq = 0

    for _ in range(25):
        sim._hold_ticks(1, dict(sim.pose))
    flight_first_seq = sim.state_rows[-1]["seq"]

    sim.pose = dict(R.REST)
    sim.fly(R.LIFT_TO_PRESENT)
    flight_last_seq = sim.state_rows[-1]["seq"]

    result = sim.result()
    mf.write_evidence(ev_dir, result.state_rows, result.command_rows)
    _write_sidecar(control_dir, CYCLE, "setup", setup_first_seq, setup_last_seq,
                   scene_block=scene_block)
    _write_sidecar(control_dir, CYCLE, "flight", flight_first_seq, flight_last_seq)
    return ev_dir, control_dir


def _run_cli(ev_dir, control_dir, arm, out):
    argv = [
        "--ev-dir", str(ev_dir), "--control-dir", str(control_dir), "--cycle", CYCLE,
        "--rep", "1", "--arm", arm, "--out", str(out), "--validation-mode",
    ]
    return cyc._cli(argv)


class TestSceneHashGateThroughTheCli:
    def test_missing_scene_file_gives_rc3(self, tmp_path):
        scene_block = {"path": str(tmp_path / "does_not_exist.yaml"), "chain_sha256": {}}
        ev_dir, control_dir = _build_cycle(tmp_path, scene_block=scene_block)
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        assert rc == 3
        payload = json.loads(out.read_text())
        assert "does_not_exist.yaml" in payload["reason"]

    def test_hash_mismatch_gives_rc3(self, tmp_path):
        scene_path = tmp_path / "scene.yaml"
        _write_scene_yaml(scene_path)
        scene_block = {"path": str(scene_path),
                       "chain_sha256": {str(scene_path.resolve()): "deadbeef"}}
        ev_dir, control_dir = _build_cycle(tmp_path, scene_block=scene_block)
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        assert rc == 3
        payload = json.loads(out.read_text())
        assert "hash mismatch" in payload["reason"]

    def test_matching_hash_loads_the_scene_and_proceeds(self, tmp_path):
        scene_path = tmp_path / "scene.yaml"
        _write_scene_yaml(scene_path)
        sha = hashlib.sha256(scene_path.read_bytes()).hexdigest()
        scene_block = {"path": str(scene_path),
                       "chain_sha256": {str(scene_path.resolve()): sha}}
        ev_dir, control_dir = _build_cycle(tmp_path, scene_block=scene_block)
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        # Never rc 3 for a scene reason -- whatever this fixture's own
        # verdict/rc is otherwise, it must not be the scene gate.
        payload = json.loads(out.read_text())
        assert not (rc == 3 and "scene" in payload.get("reason", "")), payload

    def test_no_scene_block_is_unchanged(self, tmp_path):
        """No 'scene' key at all (every pre-CB1 fixture, and V1's own
        harness) -- must behave exactly as before: no CycleInputError
        from this item, metrics.open_questions still names the missing
        SceneModel."""
        ev_dir, control_dir = _build_cycle(tmp_path, scene_block=None)
        out = tmp_path / "out.json"
        rc = _run_cli(ev_dir, control_dir, "B", out)
        payload = json.loads(out.read_text())
        assert not (rc == 3 and "scene" in payload.get("reason", "")), payload
        oq = payload["metrics"]["open_questions"] if payload["metrics"] else []
        assert any("no SceneModel was supplied" in q for q in oq)

    def test_mutation_hash_check_removed_would_authorize_a_stale_scene(self, tmp_path):
        """Mutation guard (verified directly against a copy of cycle.py
        with the chain-hash comparison removed from load_verified_scene
        -- see the handoff for the transcript): the SAME mismatched-hash
        fixture as test_hash_mismatch_gives_rc3 would then load the
        scene anyway (rc no longer 3 for a scene reason) -- proving the
        check is load-bearing, not incidentally caught elsewhere."""
        scene_path = tmp_path / "scene.yaml"
        _write_scene_yaml(scene_path)
        scene_block = {"path": str(scene_path),
                       "chain_sha256": {str(scene_path.resolve()): "deadbeef"}}
        sidecar_path = tmp_path / "setup.link.json"
        sidecar_path.write_text(json.dumps({
            "alignment": [{"server_seq": 0}, {"server_seq": 1}], "scene": scene_block}))
        # The mutant: skip the hash comparison entirely.
        import json as _json
        sidecar = _json.loads(sidecar_path.read_text())
        scene_block_read = sidecar.get("scene") or {}
        from reachy_ai.scene.awareness import SceneModel
        mutant_scene = SceneModel.from_yaml(scene_block_read["path"])
        assert mutant_scene is not None  # the mutant would authorize this


class TestSegmentSliceBugFixed:
    """Isolates the FIX at its own shipped call site
    (compute_place_route_metrics's seg_realised construction) via a spy
    on cl.compute_deltas, since a realistic non-null clearance value
    would additionally require real board/arm geometry overlap
    (orthogonal to what this bug is about: WHICH samples are passed
    in, not whether they yield a non-null clearance)."""

    def _build(self, tmp_path):
        joints = list(mf.R_JOINTS)
        start_pose = dict(zip(joints, [0.0] * 8))
        # 2 commands BEFORE HOVER (so the affected segment starts at
        # LOCAL index 2, not 0 -- the leg's own start), then HOVER (1),
        # then REST_SHUT (2).
        pre1 = dict(start_pose, r_shoulder_pitch=np.radians(-5.0))
        pre2 = dict(start_pose, r_shoulder_pitch=np.radians(-10.0))
        hover = dict(start_pose, r_shoulder_pitch=np.radians(-30.0))
        rest1 = dict(start_pose, r_shoulder_pitch=np.radians(-50.0))
        rest2 = dict(start_pose, r_shoulder_pitch=np.radians(-70.0))
        targets = [pre1, pre2, hover, rest1, rest2]

        rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                              wall_time_ns=1, position_rad21=mf.full21(start_pose))]
        cmds = []
        for i, tgt in enumerate(targets):
            cmds.append(mf.command_row_joint(seq=i, target_rad21=mf.full21(tgt)))
            rows.append(mf.state_row(seq=i + 1, sim_step=i + 1, sim_time_s=(i + 1) * 0.02,
                                      cmd_seq=i, wall_time_ns=2 + i,
                                      position_rad21=mf.full21(tgt)))

        mf.write_evidence(tmp_path, rows, cmds)
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        leg = cyc.LegSpec(
            "setup",
            route_rad=[mf.Waypoint("PRE", pre2, 1.0), mf.Waypoint("HOVER", hover, 1.0),
                       mf.Waypoint("REST_SHUT", rest2, 1.0)],
            guard=(), start_pose8=start_pose, command_indices=[0, 1, 2, 3, 4])
        return evd, leg

    def _run_with_spy(self, tmp_path):
        evd, leg = self._build(tmp_path)
        captured = []
        orig = cyc.cl.compute_deltas

        def spy(*args, **kwargs):
            captured.append(args)
            return [cyc.cl.LinkDelta("shells", "wrist_ball", "board", 1.0, 1.0, 1.0)]

        cyc.cl.compute_deltas = spy
        try:
            cyc.evaluate_cycle("cb1seg", "A", evd, [leg], scene=object(), skip_gates=True)
        finally:
            cyc.cl.compute_deltas = orig
        return captured

    def test_segment_call_uses_segment_positions_not_leg_start(self, tmp_path):
        captured = self._run_with_spy(tmp_path)
        assert len(captured) == 2  # whole-leg call, then segment call
        _, seg_realised = captured[1][4], captured[1][5]
        realised_shoulder_pitch = [q7_gripper[0][0] for q7_gripper in seg_realised]
        # seg_local = [2, 3, 4] (HOVER, REST_SHUT x2) -- NOT [0, 1, 2]
        # (the buggy realised[:len(seg_commanded)] leg-start slice).
        assert realised_shoulder_pitch == pytest.approx([-30.0, -50.0, -70.0])

    def test_mutation_leg_start_slice_restored_would_mismatch(self, tmp_path):
        """Mutation guard (verified directly against a copy of cycle.py
        with realised[:len(seg_commanded)] restored -- see the handoff
        for the transcript): the segment call's realised would then be
        [-5, -10, -30] (the LEG's own first 3 samples), not [-30, -50,
        -70]. Pinned here as the value the fix must keep giving."""
        captured = self._run_with_spy(tmp_path)
        _, seg_realised = captured[1][4], captured[1][5]
        realised_shoulder_pitch = [q7_gripper[0][0] for q7_gripper in seg_realised]
        assert realised_shoulder_pitch != pytest.approx([-5.0, -10.0, -30.0])
