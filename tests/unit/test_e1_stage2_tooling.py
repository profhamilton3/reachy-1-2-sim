"""Stage 2 tooling (decision note
outputs/e1-stage2-decision-2026-09-15.md): board parameterization,
repetition-aware identities, reused-identity/stale-marker refusal, the
policy-A start-variant gate, and the Stage 0 arm-on cell. Offline
throughout -- same posture as test_e1_stage1_notebook.py: notebook
generation and cell compilation via `nbformat`/`compile()`, gating logic
via `exec`-ing rendered cell source into a stub namespace. No server, no
SDK, no notebook executed as a whole, no Docker.
"""

import ast
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import traceback
import types

import nbformat
import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../scripts"))
sys.path.insert(0, os.path.join(_HERE, "../../src"))

from e1_stage1 import gating, make_cycle_notebook, plan, start_variant  # noqa: E402
from reachy_ai.motion import rig_routes  # noqa: E402


# ── Helpers (mirrors test_e1_stage1_notebook.py) ────────────────────────────

def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                    capture_output=True, text=True)


def _init_repo_two_commits(root):
    root.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", cwd=root)
    _git("config", "user.email", "t@example.com", cwd=root)
    _git("config", "user.name", "t", cwd=root)
    (root / "a.txt").write_text("a")
    _git("add", "a.txt", cwd=root)
    _git("commit", "-q", "-m", "first", cwd=root)
    first = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True,
                            check=True).stdout.strip()
    (root / "b.txt").write_text("b")
    _git("add", "b.txt", cwd=root)
    _git("commit", "-q", "-m", "second", cwd=root)
    second = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                             capture_output=True, text=True,
                             check=True).stdout.strip()
    return first, second


@pytest.fixture
def repo_two_commits(tmp_path):
    root = tmp_path / "throwaway_repo"
    first, second = _init_repo_two_commits(root)
    return root, first, second


# ── plan.py: board registry ─────────────────────────────────────────────────

class TestBoardRegistry:

    @pytest.mark.parametrize("board", plan.BOARD_ORDER)
    def test_authorized_boards_resolve(self, board):
        assert plan.board_scene_rel(board) == plan.BOARDS[board]
        assert plan.board_scene_rel(board).startswith("scenes/e1_boards/")

    @pytest.mark.parametrize("board", ["B3", "B5", "B6"])
    def test_unauthorized_boards_refused_by_name(self, board):
        with pytest.raises(ValueError, match=board):
            plan.board_scene_rel(board)

    def test_unknown_board_refused(self):
        with pytest.raises(ValueError, match="unknown board"):
            plan.board_scene_rel("B99")


# ── plan.py: SHAPES must not drift from the legacy CYCLES table ────────────

class TestShapesMatchLegacyCycles:
    """SHAPES is a second copy of the same three leg chains CYCLES already
    has (kept separate on purpose so a Stage 2 change can never touch
    Stage 1's already-evidenced structure). This pins them equal so the
    two tables cannot silently diverge."""

    _SHAPE_TO_LEGACY_CYCLE = {"a": "S1a", "b": "S1b", "c": "S1c"}

    @pytest.mark.parametrize("shape,legacy_cycle", sorted(_SHAPE_TO_LEGACY_CYCLE.items()))
    def test_shape_matches_cycle_routes_tools_desc_check(self, shape, legacy_cycle):
        specs = plan.SHAPES[shape]
        legacy = plan.CYCLES[legacy_cycle]
        assert len(specs) == len(legacy)
        for spec, leg in zip(specs, legacy):
            assert spec.route == leg.route
            assert spec.tool == leg.tool
            assert spec.desc == leg.desc
            assert spec.check == leg.check
            assert leg.name == f"{spec.kind}_{shape}"


# ── plan.py: repetition-aware identities ───────────────────────────────────

class TestCycleIdAndStage2Legs:

    def test_cycle_id_shape(self):
        assert plan.cycle_id("B4", "a", 3) == "S2-B4-a-r3"

    def test_rejects_unknown_shape(self):
        with pytest.raises(ValueError, match="unknown shape"):
            plan.cycle_id("B4", "z", 1)

    def test_rejects_non_positive_rep(self):
        with pytest.raises(ValueError, match="rep must be"):
            plan.cycle_id("B4", "a", 0)

    def test_rejects_unauthorized_board(self):
        with pytest.raises(ValueError, match="B6"):
            plan.cycle_id("B6", "a", 1)

    @pytest.mark.parametrize("board", plan.BOARD_ORDER)
    @pytest.mark.parametrize("shape", plan.SHAPE_ORDER)
    def test_stage2_legs_names_embed_board_shape_rep(self, board, shape):
        legs = plan.stage2_legs(board, shape, 5)
        cid = f"S2-{board}-{shape}-r5"
        for leg in legs:
            assert leg.name.startswith(cid)
            assert board in leg.name and shape in leg.name and "r5" in leg.name

    def test_different_reps_never_share_a_leg_name(self):
        names_r1 = {leg.name for leg in plan.stage2_legs("B4", "a", 1)}
        names_r2 = {leg.name for leg in plan.stage2_legs("B4", "a", 2)}
        assert names_r1.isdisjoint(names_r2)

    def test_different_boards_never_share_a_leg_name(self):
        names_b4 = {leg.name for leg in plan.stage2_legs("B4", "a", 1)}
        names_b1 = {leg.name for leg in plan.stage2_legs("B1", "a", 1)}
        assert names_b4.isdisjoint(names_b1)


# ── plan.py: policy-A start-variant classification ─────────────────────────

class TestClassifyStartVariant:

    _GROSS = {j: 0.0 for j in plan._R.GROSS_JOINTS}

    def test_stiff_zero(self):
        pose = dict(self._GROSS, r_gripper=0.0, r_wrist_roll=0.4)
        assert plan.classify_start_variant(pose) == "stiff-zero"

    def test_stiff_zero_tolerance_boundary(self):
        pose = dict(self._GROSS, r_gripper=1.0, r_wrist_roll=-1.0)
        assert plan.classify_start_variant(pose) == "stiff-zero"

    def test_keyframe_sag(self):
        pose = dict(self._GROSS, r_gripper=-36.0, r_wrist_roll=39.0)
        assert plan.classify_start_variant(pose) == "keyframe-sag"

    def test_keyframe_sag_within_tolerance(self):
        pose = dict(self._GROSS, r_gripper=-40.0, r_wrist_roll=35.5)
        assert plan.classify_start_variant(pose) == "keyframe-sag"

    def test_neither_variant_is_none(self):
        pose = dict(self._GROSS, r_gripper=-15.0, r_wrist_roll=15.0)
        assert plan.classify_start_variant(pose) is None

    def test_gross_joint_off_breaks_stiff_zero_even_if_wrist_and_gripper_match(self):
        pose = dict(self._GROSS, r_shoulder_pitch=10.0, r_gripper=0.0, r_wrist_roll=0.0)
        assert plan.classify_start_variant(pose) is None

    def test_gross_joint_off_breaks_keyframe_sag_even_if_wrist_and_gripper_match(self):
        """PR #124 review, M3: before the fix, a wildly-off gross posture
        (r_shoulder_pitch=45 deg here) that happened to share
        keyframe-sag's gripper/wrist_roll values still classified as
        'keyframe-sag', exit 0, no stop. Both variants share the same
        reset-keyframe gross posture (decision note §3's Stage 1 record);
        requiring it for keyframe-sag too closes the gap."""
        pose = dict(self._GROSS, r_shoulder_pitch=45.0,
                    r_gripper=-36.0, r_wrist_roll=39.0)
        assert plan.classify_start_variant(pose) is None

    def test_missing_key_is_none(self):
        assert plan.classify_start_variant({"r_gripper": 0.0}) is None


# ── start_variant.py: the CLI wrapper over a recorder log ──────────────────

class TestStartVariantScript:

    def _write_recording(self, tmp_path, joints):
        path = tmp_path / "recording.json"
        path.write_text(json.dumps({"samples": [{"t": 0.0, "joints": joints}]}))
        return path

    def test_stiff_zero_recording_exits_zero(self, tmp_path, capsys):
        gross = {j: 0.0 for j in plan._R.GROSS_JOINTS}
        path = self._write_recording(tmp_path, dict(gross, r_gripper=0.0, r_wrist_roll=0.0))
        rc = start_variant.main([str(path)])
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["start_variant"] == "stiff-zero"

    def test_unclassifiable_recording_exits_nonzero(self, tmp_path, capsys):
        gross = {j: 0.0 for j in plan._R.GROSS_JOINTS}
        path = self._write_recording(tmp_path, dict(gross, r_gripper=-15.0, r_wrist_roll=15.0))
        rc = start_variant.main([str(path)])
        assert rc == 1
        assert json.loads(capsys.readouterr().out)["start_variant"] is None

    def test_no_samples_exits_nonzero(self, tmp_path):
        path = tmp_path / "recording.json"
        path.write_text(json.dumps({"samples": []}))
        assert start_variant.main([str(path)]) == 1

    def test_wrong_argc_exits_two(self):
        assert start_variant.main([]) == 2

    def test_cycle_flag_missing_value_exits_two(self):
        assert start_variant.main(["x.json", "--cycle"]) == 2

    def test_cycle_flag_is_embedded_in_output(self, tmp_path, capsys):
        gross = {j: 0.0 for j in plan._R.GROSS_JOINTS}
        path = self._write_recording(tmp_path, dict(gross, r_gripper=0.0, r_wrist_roll=0.0))
        rc = start_variant.main([str(path), "--cycle", "S2-B4-a-r3"])
        assert rc == 0
        doc = json.loads(capsys.readouterr().out)
        assert doc["cycle"] == "S2-B4-a-r3"
        assert doc["start_variant"] == "stiff-zero"


# ── gating.py: stop outranks any later marker ──────────────────────────────

class TestGatingWaitFor:

    def test_ready_when_predicate_true(self, tmp_path):
        assert gating.wait_for(tmp_path, lambda: True, timeout_s=1, period=0.01) == "ready"

    def test_timeout_when_predicate_never_true(self, tmp_path):
        calls = {"n": 0}

        def clock():
            calls["n"] += 1
            return calls["n"] * 0.5

        result = gating.wait_for(tmp_path, lambda: False, timeout_s=1,
                                  period=0.01, now=clock, sleep=lambda s: None)
        assert result == "timeout"

    def test_stop_outranks_a_true_predicate(self, tmp_path):
        """The decision note's 'initialization failure preventing
        progression' property: once something upstream (Stage 0's arm-on
        step, an earlier leg) writes control/stop, wait_for must report
        'stop' even when the caller's own readiness marker is already
        present -- a later cycle's fresh go-marker can never override an
        outstanding stop."""
        (tmp_path / "stop").write_text("STOP: compliance_check")
        (tmp_path / "go_S2-B4-a-r2-setup").touch()
        result = gating.wait_for(
            tmp_path, lambda: (tmp_path / "go_S2-B4-a-r2-setup").exists(),
            timeout_s=1, period=0.01)
        assert result == "stop"


def _if_ancestors_of_call(tree, predicate_on_call):
    """Yield every ast.If node whose subtree contains a Call matching
    `predicate_on_call` -- copied from test_e1_stage1_notebook.py (kept
    self-contained per this repo's no-conftest test-isolation convention,
    not imported cross-file)."""
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and predicate_on_call(node):
            n = node
            while n in parents:
                n = parents[n]
                if isinstance(n, ast.If):
                    yield n


class TestStage2MotionCellGate:
    """'No route call before current readiness', proven directly on a real
    Stage 2-generated notebook (not just via the legacy _leg_source helper
    in test_e1_stage1_notebook.py) -- motion_cell_source is the same
    function either way, but this pins the invariant against the actual
    generate_repetition output so a future change to that entry point
    cannot silently lose the gate."""

    @pytest.mark.parametrize("shape", plan.SHAPE_ORDER)
    def test_every_stage2_leg_route_call_is_gated_on_chk_ok(
            self, shape, repo_two_commits, tmp_path):
        repo, first, second = repo_two_commits
        out = make_cycle_notebook.generate_repetition(
            "B4", shape, 1, session="sess1", repo=str(repo),
            evidence_dir=str(tmp_path / "evidence"), required_sha=first)
        nb = nbformat.read(str(out), as_version=4)
        legs = plan.stage2_legs("B4", shape, 1)
        motion_cells = [c for c in nb.cells
                        if c.cell_type == "code" and c.source.startswith("# Leg ")]
        assert len(motion_cells) == len(legs)
        for leg, cell in zip(legs, motion_cells):
            src = cell.source
            assert "require_compliance(" in src
            assert "min_cmd_seq=None" not in src
            route_fn = leg.tool.split(".")[-1]
            tree = ast.parse(src)

            def is_route_call(node, route_fn=route_fn):
                return isinstance(node.func, ast.Attribute) and node.func.attr == route_fn

            ancestor_ifs = list(_if_ancestors_of_call(tree, is_route_call))
            assert ancestor_ifs, f"no route call found for {route_fn} in:\n{src}"
            assert any("chk" in ast.unparse(n.test) and "ok" in ast.unparse(n.test)
                       for n in ancestor_ifs)


# ── make_cycle_notebook.py: board propagation into the generated notebook ──

class TestBoardPropagation:

    @pytest.mark.parametrize("board", plan.BOARD_ORDER)
    def test_stage2_connect_cell_scene_matches_board(self, board, repo_two_commits, tmp_path):
        repo, first, second = repo_two_commits
        out = make_cycle_notebook.generate_repetition(
            board, "b", 1, session="sess1", repo=str(repo),
            evidence_dir=str(tmp_path / "evidence"), required_sha=first)
        nb = nbformat.read(str(out), as_version=4)
        connect_src = nb.cells[1].source
        tree = ast.parse(connect_src)
        scene_literal = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "SCENE":
                        scene_literal = ast.literal_eval(node.value)
        assert scene_literal == f"{repo}/{plan.BOARDS[board]}"

    @pytest.mark.parametrize("board", plan.BOARD_ORDER)
    def test_stage2_plan_doc_records_board_and_scene(self, board, repo_two_commits, tmp_path):
        repo, first, second = repo_two_commits
        evidence_dir = tmp_path / "evidence"
        make_cycle_notebook.generate_repetition(
            board, "b", 1, session="sess1", repo=str(repo),
            evidence_dir=str(evidence_dir), required_sha=first)
        plan_doc = json.loads((evidence_dir / "plan_S2-{}-b-r1.json".format(board)).read_text())
        assert plan_doc["board"] == board
        assert plan_doc["scene_rel"] == plan.BOARDS[board]

    def test_legacy_generate_defaults_to_b4_scene_unchanged(self, repo_two_commits, tmp_path):
        """No behaviour change for Stage 1: the default board is B4, and
        the SCENE literal is byte-identical to the pre-Stage-2 hard-coded
        string."""
        repo, first, second = repo_two_commits
        out = make_cycle_notebook.generate(
            "S1a", repo=str(repo), evidence_dir=str(tmp_path / "evidence"),
            required_sha=first)
        nb = nbformat.read(str(out), as_version=4)
        assert 'SCENE = "' + str(repo) + '/scenes/e1_boards/B4_pool_box_1_r2c3.yaml"' \
            in nb.cells[1].source


# ── make_cycle_notebook.py: repetition isolation ────────────────────────────

class TestRepetitionIsolation:

    def test_two_repetitions_generate_independently(self, repo_two_commits, tmp_path):
        repo, first, second = repo_two_commits
        evidence_dir = str(tmp_path / "evidence")
        out1 = make_cycle_notebook.generate_repetition(
            "B4", "a", 1, session="sess1", repo=str(repo),
            evidence_dir=evidence_dir, required_sha=first)
        out2 = make_cycle_notebook.generate_repetition(
            "B4", "a", 2, session="sess1", repo=str(repo),
            evidence_dir=evidence_dir, required_sha=first)
        assert out1 != out2
        assert out1.exists() and out2.exists()

    def test_an_earlier_repetitions_leftover_go_marker_cannot_satisfy_a_later_one(
            self, repo_two_commits, tmp_path):
        """The exact bug decision note §6 item 2 describes: rep 1's
        `go_<leg>` marker must never be found by rep 2's motion cell,
        because the two reps' leg names differ."""
        repo, first, second = repo_two_commits
        evidence_dir = tmp_path / "evidence"
        out1 = make_cycle_notebook.generate_repetition(
            "B4", "a", 1, session="sess1", repo=str(repo),
            evidence_dir=str(evidence_dir), required_sha=first)
        control = evidence_dir / "control"
        control.mkdir(parents=True, exist_ok=True)
        rep1_legs = plan.stage2_legs("B4", "a", 1)
        for leg in rep1_legs:
            (control / f"go_{leg.name}").touch()
            (control / f"{leg.name}_done").write_text("{}")

        out2 = make_cycle_notebook.generate_repetition(
            "B4", "a", 2, session="sess1", repo=str(repo),
            evidence_dir=str(evidence_dir), required_sha=first)
        nb2 = nbformat.read(str(out2), as_version=4)
        rep2_legs = plan.stage2_legs("B4", "a", 2)
        motion_cells = [c for c in nb2.cells
                        if c.cell_type == "code" and c.source.startswith("# Leg ")]
        for leg, cell in zip(rep2_legs, motion_cells):
            assert f'go_{leg.name}' in cell.source
            for rep1_leg in rep1_legs:
                assert rep1_leg.name not in cell.source


# ── make_cycle_notebook.py: reused-identity / stale-marker refusal ─────────

class TestReusedIdentityRefusal:

    def test_regenerating_the_same_repetition_is_refused(self, repo_two_commits, tmp_path):
        repo, first, second = repo_two_commits
        evidence_dir = str(tmp_path / "evidence")
        make_cycle_notebook.generate_repetition(
            "B4", "a", 1, session="sess1", repo=str(repo),
            evidence_dir=evidence_dir, required_sha=first)
        with pytest.raises(make_cycle_notebook.GenerationRefused, match="already exists"):
            make_cycle_notebook.generate_repetition(
                "B4", "a", 1, session="sess1", repo=str(repo),
                evidence_dir=evidence_dir, required_sha=first)

    def test_stale_marker_without_a_plan_file_is_also_refused(self, repo_two_commits, tmp_path):
        """Simulates a crashed prior attempt that wrote a marker but never
        got as far as (or lost) its plan file -- the refusal must not rely
        on the plan file alone."""
        repo, first, second = repo_two_commits
        evidence_dir = tmp_path / "evidence"
        legs = plan.stage2_legs("B4", "a", 1)
        control = evidence_dir / "control"
        control.mkdir(parents=True)
        (control / f"go_{legs[0].name}").touch()
        with pytest.raises(make_cycle_notebook.GenerationRefused, match="stale marker"):
            make_cycle_notebook.generate_repetition(
                "B4", "a", 1, session="sess1", repo=str(repo),
                evidence_dir=str(evidence_dir), required_sha=first)

    def test_a_different_session_for_an_already_used_board_is_refused(
            self, repo_two_commits, tmp_path):
        repo, first, second = repo_two_commits
        evidence_dir = str(tmp_path / "evidence")
        make_cycle_notebook.generate_repetition(
            "B4", "a", 1, session="sess1", repo=str(repo),
            evidence_dir=evidence_dir, required_sha=first)
        with pytest.raises(make_cycle_notebook.GenerationRefused, match="session"):
            make_cycle_notebook.generate_repetition(
                "B4", "b", 1, session="sess2", repo=str(repo),
                evidence_dir=evidence_dir, required_sha=first)

    def test_the_same_session_for_the_same_board_is_allowed_across_cycles(
            self, repo_two_commits, tmp_path):
        repo, first, second = repo_two_commits
        evidence_dir = str(tmp_path / "evidence")
        make_cycle_notebook.generate_repetition(
            "B4", "a", 1, session="sess1", repo=str(repo),
            evidence_dir=evidence_dir, required_sha=first)
        out = make_cycle_notebook.generate_repetition(
            "B4", "b", 1, session="sess1", repo=str(repo),
            evidence_dir=evidence_dir, required_sha=first)
        assert out.exists()

    def test_stage0_regeneration_for_the_same_board_session_is_refused(
            self, repo_two_commits, tmp_path):
        repo, first, second = repo_two_commits
        evidence_dir = str(tmp_path / "evidence")
        make_cycle_notebook.generate_stage0(
            "B4", "sess1", repo=str(repo), evidence_dir=evidence_dir,
            required_sha=first)
        with pytest.raises(make_cycle_notebook.GenerationRefused):
            make_cycle_notebook.generate_stage0(
                "B4", "sess1", repo=str(repo), evidence_dir=evidence_dir,
                required_sha=first)


# ── make_cycle_notebook.py: Stage 0 arm-on cell ─────────────────────────────

class TestArmonCell:

    def test_stage0_notebook_has_the_expected_cell_shape(self, repo_two_commits, tmp_path):
        repo, first, second = repo_two_commits
        out = make_cycle_notebook.generate_stage0(
            "B4", "sess1", repo=str(repo), evidence_dir=str(tmp_path / "evidence"),
            required_sha=first)
        nb = nbformat.read(str(out), as_version=4)
        nbformat.validate(nb)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert len(code_cells) == 4  # connect, binding, armon, final
        for i, cell in enumerate(code_cells):
            compile(cell.source, f"<stage0 cell {i}>", "exec")
        assert code_cells[2].source.startswith("# Stage 0")

    def test_armon_cell_never_calls_any_route_function(self):
        src = make_cycle_notebook.armon_cell_source("stage0-B4-sess1-armon")
        tree = ast.parse(src)
        route_fns = {call.split(".")[-1] for call in plan.TOOL_TO_CALL}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in route_fns

    def test_armon_cell_requires_compliance_gate_before_turn_on(self):
        src = make_cycle_notebook.armon_cell_source("stage0-B4-sess1-armon")
        assert "require_compliance(" in src
        assert "min_cmd_seq=" in src
        tree = ast.parse(src)
        # turn_on must be nested under the baseline-cmd_seq validity check,
        # same shape as every leg's motion cell.
        found_turn_on_if = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                src_if = ast.unparse(node.test)
                if "baseline_cmd_seq" in src_if or "isinstance" in src_if:
                    if "turn_on" in ast.unparse(node):
                        found_turn_on_if = True
        assert found_turn_on_if

    # ── exec-based gating tests, mirroring test_e1_stage1_notebook.py's
    # _build_namespace but for the no-route armon cell ──────────────────────

    def _build_armon_namespace(self, tmp_path, *, prev_ok, baseline,
                                require_compliance_result, calls,
                                leg_name="stage0-B4-sess1-armon"):
        (tmp_path / f"go_{leg_name}").touch()
        (tmp_path / f"recorder_{leg_name}.log").write_text("fly the route now")

        def fake_read_last_state(path):
            return baseline

        def fake_require_compliance(run_dir, joints, *, compliant=False,
                                     timeout_s=None, min_cmd_seq=None):
            calls["require_compliance"].append(
                {"run_dir": run_dir, "min_cmd_seq": min_cmd_seq})
            return require_compliance_result

        def fake_turn_on(*a, **kw):
            calls["turn_on"].append((a, kw))

        def wait_for(pred, timeout_s, period=0.25):
            return "ready" if pred() else "timeout"

        def _pose():
            return {"r_shoulder_pitch": 0.0}

        return {
            "reachy": types.SimpleNamespace(
                turn_on=fake_turn_on, r_arm=types.SimpleNamespace()),
            "e1_identity": types.SimpleNamespace(
                _read_last_state=fake_read_last_state,
                require_compliance=fake_require_compliance),
            "R": types.SimpleNamespace(R_JOINTS=("r_shoulder_pitch",)),
            "ident": types.SimpleNamespace(run_dir=str(tmp_path)),
            "CTRL": tmp_path,
            "wait_for": wait_for,
            "_pose": _pose,
            "PREV_OK": prev_ok,
            "CYCLE": "test_stage0",
            "LEAD_IN_S": 0.0,
            "COMPLIANCE_TIMEOUT_S": 0.0,
            "time": types.SimpleNamespace(
                monotonic=time.monotonic, monotonic_ns=time.monotonic_ns,
                time_ns=time.time_ns, sleep=lambda s: None),
            "json": json,
            "pathlib": __import__("pathlib"),
            "traceback": traceback,
        }

    class _FakeChk:
        def __init__(self, ok):
            self.ok = ok

        def as_dict(self):
            return {"ok": self.ok}

    def test_invalid_baseline_stops_before_turn_on(self, tmp_path):
        calls = {"turn_on": [], "require_compliance": []}
        ns = self._build_armon_namespace(
            tmp_path, prev_ok=True, baseline={}, require_compliance_result=None,
            calls=calls)
        src = make_cycle_notebook.armon_cell_source("stage0-B4-sess1-armon")
        exec(compile(src, "<armon invalid baseline>", "exec"), ns)
        assert calls["turn_on"] == []
        assert calls["require_compliance"] == []
        assert ns["LEG"]["outcome"].startswith("STOP no_valid_baseline")
        assert (tmp_path / "stop").exists()
        assert ns["PREV_OK"] is False

    def test_compliance_failure_stops_and_writes_stop_marker(self, tmp_path):
        calls = {"turn_on": [], "require_compliance": []}
        ns = self._build_armon_namespace(
            tmp_path, prev_ok=True, baseline={"cmd_seq": 3},
            require_compliance_result=self._FakeChk(ok=False), calls=calls)
        src = make_cycle_notebook.armon_cell_source("stage0-B4-sess1-armon")
        exec(compile(src, "<armon compliance fails>", "exec"), ns)
        assert len(calls["turn_on"]) == 1
        assert ns["LEG"]["outcome"] == "STOP compliance_check"
        assert (tmp_path / "stop").exists()
        assert ns["PREV_OK"] is False

    def test_compliance_success_marks_returned_with_no_route_call(self, tmp_path):
        calls = {"turn_on": [], "require_compliance": []}
        ns = self._build_armon_namespace(
            tmp_path, prev_ok=True, baseline={"cmd_seq": 3},
            require_compliance_result=self._FakeChk(ok=True), calls=calls)
        src = make_cycle_notebook.armon_cell_source("stage0-B4-sess1-armon")
        exec(compile(src, "<armon compliance ok>", "exec"), ns)
        assert len(calls["turn_on"]) == 1
        assert ns["LEG"]["outcome"] == "returned"
        assert not (tmp_path / "stop").exists()
        assert ns["PREV_OK"] is True

    def test_stage0_stop_blocks_a_later_cycles_go_wait(self, tmp_path):
        """End-to-end-ish: a failed armon compliance check writes
        control/stop; a subsequent (real) cycle's own gating.wait_for --
        the exact function every generated binding cell calls -- must
        report 'stop' rather than 'ready' even though its own fresh go
        marker is present. This is 'initialization failure preventing
        progression'."""
        calls = {"turn_on": [], "require_compliance": []}
        ns = self._build_armon_namespace(
            tmp_path, prev_ok=True, baseline={"cmd_seq": 3},
            require_compliance_result=self._FakeChk(ok=False), calls=calls)
        src = make_cycle_notebook.armon_cell_source("stage0-B4-sess1-armon")
        exec(compile(src, "<armon compliance fails>", "exec"), ns)
        assert (tmp_path / "stop").exists()

        next_leg_name = plan.stage2_legs("B4", "a", 1)[0].name
        (tmp_path / f"go_{next_leg_name}").touch()
        result = gating.wait_for(
            tmp_path, lambda: (tmp_path / f"go_{next_leg_name}").exists(),
            timeout_s=1, period=0.01)
        assert result == "stop"


# ── M1 (PR #124 review): Stage 0 armon markers are identity-scoped ─────────

class TestStage0IdentityScopedMarkers:
    """Before this fix every board's Stage 0 notebook used the fixed
    marker names go_armon/recorder_armon.log/armon_done, so one board's
    leftover Stage 0 markers could satisfy a DIFFERENT board's armon
    cell -- the one action this package insists is not motion-free.
    Proves B4's Stage 0 artifacts cannot authorize B1's turn_on."""

    def test_b4_stage0_leftover_markers_do_not_appear_in_b1s_armon_cell(
            self, repo_two_commits, tmp_path):
        repo, first, second = repo_two_commits
        evidence_dir = tmp_path / "evidence"
        make_cycle_notebook.generate_stage0(
            "B4", "sess1", repo=str(repo), evidence_dir=str(evidence_dir),
            required_sha=first)
        b4_leg = plan.stage0_armon_leg_name("B4", "sess1")
        control = evidence_dir / "control"
        control.mkdir(parents=True, exist_ok=True)
        (control / f"go_{b4_leg}").touch()
        (control / f"recorder_{b4_leg}.log").write_text("fly the route now")
        (control / f"{b4_leg}_done").write_text("{}")

        out_b1 = make_cycle_notebook.generate_stage0(
            "B1", "sess1", repo=str(repo), evidence_dir=str(evidence_dir),
            required_sha=first)
        nb_b1 = nbformat.read(str(out_b1), as_version=4)
        armon_cell = [c for c in nb_b1.cells
                      if c.cell_type == "code" and c.source.startswith("# Stage 0")][0]
        b1_leg = plan.stage0_armon_leg_name("B1", "sess1")
        assert b1_leg != b4_leg
        assert f"go_{b1_leg}" in armon_cell.source
        assert b4_leg not in armon_cell.source

    def test_b4_stage0_leftover_markers_do_not_satisfy_b1s_gate_at_runtime(
            self, tmp_path):
        """Exec-based: B4's leftover go/recorder markers sit on disk; B1's
        own rendered armon cell must not find them ready, so turn_on is
        never reached."""
        b4_leg = plan.stage0_armon_leg_name("B4", "sess1")
        b1_leg = plan.stage0_armon_leg_name("B1", "sess1")
        (tmp_path / f"go_{b4_leg}").touch()
        (tmp_path / f"recorder_{b4_leg}.log").write_text("fly the route now")

        calls = {"turn_on": []}

        def fake_turn_on(*a, **kw):
            calls["turn_on"].append((a, kw))

        def wait_for(pred, timeout_s, period=0.25):
            return "ready" if pred() else "timeout"

        ns = {
            "reachy": types.SimpleNamespace(
                turn_on=fake_turn_on, r_arm=types.SimpleNamespace()),
            "e1_identity": types.SimpleNamespace(),
            "R": types.SimpleNamespace(R_JOINTS=("r_shoulder_pitch",)),
            "ident": types.SimpleNamespace(run_dir=str(tmp_path)),
            "CTRL": tmp_path,
            "wait_for": wait_for,
            "_pose": lambda: {"r_shoulder_pitch": 0.0},
            "PREV_OK": True,
            "CYCLE": "stage0-B1-sess1",
            "LEAD_IN_S": 0.0,
            "COMPLIANCE_TIMEOUT_S": 0.0,
            "time": types.SimpleNamespace(
                monotonic=time.monotonic, monotonic_ns=time.monotonic_ns,
                time_ns=time.time_ns, sleep=lambda s: None),
            "json": json, "pathlib": __import__("pathlib"),
            "traceback": traceback,
        }
        src = make_cycle_notebook.armon_cell_source(b1_leg)
        exec(compile(src, "<b1 armon>", "exec"), ns)
        assert ns["LEG"]["go"] == "timeout"
        assert calls["turn_on"] == []
        assert ns["LEG"]["outcome"] == "not_attempted"


# ── M2 (PR #124 review): refusals happen before any write; the session
# ledger is recorded atomically and is concurrency-safe ────────────────────

class TestSessionLedgerRefusesBeforeWriting:
    """Every refusal this package can raise for generate_repetition/
    generate_stage0 must leave the evidence directory -- and the session
    ledger specifically -- exactly as it found it. A rejected call must
    never reserve a session or otherwise dirty state."""

    def _ledger(self, evidence_dir):
        path = pathlib.Path(evidence_dir) / "control" / "e1_stage2_sessions.json"
        return json.loads(path.read_text()) if path.is_file() else None

    def test_bad_repo_refusal_leaves_no_ledger_and_a_correct_retry_succeeds(
            self, repo_two_commits, tmp_path):
        repo, first, second = repo_two_commits
        evidence_dir = str(tmp_path / "evidence")
        with pytest.raises(make_cycle_notebook.GenerationRefused,
                            match="git rev-parse"):
            make_cycle_notebook.generate_repetition(
                "B4", "a", 1, session="s1", repo=str(tmp_path / "not_a_repo"),
                evidence_dir=evidence_dir, required_sha=first)
        assert self._ledger(evidence_dir) is None

        out = make_cycle_notebook.generate_repetition(
            "B4", "a", 1, session="s2", repo=str(repo),
            evidence_dir=evidence_dir, required_sha=first)
        assert out.exists()
        assert self._ledger(evidence_dir)["B4"] == "s2"

    def test_docs_reviews_refusal_leaves_the_repo_clean(
            self, repo_two_commits, tmp_path):
        repo, first, second = repo_two_commits
        bad_evidence_dir = repo / "docs" / "reviews" / "historical-evidence"
        with pytest.raises(make_cycle_notebook.GenerationRefused,
                            match="docs/reviews"):
            make_cycle_notebook.generate_repetition(
                "B4", "a", 1, session="s1", repo=str(repo),
                evidence_dir=str(bad_evidence_dir), required_sha=first)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                                 capture_output=True, text=True, check=True)
        assert status.stdout.strip() == ""
        assert not bad_evidence_dir.exists()

    def test_stale_marker_refusal_does_not_pin_the_session(
            self, repo_two_commits, tmp_path):
        repo, first, second = repo_two_commits
        evidence_dir = tmp_path / "evidence"
        legs = plan.stage2_legs("B4", "a", 1)
        control = evidence_dir / "control"
        control.mkdir(parents=True)
        (control / f"go_{legs[0].name}").touch()
        with pytest.raises(make_cycle_notebook.GenerationRefused,
                            match="stale marker"):
            make_cycle_notebook.generate_repetition(
                "B4", "a", 1, session="typo-session", repo=str(repo),
                evidence_dir=str(evidence_dir), required_sha=first)
        assert self._ledger(evidence_dir) is None


class TestConcurrentSessionReservation:
    """PR #124 review, M2: a 12-way concurrent race for the SAME board
    with DIFFERENT session ids used to produce more than one 'winner'
    (demonstrated in the review as 1 OK / 11 refused -- a narrow but real
    window). Locking + atomic write + a re-check under the lock must make
    this exactly one winner, every time, with no corrupt ledger and no
    orphan notebook/plan file for a call that ultimately lost the race."""

    def test_concurrent_generate_repetition_same_board_exactly_one_session_wins(
            self, repo_two_commits, tmp_path):
        repo, first, second = repo_two_commits
        evidence_dir = str(tmp_path / "evidence")
        n = 12
        results = [None] * n
        barrier = threading.Barrier(n)

        def worker(i):
            barrier.wait()
            try:
                make_cycle_notebook.generate_repetition(
                    "B4", "a", i + 1, session=f"sess{i}", repo=str(repo),
                    evidence_dir=evidence_dir, required_sha=first)
                results[i] = "ok"
            except make_cycle_notebook.GenerationRefused as exc:
                results[i] = f"refused: {exc}"

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        oks = [r for r in results if r == "ok"]
        assert len(oks) == 1, results

        ledger_path = pathlib.Path(evidence_dir) / "control" / "e1_stage2_sessions.json"
        ledger = json.loads(ledger_path.read_text())
        assert set(ledger.keys()) == {"B4"}
        winner_index = results.index("ok")
        assert ledger["B4"] == f"sess{winner_index}"

        for i, r in enumerate(results):
            if r != "ok":
                assert "session" in r
            else:
                identity = plan.cycle_id("B4", "a", i + 1)
                assert (pathlib.Path(evidence_dir) / f"plan_{identity}.json").is_file()
            if i != winner_index:
                identity = plan.cycle_id("B4", "a", i + 1)
                assert not (pathlib.Path(evidence_dir) / f"plan_{identity}.json").is_file()


# ── M3 (PR #124 review): the per-cycle start-variant gate, bound to the
# cycle, in the generated execution path ────────────────────────────────────

class TestStartVariantGate:
    """plan.start_variant_gate: missing, corrupt, wrong-cycle, tampered,
    and non-stiff-zero (including keyframe-sag) evidence must all refuse;
    only a stiff-zero pose bound to the exact cycle passes."""

    _STIFF_ZERO_POSE = dict({j: 0.0 for j in plan._R.GROSS_JOINTS},
                             r_gripper=0.0, r_wrist_roll=0.0)
    _KEYFRAME_SAG_POSE = dict({j: 0.0 for j in plan._R.GROSS_JOINTS},
                              r_gripper=-36.0, r_wrist_roll=39.0)

    def _write(self, ctrl, cycle, *, start_variant, pose, write_cycle=None):
        (ctrl / f"start_variant_{cycle}.json").write_text(json.dumps({
            "start_variant": start_variant, "pose": pose,
            "cycle": cycle if write_cycle is None else write_cycle,
        }))

    def test_missing_evidence_refuses(self, tmp_path):
        ok, reason, doc = plan.start_variant_gate(tmp_path, "S2-B4-a-r1")
        assert ok is False
        assert "missing" in reason
        assert doc is None

    def test_corrupt_evidence_refuses(self, tmp_path):
        (tmp_path / "start_variant_S2-B4-a-r1.json").write_text("{not json")
        ok, reason, doc = plan.start_variant_gate(tmp_path, "S2-B4-a-r1")
        assert ok is False
        assert "corrupt" in reason

    def test_wrong_cycle_evidence_refuses(self, tmp_path):
        self._write(tmp_path, "S2-B4-a-r1", start_variant="stiff-zero",
                    pose=self._STIFF_ZERO_POSE, write_cycle="S2-B4-a-r2")
        ok, reason, doc = plan.start_variant_gate(tmp_path, "S2-B4-a-r1")
        assert ok is False
        assert "bound to cycle 'S2-B4-a-r2'" in reason

    def test_keyframe_sag_evidence_refuses_under_policy_a(self, tmp_path):
        self._write(tmp_path, "S2-B4-a-r1", start_variant="keyframe-sag",
                    pose=self._KEYFRAME_SAG_POSE)
        ok, reason, doc = plan.start_variant_gate(tmp_path, "S2-B4-a-r1")
        assert ok is False
        assert "not 'stiff-zero'" in reason

    def test_tampered_declared_variant_refuses(self, tmp_path):
        """Declared 'stiff-zero' but the recorded pose actually reclassifies
        differently -- the gate must not trust the declared field verbatim."""
        self._write(tmp_path, "S2-B4-a-r1", start_variant="stiff-zero",
                    pose=self._KEYFRAME_SAG_POSE)
        ok, reason, doc = plan.start_variant_gate(tmp_path, "S2-B4-a-r1")
        assert ok is False
        assert "does not match" in reason

    def test_valid_stiff_zero_bound_to_this_cycle_passes(self, tmp_path):
        self._write(tmp_path, "S2-B4-a-r1", start_variant="stiff-zero",
                    pose=self._STIFF_ZERO_POSE)
        ok, reason, doc = plan.start_variant_gate(tmp_path, "S2-B4-a-r1")
        assert ok is True
        assert reason is None
        assert doc["start_variant"] == "stiff-zero"


class TestStartVariantGateWiredIntoGeneratedNotebooks:

    @pytest.mark.parametrize("shape", plan.SHAPE_ORDER)
    def test_generate_repetition_binding_cell_calls_the_gate(
            self, shape, repo_two_commits, tmp_path):
        repo, first, second = repo_two_commits
        out = make_cycle_notebook.generate_repetition(
            "B4", shape, 1, session="sess1", repo=str(repo),
            evidence_dir=str(tmp_path / "evidence"), required_sha=first)
        nb = nbformat.read(str(out), as_version=4)
        binding_src = nb.cells[2].source
        assert "plan.start_variant_gate(CTRL, CYCLE)" in binding_src
        assert "PREV_OK = PREV_OK and START_VARIANT_OK" in binding_src
        compile(binding_src, "<binding>", "exec")

    def test_legacy_generate_binding_cell_has_no_start_variant_gate(
            self, repo_two_commits, tmp_path):
        """Stage 1 compatibility (PR #124 review, M4): the gate is a Stage
        2 addition and must not change Stage 1's already-evidenced
        notebook shape."""
        repo, first, second = repo_two_commits
        out = make_cycle_notebook.generate(
            "S1a", repo=str(repo), evidence_dir=str(tmp_path / "evidence"),
            required_sha=first)
        nb = nbformat.read(str(out), as_version=4)
        binding_src = nb.cells[2].source
        assert "start_variant_gate" not in binding_src

    def test_stage0_binding_cell_has_no_start_variant_gate(
            self, repo_two_commits, tmp_path):
        """Stage 0 is what MAKES the arm stiff-zero in the first place --
        there is no preceding parked recording for it to check."""
        repo, first, second = repo_two_commits
        out = make_cycle_notebook.generate_stage0(
            "B4", "sess1", repo=str(repo),
            evidence_dir=str(tmp_path / "evidence"), required_sha=first)
        nb = nbformat.read(str(out), as_version=4)
        binding_src = nb.cells[2].source
        assert "start_variant_gate" not in binding_src


# ── R2 (2026-09-16 re-review): fresh pre-cycle compliance, alongside the
# stiff-zero posture evidence, before any cycle command ─────────────────────

class TestPreCycleComplianceGate:
    """`start_variant_gate` reads only joint angles; a compliant arm reads
    'zero within 1 deg' for part of its own settle window too (README,
    already-stiff-arm investigation; decision note §3), so a stiff-zero
    posture alone does not prove the arm is currently stiff. The binding
    cell must also read a FRESH server-state sample (no `min_cmd_seq` --
    nothing commanded yet this cycle) and require `compliant=False` for
    every required joint, folded into `PREV_OK` alongside
    `START_VARIANT_OK` so neither `turn_on` nor a route call is reachable
    on missing, malformed, stale, or explicitly compliant evidence.
    `require_compliance`'s own fail-closed handling of those four cases is
    covered by `test_e1_identity.py::TestRequireCompliance`, unchanged by
    this PR; these tests cover the NEW wiring only."""

    def test_binding_cell_calls_require_compliance_with_freshness_only(self):
        src = make_cycle_notebook.binding_cell_source(require_start_variant=True)
        # Exact call text -- pins both the arguments present (compliant=False,
        # timeout_s=COMPLIANCE_TIMEOUT_S) and, by construction, the absence of
        # min_cmd_seq (nothing has been commanded this cycle yet, so this is a
        # freshness-window-only read, unlike the per-leg check after turn_on).
        assert ("COMPLIANCE_CHECK = e1_identity.require_compliance("
                "ident.run_dir, R.R_JOINTS, compliant=False, "
                "timeout_s=COMPLIANCE_TIMEOUT_S)") in src
        assert "PREV_OK = PREV_OK and START_VARIANT_OK and COMPLIANCE_CHECK.ok" in src
        compile(src, "<binding>", "exec")

    def test_legacy_generate_binding_cell_has_no_compliance_gate(self):
        src = make_cycle_notebook.binding_cell_source(require_start_variant=False)
        assert "COMPLIANCE_CHECK" not in src

    def test_stage0_binding_cell_has_no_compliance_gate(self, repo_two_commits, tmp_path):
        """Stage 0 is what makes the arm stiff in the first place -- there
        is no preceding cycle state to freshly re-check."""
        repo, first, second = repo_two_commits
        out = make_cycle_notebook.generate_stage0(
            "B4", "sess1", repo=str(repo),
            evidence_dir=str(tmp_path / "evidence"), required_sha=first)
        nb = nbformat.read(str(out), as_version=4)
        assert "COMPLIANCE_CHECK" not in nb.cells[2].source

    def _exec_gate_fragment(self, tmp_path, *, start_variant_ok, compliance_ok,
                             reasons=()):
        """Execs the actual `require_start_variant=True` appended fragment
        (not a re-implementation), stubbing only the two I/O boundaries
        (`plan.start_variant_gate`, `e1_identity.require_compliance`) so
        the gate's own control flow -- fold into PREV_OK, write
        binding_FAIL -- runs for real."""
        full_src = make_cycle_notebook.binding_cell_source(require_start_variant=True)
        fragment = full_src.split("PREV_OK = BINDING_OK\n", 1)[1]

        def fake_start_variant_gate(ctrl_dir, cycle):
            return start_variant_ok, (None if start_variant_ok else "stub refusal"), None

        def fake_require_compliance(run_dir, joints, *, compliant=False, timeout_s=None):
            return types.SimpleNamespace(
                ok=compliance_ok, reasons=tuple(reasons),
                as_dict=lambda: {"ok": compliance_ok, "reasons": list(reasons)})

        ns = {
            "plan": types.SimpleNamespace(start_variant_gate=fake_start_variant_gate),
            "e1_identity": types.SimpleNamespace(require_compliance=fake_require_compliance),
            "R": types.SimpleNamespace(R_JOINTS=("r_shoulder_pitch",)),
            "ident": types.SimpleNamespace(run_dir=str(tmp_path)),
            "CTRL": tmp_path, "CYCLE": "S2-B4-a-r1",
            "COMPLIANCE_TIMEOUT_S": 0.0,
            "PREV_OK": True,
            "json": json,
        }
        exec(compile(fragment, "<r2 gate fragment>", "exec"), ns)
        return ns

    def test_both_gates_ok_leaves_prev_ok_true(self, tmp_path):
        ns = self._exec_gate_fragment(tmp_path, start_variant_ok=True, compliance_ok=True)
        assert ns["PREV_OK"] is True
        assert not (tmp_path / "binding_FAIL_S2-B4-a-r1").exists()

    def test_compliance_failure_alone_blocks_prev_ok(self, tmp_path):
        """Posture (start_variant) passes but the fresh compliance read
        does not -- e.g. a compliant arm reading zero mid-settle -- PREV_OK
        must still go False."""
        ns = self._exec_gate_fragment(
            tmp_path, start_variant_ok=True, compliance_ok=False,
            reasons=("r_gripper: compliant=True (want False)",))
        assert ns["PREV_OK"] is False
        marker = tmp_path / "binding_FAIL_S2-B4-a-r1"
        assert marker.exists()
        assert "pre_cycle_compliance" in marker.read_text()

    def test_start_variant_failure_alone_still_blocks_even_if_compliance_ok(self, tmp_path):
        ns = self._exec_gate_fragment(tmp_path, start_variant_ok=False, compliance_ok=True)
        assert ns["PREV_OK"] is False

    def test_both_gates_failing_blocks(self, tmp_path):
        ns = self._exec_gate_fragment(tmp_path, start_variant_ok=False, compliance_ok=False)
        assert ns["PREV_OK"] is False

    def test_compliance_failure_blocks_the_next_legs_turn_on_and_route(
            self, repo_two_commits, tmp_path):
        """Same shape as TestNoRouteCallWithoutCurrentGates, but the cause
        is specifically a failed pre-cycle compliance read: PREV_OK False
        makes every leg's motion cell not_eligible / not_attempted, with
        no turn_on/route call reachable."""
        repo, first, second = repo_two_commits
        out = make_cycle_notebook.generate_repetition(
            "B4", "a", 1, session="sess1", repo=str(repo),
            evidence_dir=str(tmp_path / "evidence"), required_sha=first)
        nb = nbformat.read(str(out), as_version=4)
        motion_cells = [c for c in nb.cells
                        if c.cell_type == "code" and c.source.startswith("# Leg ")]
        assert motion_cells

        ns = {
            "PREV_OK": False,  # what a failed pre-cycle compliance check produces
            "CTRL": tmp_path, "CYCLE": "S2-B4-a-r1", "json": json,
        }
        exec(compile(motion_cells[0].source, "<leg0 prev_ok False (compliance)>", "exec"), ns)
        assert ns["LEG"]["go"] == "not_eligible"
        assert ns["LEG"]["outcome"] == "not_attempted"


class TestNoRouteCallWithoutCurrentGates:
    """A failed start_variant_gate sets PREV_OK False in the binding cell
    (decision note §4; PR #124 review, M3). Proves that on a REAL Stage
    2-generated leg cell, PREV_OK False makes the leg not_eligible with no
    turn_on/route call reachable -- the stub namespace below deliberately
    omits `reachy`/`turn_on`/the route module, so if the gate wiring ever
    regressed and either became reachable, this raises NameError rather
    than silently passing."""

    def test_prev_ok_false_from_the_gate_blocks_turn_on_and_route(
            self, repo_two_commits, tmp_path):
        repo, first, second = repo_two_commits
        out = make_cycle_notebook.generate_repetition(
            "B4", "a", 1, session="sess1", repo=str(repo),
            evidence_dir=str(tmp_path / "evidence"), required_sha=first)
        nb = nbformat.read(str(out), as_version=4)
        motion_cells = [c for c in nb.cells
                        if c.cell_type == "code" and c.source.startswith("# Leg ")]
        assert motion_cells

        ns = {
            "PREV_OK": False,  # what a failed start_variant_gate produces
            "CTRL": tmp_path, "CYCLE": "S2-B4-a-r1", "json": json,
        }
        exec(compile(motion_cells[0].source, "<leg0 prev_ok False>", "exec"), ns)

        assert ns["LEG"]["go"] == "not_eligible"
        assert ns["LEG"]["outcome"] == "not_attempted"


# ── M5 (PR #124 review): HOME's tail-check target is not stiff-zero ────────

class TestHomeToleranceIsNotStiffZero:
    """'Do not confuse HOME's normal gripper target with stiff-zero.'
    `rig_routes.HOME`'s own r_gripper (-45 deg, OPEN) is a world apart
    from stiff-zero's |r_gripper| <= 1 deg requirement, so `HOME` was
    never a usable stand-in for "did turn_on hold" -- and, as the
    2026-09-16 re-review demonstrated, was not even reachable as an
    `end`-mode tail-check target on a fresh, compliant server (the
    gripper sags toward keyframe-sag, not toward OPEN, during the armon
    recording). The Stage 0 armon recording is now judged with
    `e1_tail_check.check_init`/`INIT` instead (see `TestCheckInit` in
    `test_e1_tail_check.py`, and `e1_stage1/README.md`'s "Policy A"
    section), which has no gripper or posture criterion at all -- so this
    class keeps only the standing invariant that made `HOME` wrong in the
    first place: it can never classify as, or be confused with,
    stiff-zero."""

    def test_home_pose_does_not_classify_as_a_known_start_variant(self):
        assert rig_routes.HOME["r_gripper"] == pytest.approx(-45.0)
        assert plan.classify_start_variant(rig_routes.HOME) is None

    def test_home_gripper_target_is_far_outside_stiff_zero_tolerance(self):
        gap = abs(rig_routes.HOME["r_gripper"]) - plan.STIFF_ZERO_TOL_DEG
        assert gap > 40.0


# ── M4 (PR #124 review): an unauthorized board's reasoned refusal reaches
# the CLI user rather than a generic argparse error ─────────────────────────

class TestCLIBoardRefusalReachesTheUser:

    def test_unauthorized_board_via_stage0_cli_prints_reasoned_refusal(
            self, repo_two_commits, tmp_path, capsys):
        repo, first, second = repo_two_commits
        rc = make_cycle_notebook.main([
            "--stage0", "--board", "B6", "--session", "sess1",
            "--repo", str(repo), "--evidence-dir", str(tmp_path / "evidence"),
        ])
        assert rc == 1
        err = capsys.readouterr().err
        assert "REFUSED:" in err
        assert "B6" in err
        assert "decision note" in err
