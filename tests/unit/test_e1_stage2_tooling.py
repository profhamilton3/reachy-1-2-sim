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
import subprocess
import sys
import time
import traceback
import types

import nbformat
import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../scripts"))
sys.path.insert(0, os.path.join(_HERE, "../../src"))

from e1_stage1 import gating, make_cycle_notebook, plan, start_variant  # noqa: E402


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
        src = make_cycle_notebook.armon_cell_source()
        tree = ast.parse(src)
        route_fns = {call.split(".")[-1] for call in plan.TOOL_TO_CALL}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in route_fns

    def test_armon_cell_requires_compliance_gate_before_turn_on(self):
        src = make_cycle_notebook.armon_cell_source()
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
                                require_compliance_result, calls):
        (tmp_path / "go_armon").touch()
        (tmp_path / "recorder_armon.log").write_text("fly the route now")

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
        src = make_cycle_notebook.armon_cell_source()
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
        src = make_cycle_notebook.armon_cell_source()
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
        src = make_cycle_notebook.armon_cell_source()
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
        src = make_cycle_notebook.armon_cell_source()
        exec(compile(src, "<armon compliance fails>", "exec"), ns)
        assert (tmp_path / "stop").exists()

        next_leg_name = plan.stage2_legs("B4", "a", 1)[0].name
        (tmp_path / f"go_{next_leg_name}").touch()
        result = gating.wait_for(
            tmp_path, lambda: (tmp_path / f"go_{next_leg_name}").exists(),
            timeout_s=1, period=0.01)
        assert result == "stop"
