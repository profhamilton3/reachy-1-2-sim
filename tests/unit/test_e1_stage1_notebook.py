"""scripts/e1_stage1: the new Stage 1 execution notebook tooling (fail-
closed cmd_seq baseline, generation-time + runtime tree provenance).

Offline throughout: notebook generation and cell compilation are tested
with `nbformat` + `compile()`; the motion cell's gating logic is tested by
`exec`-ing its rendered source into a namespace that stubs every external
(SDK, e1_identity, route functions). No server, no SDK connection, no
notebook is ever executed as a whole, no Docker.
"""

import ast
import inspect
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

from e1_stage1 import make_cycle_notebook, plan, provenance  # noqa: E402
import e1_tail_check  # noqa: E402
from reachy_ai.motion import rig_routes  # noqa: E402


# ── Helpers ────────────────────────────────────────────────────────────────

def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                    capture_output=True, text=True)


def _init_repo_two_commits(root):
    """A throwaway repo with two commits; returns (first_sha, second_sha).
    Used as `required_sha`/HEAD so tests never depend on the real 4d727c0.
    """
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


# ── Section 2 (W1): every generated cell compiles ───────────────────────────

@pytest.mark.parametrize("cycle", sorted(plan.CYCLES))
def test_every_generated_cell_compiles(cycle, repo_two_commits, tmp_path):
    repo, first, second = repo_two_commits
    out = make_cycle_notebook.generate(
        cycle, repo=str(repo), evidence_dir=str(tmp_path / "evidence"),
        required_sha=first)
    nb = nbformat.read(str(out), as_version=4)
    nbformat.validate(nb)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) == 2 + len(plan.CYCLES[cycle]) + 1
    for i, cell in enumerate(code_cells):
        compile(cell.source, f"<{cycle} cell {i}>", "exec")


def _if_ancestors_of_call(tree, predicate_on_call):
    """Yield every ast.If node whose subtree contains a Call matching
    `predicate_on_call`, walking from that Call up through its ancestors."""
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


@pytest.mark.parametrize("cycle", sorted(plan.CYCLES))
def test_generated_motion_cell_contains_the_gate(cycle, repo_two_commits, tmp_path):
    repo, first, second = repo_two_commits
    out = make_cycle_notebook.generate(
        cycle, repo=str(repo), evidence_dir=str(tmp_path / "evidence"),
        required_sha=first)
    nb = nbformat.read(str(out), as_version=4)
    legs = plan.CYCLES[cycle]
    motion_cells = [c for c in nb.cells
                    if c.cell_type == "code" and c.source.startswith("# Leg ")]
    assert len(motion_cells) == len(legs)
    for leg, cell in zip(legs, motion_cells):
        src = cell.source
        assert "require_compliance(" in src
        assert "min_cmd_seq=" in src
        assert "min_cmd_seq=None" not in src
        route_fn = leg.tool.split(".")[-1]
        tree = ast.parse(src)

        def is_route_call(node, route_fn=route_fn):
            return isinstance(node.func, ast.Attribute) and node.func.attr == route_fn

        ancestor_ifs = list(_if_ancestors_of_call(tree, is_route_call))
        assert ancestor_ifs, f"no route call found for {route_fn} in:\n{src}"
        assert any("chk" in ast.unparse(n.test) and "ok" in ast.unparse(n.test)
                   for n in ancestor_ifs), (
            f"route call for {route_fn} not nested inside an `if ...chk.ok...`:\n{src}")


# ── Section 5 (W3/M1): plan durations, derived from the route tables ───────
#
# PR #120 review, M1: the old `ROUTE_BUDGET_S` was a single 20.0s pinned to
# every route, "rounded up" from an aborted flight's elapsed time, and the
# only test here (`test_plan_durations_cover_the_gate`) checked durations
# against that same wrong number rather than against the routes themselves
# -- it could not have caught the bug it was meant to guard. These replace
# it with tests against `rig_routes`' waypoint chains and `plan.py`'s own
# named timing terms.

def test_parked_tail_matches_e1_tail_check_default():
    """`plan.PARKED_TAIL_S` exists so the tail window is a visible budget
    term; it is only honest if it actually agrees with the checker that
    enforces it."""
    default_window_s = (
        inspect.signature(e1_tail_check.check).parameters["window_s"].default)
    assert plan.PARKED_TAIL_S == default_window_s


@pytest.mark.parametrize("route", sorted(plan.ROUTE_BUDGET_S))
def test_route_budget_covers_the_nominal_waypoint_chain(route):
    nominal = sum(wp.seconds for wp in getattr(rig_routes, route))
    assert plan.ROUTE_BUDGET_S[route] >= nominal


@pytest.mark.parametrize("route", sorted(plan.ROUTE_BUDGET_S))
def test_route_budget_matches_its_own_derivation(route):
    """Not a magic number: recomputes plan.py's own formula (nominal +
    settle overhead + one-waypoint retry allowance) from the route table
    and the runner's retry/settle constants, so a change to either changes
    this expectation along with the budget instead of drifting from it."""
    waypoints = getattr(rig_routes, route)
    nominal = sum(wp.seconds for wp in waypoints)
    runner = plan.ROUTE_RUNNER[route]
    settle = (plan._RIG_MOTION_FLY_ROUTE_SETTLE_S * len(waypoints)
              if runner == "rig_motion_fly_route" else 0.0)
    retry_worst = plan.RUNNER_RETRY_WORST_S[runner]
    assert plan.ROUTE_BUDGET_S[route] == pytest.approx(nominal + settle + retry_worst)


def test_route_budget_is_not_full_worst_case_for_multi_waypoint_routes():
    """Guards the deliberate, documented choice in plan.py: the retry term
    budgets ONE bad waypoint, not every waypoint retrying to its worst
    case (which would run some routes past three minutes). A single-
    waypoint route has no "every other waypoint" to be smaller than, so
    it is excluded rather than asserted equal by coincidence."""
    for route, budget in plan.ROUTE_BUDGET_S.items():
        waypoints = getattr(rig_routes, route)
        if len(waypoints) <= 1:
            continue
        runner = plan.ROUTE_RUNNER[route]
        settle = (plan._RIG_MOTION_FLY_ROUTE_SETTLE_S * len(waypoints)
                  if runner == "rig_motion_fly_route" else 0.0)
        full_worst_case = (sum(wp.seconds for wp in waypoints) + settle
                           + plan.RUNNER_RETRY_WORST_S[runner] * len(waypoints))
        assert budget < full_worst_case


def test_route_budgets_exceed_the_old_pinned_20s():
    """The bug this section replaces: every route shared a flat 20.0s
    budget, verified only against itself. Every derived budget below must
    clear that floor -- if a future change to the route tables or runner
    policy ever brought one back down to 20s or under, that is the
    regression this guards."""
    for route, budget in plan.ROUTE_BUDGET_S.items():
        assert budget > 20.0


def test_leg_duration_covers_lead_in_compliance_budget_and_parked_tail():
    for cycle, legs in plan.CYCLES.items():
        durations = plan.cycle_durations(cycle)
        for leg in legs:
            assert durations[leg.name] >= (
                plan.LEAD_IN_S + plan.COMPLIANCE_TIMEOUT_S
                + plan.ROUTE_BUDGET_S[leg.route] + plan.PARKED_TAIL_S)


@pytest.mark.parametrize("cycle", sorted(plan.CYCLES))
def test_notebook_constants_match_plan(cycle, repo_two_commits, tmp_path):
    repo, first, second = repo_two_commits
    out = make_cycle_notebook.generate(
        cycle, repo=str(repo), evidence_dir=str(tmp_path / "evidence"),
        required_sha=first)
    nb = nbformat.read(str(out), as_version=4)
    connect_src = nb.cells[1].source
    tree = ast.parse(connect_src)
    values = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                        "LEAD_IN_S", "COMPLIANCE_TIMEOUT_S"):
                    values[target.id] = ast.literal_eval(node.value)
    assert values["LEAD_IN_S"] == plan.LEAD_IN_S
    assert values["COMPLIANCE_TIMEOUT_S"] == plan.COMPLIANCE_TIMEOUT_S


# ── Generator refusal / acceptance (W4, generation-time ancestry) ──────────

def test_generate_refuses_when_repo_does_not_contain_required_sha(tmp_path):
    root = tmp_path / "one_commit_repo"
    root.mkdir()
    _git("init", "-q", cwd=root)
    _git("config", "user.email", "t@example.com", cwd=root)
    _git("config", "user.name", "t", cwd=root)
    (root / "a.txt").write_text("a")
    _git("add", "a.txt", cwd=root)
    _git("commit", "-q", "-m", "only commit", cwd=root)

    fake_required_sha = "0" * 40  # not reachable from this repo's HEAD
    evidence_dir = tmp_path / "evidence"
    with pytest.raises(make_cycle_notebook.GenerationRefused):
        make_cycle_notebook.generate(
            "S1a", repo=str(root), evidence_dir=str(evidence_dir),
            required_sha=fake_required_sha)
    assert not evidence_dir.exists()


def test_generate_accepts_when_repo_descends_from_required_sha(repo_two_commits, tmp_path):
    repo, first, second = repo_two_commits
    evidence_dir = tmp_path / "evidence"
    out = make_cycle_notebook.generate(
        "S1a", repo=str(repo), evidence_dir=str(evidence_dir), required_sha=first)
    assert out.exists()
    assert (evidence_dir / "plan_S1a.json").exists()
    plan_doc = json.loads((evidence_dir / "plan_S1a.json").read_text())
    assert plan_doc["repo_sha"] == second
    assert plan_doc["required_sha"] == first


def test_generate_refuses_without_evidence_dir(repo_two_commits):
    repo, first, second = repo_two_commits
    with pytest.raises(make_cycle_notebook.GenerationRefused):
        make_cycle_notebook.generate(
            "S1a", repo=str(repo), evidence_dir="", required_sha=first)


def test_generate_refuses_evidence_dir_under_docs_reviews(repo_two_commits):
    repo, first, second = repo_two_commits
    with pytest.raises(make_cycle_notebook.GenerationRefused):
        make_cycle_notebook.generate(
            "S1a", repo=str(repo),
            evidence_dir=str(repo / "docs" / "reviews" / "sneaky"),
            required_sha=first)


def test_generate_refuses_evidence_dir_anywhere_inside_repo(repo_two_commits):
    """M3, PR #120 review: not just docs/reviews/ -- any evidence dir
    inside --repo leaves untracked artifacts that dirty `git status
    --porcelain` in the server's own checkout, which the runtime binding
    check correctly (but mysteriously, until this) refuses on."""
    repo, first, second = repo_two_commits
    with pytest.raises(make_cycle_notebook.GenerationRefused):
        make_cycle_notebook.generate(
            "S1a", repo=str(repo),
            evidence_dir=str(repo / "some_other_subdir" / "evidence"),
            required_sha=first)
    with pytest.raises(make_cycle_notebook.GenerationRefused):
        make_cycle_notebook.generate(
            "S1a", repo=str(repo), evidence_dir=str(repo), required_sha=first)


@pytest.mark.parametrize("cycle", sorted(plan.CYCLES))
def test_binding_cell_passes_generated_at_sha(cycle, repo_two_commits, tmp_path):
    """W4 note 1, PR #120 review: the binding cell must compare the
    server's code_sha against the exact tree the notebook was generated
    from, not just require it descend from REQUIRED_SHA."""
    repo, first, second = repo_two_commits
    out = make_cycle_notebook.generate(
        cycle, repo=str(repo), evidence_dir=str(tmp_path / "evidence"),
        required_sha=first)
    nb = nbformat.read(str(out), as_version=4)
    binding_src = nb.cells[2].source
    assert "generated_at_sha=GENERATED_AT_SHA" in binding_src


# ── Section 4: no-route-call table, exec'd offline ──────────────────────────

class _FakeChk:
    def __init__(self, ok, detail="fake"):
        self.ok = ok
        self._detail = detail

    def as_dict(self):
        return {"ok": self.ok, "detail": self._detail}


def _build_namespace(tmp_path, *, prev_ok, baseline, require_compliance_result,
                      route_should_raise, calls):
    (tmp_path / f"go_leg").touch()
    (tmp_path / f"recorder_leg.log").write_text("fly the route now")

    def fake_read_last_state(path):
        return baseline

    def fake_require_compliance(run_dir, joints, *, compliant=False,
                                 timeout_s=None, min_cmd_seq=None):
        calls["require_compliance"].append(
            {"run_dir": run_dir, "min_cmd_seq": min_cmd_seq})
        return require_compliance_result

    def fake_turn_on(*a, **kw):
        calls["turn_on"].append((a, kw))

    def fake_route(*a, **kw):
        calls["route"].append((a, kw))
        if route_should_raise:
            raise RuntimeError("route boom")
        return {"ok": True}

    def wait_for(pred, timeout_s, period=0.25):
        return "ready" if pred() else "timeout"

    def start_check(kind):
        return {"kind": kind, "ok": True, "why": "", "posture_of": "stub", "pose": {}}

    def _pose():
        return {"r_shoulder_pitch": 0.0}

    primitives_ns = types.SimpleNamespace(raise_to_side=fake_route)
    rig_motion_ns = types.SimpleNamespace(
        from_present=fake_route, deploy_to_rest=fake_route, to_present=fake_route)

    return {
        "reachy": types.SimpleNamespace(
            turn_on=fake_turn_on, r_arm=types.SimpleNamespace()),
        "e1_identity": types.SimpleNamespace(
            _read_last_state=fake_read_last_state,
            require_compliance=fake_require_compliance),
        "R": types.SimpleNamespace(R_JOINTS=("r_shoulder_pitch",)),
        "primitives": primitives_ns,
        "rig_motion": rig_motion_ns,
        "ident": types.SimpleNamespace(run_dir=str(tmp_path)),
        "CTRL": tmp_path,
        "wait_for": wait_for,
        "start_check": start_check,
        "_pose": _pose,
        "PREV_OK": prev_ok,
        "CYCLE": "test_cycle",
        "LEAD_IN_S": 0.0,
        "COMPLIANCE_TIMEOUT_S": 0.0,
        "time": types.SimpleNamespace(
            monotonic=time.monotonic, monotonic_ns=time.monotonic_ns,
            time_ns=time.time_ns, sleep=lambda s: None),
        "json": json,
        "pathlib": __import__("pathlib"),
        "traceback": traceback,
    }


def _leg_source(cycle, leg_index):
    leg = plan.CYCLES[cycle][leg_index]
    leg = leg._replace(name="leg")  # markers pre-created under a fixed name above
    return make_cycle_notebook.motion_cell_source(leg)


_CASES = [
    # (case name, baseline, require_compliance_result, route_should_raise)
    ("missing_key", {}, None, False),
    ("explicit_null", {"cmd_seq": None}, None, False),
    ("bool", {"cmd_seq": True}, None, False),
    ("string", {"cmd_seq": "3"}, None, False),
    ("float", {"cmd_seq": 3.0}, None, False),
    ("unreadable_stream", None, None, False),
]

#: Every leg index that exists in at least one cycle (0 and 1 -- the
#: longest cycles have two legs). Each test below parametrizes over this
#: and skips indices past a given cycle's own length, so the table runs for
#: every cycle's every leg, not one representative -- the four route-call
#: shapes (primitives.raise_to_side, rig_motion.from_present/deploy_to_rest
#: /to_present) differ per leg.
_ALL_LEG_INDICES = sorted({i for legs in plan.CYCLES.values() for i in range(len(legs))})


@pytest.mark.parametrize("cycle", sorted(plan.CYCLES))
@pytest.mark.parametrize("leg_index", _ALL_LEG_INDICES)
@pytest.mark.parametrize("case_name,baseline,require_compliance_result,route_should_raise", _CASES)
def test_invalid_baseline_stops_before_turn_on(
        cycle, leg_index, case_name, baseline, require_compliance_result,
        route_should_raise, tmp_path):
    if leg_index >= len(plan.CYCLES[cycle]):
        pytest.skip("cycle has fewer legs")
    calls = {"turn_on": [], "require_compliance": [], "route": []}
    ns = _build_namespace(tmp_path, prev_ok=True, baseline=baseline,
                           require_compliance_result=require_compliance_result,
                           route_should_raise=route_should_raise, calls=calls)
    src = _leg_source(cycle, leg_index)
    exec(compile(src, f"<{cycle} leg {leg_index} {case_name}>", "exec"), ns)

    assert calls["route"] == []
    assert calls["turn_on"] == []
    assert calls["require_compliance"] == []
    assert ns["LEG"]["outcome"].startswith("STOP no_valid_baseline")
    assert (tmp_path / "stop").exists()
    assert ns["PREV_OK"] is False


@pytest.mark.parametrize("cycle", sorted(plan.CYCLES))
@pytest.mark.parametrize("leg_index", _ALL_LEG_INDICES)
def test_valid_baseline_gate_refuses(cycle, leg_index, tmp_path):
    if leg_index >= len(plan.CYCLES[cycle]):
        pytest.skip("cycle has fewer legs")
    calls = {"turn_on": [], "require_compliance": [], "route": []}
    ns = _build_namespace(tmp_path, prev_ok=True, baseline={"cmd_seq": 3},
                           require_compliance_result=_FakeChk(ok=False),
                           route_should_raise=False, calls=calls)
    src = _leg_source(cycle, leg_index)
    exec(compile(src, f"<{cycle} leg {leg_index} gate_refuses>", "exec"), ns)

    assert len(calls["turn_on"]) == 1
    assert calls["require_compliance"][0]["min_cmd_seq"] == 3
    assert calls["route"] == []
    assert ns["LEG"]["outcome"] == "STOP compliance_check"
    assert (tmp_path / "stop").exists()
    assert ns["PREV_OK"] is False


@pytest.mark.parametrize("cycle", sorted(plan.CYCLES))
@pytest.mark.parametrize("leg_index", _ALL_LEG_INDICES)
def test_valid_baseline_gate_passes(cycle, leg_index, tmp_path):
    if leg_index >= len(plan.CYCLES[cycle]):
        pytest.skip("cycle has fewer legs")
    calls = {"turn_on": [], "require_compliance": [], "route": []}
    ns = _build_namespace(tmp_path, prev_ok=True, baseline={"cmd_seq": 3},
                           require_compliance_result=_FakeChk(ok=True),
                           route_should_raise=False, calls=calls)
    src = _leg_source(cycle, leg_index)
    exec(compile(src, f"<{cycle} leg {leg_index} gate_passes>", "exec"), ns)

    assert len(calls["turn_on"]) == 1
    assert len(calls["route"]) == 1
    assert ns["LEG"]["outcome"] == "returned"
    assert not (tmp_path / "stop").exists()
    assert ns["PREV_OK"] is True


@pytest.mark.parametrize("cycle", sorted(plan.CYCLES))
@pytest.mark.parametrize("leg_index", _ALL_LEG_INDICES)
def test_gate_passes_but_route_raises(cycle, leg_index, tmp_path):
    if leg_index >= len(plan.CYCLES[cycle]):
        pytest.skip("cycle has fewer legs")
    calls = {"turn_on": [], "require_compliance": [], "route": []}
    ns = _build_namespace(tmp_path, prev_ok=True, baseline={"cmd_seq": 3},
                           require_compliance_result=_FakeChk(ok=True),
                           route_should_raise=True, calls=calls)
    src = _leg_source(cycle, leg_index)
    exec(compile(src, f"<{cycle} leg {leg_index} route_raises>", "exec"), ns)

    assert len(calls["route"]) == 1
    assert ns["LEG"]["outcome"].startswith("EXC")
    assert ns["PREV_OK"] is False


@pytest.mark.parametrize("cycle", sorted(plan.CYCLES))
@pytest.mark.parametrize("leg_index", _ALL_LEG_INDICES)
def test_earlier_leg_stopped_makes_this_leg_not_eligible(cycle, leg_index, tmp_path):
    if leg_index >= len(plan.CYCLES[cycle]):
        pytest.skip("cycle has fewer legs")
    calls = {"turn_on": [], "require_compliance": [], "route": []}
    ns = _build_namespace(tmp_path, prev_ok=False, baseline={"cmd_seq": 3},
                           require_compliance_result=_FakeChk(ok=True),
                           route_should_raise=False, calls=calls)
    src = _leg_source(cycle, leg_index)
    exec(compile(src, f"<{cycle} leg {leg_index} not_eligible>", "exec"), ns)

    assert ns["LEG"]["go"] == "not_eligible"
    assert calls["route"] == []
    assert calls["turn_on"] == []
    assert ns["LEG"]["outcome"] == "not_attempted"
    assert ns["PREV_OK"] is False


# ── Provenance pure function (W4, runtime binding check) ───────────────────

class TestCheckBindingProvenance:

    def _ok_manifest(self):
        return {"code_sha": "deadbeef", "code_sha_dirty": False,
                "started_at": "2026-09-16T02:00:00+00:00"}

    def test_accepts_clean_descendant_after_merge(self):
        ok, reasons = provenance.check_binding_provenance(
            self._ok_manifest(),
            git_is_ancestor=lambda a, b: True,
            required_sha="required", merge_time_iso="2026-09-16T01:35:22Z")
        assert ok is True
        assert reasons == []

    def test_refuses_missing_code_sha(self):
        m = self._ok_manifest(); del m["code_sha"]
        ok, reasons = provenance.check_binding_provenance(
            m, git_is_ancestor=lambda a, b: True,
            required_sha="required", merge_time_iso="2026-09-16T01:35:22Z")
        assert ok is False
        assert any("code_sha" in r for r in reasons)

    def test_refuses_dirty_tree(self):
        m = self._ok_manifest(); m["code_sha_dirty"] = True
        ok, reasons = provenance.check_binding_provenance(
            m, git_is_ancestor=lambda a, b: True,
            required_sha="required", merge_time_iso="2026-09-16T01:35:22Z")
        assert ok is False
        assert any("dirty" in r for r in reasons)

    def test_refuses_unknown_dirty_state(self):
        m = self._ok_manifest(); m["code_sha_dirty"] = None
        ok, reasons = provenance.check_binding_provenance(
            m, git_is_ancestor=lambda a, b: True,
            required_sha="required", merge_time_iso="2026-09-16T01:35:22Z")
        assert ok is False

    def test_refuses_non_ancestor(self):
        ok, reasons = provenance.check_binding_provenance(
            self._ok_manifest(), git_is_ancestor=lambda a, b: False,
            required_sha="required", merge_time_iso="2026-09-16T01:35:22Z")
        assert ok is False
        assert any("descendant" in r for r in reasons)

    def test_refuses_started_before_merge(self):
        m = self._ok_manifest(); m["started_at"] = "2026-09-16T01:00:00+00:00"
        ok, reasons = provenance.check_binding_provenance(
            m, git_is_ancestor=lambda a, b: True,
            required_sha="required", merge_time_iso="2026-09-16T01:35:22Z")
        assert ok is False
        assert any("started_at" in r for r in reasons)

    def test_refuses_missing_started_at(self):
        m = self._ok_manifest(); del m["started_at"]
        ok, reasons = provenance.check_binding_provenance(
            m, git_is_ancestor=lambda a, b: True,
            required_sha="required", merge_time_iso="2026-09-16T01:35:22Z")
        assert ok is False

    # ── generated_at_sha (W4 note 1, PR #120 review) ────────────────────────

    def test_accepts_when_generated_at_sha_matches_code_sha(self):
        ok, reasons = provenance.check_binding_provenance(
            self._ok_manifest(), git_is_ancestor=lambda a, b: True,
            required_sha="required", merge_time_iso="2026-09-16T01:35:22Z",
            generated_at_sha="deadbeef")
        assert ok is True
        assert reasons == []

    def test_refuses_when_generated_at_sha_differs_from_code_sha(self):
        """The server can be a genuine descendant of required_sha and still
        not be the tree --repo pointed at when the notebook was generated
        -- ancestry alone does not rule that out."""
        ok, reasons = provenance.check_binding_provenance(
            self._ok_manifest(), git_is_ancestor=lambda a, b: True,
            required_sha="required", merge_time_iso="2026-09-16T01:35:22Z",
            generated_at_sha="a_different_tree")
        assert ok is False
        assert any("!=" in r for r in reasons)

    def test_skips_generated_at_sha_check_when_not_given(self):
        """Backward-compatible default: callers with no generation-time
        tree to compare against (e.g. these other unit tests) keep the
        looser ancestry-only check."""
        ok, reasons = provenance.check_binding_provenance(
            self._ok_manifest(), git_is_ancestor=lambda a, b: True,
            required_sha="required", merge_time_iso="2026-09-16T01:35:22Z")
        assert ok is True
