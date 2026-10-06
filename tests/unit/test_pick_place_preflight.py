"""Issue #56: one job-level preflight, used by the worker, the demo and the
notebook, before any of the job moves.

Offline.  Real planning goes through the host mirror of the simulator's IK;
the robot is a recorder, and the primitives that would command hardware are
replaced by recorders.  Nothing here starts a simulator, a notebook kernel or
an arm.
"""

import importlib.util
import json
import os
import sys

import pytest

pytest.importorskip("scipy")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "web"))

from arc_test_fakes import (  # noqa: E402
    DEMO_SCENE, LAB_SCENE, RecordingArm, RecordingRobot, Spies, board)
from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.motion.kinematics import (  # noqa: E402
    ArmClearanceError, CartesianPlanner, UnreachableError)
from reachy_ai.scene.awareness import SceneModel  # noqa: E402
from reachy_ai.tasks import pick_place_live as L  # noqa: E402

_ROOT = os.path.join(_HERE, "..", "..")


def _planner(scene):
    robot = RecordingRobot()
    return robot, CartesianPlanner(robot.r_arm, scene=scene)


def _zero_motion(robot, spies):
    assert robot.r_arm.writes == []
    assert robot.turn_on_calls == []
    assert spies.calls == []


# ── the three places a job can be refused ───────────────────────────────────

class TestJobPreflightRefusals:
    def test_refused_at_the_raise_footprint_before_any_arc_is_planned(
            self, monkeypatch):
        spies = Spies(monkeypatch)
        scene, cells = board({"red_cube": "cell_r3c3"})   # -8.9 cm vs the REST leg
        robot, planner = _planner(scene)
        with pytest.raises(L.PreflightRefused) as info:
            L.preflight_pick_place(planner, scene,
                                   [("red_cube", cells["cell_r2c2"][:2])],
                                   skip_refused=False)
        e = info.value
        assert e.segment == "RAISE_TO_SIDE footprint"
        assert (e.obstacle, e.link, e.model) == ("red_cube", "hand", "tube")
        assert e.clearance_m < 0 and e.margin_m == R.FOOTPRINT_MARGIN
        assert "RAISE_TO_SIDE" in str(e) and "red_cube" in str(e)
        assert robot.r_arm.ik_calls == 0          # no arc was planned
        _zero_motion(robot, spies)

    def test_refused_at_an_arc(self, monkeypatch):
        spies = Spies(monkeypatch)
        scene, cells = board({"red_cube": "cell_r1c1"})   # the rail beside r1c1
        robot, planner = _planner(scene)
        with pytest.raises(L.PreflightRefused) as info:
            L.preflight_pick_place(planner, scene,
                                   [("red_cube", cells["cell_r3c3"][:2])],
                                   skip_refused=False)
        e = info.value
        assert e.object_id == "red_cube"
        assert e.obstacle.startswith("rig_rail")
        assert e.segment in ("descend to grasp", "lift to clear height",
                             "descend to hover", "swing in over table")
        assert e.link in ("upper_arm", "forearm", "hand")
        assert e.clearance_m < 0 and e.model == "tube"
        assert robot.r_arm.ik_calls > 0           # it WAS planned ...
        _zero_motion(robot, spies)                # ... and nothing was sent

    def test_refused_at_the_stow_footprint_only_once_the_target_is_placed(
            self, monkeypatch):
        spies = Spies(monkeypatch)
        scene, cells = board({"red_cube": "cell_r2c2"})
        robot, planner = _planner(scene)
        # r2c2 clears the raise footprint (+8.9 cm); r2c3 does not (-5.0 cm).
        assert L._footprint_refusal(scene, L.RAISE_ROUTE) is None
        with pytest.raises(L.PreflightRefused) as info:
            L.preflight_pick_place(planner, scene,
                                   [("red_cube", cells["cell_r2c3"][:2])],
                                   skip_refused=False)
        e = info.value
        assert e.segment == "STOW_FROM_SIDE footprint"
        assert (e.obstacle, e.model) == ("red_cube", "tube")
        assert e.clearance_m < 0
        assert robot.r_arm.ik_calls > 0           # the arc passed first
        _zero_motion(robot, spies)

    def test_a_passing_job_returns_its_arcs_and_the_predicted_board(
            self, monkeypatch):
        Spies(monkeypatch)
        scene, cells = board({"red_cube": "cell_r2c2"})
        robot, planner = _planner(scene)
        before = scene.objects
        job = L.preflight_pick_place(planner, scene,
                                     [("red_cube", cells["cell_r2c1"][:2])],
                                     skip_refused=False)
        assert [a.object_id for a in job.arcs] == ["red_cube"]
        assert job.skipped == ()
        assert job.final_board["red_cube"] == job.arcs[0].place
        assert scene.objects == before            # the caller's board is untouched


class TestSkipRefusedSemantics:
    """With the arcs stubbed out, so these are about the bookkeeping."""

    def _fake_plan(self, monkeypatch, refuse=(), unreachable=()):
        seen = []

        def fake(planner, scene, object_id, place_xy):
            seen.append((object_id, {o: scene.get(o).center
                                     for o in scene.manipulable_ids()}))
            if object_id in refuse:
                raise ArmClearanceError(
                    segment="descend to place", link="hand", obstacle="x",
                    clearance_m=-0.01, margin_m=0.0, model="tube",
                    gripper_deg=-45.0, pose_index=3)
            if object_id in unreachable:
                err = UnreachableError("swing in over table: No IK solution")
                err.segment = "swing in over table"
                raise err
            place = scene.rest_point(place_xy, object_id)
            return L.ArcPlan(object_id=object_id, place_xy=tuple(place_xy),
                             place=place, start=L.present_joints(), segments=(),
                             end=L.present_joints(), scene=scene.copy())

        monkeypatch.setattr(L, "plan_pick_and_place", fake)
        return seen

    def _board(self):
        scene, cells = board({"red_cube": "cell_r2c2", "soda_can": "cell_r1c1",
                              "foam_block": "cell_r3c1"})
        return scene, cells

    def test_skip_records_the_refusal_and_leaves_the_object_where_it_is(
            self, monkeypatch):
        seen = self._fake_plan(monkeypatch, refuse=("red_cube",))
        scene, cells = self._board()
        _robot, planner = _planner(scene)
        job = L.preflight_pick_place(
            planner, scene,
            [("red_cube", cells["cell_r2c1"][:2]),
             ("soda_can", cells["cell_r1c2"][:2])], skip_refused=True)
        assert [a.object_id for a in job.arcs] == ["soda_can"]
        (skipped_id, text), = job.skipped
        assert skipped_id == "red_cube" and "descend to place" in text
        assert job.final_board["red_cube"] == scene.get("red_cube").center

    def test_accepted_arcs_carry_their_predicted_placements_into_later_arcs(
            self, monkeypatch):
        seen = self._fake_plan(monkeypatch)
        scene, cells = self._board()
        _robot, planner = _planner(scene)
        job = L.preflight_pick_place(
            planner, scene,
            [("red_cube", cells["cell_r2c1"][:2]),
             ("soda_can", cells["cell_r1c2"][:2])], skip_refused=True)
        first, second = job.arcs
        assert seen[1][1]["red_cube"] == first.place       # moved on the copy
        assert seen[0][1]["red_cube"] == scene.get("red_cube").center
        assert second.scene.get("red_cube").center == first.place
        assert scene.get("red_cube").center != first.place  # original untouched
        assert job.final_board["soda_can"] == second.place

    def test_without_skip_the_first_refusal_refuses_the_job_with_its_fields(
            self, monkeypatch):
        self._fake_plan(monkeypatch, refuse=("red_cube",))
        scene, cells = self._board()
        _robot, planner = _planner(scene)
        with pytest.raises(L.PreflightRefused) as info:
            L.preflight_pick_place(planner, scene,
                                   [("red_cube", cells["cell_r2c1"][:2])],
                                   skip_refused=False)
        e = info.value
        assert (e.segment, e.object_id, e.link, e.obstacle, e.model) == (
            "descend to place", "red_cube", "hand", "x", "tube")
        assert e.clearance_m == -0.01 and e.margin_m == 0.0

    def test_an_unreachable_point_is_a_refusal_too(self, monkeypatch):
        self._fake_plan(monkeypatch, unreachable=("red_cube",))
        scene, cells = self._board()
        _robot, planner = _planner(scene)
        job = L.preflight_pick_place(planner, scene,
                                     [("red_cube", cells["cell_r2c1"][:2])],
                                     skip_refused=True)
        assert job.arcs == () and "swing in over table" in job.skipped[0][1]


# ── the worker ───────────────────────────────────────────────────────────────

import motion_worker as W  # noqa: E402


class _SDK:
    robot = None

    def __init__(self, host=None, sdk_port=None):
        self.r_arm = _SDK.robot.r_arm
        self.turn_on = _SDK.robot.turn_on

    def _stop(self):
        pass


def _worker_job(scene, cells, target, dest):
    return {"kind": "pick_place", "target_id": target,
            "cell_xy": list(cells[dest][:2]), "scene_file": LAB_SCENE,
            "live": {oid: list(scene.get(oid).center)
                     for oid in scene.manipulable_ids()}}


@pytest.fixture
def worker_env(monkeypatch):
    monkeypatch.setattr(W, "CONNECT_SETTLE_S", 0.0)
    spies = Spies(monkeypatch)
    robot = RecordingRobot()
    _SDK.robot = robot
    spies.events = None
    return robot, spies, W.Connection("h", 1, sdk_class=_SDK)


class TestWorker:
    def test_a_refused_job_returns_the_evidence_with_no_turn_on(self, worker_env):
        robot, spies, conn = worker_env
        scene, cells = board({"red_cube": "cell_r1c1"})
        out = W.run_pick_place(_worker_job(scene, cells, "red_cube", "cell_r3c3"),
                               conn)
        assert out["status"] == "failed"
        ev = out["evidence"]
        assert ev["refused_before_motion"] is True
        for key in ("segment", "link", "obstacle", "clearance_m", "margin_m",
                    "model", "object_id"):
            assert ev[key] is not None, key
        assert ev["model"] == "tube" and ev["obstacle"].startswith("rig_rail")
        assert ev["obstacle"] in out["detail"]            # the operator sees it
        _zero_motion(robot, spies)

    def _stub(self, monkeypatch, execute):
        scene, cells = board({"red_cube": "cell_r2c2"})
        arc = L.ArcPlan(object_id="red_cube", place_xy=cells["cell_r2c1"][:2],
                        place=(0.0, 0.0, 0.0), start=L.present_joints(),
                        segments=(), end=L.present_joints(), scene=scene.copy())
        monkeypatch.setattr(
            L, "preflight_pick_place",
            lambda *a, **k: L.JobPlan(arcs=(arc,), skipped=(), final_board={}))
        monkeypatch.setattr(L, "execute_arc", execute)
        return _worker_job(scene, cells, "red_cube", "cell_r2c1")

    def test_the_whole_job_runs_in_order_and_stows_after_arrival(
            self, worker_env, monkeypatch):
        robot, spies, conn = worker_env
        order = []
        monkeypatch.setattr(W.time, "sleep", lambda s: None)
        job = self._stub(monkeypatch, lambda *a, **k: order.append("arc") or [])
        robot.events = order
        spies.calls = order
        out = W.run_pick_place(job, conn)
        assert out == {"status": "moved"}
        assert order == [("turn_on", "r_arm"), ("raise_to_side",), "arc",
                         ("go_home",)]

    def test_a_failed_arrival_is_reported_and_the_arm_is_not_stowed(
            self, worker_env, monkeypatch):
        robot, spies, conn = worker_env

        def arrives_never(*a, **k):
            raise L.ReturnArrivalError("did not arrive", worst_joint="r_elbow_pitch",
                                       off_deg=31.0, attempts=2)

        job = self._stub(monkeypatch, arrives_never)
        out = W.run_pick_place(job, conn)
        assert out["status"] == "failed"
        assert out["evidence"]["failed_arrival"] is True
        assert out["evidence"]["worst_joint"] == "r_elbow_pitch"
        assert out["evidence"]["attempts"] == 2
        assert spies.count("go_home") == 0        # the stow's precondition failed
        assert spies.count("raise_to_side") == 1

    def test_a_cancel_is_reported_after_the_arm_is_parked(
            self, worker_env, monkeypatch):
        robot, spies, conn = worker_env
        job = self._stub(monkeypatch,
                         lambda robot, arc, planner, attacher, abort, phase, **k:
                         abort() and [])
        out = W.run_pick_place(job, conn, should_abort=lambda: True)
        assert out["status"] == "cancelled"
        assert spies.count("go_home") == 1


# ── the demo ─────────────────────────────────────────────────────────────────

@pytest.fixture
def demo(monkeypatch, tmp_path):
    """scripts/demo_pick_place.py imported with a fake SDK (it exits at import
    without one) and with its marker file pointed away from /tmp."""
    import types
    fake = types.ModuleType("reachy_sdk")
    fake.ReachySDK = lambda host=None, sdk_port=None: None
    monkeypatch.setitem(sys.modules, "reachy_sdk", fake)
    monkeypatch.setenv("REACHY_SIM_BACKEND", "mujoco-remote")
    monkeypatch.setenv("REACHY_SIM_SCENE_OVERRIDES", str(tmp_path / "o.json"))
    spec = importlib.util.spec_from_file_location(
        "demo_pick_place_under_test",
        os.path.join(_ROOT, "scripts", "demo_pick_place.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    spies = Spies(monkeypatch)
    robot = RecordingRobot()
    mod.ReachySDK = lambda host=None, sdk_port=None: robot
    events = []
    robot.events = events
    spies.calls = events
    return mod, robot, spies, events


class TestDemo:
    def test_the_demo_board_is_refused_outright_with_no_motion(self, demo):
        mod, robot, spies, events = demo
        with pytest.raises(SystemExit) as info:
            mod.run_demo("h", 1, DEMO_SCENE)
        assert info.value.code == 1
        assert robot.r_arm.writes == [] and robot.turn_on_calls == []
        assert events == []

    def _preflight_returns(self, mod, monkeypatch, events, arcs, skipped=()):
        def fake(planner, scene, moves, *, skip_refused):
            events.append("preflight")
            assert skip_refused is True
            return L.JobPlan(arcs=tuple(arcs), skipped=tuple(skipped),
                             final_board={})
        monkeypatch.setattr(mod, "preflight_pick_place", fake)

    def _arc(self):
        scene, cells = board({"red_cube": "cell_r2c2"})
        return L.ArcPlan(object_id="red_cube", place_xy=(0.42, -0.02),
                         place=(0.42, -0.02, 0.77), start=L.present_joints(),
                         segments=(), end=L.present_joints(), scene=scene)

    def test_the_preflight_is_made_before_the_arm_is_turned_on(
            self, demo, monkeypatch):
        mod, robot, spies, events = demo
        self._preflight_returns(mod, monkeypatch, events, [self._arc()],
                                skipped=[("blue_cylinder", "why")])
        monkeypatch.setattr(
            mod, "execute_arc", lambda *a, **k: events.append("execute_arc"))
        mod.run_demo("h", 1, DEMO_SCENE)
        assert events[0] == "preflight"
        assert events.index("preflight") < events.index(("turn_on", "r_arm"))
        assert [e for e in events if e != ("look_at",)] == [
            "preflight", ("turn_on", "r_arm"), ("raise_to_side",),
            "execute_arc", ("go_home",)]

    def test_nothing_accepted_means_no_motion(self, demo, monkeypatch):
        mod, robot, spies, events = demo
        self._preflight_returns(mod, monkeypatch, events, [],
                                skipped=[("red_cube", "refused")])
        with pytest.raises(SystemExit):
            mod.run_demo("h", 1, DEMO_SCENE)
        assert events == ["preflight"] and robot.turn_on_calls == []

    def test_a_failed_arrival_stops_the_demo_without_stowing(
            self, demo, monkeypatch):
        mod, robot, spies, events = demo
        self._preflight_returns(mod, monkeypatch, events, [self._arc()])

        def never(*a, **k):
            raise L.ReturnArrivalError("x", worst_joint="r_elbow_pitch",
                                       off_deg=30.0, attempts=2)

        monkeypatch.setattr(mod, "execute_arc", never)
        with pytest.raises(SystemExit):
            mod.run_demo("h", 1, DEMO_SCENE)
        assert ("go_home",) not in events

    def test_the_demo_board_and_place_sites_are_unchanged(self, demo):
        mod, *_ = demo
        assert mod._PLACE_XY == {"red_cube": (0.42, -0.02),
                                 "blue_cylinder": (0.40, -0.20)}
        assert mod._CLEAR_Z == 1.00


# ── the notebook ─────────────────────────────────────────────────────────────

_NB = os.path.join(_ROOT, "notebooks", "pick_and_place_training.ipynb")


class TestNotebook:
    @pytest.fixture(scope="class")
    def nb(self):
        with open(_NB, encoding="utf-8") as f:
            raw = f.read()
        return raw, json.loads(raw)

    def _code(self, nb):
        return [(i, "".join(c["source"])) for i, c in enumerate(nb[1]["cells"])
                if c["cell_type"] == "code"]

    def test_it_round_trips_through_json_unchanged(self, nb):
        raw, doc = nb
        assert json.dumps(doc, indent=1, ensure_ascii=False) + "\n" == raw
        assert doc["nbformat"] == 4 and "kernelspec" in doc["metadata"]
        ids = [c["id"] for c in doc["cells"]]
        assert len(ids) == len(set(ids))

    def test_no_code_cell_plans_or_flies_a_segment_on_its_own(self, nb):
        for i, text in self._code(nb):
            for banned in ("plan_segment(", "execute_trajectory(",
                           "smooth_move(", "r_forearm_yaw"):
                assert banned not in text, (i, banned)

    def test_there_is_exactly_one_raise_and_it_follows_the_preflight(self, nb):
        code = self._code(nb)
        raises = [i for i, t in code if "raise_to_side(" in t]
        pre = [i for i, t in code if "preflight_pick_place(" in t]
        assert len(raises) == 1
        assert min(pre) < raises[0]

    def test_motion_cells_preflight_before_they_execute(self, nb):
        code = self._code(nb)
        first_pre = min(i for i, t in code if "preflight_pick_place(" in t)
        first_turn_on = min(i for i, t in code if "turn_on(" in t
                            and "def " not in t)
        first_exec = min(i for i, t in code if "execute_arc(" in t)
        assert first_pre < first_turn_on < first_exec
        # the episode loop preflights inside the episode, before its execute
        loop = [t for i, t in code if "def _run_pick_place_episode" in t][0]
        assert loop.index("preflight_pick_place(") < loop.index("execute_arc(")
        assert "close_hold_s=hold_steps / 500.0" in loop

    def test_the_old_return_and_hub_are_gone_from_the_text(self, nb):
        text = "\n".join("".join(c["source"]) for c in nb[1]["cells"])
        assert "SIDE_HIGH" not in text
        assert "pad-only" not in text
        assert "#55" in text                       # the jaw twist is deferred
