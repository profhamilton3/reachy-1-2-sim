"""E1 readiness (assignment 2026-09-14, work item 1): the six committed
board scene YAMLs under `scenes/e1_boards/`.

Offline throughout: MuJoCo is used only to `mj_forward` a compiled model
(compile-and-read qpos0, never mj_step), and no server is started.
"""

import os
import sys

import pytest

mujoco = pytest.importorskip("mujoco")

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../src"))
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))
sys.path.insert(0, os.path.join(_HERE, "../../scripts"))
sys.path.insert(0, _HERE)

import make_e1_boards as gen  # noqa: E402
import test_footprint_boards as tfb  # noqa: E402
from objects import build_scene_model_xml  # noqa: E402
from scene_io import load_scene  # noqa: E402
from reachy_ai.scene.awareness import SceneModel  # noqa: E402

_OUT_DIR = os.path.join(_HERE, "../../scenes/e1_boards")
_ROBOT_MODEL = os.path.join(_HERE, "../../native_mujoco/model/reachy_1_2.xml")

#: {board_id: (object_id, expected_tube_cm, expected_shells_cm)} -- from the
#: assignment's fixture table (== TestNamedBoards's parametrize in
#: test_footprint_boards.py).
_FIXTURE_TABLE = {
    "B4_pool_box_1_r2c3": ("pool_box_1", -3.719, 6.258),
    "B5_pool_cyl_1_r2c3": ("pool_cyl_1", -3.079, 5.936),
    "B3_soda_can_r2c3": ("soda_can", -4.379, 4.535),
    "B1_evidence": ("foam_block", -5.699, 4.479),
    "B2_foam_r3c3": ("foam_block", -9.369, 2.106),
    "B6_incident": ("foam_block", -11.944, -2.414),
}


def _board_path(board_id: str) -> str:
    return os.path.join(_OUT_DIR, f"{board_id}.yaml")


class TestGeneratorIsIdempotent:
    def test_regenerating_reproduces_committed_files_byte_for_byte(self, tmp_path):
        written = gen.generate(tmp_path)
        assert set(written) == set(gen.BOARDS) | {gen.INCIDENT_ID}
        for board_id, fresh_path in written.items():
            committed = _board_path(board_id)
            assert os.path.isfile(committed), f"{committed} not committed"
            assert fresh_path.read_text() == open(committed).read(), (
                f"{board_id}: regenerated content differs from the committed "
                "file -- someone hand-edited it, or the generator changed "
                "without regenerating")

    def test_all_six_boards_are_named(self):
        assert len(gen.BOARDS) == 5
        assert gen.INCIDENT_ID == "B6_incident"
        assert set(gen.BOARDS) | {gen.INCIDENT_ID} == set(_FIXTURE_TABLE)


class TestBoardsMatchFixtureTable:
    """`_board()`'s in-memory boards and the committed YAML files must agree
    on worst clearance, over any guarded route, both hand models -- to
    1e-6 m, and to 1e-3 cm against the assignment's own fixture table."""

    @pytest.mark.parametrize("board_id", sorted(_FIXTURE_TABLE))
    def test_scene_model_matches_in_memory_board(self, board_id):
        object_id, expected_tube_cm, expected_shells_cm = _FIXTURE_TABLE[board_id]
        from_yaml = SceneModel.from_yaml(_board_path(board_id))

        if board_id == "B6_incident":
            xy = SceneModel.from_yaml(
                str(gen._PARENT_SCENE_PATH)).cell_center("cell_r2c3")[:2]
            in_memory_objects = {object_id: (xy[0], xy[1] + gen.INCIDENT_Y_OFFSET_M)}
        elif board_id == "B1_evidence":
            base = SceneModel.from_yaml(str(gen._PARENT_SCENE_PATH))
            in_memory_objects = {
                "soda_can": base.cell_center("cell_r1c1")[:2],
                "foam_block": base.cell_center("cell_r2c3")[:2],
            }
        else:
            cell = gen.BOARDS[board_id][object_id]
            xy = SceneModel.from_yaml(str(gen._PARENT_SCENE_PATH)).cell_center(cell)[:2]
            in_memory_objects = {object_id: xy}

        in_memory = tfb._board(in_memory_objects)

        for hand in ("tube", "shells"):
            from_yaml_clearance = tfb._worst_over_any_guarded_route(
                from_yaml, object_id, hand)
            in_memory_clearance = tfb._worst_over_any_guarded_route(
                in_memory, object_id, hand)
            assert from_yaml_clearance == pytest.approx(
                in_memory_clearance, abs=1e-6)

        tube_cm = tfb._worst_over_any_guarded_route(from_yaml, object_id, "tube") * 100.0
        shells_cm = tfb._worst_over_any_guarded_route(
            from_yaml, object_id, "shells") * 100.0
        assert tube_cm == pytest.approx(expected_tube_cm, abs=1e-3)
        assert shells_cm == pytest.approx(expected_shells_cm, abs=1e-3)


class TestNativeLoaderAgreesWithSceneModel:
    """`scene_io.load_scene` (native side) resolves `extends` the same way
    `SceneModel.from_yaml` does (which delegates to the SAME function --
    see make_e1_boards.py's module docstring), and the compiled model's
    qpos0 for the overridden object matches the YAML pose.

    Tolerance 1e-6 m, not 1e-9: `scene_compiler._pos_str` formats every MJCF
    `pos=` attribute with `.6f` (6 decimal places), so 1e-9 agreement is not
    achievable through the compiled XML regardless of how precisely the YAML
    itself stores the number -- verified empirically here (measured diffs
    were ~1e-17, i.e. within float64 noise of the `.6f`-rounded value, but
    the honest bound the compiler's own formatting guarantees is 1e-6).
    """

    @pytest.mark.parametrize("board_id", sorted(_FIXTURE_TABLE))
    def test_qpos0_matches_yaml_pose(self, board_id):
        path = _board_path(board_id)
        object_id = _FIXTURE_TABLE[board_id][0]

        doc = load_scene(path)
        xml = build_scene_model_xml(doc, _ROBOT_MODEL)
        model = mujoco.MjModel.from_xml_string(xml)

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_id)
        assert body_id >= 0, f"{object_id} not compiled into the model"
        joint_id = model.body_jntadr[body_id]
        assert joint_id >= 0, f"{object_id} has no free joint"
        qpos_adr = model.jnt_qposadr[joint_id]
        qpos0_xyz = tuple(float(v) for v in model.qpos0[qpos_adr:qpos_adr + 3])

        yaml_xyz = SceneModel.from_yaml(path).get(object_id).center
        for a, b in zip(qpos0_xyz, yaml_xyz):
            assert a == pytest.approx(b, abs=1e-6)

    def test_board_overrides_only_the_named_object(self):
        """Every other object in the board keeps the parent's pose --
        `_merge_objects` patches by id, it does not replace the whole list."""
        parent = SceneModel.from_yaml(str(gen._PARENT_SCENE_PATH))
        board = SceneModel.from_yaml(_board_path("B4_pool_box_1_r2c3"))
        for object_id, obj in parent.objects.items():
            if object_id == "pool_box_1":
                continue
            assert board.get(object_id).center == obj.center, (
                f"{object_id} moved on a board that only overrides pool_box_1")
