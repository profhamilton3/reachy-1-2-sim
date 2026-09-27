"""K1 (2026-09-27 pr144-k1 assignment, Option A): equivalence between the
old (b473f6d) whole-file loader and the new streaming loader in
``tools/goalfix_cmp/evidence.py``.

Vendors the b473f6d ``_read_jsonl``/``load_states``/``load_commands``
verbatim (as the algorithm was at that head) as fixture functions below --
``_read_jsonl_old``/``_load_states_old``/``_load_commands_old`` -- and
compares old vs new on generated files. Only the pieces that did NOT
change at K1 are referenced from the real module rather than re-copied:
``EvidenceError``, ``States``/``Commands``, ``_JOINT_ORDER``,
``assign_state_epochs``/``assign_command_epochs`` -- none of those are
part of what this assignment touches.

Every comparison is either: identical arrays (``np.array_equal``,
including dtype and shape) and identical ``truncated``/
``duplicate_indices``/``epoch``, OR the same exception type and the exact
same message.
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

from tools.goalfix_cmp import evidence as ev  # noqa: E402


# ---------------------------------------------------------------------------
# Vendored b473f6d functions (verbatim algorithm; only EvidenceError/States/
# Commands/_JOINT_ORDER/assign_*_epochs are taken from the real module,
# since those did not change).
# ---------------------------------------------------------------------------

def _read_jsonl_old(path):
    text = Path(path).read_text()
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]  # trailing newline, not a truncation
    rows: List[dict] = []
    truncated = False
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                truncated = True
                continue
            raise ev.EvidenceError(
                f"{path}: malformed JSON on line {i + 1} of {len(lines)} "
                "(not the final line -- a truncated recording only ever "
                "affects the last one)")
    return rows, truncated


def _load_states_old(path):
    rows, truncated = _read_jsonl_old(path)
    if not rows:
        raise ev.EvidenceError(f"{path}: no usable state rows")
    try:
        seq = np.array([r["seq"] for r in rows], dtype=np.int64)
        sim_step = np.array([r["sim_step"] for r in rows], dtype=np.int64)
        sim_time_s = np.array([r["sim_time_s"] for r in rows], dtype=np.float64)
        wall_time_ns = np.array([r["wall_time_ns"] for r in rows], dtype=np.int64)
        cmd_seq = np.array([r["cmd_seq"] for r in rows], dtype=np.int64)
        position_rad = np.empty((len(rows), 21), dtype=np.float64)
        compliant = np.empty((len(rows), 21), dtype=bool)
        for i, r in enumerate(rows):
            by_name = {j["name"]: j for j in r["joints"]}
            for k, name in enumerate(ev._JOINT_ORDER):
                jd = by_name[name]
                position_rad[i, k] = jd["position_rad"]
                compliant[i, k] = bool(jd["compliant"])
    except (KeyError, TypeError) as exc:
        raise ev.EvidenceError(f"{path}: state row missing an expected field: {exc}")

    epoch = ev.assign_state_epochs(sim_step)
    duplicate_indices: List[int] = []
    for e in np.unique(epoch):
        mask = np.nonzero(epoch == e)[0]
        seen = {}
        for idx in mask:
            s = int(sim_step[idx])
            if s in seen:
                duplicate_indices.append(int(idx))
            else:
                seen[s] = int(idx)

    return ev.States(seq, sim_step, sim_time_s, wall_time_ns, cmd_seq,
                      position_rad, compliant, epoch, duplicate_indices, truncated)


def _load_commands_old(path):
    rows, truncated = _read_jsonl_old(path)
    if not rows:
        raise ev.EvidenceError(f"{path}: no usable command rows")
    kind: List[str] = []
    seq: List[int] = []
    target_rad = np.full((len(rows), 21), np.nan, dtype=np.float64)
    compliant: List[Optional[List[Optional[bool]]]] = []
    reset_sim_step: List[Optional[int]] = []
    for i, r in enumerate(rows):
        t = r.get("type")
        if t == "joint_command":
            try:
                row_seq = int(r["seq"])
                tgt = r["target_rad"]
                if len(tgt) != 21:
                    raise ev.EvidenceError(
                        f"{path}: joint_command seq={r.get('seq')} has "
                        f"{len(tgt)} target_rad entries, expected 21")
                tgt_row = [float(v) for v in tgt]
            except (KeyError, TypeError, ValueError) as exc:
                raise ev.EvidenceError(
                    f"{path}: joint_command on row {i} is malformed: {exc}") from exc
            kind.append("joint_command")
            seq.append(row_seq)
            target_rad[i, :] = tgt_row
            compliant.append(r.get("compliant"))
            reset_sim_step.append(None)
        elif t == "reset":
            kind.append("reset")
            seq.append(-1)
            compliant.append(None)
            reset_sim_step.append(r.get("sim_step"))
        else:
            raise ev.EvidenceError(f"{path}: unknown command type {t!r} on row {i}")
    epoch = ev.assign_command_epochs(kind)
    return ev.Commands(kind, np.array(seq, dtype=np.int64), target_rad, compliant,
                        epoch, reset_sim_step, truncated_final_line=truncated)


# ---------------------------------------------------------------------------
# Minimal row builders (only the fields the loaders actually read).
# ---------------------------------------------------------------------------

def _joint(name, position_rad=0.0, compliant=False):
    return {"name": name, "position_rad": position_rad, "compliant": compliant}


def _state_row(seq=0, sim_step=0, sim_time_s=0.0, wall_time_ns=1, cmd_seq=-1,
                joint_names=None, joints=None):
    if joints is None:
        names = list(joint_names) if joint_names is not None else list(ev._JOINT_ORDER)
        joints = [_joint(n, float(i) * 0.001) for i, n in enumerate(names)]
    return {"seq": seq, "sim_step": sim_step, "sim_time_s": sim_time_s,
            "wall_time_ns": wall_time_ns, "cmd_seq": cmd_seq, "joints": joints}


def _cmd_row(seq=0, target_rad=None):
    tgt = list(target_rad) if target_rad is not None else [0.01 * i for i in range(21)]
    return {"type": "joint_command", "seq": seq, "target_rad": tgt}


def _reset_row(sim_step=0):
    return {"type": "reset", "sim_step": sim_step}


def _write(path: Path, content: str):
    path.write_text(content, newline="")  # never let write_text translate "\n"/"\r\n"


def _lines_text(lines, newline="\n", trailing_newline=True):
    text = newline.join(lines)
    if lines and trailing_newline:
        text += newline
    return text


# ---------------------------------------------------------------------------
# Comparison harnesses
# ---------------------------------------------------------------------------

def _run(fn, path):
    try:
        return fn(path), None
    except BaseException as exc:  # noqa: BLE001 -- ValueError can be uncaught by design
        return None, exc


def _assert_same_outcome_or_raise(old_pair, new_pair):
    (old_result, old_exc), (new_result, new_exc) = old_pair, new_pair
    if old_exc is not None or new_exc is not None:
        assert old_exc is not None, f"old succeeded, new raised {new_exc!r}"
        assert new_exc is not None, f"new succeeded, old raised {old_exc!r}"
        assert type(old_exc) is type(new_exc), (type(old_exc), type(new_exc))
        assert str(old_exc) == str(new_exc), (str(old_exc), str(new_exc))
        return None
    return old_result, new_result


def _compare_load_states(path) -> None:
    outcome = _assert_same_outcome_or_raise(
        _run(_load_states_old, path), _run(ev.load_states, path))
    if outcome is None:
        return
    old, new = outcome
    for field_name in ("seq", "sim_step", "sim_time_s", "wall_time_ns", "cmd_seq",
                        "position_rad", "compliant", "epoch"):
        a, b = getattr(old, field_name), getattr(new, field_name)
        assert a.dtype == b.dtype, (field_name, a.dtype, b.dtype)
        assert a.shape == b.shape, (field_name, a.shape, b.shape)
        assert np.array_equal(a, b), field_name
    assert old.duplicate_indices == new.duplicate_indices
    assert old.truncated_final_line == new.truncated_final_line


def _compare_load_commands(path) -> None:
    outcome = _assert_same_outcome_or_raise(
        _run(_load_commands_old, path), _run(ev.load_commands, path))
    if outcome is None:
        return
    old, new = outcome
    assert old.kind == new.kind
    assert np.array_equal(old.seq, new.seq)
    assert np.array_equal(old.target_rad, new.target_rad, equal_nan=True)
    assert old.compliant == new.compliant
    assert np.array_equal(old.epoch, new.epoch)
    assert old.reset_sim_step == new.reset_sim_step
    assert old.truncated_final_line == new.truncated_final_line


# ---------------------------------------------------------------------------
# Clean files
# ---------------------------------------------------------------------------

class TestCleanFiles:
    def test_clean_states_file(self, tmp_path):
        rows = [_state_row(seq=i, sim_step=i, sim_time_s=i * 0.02, cmd_seq=i - 1)
                 for i in range(5)]
        p = tmp_path / "states.jsonl"
        _write(p, _lines_text([json.dumps(r) for r in rows]))
        _compare_load_states(p)

    def test_clean_states_file_joints_out_of_order(self, tmp_path):
        shuffled = list(reversed(ev._JOINT_ORDER))
        rows = [_state_row(seq=i, sim_step=i, sim_time_s=i * 0.02, cmd_seq=i - 1,
                            joint_names=shuffled) for i in range(3)]
        p = tmp_path / "states.jsonl"
        _write(p, _lines_text([json.dumps(r) for r in rows]))
        _compare_load_states(p)

    def test_clean_commands_file(self, tmp_path):
        rows = [_cmd_row(seq=0), _reset_row(sim_step=5), _cmd_row(seq=1)]
        p = tmp_path / "commands.jsonl"
        _write(p, _lines_text([json.dumps(r) for r in rows]))
        _compare_load_commands(p)


# ---------------------------------------------------------------------------
# Line-level cases
# ---------------------------------------------------------------------------

class TestLineLevel:
    def test_truncated_final_line_no_trailing_newline(self, tmp_path):
        p = tmp_path / "states.jsonl"
        text = json.dumps(_state_row(seq=0)) + "\n" + '{"incomplete'
        _write(p, text)
        _compare_load_states(p)

    def test_truncated_final_line_with_trailing_newline(self, tmp_path):
        p = tmp_path / "states.jsonl"
        text = json.dumps(_state_row(seq=0)) + "\n" + '{"incomplete' + "\n"
        _write(p, text)
        _compare_load_states(p)

    def test_malformed_middle_line(self, tmp_path):
        p = tmp_path / "states.jsonl"
        lines = [json.dumps(_state_row(seq=0)), "{not json",
                 json.dumps(_state_row(seq=2))]
        _write(p, _lines_text(lines))
        _compare_load_states(p)

    def test_malformed_line_followed_only_by_blank_lines(self, tmp_path):
        p = tmp_path / "states.jsonl"
        lines = [json.dumps(_state_row(seq=0)), "{not json", "", "   "]
        _write(p, _lines_text(lines))
        _compare_load_states(p)

    def test_trailing_blank_lines(self, tmp_path):
        p = tmp_path / "states.jsonl"
        lines = [json.dumps(_state_row(seq=0)), json.dumps(_state_row(seq=1)), "", ""]
        _write(p, _lines_text(lines))
        _compare_load_states(p)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "states.jsonl"
        _write(p, "")
        _compare_load_states(p)

    def test_file_of_only_blanks(self, tmp_path):
        p = tmp_path / "states.jsonl"
        _write(p, "\n\n   \n\t\n")
        _compare_load_states(p)

    def test_crlf_line_endings(self, tmp_path):
        p = tmp_path / "states.jsonl"
        rows = [_state_row(seq=0), _state_row(seq=1)]
        text = "\r\n".join(json.dumps(r) for r in rows) + "\r\n"
        _write(p, text)
        _compare_load_states(p)

    def test_commands_malformed_non_final_line(self, tmp_path):
        p = tmp_path / "commands.jsonl"
        lines = [json.dumps(_cmd_row(seq=0)), "{not json", json.dumps(_cmd_row(seq=1))]
        _write(p, _lines_text(lines))
        _compare_load_commands(p)

    def test_commands_truncated_final_line(self, tmp_path):
        p = tmp_path / "commands.jsonl"
        text = json.dumps(_cmd_row(seq=0)) + "\n" + '{"incomplete'
        _write(p, text)
        _compare_load_commands(p)


# ---------------------------------------------------------------------------
# Field-level cases
# ---------------------------------------------------------------------------

class TestFieldLevel:
    @pytest.mark.parametrize(
        "column", ["seq", "sim_step", "sim_time_s", "wall_time_ns", "cmd_seq"])
    def test_missing_scalar_field(self, tmp_path, column):
        row0 = _state_row(seq=0)
        row1 = _state_row(seq=1)
        del row1[column]
        p = tmp_path / "states.jsonl"
        _write(p, _lines_text([json.dumps(row0), json.dumps(row1)]))
        _compare_load_states(p)

    def test_missing_field_early_row_precedes_malformed_line_later(self, tmp_path):
        row0 = _state_row(seq=0)
        del row0["seq"]
        row1 = _state_row(seq=1)
        row2 = _state_row(seq=2)
        p = tmp_path / "states.jsonl"
        lines = [json.dumps(row0), json.dumps(row1), "{not json", json.dumps(row2)]
        _write(p, _lines_text(lines))
        _compare_load_states(p)

    def test_two_columns_missing_in_different_rows(self, tmp_path):
        row0 = _state_row(seq=0)
        del row0["sim_step"]  # column index 1
        row1 = _state_row(seq=1)
        del row1["seq"]       # column index 0 -- must win despite the later row
        p = tmp_path / "states.jsonl"
        _write(p, _lines_text([json.dumps(row0), json.dumps(row1)]))
        _compare_load_states(p)

    def test_missing_joint(self, tmp_path):
        names = list(ev._JOINT_ORDER)
        dropped = names[3]
        joints = [_joint(n) for n in names if n != dropped]
        row = _state_row(joints=joints)
        p = tmp_path / "states.jsonl"
        _write(p, _lines_text([json.dumps(row)]))
        _compare_load_states(p)

    def test_missing_position_rad(self, tmp_path):
        joints = [_joint(n) for n in ev._JOINT_ORDER]
        del joints[5]["position_rad"]
        row = _state_row(joints=joints)
        p = tmp_path / "states.jsonl"
        _write(p, _lines_text([json.dumps(row)]))
        _compare_load_states(p)

    def test_missing_compliant(self, tmp_path):
        joints = [_joint(n) for n in ev._JOINT_ORDER]
        del joints[5]["compliant"]
        row = _state_row(joints=joints)
        p = tmp_path / "states.jsonl"
        _write(p, _lines_text([json.dumps(row)]))
        _compare_load_states(p)

    def test_joints_not_a_list(self, tmp_path):
        row = _state_row()
        row["joints"] = 42
        p = tmp_path / "states.jsonl"
        _write(p, _lines_text([json.dumps(row)]))
        _compare_load_states(p)

    def test_row_is_json_array(self, tmp_path):
        p = tmp_path / "states.jsonl"
        _write(p, _lines_text([json.dumps([1, 2, 3])]))
        _compare_load_states(p)

    def test_none_value_in_numeric_column(self, tmp_path):
        row = _state_row(seq=None)
        p = tmp_path / "states.jsonl"
        _write(p, _lines_text([json.dumps(row)]))
        _compare_load_states(p)

    def test_string_value_in_numeric_column(self, tmp_path):
        row = _state_row(seq="abc")
        p = tmp_path / "states.jsonl"
        _write(p, _lines_text([json.dumps(row)]))
        _compare_load_states(p)

    def test_earlier_column_conversion_error_beats_later_column_missing_key(self, tmp_path):
        """R1 (coordinator review, 2026-09-27): the old code fully resolved
        one column -- extraction across every row, then its own
        np.array(...) conversion -- before even starting the next column's
        extraction. So an earlier column's CONVERSION error (here: seq=None,
        a KeyError/TypeError-wrapped EvidenceError) must win over a later
        column's plain missing-key extraction failure (sim_step absent from
        row 1), even though the missing key would otherwise look like the
        "simpler"/earlier-detected problem in a row-major reading."""
        row0 = _state_row(seq=None)
        row1 = _state_row(seq=1)
        del row1["sim_step"]
        p = tmp_path / "states.jsonl"
        _write(p, _lines_text([json.dumps(row0), json.dumps(row1)]))
        _compare_load_states(p)

    def test_earlier_column_raw_valueerror_beats_later_column_missing_key(self, tmp_path):
        """R1: same as above, but the earlier column's conversion error is
        a raw (uncaught) ValueError -- confirms it is left exactly as
        uncaught as the old code left it, still ahead of the later
        column's missing key."""
        row0 = _state_row(seq="abc")
        row1 = _state_row(seq=1)
        del row1["sim_step"]
        p = tmp_path / "states.jsonl"
        _write(p, _lines_text([json.dumps(row0), json.dumps(row1)]))
        _compare_load_states(p)

    def test_middle_column_conversion_error_beats_later_column_missing_key(self, tmp_path):
        """R1: the earlier-wins rule isn't specific to column 0 -- a
        conversion error in sim_step (column index 1) must still beat a
        missing wall_time_ns (column index 3) in a different row."""
        row0 = _state_row(seq=0, sim_step="not-a-number")
        row1 = _state_row(seq=1)
        del row1["wall_time_ns"]
        p = tmp_path / "states.jsonl"
        _write(p, _lines_text([json.dumps(row0), json.dumps(row1)]))
        _compare_load_states(p)
