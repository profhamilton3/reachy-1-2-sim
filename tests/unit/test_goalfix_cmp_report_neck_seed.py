"""Synthetic tests of the report-only neck/seed extractor
(tools/goalfix_cmp_report/neck_seed.py).  Offline, compact hand-written
evidence; no recorded evidence is read."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "native_mujoco"),
           os.path.join(_ROOT, "scripts"),
           os.path.join(_ROOT, "tests", "fixtures", "goalfix_cmp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import pathcheck as pc  # noqa: E402
from tools.goalfix_cmp._io import read_result  # noqa: E402
from tools.goalfix_cmp._simtime import Bracket  # noqa: E402
from tools.goalfix_cmp_report import neck_seed as ns  # noqa: E402

CYCLE = "S-c1"
KF_QPOS = [0.0] * 8 + [0.001 * k for k in range(1, 9)] + [0.0, 0.45, 0.0, 0.0, 0.0]
assert len(KF_QPOS) == 21

# Applied-at state indices within epoch 1
SETUP_APPLIED = [10, 11, 12, 13, 14, 15]           # bridge seq 1..6
FLIGHT_APPLIED = [30, 31, 32, 33]                  # bridge seq 7..10
SETUP_SPAN = (8, 20)
FLIGHT_SPAN = (28, 40)
SETUP_NECK = (0.011, 0.462, 0.033)
FLIGHT_NECK = (0.021, 0.472, 0.043)


def _neck_pos(s):
    return (0.1 + 0.001 * s, 0.45 + 0.002 * s, -0.3 + 0.003 * s)


def _pos21(s):
    v = [0.01 * (k + 1) for k in range(21)]
    v[16], v[17], v[18] = _neck_pos(s)
    return v


def _target21(neck, other_shift=0.0):
    v = [0.0] * 21
    for k in range(8):
        v[k] = 0.05 * (k + 1)
    for k in range(8, 16):
        v[k] = 0.002 * k + other_shift
    v[16], v[17], v[18] = neck
    v[19] = v[20] = 0.0
    return v


def _write_keyframe(path, qpos=None, name="home"):
    q = KF_QPOS if qpos is None else qpos
    path.write_text('<mujoco><keyframe><key name="%s" qpos="%s"/></keyframe></mujoco>'
                    % (name, " ".join(repr(float(x)) for x in q)))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_session(tmp_path, *, n_states=700, dup_p4=False, with_sidecars=True):
    ev_dir = tmp_path / "ev"
    control = tmp_path / "control"
    ev_dir.mkdir()
    control.mkdir()
    states = []
    seq = 0
    # epoch 0: 5 states, steps 0..40, no commands
    for s in range(5):
        seq += 1
        states.append(mf.state_row(seq=seq, sim_step=10 * s, sim_time_s=0.02 * s, cmd_seq=0,
                                   wall_time_ns=seq, position_rad21=_pos21(0)))
    e1_seq = {}
    applied = {}
    for i, a in enumerate(SETUP_APPLIED):
        applied[a] = 1 + i
    for i, a in enumerate(FLIGHT_APPLIED):
        applied[a] = 7 + i
    cmd_seq_now = 0
    for s in range(n_states):
        seq += 1
        e1_seq[s] = seq
        cmd_seq_now = applied.get(s, cmd_seq_now)
        states.append(mf.state_row(seq=seq, sim_step=10 * s, sim_time_s=0.02 * s,
                                   cmd_seq=cmd_seq_now, wall_time_ns=seq,
                                   position_rad21=_pos21(s)))
    if dup_p4:
        # a paused/duplicated sim_step: state 11 repeats state 10's sim_step
        states[5 + 11]["sim_step"] = states[5 + 10]["sim_step"]
    commands = [mf.command_row_reset(seed=1, sim_step=50, wall_time_s=0.0)]
    for i in range(6):
        commands.append(mf.command_row_joint(
            seq=1 + i, target_rad21=_target21(SETUP_NECK)))
    for i in range(4):
        commands.append(mf.command_row_joint(
            seq=7 + i, target_rad21=_target21(FLIGHT_NECK, other_shift=0.0)))
    mf.write_evidence(ev_dir, states, commands)

    run_dir = str(ev_dir.resolve())
    if with_sidecars:
        for label, route, span in (("setup", "PLACE_ROUTE", SETUP_SPAN),
                                   ("flight", "LIFT_TO_PRESENT", FLIGHT_SPAN)):
            (control / f"{label}.link.json").write_text(json.dumps({
                "alignment": [{"server_seq": e1_seq[span[0]]}, {"server_seq": e1_seq[span[1]]}],
                "log": f"route_clearance_{route}_t.log", "server_run_dir": run_dir}))
    (control / f"cycle_{CYCLE}.json").write_text(json.dumps({
        "rep": 1, "cycle": CYCLE, "setup_sidecar": "setup.link.json",
        "flight_sidecar": "flight.link.json"}))
    kf = tmp_path / "reachy_1_2.xml"
    return ev_dir, control, kf, _write_keyframe(kf)


def _run(ev_dir, control, kf, sha, **kw):
    args = dict(ev_dir=str(ev_dir), run=None, states_rel=None, commands_rel=None,
                sha256sums="SHA256SUMS", control_dir=str(control), cycle=CYCLE, arm="B",
                epoch=1, keyframe_xml=str(kf), keyframe_name="home", keyframe_sha256=sha,
                keyframe_source_rev="abc1234", state_every=100, fixed_sim_step=5000)
    args.update(kw)
    return ns.extract(**args)


def _all_fields(payload):
    for grp in ("neck_targets", "neck_actual", "seed_deviation_13"):
        for k, f in payload[grp].items():
            yield f"{grp}.{k}", f


def _assert_no_fill(payload):
    """An incomplete field is null with a reason; nothing is ever a fabricated value."""
    for name, f in _all_fields(payload):
        assert f["status"] in ("ok", "incomplete"), name
        if f["status"] == "incomplete":
            assert f["value"] is None, name
            assert f["reason"], name


def _exp_minus(vec):
    return {n: vec[i] - KF_QPOS[i] for n, i in zip(ns.NECK_NAMES, ns.NECK_IDX)}


def _exp_neck(vec3):
    return dict(zip(ns.NECK_NAMES, vec3))


# ---------------------------------------------------------------------------

def test_all_ok_exact_values(tmp_path):
    ev_dir, control, kf, sha = make_session(tmp_path)
    payload, rc = _run(ev_dir, control, kf, sha)
    assert rc == 0 and payload["all_ok"] is True and payload["report_only"] is True
    assert payload["keyframe"]["sha256"] == sha and payload["keyframe"]["source_rev"] == "abc1234"
    assert all(f["status"] == "ok" for _, f in _all_fields(payload))

    # neck targets of each leg's first command, and target - keyframe
    for label, neck, first_seq in (("setup", SETUP_NECK, 1), ("flight", FLIGHT_NECK, 7)):
        v = payload["neck_targets"][label]["value"]
        assert v["seq"] == first_seq and v["epoch"] == 1
        assert v["target_rad"] == _exp_neck(neck)
        assert v["minus_keyframe"] == _exp_minus(list(_target21(neck)))
        assert v["turn_on_match_local_index"] in (None, 0, 1, 2, 3, 4, 5)

    # actual neck positions and actual - keyframe at the five state kinds
    def check(field, s):
        v = payload["neck_actual"][field]["value"]
        assert v["position_rad"] == _exp_neck(_neck_pos(s)), field
        assert v["minus_keyframe"] == _exp_minus(_pos21(s)), field
        assert v["epoch"] == 1 and v["sim_step"] == 10 * s, field

    check("p4", 10)                      # first state with sim_step >= 100
    check("p5", 500)                     # first state with sim_step >= 5000
    check("setup_first_cmd_t_hi", 10)    # first state reporting cmd_seq 1
    check("setup_last_aligned", SETUP_SPAN[1])
    check("flight_first_cmd_t_hi", 30)
    check("flight_last_aligned", FLIGHT_SPAN[1])


def test_seed_deviation_equals_check_c8_second_value(tmp_path):
    ev_dir, control, kf, sha = make_session(tmp_path)
    payload, _ = _run(ev_dir, control, kf, sha)
    evd = ev.verify_and_load(str(ev_dir), "states.jsonl", "commands.jsonl", "SHA256SUMS")
    for label, rows in (("setup", range(1, 7)), ("flight", range(7, 11))):
        targets21 = np.asarray([evd.commands.target_rad[i] for i in rows])
        _res, dev = pc.check_c8(targets21, KF_QPOS)
        assert payload["seed_deviation_13"][label]["value"]["max_abs_rad"] == dev
    # independently: max |first[8:21] - keyframe[8:21]|
    first = _target21(SETUP_NECK)
    exp = max(abs(first[i] - KF_QPOS[i]) for i in range(8, 21))
    assert payload["seed_deviation_13"]["setup"]["value"]["max_abs_rad"] == exp
    assert exp > 0


def test_seed_deviation_is_taken_from_the_shipped_function(tmp_path, monkeypatch):
    ev_dir, control, kf, sha = make_session(tmp_path)
    calls = []

    def spy(targets21, kf21):
        calls.append(len(targets21))
        return pc.CheckResult("C8", True), 12.5
    monkeypatch.setattr(ns.pc, "check_c8", spy)
    payload, _ = _run(ev_dir, control, kf, sha)
    assert calls == [6, 4]
    assert payload["seed_deviation_13"]["setup"]["value"]["max_abs_rad"] == 12.5


def test_keyframe_sha_mismatch_is_incomplete_everywhere(tmp_path):
    ev_dir, control, kf, sha = make_session(tmp_path)
    payload, rc = _run(ev_dir, control, kf, "0" * 64)
    assert rc == 3 and payload["all_ok"] is False
    assert "sha256 mismatch" in payload["keyframe"]["reason"]
    assert all(f["status"] == "incomplete" for _, f in _all_fields(payload))
    _assert_no_fill(payload)


def test_keyframe_with_20_values_is_incomplete(tmp_path):
    ev_dir, control, kf, _ = make_session(tmp_path)
    sha = _write_keyframe(kf, qpos=KF_QPOS[:20])
    payload, rc = _run(ev_dir, control, kf, sha)
    assert rc == 3 and "exactly 21" in payload["keyframe"]["reason"]
    _assert_no_fill(payload)


def test_keyframe_missing_name_or_file_is_incomplete(tmp_path):
    ev_dir, control, kf, sha = make_session(tmp_path)
    payload, rc = _run(ev_dir, control, kf, sha, keyframe_name="nope")
    assert rc == 3 and "found 0" in payload["keyframe"]["reason"]
    payload, rc = _run(ev_dir, control, tmp_path / "absent.xml", sha)
    assert rc == 3 and "does not exist" in payload["keyframe"]["reason"]


def test_epoch_mismatch_is_incomplete(tmp_path):
    ev_dir, control, kf, sha = make_session(tmp_path)
    # Epoch 0 has states (steps 0..40) but no legs and no state with sim_step >= 5000.
    payload, rc = _run(ev_dir, control, kf, sha, epoch=0, state_every=0)
    assert rc == 3
    assert payload["neck_actual"]["p4"]["status"] == "ok"
    assert payload["neck_actual"]["p4"]["value"]["epoch"] == 0
    assert payload["neck_actual"]["p5"]["status"] == "incomplete"
    for label in ("setup", "flight"):
        assert payload["neck_targets"][label]["status"] == "incomplete"
        assert "expected 0" in payload["neck_targets"][label]["reason"]
        assert payload["seed_deviation_13"][label]["status"] == "incomplete"
        assert payload["neck_actual"][f"{label}_last_aligned"]["status"] == "incomplete"
    _assert_no_fill(payload)


def test_no_state_with_sim_step_5000_is_incomplete_and_not_zero(tmp_path):
    ev_dir, control, kf, sha = make_session(tmp_path, n_states=300)     # max sim_step 2990
    payload, rc = _run(ev_dir, control, kf, sha)
    assert rc == 3
    p5 = payload["neck_actual"]["p5"]
    assert p5["status"] == "incomplete" and p5["value"] is None
    assert "sim_step >= 5000" in p5["reason"]
    assert payload["neck_actual"]["p4"]["status"] == "ok"        # the others are unaffected
    assert payload["neck_targets"]["setup"]["status"] == "ok"
    _assert_no_fill(payload)


def test_duplicate_sim_step_at_p4_is_ambiguous(tmp_path):
    ev_dir, control, kf, sha = make_session(tmp_path, dup_p4=True)
    payload, rc = _run(ev_dir, control, kf, sha)
    assert rc == 3
    p4 = payload["neck_actual"]["p4"]
    assert p4["status"] == "incomplete" and "ambiguous" in p4["reason"] and p4["value"] is None
    _assert_no_fill(payload)


def test_missing_sidecar_makes_that_leg_incomplete_only(tmp_path):
    ev_dir, control, kf, sha = make_session(tmp_path)
    os.remove(control / "setup.link.json")
    payload, rc = _run(ev_dir, control, kf, sha)
    assert rc == 3
    for f in (payload["neck_targets"]["setup"], payload["seed_deviation_13"]["setup"],
              payload["neck_actual"]["setup_first_cmd_t_hi"],
              payload["neck_actual"]["setup_last_aligned"]):
        assert f["status"] == "incomplete" and "missing sidecar" in f["reason"]
    assert payload["neck_targets"]["flight"]["status"] == "ok"
    assert payload["neck_actual"]["p4"]["status"] == "ok"
    _assert_no_fill(payload)


def test_missing_manifest_is_incomplete(tmp_path):
    ev_dir, control, kf, sha = make_session(tmp_path)
    os.remove(control / f"cycle_{CYCLE}.json")
    payload, rc = _run(ev_dir, control, kf, sha)
    assert rc == 3
    assert payload["neck_targets"]["setup"]["status"] == "incomplete"
    assert "manifest" in payload["neck_targets"]["setup"]["reason"]
    _assert_no_fill(payload)


def test_leg_with_no_placeable_command_is_incomplete(tmp_path):
    """Setup sidecar aligned over states that apply no command at all: the shipped
    resolve_leg refuses ('no commands found'), so nothing is invented."""
    ev_dir, control, kf, sha = make_session(tmp_path)
    doc = json.loads((control / "setup.link.json").read_text())
    evd = ev.verify_and_load(str(ev_dir), "states.jsonl", "commands.jsonl", "SHA256SUMS")
    idx = {int(s): i for i, s in enumerate(evd.states.seq)}
    seqs = sorted(idx)
    # two states well after every command was applied and before the flight span
    doc["alignment"] = [{"server_seq": seqs[5 + 100]}, {"server_seq": seqs[5 + 110]}]
    (control / "setup.link.json").write_text(json.dumps(doc))
    payload, rc = _run(ev_dir, control, kf, sha)
    assert rc == 3
    assert payload["neck_targets"]["setup"]["status"] == "incomplete"
    assert "no commands found" in payload["neck_targets"]["setup"]["reason"]
    _assert_no_fill(payload)


def _located(tmp_path):
    ev_dir, control, kf, sha = make_session(tmp_path)
    evd = ev.verify_and_load(str(ev_dir), "states.jsonl", "commands.jsonl", "SHA256SUMS")
    manifest = json.loads((control / f"cycle_{CYCLE}.json").read_text())
    run_dir = str(ev_dir.resolve())
    out = {}
    for label, key, route in ns._LEGS:
        spec, leg, why = ns._locate_leg(evd, control, manifest, label, key, route, run_dir)
        assert spec is not None, why
        out[label] = (spec, leg)
    return evd, out


def test_unplaceable_row_before_the_located_first_command_is_ambiguous(tmp_path):
    evd, legs = _located(tmp_path)
    spec, leg = legs["flight"]
    first = spec.command_indices[0]
    assert evd.commands.kind[first - 1] == "joint_command"
    evd.brackets[first - 1] = Bracket(1, None, None, None)             # unplaceable
    nt, t_hi, last, seed = ns._leg_fields(evd, spec, leg, 1, KF_QPOS, "flight")
    assert nt["status"] == "incomplete" and "unplaceable" in nt["reason"] and nt["value"] is None
    assert t_hi["status"] == "incomplete" and t_hi["value"] is None
    assert last["status"] == "ok"                       # independent of the first command


def test_bracket_spanning_epochs_is_incomplete(tmp_path):
    evd, legs = _located(tmp_path)
    spec, leg = legs["setup"]
    first = spec.command_indices[0]
    b = evd.brackets[first]
    evd.brackets[first] = Bracket(0, b.t_lo, b.t_hi, b.hi_state_index)   # bracket epoch != state epoch
    nt, t_hi, last, seed = ns._leg_fields(evd, spec, leg, 1, KF_QPOS, "setup")
    assert t_hi["status"] == "incomplete" and "spans epochs" in t_hi["reason"]
    assert t_hi["value"] is None


def test_evidence_integrity_failure_is_incomplete(tmp_path):
    ev_dir, control, kf, sha = make_session(tmp_path)
    (ev_dir / "commands.jsonl").write_text((ev_dir / "commands.jsonl").read_text() + " ")
    payload, rc = _run(ev_dir, control, kf, sha)
    assert rc == 3 and "sha256 mismatch" in payload["neck_targets"]["setup"]["reason"]
    _assert_no_fill(payload)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _argv(ev_dir, control, kf, sha, out, **over):
    a = {"--ev-dir": str(ev_dir), "--control-dir": str(control), "--cycle": CYCLE, "--arm": "A",
         "--epoch": "1", "--keyframe-xml": str(kf), "--keyframe-name": "home",
         "--keyframe-sha256": sha, "--keyframe-source-rev": "abc1234",
         "--state-every": "100", "--fixed-sim-step": "5000", "--out": str(out)}
    a.update(over)
    return [x for kv in a.items() for x in kv]


def test_cli_rc_0_and_json(tmp_path):
    ev_dir, control, kf, sha = make_session(tmp_path)
    out = tmp_path / "neck.json"
    rc = ns._cli(_argv(ev_dir, control, kf, sha, out))
    assert rc == 0
    doc = read_result(out)
    assert doc["rc"] == 0 and doc["report_only"] is True and doc["arm_label"] == "A"


def test_cli_rc_3_when_any_field_incomplete(tmp_path):
    ev_dir, control, kf, sha = make_session(tmp_path, n_states=300)
    out = tmp_path / "neck.json"
    assert ns._cli(_argv(ev_dir, control, kf, sha, out)) == 3
    assert read_result(out)["rc"] == 3


def test_cli_rc_2_on_usage_errors(tmp_path):
    ev_dir, control, kf, sha = make_session(tmp_path)
    out = tmp_path / "neck.json"
    assert ns._cli(_argv(ev_dir, control, kf, "xyz", out)) == 2                 # bad sha argument
    assert ns._cli(_argv(ev_dir, control, kf, sha, out, **{"--epoch": "-1"})) == 2
    argv = _argv(ev_dir, control, kf, sha, out)
    i = argv.index("--fixed-sim-step")
    with pytest.raises(SystemExit) as exc:                                       # required, no default
        ns._cli(argv[:i] + argv[i + 2:])
    assert exc.value.code == 2
