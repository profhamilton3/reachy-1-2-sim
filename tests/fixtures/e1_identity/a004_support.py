"""Test-only helpers over a004_bridge_restart.json (real a004 rows).

The fixture states are a PROJECTION of the sealed lines: the gate's fields
(type, seq, sim_step, wall_time_ns, cmd_seq) plus the 8 right-arm joints'
name/position_rad/effort/compliant; their values equal the sealed lines.
The 4 command rows are the sealed rows verbatim. At test time only
`wall_time_ns` is rewritten (to `time.monotonic_ns()`, the clock the native
server stamps states with), so the freshness check is genuinely satisfied.
"""
import copy
import json
import pathlib
import time

FIXTURE = pathlib.Path(__file__).with_name("a004_bridge_restart.json")


def load():
    return json.loads(FIXTURE.read_text())


def fresh(state, *, now_ns=None):
    s = copy.deepcopy(state)
    s["wall_time_ns"] = time.monotonic_ns() if now_ns is None else now_ns
    return s


def line(obj):
    return json.dumps(obj, separators=(",", ":")) + "\n"


def write_pre_restart_run_dir(run_dir):
    """The run dir as the r1 setup leg found it: commands rows 1-3 (two
    old-bridge joint_commands and the reset), states up to the baseline
    state (cmd_seq=2). The 4th command row and the post state are appended
    by the fake `turn_on` / by the test."""
    d = load()
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "commands.jsonl").write_text("".join(line(c) for c in d["commands"][:3]))
    (run_dir / "states.jsonl").write_text(line(fresh(d["baseline_state"])))
    return d


def append_commands(run_dir, rows):
    with open(pathlib.Path(run_dir) / "commands.jsonl", "a") as f:
        for r in rows:
            f.write(line(r))


def append_states(run_dir, states):
    with open(pathlib.Path(run_dir) / "states.jsonl", "a") as f:
        for s in states:
            f.write(line(s))


def write_baseline_streams(run_dir, baseline):
    """Real files for a notebook-level stub test whose `baseline` is the old
    injected `_read_last_state` return value: None -> no states file
    (unreadable), otherwise one complete state line carrying exactly that
    dict (plus a `seq`), and an empty commands.jsonl."""
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "commands.jsonl").write_text("")
    if baseline is not None:
        (run_dir / "states.jsonl").write_text(line({"seq": 7, **baseline}))
