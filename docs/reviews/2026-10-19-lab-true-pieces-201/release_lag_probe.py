"""The live cylinder release, replayed offline with the live link's timing
(#201): pool_cyl_1 r2c3 -> r2c2, best grasp.

Live (trial-2026-10-19-issue201-live, run 2) the release let go at -18.4 deg,
one ramp step from where the hand stood, and the pads were still on the
cylinder at 29.7 N.  Offline (no lag) it let go at -28.0 deg with 0 N.

What the live link adds, modelled here:
  * observations are SAMPLED: the server pushes a state every 10 physics steps
    (50 Hz, `server.py` `_STATE_HZ`), and the grip force, the object pose and
    the gripper angle the executor reads are the latest push, ``obs_s`` old;
  * commands arrive ``cmd_s`` after they are sent (SDK -> bridge -> server);
  * ``dropout``: the grip force reads 0.0 for one push at the first push after
    the release starts -- what `motion_worker._observer` returns when the
    gripper entry is missing (``g.get("grip_force_n", 0.0)``) or what a
    contact bounce looks like at 50 Hz.

Prints one JSON row per case: the let-go angle, the grip force after the
release, the criteria, and the force/angle trace through the release.

Run: python3 release_lag_probe.py OUT.jsonl CASE [CASE ...]   (names in CASES)
"""
import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO, REPO / "src", REPO / "native_mujoco", REPO / "tests" / "unit", HERE):
    sys.path.insert(0, str(p))

CASES = {
    "none": dict(obs_s=0.0, cmd_s=0.0, dropout=False),
    "sampled": dict(obs_s=0.02, cmd_s=0.0, dropout=False),
    "obs100": dict(obs_s=0.1, cmd_s=0.0, dropout=False),
    "obs100_cmd100": dict(obs_s=0.1, cmd_s=0.1, dropout=False),
    "obs200_cmd200": dict(obs_s=0.2, cmd_s=0.2, dropout=False),
    "cmd200": dict(obs_s=0.02, cmd_s=0.2, dropout=False),
    "dropout": dict(obs_s=0.02, cmd_s=0.0, dropout=True),
    "dropout_obs100_cmd100": dict(obs_s=0.1, cmd_s=0.1, dropout=True),
}
PUSH_STEPS = 10           # the server's state push: every 10 steps at 500 Hz


def patch(case):
    from tools.withdrawal_replay import rebuild
    W = rebuild.World
    init, target, grip, deg = W.__init__, W.target, W.grip_force, W.deg
    st = {"trace": [], "releasing": False, "dropped": False}

    def __init__(self):
        init(self)
        self._pushes = collections.deque(maxlen=400)
        self._pending = collections.deque()
        self._n = 0
        adv = self.core.advance

        def advance():
            adv()
            self._n += 1
            t = self._n * self.m.opt.timestep
            while self._pending and self._pending[0][0] <= t:
                _, j, v = self._pending.popleft()
                target(self, j, v)
            if self._n % PUSH_STEPS == 0:
                f = grip(self)
                if case["dropout"] and st["releasing"] and not st["dropped"]:
                    st["dropped"] = True
                    f_seen = 0.0
                else:
                    f_seen = f
                self._pushes.append((t, f_seen, deg(self, "r_gripper")))
                if st["releasing"]:
                    st["trace"].append(dict(t=round(t, 3), force=round(f, 2),
                                            gripper=round(deg(self, "r_gripper"), 2),
                                            goal=round(self._goal_g, 2)))
        self.core.advance = advance
        self._goal_g = deg(self, "r_gripper")

    def _seen(self):
        t = self._n * self.m.opt.timestep
        for row in reversed(self._pushes):
            if row[0] <= t - case["obs_s"] + 1e-9:
                return row
        return None

    def target_(self, joint, v):
        if joint == "r_gripper":
            self._goal_g = v
        if case["cmd_s"] <= 0.0:
            return target(self, joint, v)
        t = self._n * self.m.opt.timestep
        self._pending.append((t + case["cmd_s"], joint, v))

    def grip_(self):
        row = _seen(self)
        return grip(self) if row is None else row[1]

    def deg_(self, joint):
        if joint == "r_gripper":
            row = _seen(self)
            if row is not None:
                return row[2]
        return deg(self, joint)

    W.__init__, W.target, W.grip_force, W.deg = __init__, target_, grip_, deg_
    return st, (init, target, grip, deg)


def run(name):
    from tools.place_sweep import fly as F
    from tools.withdrawal_replay import rebuild
    case = CASES[name]
    st, saved = patch(case)

    from reachy_ai.motion import primitives as P
    rel = (P.release_grip, P.release_grip_retreating, P.release_to)

    def wrap(fn):
        def w(*a, **k):
            st["releasing"] = True
            try:
                return fn(*a, **k)
            finally:
                st["releasing"] = False
        return w
    P.release_grip, P.release_grip_retreating, P.release_to = (wrap(rel[0]), wrap(rel[1]),
                                                               wrap(rel[2]))
    try:
        r = F.fly(oid="pool_cyl_1", origin="r2c3", dest="r2c2", grasp="best")
    finally:
        rebuild.World.__init__, rebuild.World.target, rebuild.World.grip_force, rebuild.World.deg = saved
        P.release_grip, P.release_grip_retreating, P.release_to = rel
    rep = (r.get("outcome") or {}).get("replaced") or {}
    return dict(case=name, **case, let_go_deg=rep.get("let_go_deg"),
                grip_force_after_open_n=rep.get("grip_force_after_open_n"),
                opening_deg=rep.get("opening_deg"), criteria=r.get("criteria"),
                halt=(r.get("outcome") or {}).get("halt"), error_mm=r.get("error_mm"),
                trace=st["trace"][:150])


if __name__ == "__main__":
    out, *names = sys.argv[1:]
    with open(out, "a") as f:
        for n in names:
            row = run(n)
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            print(json.dumps({k: v for k, v in row.items() if k != "trace"}, default=str),
                  flush=True)
