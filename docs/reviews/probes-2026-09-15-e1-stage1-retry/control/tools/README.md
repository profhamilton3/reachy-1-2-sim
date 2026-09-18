# E1 Stage 1 execution notebook tooling

Generates the marker-gated Stage 1 cycle notebooks (`S1a`, `S1b`, `S1c`)
from `plan.py`'s single source of truth for timing and provenance. This
tree is code, not evidence: notebooks are written into an evidence
directory the operator names at run time, never into `docs/reviews/`.

Nothing here starts a server, connects an SDK, or executes a notebook.
Generation and the fail-closed gating logic are covered by
`tests/unit/test_e1_stage1_notebook.py`.

## `settle_wait.py`

Not carried over from PR #117: this package's `leg.sh` does not call it.
If a future cycle's operator flow needs a settle-wait step, copy it from
`~/reachy-1-2-sim-stage1/docs/reviews/probes-2026-09-15-e1-stage1-b4/control/tools/settle_wait.py`
(read-only reference) rather than reconstructing it.

## Operator flow, in order

1. **Stop the native server.**
2. **Confirm the tree is at or after this PR's merge** (`plan.REQUIRED_SHA`,
   currently `3d7fc74`) —
   `git -C <server tree> merge-base --is-ancestor <REQUIRED_SHA> HEAD`.
   `4d727c0` (PR #118's merge, the `cmd_seq` fix) is necessary but **not**
   sufficient: `native_mujoco/recorder.py` did not stamp `code_sha` into
   the manifest until `3d7fc74`, this PR's own first commit. A server tree
   at exactly `4d727c0` runs a `recorder.py` with no `code_sha` field at
   all, and the binding cell's honest "manifest has no code_sha" refusal
   reads like a mystery STOP for that reason alone (PR #120 review, M2) —
   `3d7fc74` is a descendant of `4d727c0`, so requiring it still implies
   the `cmd_seq` fix.
3. **Choose an evidence directory outside this repo's checkout entirely**
   (not a subdirectory of it, and not under any other repo's `docs/reviews/`
   either). `_code_provenance()` runs `git status --porcelain` in the
   server's own tree; `start_sim.sh` creates `$REACHY_SIM_RECORD` (and
   `leg.sh`'s `--record-root`) before the server starts, and an untracked
   directory anywhere inside the checkout — `e1_server_runs/` is not
   gitignored — makes `git status --porcelain` non-empty, which makes
   `code_sha_dirty: true` on every run, which the binding cell refuses
   (correctly, but for a reason nothing here used to say — PR #120 review,
   M3). `make_cycle_notebook`'s generator now refuses at generation time,
   with nothing written, if `--evidence-dir` is inside `--repo`.
4. **Start the server from that tree with `--record` pointed at the chosen
   evidence directory.** `native_mujoco/recorder.py` stamps the new run's
   `manifest.json` with `code_sha` (this branch's own tree, best-effort
   `git rev-parse HEAD`) and `code_sha_dirty` (whether that tree had
   uncommitted changes). A dirty or unreadable tree makes the sha alone
   unprovable as runtime identity, so the notebook's binding cell refuses
   on `code_sha_dirty != False` exactly as it refuses on a missing
   `code_sha` — commit or stash before starting the server for this
   experiment.
5. **Confirm the new run's `manifest.json` has `code_sha` set and
   `code_sha_dirty: false`.**
6. **Only then generate the notebook, with `--repo` pointed at the exact
   checkout the server in step 4 is running from** — not merely a clone at
   the same commit:
   ```
   python -m scripts.e1_stage1.make_cycle_notebook S1a \
       --repo /path/to/reachy-1-2-sim \
       --evidence-dir /path/to/a/new/evidence/directory
   ```
   The generator refuses (no files written) if `--repo`'s `HEAD` does not
   descend from `plan.REQUIRED_SHA`, or if `--evidence-dir` is under
   `docs/reviews/` or anywhere inside `--repo` (step 3). It also embeds
   `--repo`'s HEAD as `GENERATED_AT_SHA`; the generated notebook's binding
   cell requires the server's `code_sha` to equal `GENERATED_AT_SHA`
   **exactly**, not merely descend from `REQUIRED_SHA` — so a server
   running a different tree than the one `--repo` pointed at (even a
   related one) fails closed at binding time instead of silently (PR #120
   review, W4 note 1).
7. Run the operator's per-leg loop (`leg.sh <repo> <evidence-dir> <cycle>
   <leg-name> <ROUTE> <TAIL_TARGET>` per leg; `reset.sh` between cycles) and
   drive the generated notebook by hand, writing `go_<leg>` markers as each
   gate passes. `leg.sh` reads each leg's duration from
   `plan_<cycle>.json` (written alongside the notebook) rather than a
   positional argument.

   **Run `leg.sh` (and the notebook kernel) from the documented host SDK
   environment ("e1venv", `reachy_sdk` 0.7.0), not the system Python** (M4,
   PR #120 review): `measure_route_clearance.py` imports `reachy_sdk`,
   which the system `python3` does not have, so a bare `python3` exits
   non-zero on the first leg. Set `E1_PYTHON` to the venv's interpreter
   before calling `leg.sh`:
   ```
   export E1_PYTHON=/path/to/e1venv/bin/python3
   ```
   and select the `e1venv` kernel for the generated notebook (its
   `kernelspec` names it, but the kernel itself has to exist on the
   operator's Jupyter).

Step 5's requirement is new relative to PR #117's flow, which had no way to
prove which tree the server ran from at all (W4). The notebook's binding
cell (after `verify_simulator_identity`) re-checks the manifest at runtime:
`code_sha` present, clean, a descendant of `REQUIRED_SHA`, equal to
`GENERATED_AT_SHA`, and `manifest.started_at` after `REQUIRED_SHA`'s commit
time (`plan.MERGE_TIME_ISO`). Any failure writes `binding_FAIL_<cycle>` and
every leg in the cycle becomes `not_eligible` -- no motion is attempted.

## Already-stiff arm -- read-only investigation (assignment section 6)

Question: on an arm whose cached `compliant` is already `False`,
does `reachy.turn_on("r_arm")` still advance `cmd_seq`, and can that make
`require_compliance(min_cmd_seq=baseline)` falsely time out?

1. **Does `turn_on` on an already-stiff arm still enqueue a `compliant`
   register for sync?** `reachy_sdk/joint.py` `Joint.__setitem__` (`:79-91`)
   writes through a `Register` descriptor that marks the register dirty
   for the next sync **unconditionally on write** -- it does not compare
   against the joint's last-known value first. `reachy_sdk.py`
   `_change_compliancy` (`:269-287`) calls `joint.compliant = False` (or
   `True`) for every named joint on `turn_on`/`turn_off`, with no
   short-circuit for a joint that is already in that state. So yes: `turn_on`
   on an already-stiff arm still enqueues a `compliant` write.
   Side effect in the same method (`joint.py:83-84`): setting
   `compliant = False` (going stiff) also sets `goal_position =
   present_position` for that joint -- the arm is pinned to hold wherever it
   currently is, rather than snapping to a stale prior `goal_position`.
2. **Does the sync loop then emit a `JointsCommand`, and does the bridge
   produce a new `seq`?** `reachy_sdk.py`'s background sync loop (`~185-265`)
   flushes every dirty register on its next tick, batching them into one
   `JointsCommand` per sync period -- it does not check whether the value
   being sent differs from the simulator's last-known state, only whether
   the local register is marked dirty. On the fake server side,
   `fake_reachy_server._submit_batch` forwards to
   `MujocoRemoteBackend.submit_command`, whose `_send`
   (`mujoco_remote_backend.py:332-378`) increments `_cmd_seq` and stamps it
   on the message unconditionally, for any non-empty batch -- it does not
   de-duplicate against the previously applied `compliant` value either.
3. **Therefore:** yes, `cmd_seq` advances after `turn_on` even when the arm
   was already stiff, because neither the SDK's register write nor the
   bridge's send path compares against the prior value before emitting.
   `require_compliance(min_cmd_seq=baseline)` does **not** falsely time out
   for *this* reason: the already-stiff case still produces a fresh
   `cmd_seq` past the baseline, same as a genuine compliance transition.
4. **Residual false STOP that does exist** (separate from the already-stiff
   path, not "fixed" by this branch, per the assignment's scope): a bridge
   container restart resets `MujocoRemoteBackend._cmd_seq` to 0 while the
   native server's own counter and any baseline read from an existing run
   directory stay at their old, higher value. The operator symptom is
   `require_compliance`'s reason string, `cmd_seq=k has not advanced past
   the pre-call baseline (min_cmd_seq=N)` with `k < N`, persisting past
   `timeout_s` on the very first attempt after the restart. Recovery:
   restart the native server too, so both counters start from 0 together --
   never lower the baseline by hand to work around it, since that reopens
   the gap `min_cmd_seq` exists to close.

(Evidence trail: `reachy_sdk` 0.7.0 source read read-only from the copy
noted in the assignment's verified facts; `mujoco_remote_backend.py` and
`fake_reachy_server.py` read from this tree at `4d727c0`. No SDK was
instantiated and no server was started to answer this section.)
