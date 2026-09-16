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
2. **Confirm the tree is at or after `4d727c0`** (`plan.REQUIRED_SHA`) —
   `git -C <server tree> merge-base --is-ancestor <REQUIRED_SHA> HEAD`.
   A server built off an older tree never writes `cmd_seq` into `State`
   (base server bug, fixed by `a3ee2c9`); the compliance gate would refuse
   forever without explaining why (W4, PR #118 re-review).
3. **Start the server from it with `--record` to a fresh `e1_server_runs`.**
   `native_mujoco/recorder.py` now stamps the new run's `manifest.json` with
   `code_sha` (this branch's own tree, best-effort `git rev-parse HEAD`) and
   `code_sha_dirty` (whether that tree had uncommitted changes). A dirty or
   unreadable tree makes the sha alone unprovable as runtime identity, so
   the notebook's binding cell refuses on `code_sha_dirty != False` exactly
   as it refuses on a missing `code_sha` — commit or stash before starting
   the server for this experiment.
4. **Confirm the new run's `manifest.json` has `code_sha` set and
   `code_sha_dirty: false`.**
5. **Only then generate the notebook:**
   ```
   python -m scripts.e1_stage1.make_cycle_notebook S1a \
       --repo /path/to/reachy-1-2-sim \
       --evidence-dir /path/to/a/new/evidence/directory
   ```
   The generator itself refuses (no files written) if `--repo`'s `HEAD`
   does not descend from `4d727c0`.
6. Run the operator's per-leg loop (`leg.sh <repo> <evidence-dir> <cycle>
   <leg-name> <ROUTE> <TAIL_TARGET>` per leg; `reset.sh` between cycles) and
   drive the generated notebook by hand, writing `go_<leg>` markers as each
   gate passes. `leg.sh` reads each leg's duration from
   `plan_<cycle>.json` (written alongside the notebook) rather than a
   positional argument.

Step 4's requirement is new relative to PR #117's flow, which had no way to
prove which tree the server ran from at all (W4). The notebook's binding
cell (after `verify_simulator_identity`) re-checks the manifest at runtime:
`code_sha` present and clean, a descendant of `4d727c0`, and
`manifest.started_at` after `4d727c0`'s merge (`2026-09-16T01:35:22Z`,
`plan.MERGE_TIME_ISO`). Either failure writes `binding_FAIL_<cycle>` and
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
