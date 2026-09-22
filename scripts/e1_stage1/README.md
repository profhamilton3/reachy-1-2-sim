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

   `leg.sh` takes an optional 7th argument, `MODE` (`start` or `end`,
   default `end`), carried over from PR #117 for pre-flight parked
   recordings: in `start` mode a failed tail check is informational only
   and does not write `control/stop`. **Never pass `start` for a flown
   leg** -- omit the argument (or pass `end` explicitly) so an overrun at
   the end of an actual flight still writes `control/stop` rather than
   being silently logged and ignored.

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

   **`reset.sh`'s verification contract:** `reset.sh` snapshots the run's
   last state and current reset count (`reset_verify.py snapshot`) before
   sending the sentinel request, then, once the ack poll matches, demands
   *fresh evidence of the requested reset* (`reset_verify.py verify`)
   before printing `reset gen=... ok` and returning 0: (1) the ack equals
   the requested generation; (2) `commands.jsonl`'s reset count is
   *exactly* one more than the snapshot's (two resets is ambiguous
   evidence of *this* one, not sufficient), and that new reset line's
   `sim_step` is an `int` at or after the snapshot's; (3) the last
   *complete* state (read from the tail, tolerant of a torn last line
   from the server's concurrent ~20 Hz writer) has `seq`/`wall_time_ns`
   strictly past the snapshot's and `sim_step` restarted below both the
   snapshot's and the reset line's; (4) that state is live -- recorded
   within `max_state_age_s` (default 1.0 s) of now. It polls for up to
   `--timeout-s` (10 s default; override with `RESET_VERIFY_TIMEOUT_S`/
   `RESET_VERIFY_POLL_S`, an env knob like `E1_PYTHON`, not a new
   positional argument) before giving up. Any check that doesn't hold --
   malformed JSON, a `bool`/`NaN` where an `int` is required, a missing
   or stale sample -- writes `STOP: reset <gen> not verified: <reason>`
   to `control/stop` and exits 9, with one of a fixed set of reasons
   (`ack mismatch`, `reset not recorded`, `more than one reset recorded`,
   `reset line malformed`, `no complete state`, `state malformed:
   <field>`, `state not fresh`, `sim_step not restarted`, `state stale
   (age ...)`). **A stop from this step is a stop, exactly as before --
   there is no retry**, and this change only makes the read of the
   evidence robust to the server's own concurrent writing; it does not
   make the verification more permissive.

Step 5's requirement is new relative to PR #117's flow, which had no way to
prove which tree the server ran from at all (W4). The notebook's binding
cell (after `verify_simulator_identity`) re-checks the manifest at runtime:
`code_sha` present, clean, a descendant of `REQUIRED_SHA`, equal to
`GENERATED_AT_SHA`, and `manifest.started_at` after `REQUIRED_SHA`'s commit
time (`plan.MERGE_TIME_ISO`). Any failure writes `binding_FAIL_<cycle>` and
every leg in the cycle becomes `not_eligible` -- no motion is attempted.

## When to regenerate

A generated notebook and its `plan_<cycle>.json` bake in `GENERATED_AT_SHA`,
`REPO`, the evidence directory, and each leg's `dur_s`. All of the following
invalidate that snapshot -- regenerate (step 6) before continuing:

1. **The server checkout's `HEAD` moves** -- a pull, checkout, commit, or
   stash pop in the tree the server is running from. Regenerating alone
   after a pull is **not enough**: `code_sha` is captured once, at server
   start (step 4), so the server must also be **restarted** from the moved
   tree. Cell 1 refuses if the kernel's own tree no longer matches
   `GENERATED_AT_SHA`; cell 2 (the binding cell) refuses if the server's
   `code_sha` no longer matches it either -- but only a server restart
   fixes the second one.
2. **`plan.py` changes** -- any edit to the budgets or timing constants
   (`ROUTE_BUDGET_S` and everything it is derived from). `leg.sh` reads
   each leg's `dur_s` from `plan_<cycle>.json`, written at generation time,
   not from `plan.py` directly, so a `plan.py` change has no effect on an
   already-generated notebook until it is regenerated.
3. **The evidence directory moves** -- `--evidence-dir`'s absolute path is
   baked into the generated notebook and `plan_<cycle>.json`.
4. **`plan.REQUIRED_SHA` changes merge method.** `REQUIRED_SHA`
   (currently `3d7fc74`) is a commit on the PR branch that this tooling's
   generation-time and binding-time checks both require as an ancestor. A
   merge commit or a fast-forward keeps that commit in `main`'s history, so
   generation continues to work after the PR lands -- this repo's
   convention is merge-commit merges, and this tooling assumes it. A
   **squash or rebase merge does not**: it replaces `3d7fc74` with a new
   commit that does not contain it, so every generation call refuses
   ("`<repo> HEAD <sha> does not contain 3d7fc74`") until `REQUIRED_SHA`
   and `MERGE_TIME_ISO` in `plan.py` are re-pointed at the new merge
   commit and its landing time. If this repo's merge convention ever
   changes, update `plan.py`'s constants as part of that change, not
   after the first mystery refusal.

Cosmetic note on the refusal text: the strict-equality check's message
(`provenance.check_binding_provenance`) always reads "regenerate against
the server's actual tree", worded for the case where the notebook is
stale. When the mismatch is the other way around -- the server is running
an older or different tree than the one `--repo` pointed at when the
notebook was generated -- the right action is to **restart the server**
from the tree `--repo` names, not to regenerate again; regenerating just
produces a second notebook the same server still won't satisfy.

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

## Stage 2 additions (decision note `outputs/e1-stage2-decision-2026-09-15.md`)

Stage 1 flew one board (B4) once. Stage 2 flies three boards, 18 cycles
each (6 repetitions of shapes `a`/`b`/`c`), so this package gained the
identity and preparation machinery Stage 1 never needed. Nothing below
changes a guard, margin, tolerance, route, or stop rule -- it is
generator/shell tooling, same as everything above.

### Board parameterization

`plan.BOARDS` (`B4`/`B1`/`B2` -> `scenes/e1_boards/<file>.yaml`) is the one
place a board id maps to a scene path. `make_cycle_notebook.generate`
(legacy, defaults to `board="B4"`), `generate_repetition`, and
`generate_stage0` all resolve the scene through `plan.board_scene_rel`,
freeze the result into `plan_<identity>.json` as `scene_rel`, and
`leg.sh`/`parked.sh` read it from there (or, for `parked.sh`, resolve it
directly via `--board`) rather than carrying an independent path literal.
`B3`/`B5` (never authorized for Stage 2) and `B6` (the
deliberately-unflyable incident board) are refused by name with a reason,
not silently treated as "unknown" -- including from the CLI: `--board`
does not restrict argparse `choices`, so an unauthorized board reaches
`plan.board_scene_rel`'s reasoned `ValueError` (which `main()` prints as
`REFUSED: ...`) instead of a generic "invalid choice" (PR #124 review,
M4).

For `generate`'s default `board="B4"`, the generated `SCENE` literal is
byte-identical to Stage 1's pre-Stage-2 hard-coded string
(`test_legacy_generate_defaults_to_b4_scene_unchanged`). The rest of
`generate`'s cell 2 is not byte-identical: it now imports `gating` and
delegates `wait_for` to `gating.wait_for` (same poll order and behaviour
as the inline closure it replaced), and `plan_S1a.json` gains `board`/
`scene_rel` keys with no existing value changed. "Byte-identical" applied
only to the `SCENE` string, not the whole notebook (PR #124 review, M4).

### Repetition-aware identity (decision note §6 item 2)

Stage 1's leg names (`setup_a`, `flight_a`, ...) were fixed strings --
fine for one cycle per shape ever generated. Stage 2 flies six
repetitions of each shape per board; reusing those names would reuse
`go_<leg>`/`<leg>_done`/`recorder_<leg>.log` marker names too, so a second
repetition could find the first repetition's leftover `go_setup_a`
already on disk and fire its motion cell without a fresh operator signal.
`plan.stage2_legs(board, shape, rep)` gives every leg a name built from
`plan.cycle_id(board, shape, rep)` (`S2-<board>-<shape>-r<rep>`), so every
marker is unique by construction. The same is true of Stage 0's own armon
leg: `plan.stage0_armon_leg_name(board, session)` builds
`stage0-<board>-<session>-armon` -- before this fix every Stage 0
notebook used the fixed names `go_armon`/`recorder_armon.log`/
`armon_done`, so one board's leftover Stage 0 markers could satisfy a
*different* board's armon cell and fire `turn_on` with no fresh operator
signal and no recorder running (PR #124 review, M1).

`make_cycle_notebook.generate_repetition`/`generate_stage0` refuse,
before writing anything, if this identity's plan file or any of its
legs' markers already exist in the evidence directory (a stale marker
from a crashed prior attempt, not just a plan file, is enough to refuse),
and if the evidence directory has already recorded a *different* session
id for this board (`control/e1_stage2_sessions.json` -- one server
session per board, per decision note §5). Every one of these checks is
read-only and runs before any file is written; the session ledger itself
is recorded last, right before the notebook/plan files, under a file
lock with an atomic (`os.replace`) write and a re-check inside the lock,
so a refused call never reserves a session or dirties the evidence
directory, and two concurrent calls naming different sessions for the
same board can never both win (PR #124 review, M2).

### Policy A: Stage 0 arm-on, and the per-cycle start-variant gate (decision note §4)

`turn_on("r_arm")` is **not** treated as motion-free here, even though
going stiff pins `goal_position` to `present_position` for each joint (see
the already-stiff-arm investigation above) -- the only way to know nothing
moved is to record it and check, not to assert it. `make_cycle_notebook
.generate_stage0(board, session, ...)` generates a small, once-per-board-
session notebook (`armon_cell_source`) that runs `turn_on("r_arm")` +
`require_compliance` gated exactly like a leg (go-marker, recorder,
baseline `cmd_seq`) but calls no route function at all, and is recorded
via the same `leg.sh` chain (its leg name is the identity-scoped
`plan.stage0_armon_leg_name(board, session)`, a key in that notebook's own
`plan_stage0-<board>-<session>.json` -- note the hyphens, not
underscores).

**The exact `leg.sh` invocation for the armon recording** (M5, corrected
by the 2026-09-16 re-review's R1 -- `HOME` was wrong here, see below):

```
E1_PYTHON=<e1venv python> scripts/e1_stage1/leg.sh <repo> <evidence-dir> \
    stage0-<board>-<session> stage0-<board>-<session>-armon \
    RAISE_TO_SIDE INIT end
```

`RAISE_TO_SIDE` there is a clearance-table label only -- nothing in the
armon cell can fly it (`test_armon_cell_never_calls_any_route_function`).
`end` mode is required: it is the only mode in which a failed tail check
writes `control/stop`; in `start` mode the check is merely informational.

**`INIT` is a dedicated Stage 0 initialization acceptance check
(`e1_tail_check.check_init`), not a posture target, and it is distinct
from the post-reset stiff-zero gate below.** The invocation used to name
`HOME` here, on the theory that "did the arm hold still" could piggyback
on an existing posture target. It cannot: `e1_tail_check.check(...,
"HOME")` requires the last sample's `r_gripper` within `GRIPPER_TOL_DEG`
(3°) of `rig_routes.HOME`'s own `r_gripper` target, `OPEN = -45.0°`
(`rig_routes.pose()`: "gripper open unless told otherwise"). But on a
fresh, compliant server the gripper is still sagging toward the
keyframe-sag pose during the recording (decision note §3: ~65 s to
settle; the armon recording is 14 s), so it sits somewhere in
`[-40°, 0°]` throughout -- disjoint from `HOME`'s `[-48°, -42°]` window.
`HOME` in `end` mode was therefore unpassable by construction: every
Stage 1 parked `HOME` tail check on a comparably fresh arm failed on
exactly this criterion (2026-09-16 re-review, R1), which meant the
documented invocation stopped the board session before cycle 1, every
time.

`INIT` allows that sag: Stage 0's `turn_on` pins `goal_position` to
wherever the arm already is (see the already-stiff-arm investigation
above), which does not forbid the arm moving *before* that instant, only
require it settled *after*. `check_init` asks two questions instead of a
posture match: is the recording's final `window_s` (default 3 s, the
same window `PARKED_TAIL_S` reserves for it) still, and does the
recording's own linked evidence pass the shared experiment-acceptance
gate (`scripts/experiment_gate.py`, below -- complete evidence, no
recorded contact, every tracked board object within displacement
tolerance). It requires no first-to-last invariance -- the lead-in sag is
exactly the motion such a requirement would have to forbid, and Stage 0
has no preceding established pose to be invariant relative to; that is
what this step establishes. `INIT` does **not** assert stiff-zero
(`plan.classify_start_variant` applied to `rig_routes.HOME` itself
returns `None`, matching neither variant, by construction -- and
`check_init` looks at stillness and contact evidence, never gripper
angle, so it cannot be conflated with `stiff-zero` either). Stage 0's
`turn_on` only has to prove the arm settled while going stiff; it is
what makes the *following* reset settle to stiff-zero rather than sag to
keyframe-sag (decision note §3), not something that asserts stiff-zero
itself. Whether stiff-zero was actually reached is checked separately,
per cycle, by the parked-recording gate below -- and, since the
2026-09-16 re-review's R2, by a fresh-sample compliance check as well
(below).

Every cycle's *preceding* parked recording is verified with
`start_variant.py` (invoked from the promoted `parked.sh`, below): it
classifies the recording's last sample with `plan.classify_start_variant`
into `"stiff-zero"`, `"keyframe-sag"`, or neither, and embeds the cycle id
(`--cycle`) it is being recorded for into
`control/start_variant_<cycle>.json`. `parked.sh` STOPs unless the
classification is *exactly* `"stiff-zero"` -- under policy A,
`"keyframe-sag"` means Stage 0's `turn_on` did not hold (motors off,
server restarted, or Stage 0 skipped) and must prevent motion the same as
matching neither variant (PR #124 review, M3; this was previously an open
policy question -- the code now enforces policy A's own text literally).

**The generated Stage 2 cycle notebook enforces this gate itself**, not
only `parked.sh` at recording time: the binding cell (once per cycle,
before any leg is eligible) calls `plan.start_variant_gate(CTRL, CYCLE)`,
which requires `control/start_variant_<CYCLE>.json` to exist, parse, be
bound to *this* cycle (not a stale or wrong-cycle file), and independently
re-classify its own recorded `pose` to `stiff-zero` -- the declared
`start_variant` field is never trusted verbatim. Missing, corrupt,
wrong-cycle, or non-stiff-zero (including `keyframe-sag`) evidence folds
into `PREV_OK`, so every leg's `go = wait_for(...) if PREV_OK else
"not_eligible"` short-circuits and no `turn_on`/route call is reachable
that cycle. This gate is Stage 2-only: legacy `generate` and
`generate_stage0` do not carry it (Stage 0 is what *establishes*
stiff-zero for the first cycle; there is no preceding parked recording
for it to check).

**A stiff-zero posture alone does not prove the arm is currently
stiff** (2026-09-16 re-review, R2): `plan.start_variant_gate` reads only
joint angles, and a *compliant* arm reads "zero within 1°" for part of
its settle window too -- it starts at the reset keyframe (every joint at
0) and sags toward keyframe-sag over ~65 s (decision note §3), so early
in that sag it is still within stiff-zero's 1° tolerance, and a
`parked.sh` recording taken in that window would read the same as a
genuinely stiff arm. So the binding cell also runs a fresh-sample
compliance check, right alongside the posture gate above, before any
cycle command:

```python
COMPLIANCE_CHECK = e1_identity.require_compliance(
    ident.run_dir, R.R_JOINTS, compliant=False, timeout_s=COMPLIANCE_TIMEOUT_S)
PREV_OK = PREV_OK and START_VARIANT_OK and COMPLIANCE_CHECK.ok
```

No `min_cmd_seq` -- nothing has been commanded yet this cycle, so this is
a freshness-window read of whatever the server is reporting right now,
not proof tied to a specific command the way the per-leg post-`turn_on`
check is. `require_compliance`'s existing fail-closed contract does the
rest: a missing sample, a non-bool `compliant` field (state stream not
reporting compliance at all), a stale sample (older than
`max_state_age_s`), or a sample that explicitly reports `compliant=True`
for any of the 8 `R_JOINTS` all refuse. Folded into `PREV_OK` alongside
`START_VARIANT_OK`, so neither this cycle's `turn_on` nor its route call
is reachable without both the posture evidence and a fresh stiff reading
agreeing.

### Experiment-acceptance gate (2026-09-16 re-review, R3)

`leg.sh`/`parked.sh` always ran the linker (`scripts/link_e1_flight.py`)
and STOPped on a non-zero exit, but a non-zero linker exit only means the
contact *evidence* is missing or malformed (`contacts_recorded` could not
be computed). A recording whose evidence was complete and said a contact
happened, or that a board moved during the flight, still reached `LEG
ok`/`PARKED ok`: `contacts_recorded=True` is a completeness verdict, not
a no-contact verdict, and nothing read `contacts`/`displacement_m`
themselves anywhere in the chain (the decision note's own gate/stop rows
for this were never enforced).

`scripts/experiment_gate.py` closes that gap -- one shared check, called
by `leg.sh`, `parked.sh`, and `e1_tail_check.check_init` (Stage 0's own
acceptance check) alike, right after the linker succeeds:

* **evidence invalid** (missing sidecar, unparseable, not tied to THIS
  recording by its `log` field, or not the shape a successful linker run
  produces) -- refuse; the gate cannot tell whether the experiment was
  clean.
* **experiment not accepted** -- the evidence is valid and complete, and
  it says either a contact happened (`contacts` non-empty) or some
  tracked board object moved more than 1 mm
  (`experiment_gate.DISPLACEMENT_TOL_M`) between the first and last
  sample -- refuse; the experiment it describes is rejected.
* **accepted** -- evidence valid, no contact, every tracked object's
  displacement finite and within tolerance.

Either kind of refusal is a STOP, unconditionally -- unlike the tail
check, the gate does not go informational in `start` mode: a recorded
contact or a disturbed board is a safety fact about the leg that just
flew, independent of its position in the cycle. `leg.sh`/`parked.sh` both
archive the recording to `recorder_logs/` *before* calling the gate, so a
rejected recording's log and sidecar are preserved for review, not lost
to the STOP -- evidence validity and experiment acceptance are
deliberately kept distinct (see the module's docstring): `evaluate()`
never writes to the sidecar or the log, only reads them.

### `parked.sh` promoted into the package (decision note §6 item 3)

C3 parked recordings used to live only in each evidence directory's own
`control/tools/parked.sh` (hand-copied, PR #117's shape). `parked.sh` is
now versioned here, board-parameterized (`--board`, resolved via
`plan.board_scene_rel`, not a hard-coded path), and runs the
`start_variant.py` check described above after its (informational,
`mode=start`) tail check. Usage:

```
parked.sh <repo> <evidence-dir> <name> <board> <route-label> <cycle>
```

`<route-label>` is the upcoming cycle's first leg route -- the recorder's
planned-vs-realised clearance reference only; nothing is flown regardless
of its value, same as Stage 1's parked recordings. `<cycle>` (PR #124
review, M3) is the upcoming Stage 2 cycle id this recording will
authorize (e.g. `S2-B4-a-r3`) and must match the `CYCLE` variable baked
into that cycle's generated notebook -- it binds the evidence file
(`start_variant_<cycle>.json`) independent of `<name>`, which remains
free-form for the recorder log's own filename. Re-recording for a cycle
that already has `start_variant_<cycle>.json` is refused (no re-fly, no
recovery -- decision note §5).

### CLI summary

```
# Legacy Stage 1 (unchanged):
python -m scripts.e1_stage1.make_cycle_notebook S1a --repo <repo> --evidence-dir <dir>

# Stage 2, one repetition:
python -m scripts.e1_stage1.make_cycle_notebook \
    --board B4 --shape a --rep 3 --session <session-id> \
    --repo <repo> --evidence-dir <dir>

# Stage 2, Stage 0 arm-on setup (once per board/session, before reset #1):
python -m scripts.e1_stage1.make_cycle_notebook \
    --stage0 --board B4 --session <session-id> \
    --repo <repo> --evidence-dir <dir>
```
