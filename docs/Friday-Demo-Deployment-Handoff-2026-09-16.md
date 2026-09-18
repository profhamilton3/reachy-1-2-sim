# Friday demo deployment handoff — 2026-09-16

Prepared by Claude Code (Sonnet) per assignment
`assignment-2026-09-16-friday-demo-deployment-sonnet.md`. Integration only:
no service launch, image build, motor enable, motion, or merge happened in
this pass.

## 1. Branch, base, and candidate

| Item | Value |
|---|---|
| Base | `origin/main` at `eb34238` (merge of #120 — verified identical to the assignment's recorded fact; no newer main commits existed at fetch time) |
| Deployment delta | `0d09ce8` (`chore/deployment-panel-features`, "Carry the panel's features into the container, and persist what they record") |
| Workspace | New worktree: `/Users/terrancehamilton/reachy-1-2-sim-demo-friday` (approved by user before creation) |
| Branch | `demo/friday-deployment-2026-09-16` |
| **Candidate commit** | **`0d5c580`** — cherry-pick of `0d09ce8` onto `eb34238` |
| PR | See "Deliverable" section below |

The primary checkout (`/Users/terrancehamilton/reachy-1-2-sim`, on
`chore/deployment-panel-features` at `0d09ce8`) was **not modified**. Its
uncommitted state is preserved exactly as found:
- Modified: `notebooks/place_objects.ipynb`
- Untracked: `notebooks/demo_pick_place.py`, `outputs/`

Evidence worktrees (`reachy-1-2-sim-e1`, `reachy-1-2-sim-stage1`,
`reachy-1-2-sim-issue116`) and the Stage 2 worktree
(`reachy-1-2-sim-stage2-tooling`, #124 at `2289ce9`) were not touched. #124
and Stage 2 remain paused, out of scope for this integration.

## 2. Diff summary

Cherry-pick applied cleanly except for one additive conflict in `.gitignore`
(main had independently added a `docs/reviews/probes-*/*.json` ignore rule;
resolved by keeping both blocks). `docker-compose.yml` auto-merged with no
conflict — main's removal of `REACHY_SIM_ALLOW_FIXTURE_FALLBACK=true` (a
newer main behavior, unrelated to this delta) is preserved on the candidate.

```
 .env.example       | 39 +++++++++++++++++++++++++++++++++++++++
 .gitignore         |  4 ++++
 docker-compose.yml | 44 ++++++++++++++++++++++++++++++++++++++++++++
 3 files changed, 87 insertions(+)
```

Confirmed: exactly the three files, 87 lines added, matching `0d09ce8`
verbatim in scope. No secrets, no unrelated hunks. Additions:
- Read-only bind mount `./native_mujoco/model:/opt/native_mujoco/model:ro`
- Persistent bind mount `./runs:/opt/runs`
- Compose pass-through of `REACHY_PANEL_EPISODES(_DB|_KEEP)`,
  `REACHY_PANEL_RECIPES(_DB)`, `REACHY_PANEL_LANGUAGE(_MODEL|_MIN_CONFIDENCE|_TIMEOUT_S)`,
  `ANTHROPIC_API_KEY`
- New `.env.example` (committed, secret-free) and `.env` ignore rule

## 3. Verification performed (offline / config-only)

All checks below were run against the candidate worktree. No service was
started, no image was built, no motion occurred.

**Diff/secret inspection** — `git diff eb34238..0d5c580`: 3 files, 87
insertions, no deletions; no `api_key=`, `sk-ant`, or `password=` literals
found anywhere in the diff.

**Compose config validation** — `docker compose --env-file <temp secret-free
file> config`, resolved and inspected only an allowlisted projection:

| Setting | Resolved |
|---|---|
| `REACHY_SIM_BACKEND` | `mujoco-remote` |
| `REACHY_SIM_SCENE_FILE` | `/opt/scenes/FWDCenterLabSivaPool.yaml` |
| `REACHY_PANEL_EXECUTOR` | `1` |
| `REACHY_PANEL_EPISODES` | `1` |
| `REACHY_PANEL_RECIPES` | `1` |
| `REACHY_PANEL_LANGUAGE` | `0` (test value; see §4 for the real deployment's actual value) |
| Mounts | `./native_mujoco/model → /opt/native_mujoco/model` (ro), `./runs → /opt/runs`, plus existing `./scenes`, `./web` |
| `ANTHROPIC_API_KEY` | present in resolved config (value withheld; test placeholder only) |

The temporary env file used only placeholder values and lived at
`<scratchpad>/demo.env`, never in the repo. No `docker inspect`, no shell
`env` dump, no full expanded config was printed — only the fields above.

**Model mount vs. source** — `web/panel_provenance.py:72` hashes
`/opt/native_mujoco/model/reachy_1_2.xml`. That file exists on the host at
`native_mujoco/model/reachy_1_2.xml`, which the new mount places at exactly
that container path. Identity will no longer be the empty/degenerate hash.

**Database path vs. source** — `web/panel_episodes.py` and
`web/panel_recipes.py` resolve an empty `_DB` env var to
`<repo_root>/runs/panel_episodes.db` and `<repo_root>/runs/promoted_recipes.db`
respectively, where `repo_root` is relative to the module's own location
(i.e., `/opt` in the container). Both resolve under the new `/opt/runs`
mount, so both persist across a container recreate.
**`runs/promoted_recipes.db` does not currently exist anywhere in the repo or
worktrees.** `REACHY_PANEL_RECIPES=1` is safe to leave on for the demo: with
no store present, `build_library` still returns a working (empty) library —
it looks, finds nothing, and every ability flies its normal route. There is
no compatible promoted store to demonstrate retrieval from; only recording
and reuse-safe no-op retrieval are demoable, not an actual promoted-recipe
hit.

**Focused tests — isolated, one file at a time, offline, no network:**

| File | Result |
|---|---|
| `tests/unit/test_camera_server_status.py` | 9 passed |
| `tests/unit/test_panel_episodes.py` | 24 passed |
| `tests/unit/test_panel_recipes.py` | 29 passed |
| `tests/unit/test_panel_language.py` | 66 passed (no network calls; adapter is stubbed) |

Each file inserts its own `sys.path` (this repo has no `conftest.py`, and
these files self-document that a cross-test import would only resolve when
the whole suite runs), so running them individually is the correct isolation
check, not a shortcut. No full-suite rerun was performed — not needed for a
three-file config integration with no implementation-code changes, per
assignment scope.

**Scene-path resolution** — `bash tests/unit/test_start_sim_scene_path.sh`
(pure, no docker/lsof/side effects): all 3 cases pass, including confirming
`REACHY_SIM_SCENE=FWDCenterLabSivaPool` resolves to container path
`/opt/scenes/FWDCenterLabSivaPool.yaml` — the same path set in `.env.example`
and the real `.env` (see §4).

## 4. Configuration — non-secret report

**Real `.env`** (`/Users/terrancehamilton/reachy-1-2-sim/.env`) exists.
Per the assignment, only named non-secret settings and credential *presence*
were read — the file itself was never dumped or printed in full:

| Key | Value |
|---|---|
| `REACHY_SIM_BACKEND` | `mujoco-remote` |
| `REACHY_SIM_SCENE_FILE` | `/opt/scenes/FWDCenterLabSivaPool.yaml` |
| `REACHY_PANEL_EXECUTOR` | `1` |
| `REACHY_PANEL_EPISODES` | `1` |
| `REACHY_PANEL_RECIPES` | `1` |
| `REACHY_PANEL_LANGUAGE` | `1` |
| `ANTHROPIC_API_KEY` | **absent or empty** |

**This is a live demo-relevant fact, not just a config note:** with
`REACHY_PANEL_LANGUAGE=1` and no key, `web/panel_language.py:build_adapter()`
logs a startup warning ("`REACHY_PANEL_LANGUAGE is on but no API key is set;
the panel will use the registry only`") and returns `None` — the panel
silently falls back to the deterministic registry only. This is not a bug;
it is the documented, deliberate fallback. If language interpretation is
wanted for Friday, the operator must add `ANTHROPIC_API_KEY=...` to the real
`.env` themselves — this assignment does not modify that file. If it is not
added, the rehearsal and demo proceed on deterministic commands only, which
fully satisfies the Friday target (language is explicitly optional).

**The real `.env` was not modified, sourced as shell code, or otherwise
touched by this integration.**

**Demo values used for validation** (placeholders only, never the real key,
written to a temp file, not committed):

| Setting | Demo intent | Notes |
|---|---|---|
| `REACHY_SIM_BACKEND` | `mujoco-remote` | Compose var |
| `REACHY_SIM_SCENE` | `FWDCenterLabSivaPool` | **Shell var** consumed only by `scripts/start_sim.sh` — not read by Compose at all |
| `REACHY_SIM_SCENE_FILE` | `/opt/scenes/FWDCenterLabSivaPool.yaml` | Compose var; must match what `REACHY_SIM_SCENE` resolves to, or RViz/noVNC and physics show different scenes |
| `REACHY_PANEL_EXECUTOR` | `1` | Compose var |
| `REACHY_PANEL_EPISODES` | `1` | Compose var |
| `REACHY_PANEL_RECIPES` | `1` | Compose var; see promoted-store note in §3 |
| `REACHY_PANEL_LANGUAGE` | Optional | Compose var; inert without `ANTHROPIC_API_KEY` (see above) |
| `REACHY_SIM_DEPTH` / `REACHY_SIM_SEGMENTATION` | `0` (i.e., unset) | **Shell vars**, consumed only by `start_sim.sh` on the host — not passed through Compose at all. Turning both on simultaneously breaks the container's websocket bridge (>1 MiB frame, silent reconnect loop) — leave both off for the panel demo. |
| `REACHY_SIM_EFFECTS` | unset | Shell var, `start_sim.sh` only; no default profile exists |

### Shell-vs-Compose precedence (the documented "trap" this delta fixes)

- `docker compose` auto-reads `.env` in the working directory for
  `${VAR:-default}` substitution in `docker-compose.yml`. This is why a
  committed `.env.example` + a real `.env` makes a bare `docker compose up
  -d` reproduce the intended deployment instead of silently reverting to
  kinematic/wrong-scene/executor-off defaults.
- `scripts/start_sim.sh` reads `REACHY_SIM_SCENE` **only from the shell
  environment** (`SCENE_IN="${REACHY_SIM_SCENE:-tabletop_demo}"`), not from
  `.env` — Compose never sees this variable. It resolves the scene, then
  invokes `docker compose up -d` with `REACHY_SIM_BACKEND` and
  `REACHY_SIM_SCENE_FILE` **set inline on that command line**, which
  overrides whatever is in `.env` for those two keys specifically (so the
  container's noVNC/RViz scene always matches the native server it just
  started). Every other Compose var (`REACHY_PANEL_*`, `ANTHROPIC_API_KEY`)
  is *not* set inline by the script, so those come from `.env` alone.
- **Practical consequence:** setting `REACHY_SIM_SCENE_FILE` in `.env` alone
  (without also exporting `REACHY_SIM_SCENE` in the shell that runs
  `start_sim.sh`) selects the right scene for a bare `docker compose up -d`
  but does **not** select the host-side native server's scene if
  `start_sim.sh` is used instead — the two paths must agree deliberately.

**Documented startup command that explicitly selects the same host/container
scene** (not executed in this assignment):

```bash
REACHY_SIM_SCENE=FWDCenterLabSivaPool ./scripts/start_sim.sh
```

This is the only supported way to guarantee the native server and the
container render the same scene together; a bare `docker compose up -d`
alone starts no native server and the bridge reconnects forever in DEGRADED
(per the script's own header note, R6).

## 5. Deployment plan (prepared, not executed)

- **Candidate:** commit `0d5c580` on branch `demo/friday-deployment-2026-09-16`.
- **Execution checkout:** a fresh checkout of that commit, or the demo
  worktree itself (`/Users/terrancehamilton/reachy-1-2-sim-demo-friday`) —
  operator's choice; do not build from the primary checkout, which still
  carries uncommitted notebook/output changes.
- **Scene:** `FWDCenterLabSivaPool`.
- **Image tag:** tag the build explicitly, e.g.
  `reachy-1-2-sim:demo-2026-09-16-0d5c580`, or record the resulting image ID
  (`docker images --no-trunc reachy-1-2-sim:latest`) immediately after
  building so the exact candidate is traceable — **do not infer which code an
  existing image contains from its tag or age.**
- **Build:** `docker compose build` from the execution checkout, normal
  layer cache, no `--no-cache`, no dependency upgrade.
- **Consistency requirement:** the native server (host, from
  `native_mujoco/`), the Compose mounts (`native_mujoco/model`, `runs`,
  `scenes`, `web`), and the built image must all draw from this same
  checkout — a stale image with fresh mounts (or vice versa) will mismatch
  baked-in files (`mujoco_remote_backend.py`, `fake_reachy_server.py`) against
  mounted ones.
- **Container name / Compose project:** before replacing anything, identify
  the currently running container's owning Compose project and state
  (`docker compose ls`, `docker ps --filter name=reachy-sim` — read-only
  inspection only). **Do not delete containers or volumes speculatively.**
- **Shutdown:** stop only this demo's own services
  (`docker compose down` from the correct project directory) — do not run
  the handoff's broad port-kill commands unless the specific port is
  confirmed occupied by a leftover process.
- **Known precondition:** `start_sim.sh` unconditionally kills whatever
  listens on `:8765` before starting the native server (`lsof -tiTCP:8765 …
  | xargs kill -9`). Flag this for the operator; it was not invoked during
  this integration.
- **Restart/persistence verification:** during the *later* rehearsal only —
  restart the container and confirm `runs/panel_episodes.db` (and, if
  present, `runs/promoted_recipes.db`) survive and grow, not attempted here.
- **Rollback:** the old primary checkout (`chore/deployment-panel-features`
  @ `0d09ce8`) and any existing image are rollback references only — they are
  not automatically a validated fallback and were not overwritten. Preserve
  both.

## 6. Feature matrix

| Feature | Configured | Tested offline | Rehearsed live |
|---|---|---|---|
| MuJoCo physics backend + scene mount | ✅ | ✅ (scene-path resolution test) | ❌ |
| Model identity mount (`/opt/native_mujoco/model`) | ✅ | ✅ (path cross-checked against `panel_provenance.py`) | ❌ |
| Persistent `runs/` mount | ✅ | ✅ (DB path resolution cross-checked) | ❌ |
| Episode recording (`REACHY_PANEL_EPISODES`) | ✅ (on) | ✅ (24 unit tests) | ❌ |
| Recipe retrieval (`REACHY_PANEL_RECIPES`) | ✅ (on) | ✅ (29 unit tests) — no promoted store exists; retrieval is a safe no-op, not a demoable hit | ❌ |
| Camera status reporting | n/a (pre-existing) | ✅ (9 unit tests) | ❌ |
| Language adapter (`REACHY_PANEL_LANGUAGE`) | ⚠️ flag on in real `.env`, but **no API key present** → inert fallback | ✅ (66 unit tests, stubbed) | ❌, and will remain deterministic-only until operator adds a key |
| Executor (motion enablement) | ✅ (on) | not exercised (no motion in this assignment) | ❌ |
| Live browser-panel command flow | n/a | n/a | ❌ — not attempted; recent B4 route trials are route evidence, not browser-panel proof |

## 7. Rehearsal card (for the next, separately authorized session)

Use **FWDCenterLabSivaPool** exclusively — never substitute `tabletop_demo`
or a kinematic scene for this rehearsal. The B4 evidence board is
tube-refused and is not an approved production-panel motion layout; derive
the actual clear-board layout from FWDCenterLabSivaPool's current geometry
before rehearsing.

Proposed sequence (verify later, not now):
1. Confirm native physics identity and fresh stereo frames; RViz/noVNC
   matches the launched scene.
2. Greeting and text planning/confirmation/cancellation (deterministic
   registry).
3. Scene-object placement and recall — arm stowed, no motion concurrent with
   placement.
4. Rest, wave, and stow through the panel on a deliberately clear board.
5. Point to a cell/object — only after a suitable layout passes the
   unchanged guard. Off-board pool objects must still be checked against the
   arm corridor; off-board is not the same as clear.
6. Episode written with scene/model identity, and confirmed to persist
   through a controlled container restart.
7. (Optional, only if a real `ANTHROPIC_API_KEY` is added) broader language
   phrasing, with an explicit plan-source indicator or other verifiable
   evidence that the adapter (not the deterministic registry) produced the
   plan.

**Expected refusals to keep, not bypass:** the executor's lease check, the
footprint/tube guard on off-board or ambiguous layouts, and the language
adapter's silent fallback with no key. None of these should be worked around
to "make the demo work."

**Acceptance for the rehearsal step:** two consecutive full panel sequences
on the fixed candidate image/commit, with no contact, no unexpected object
displacement, no hanging command, no stale camera feed, and no unplanned
recovery. This is practical demo confidence for Friday, not a statistical
reliability claim. Record any excluded feature (e.g., language, if no key is
added) honestly in the rehearsal notes rather than omitting it.

**Recording:** an existing successful recording should be kept as fallback;
a new screen recording should be captured during the actual rehearsal. **No
screen recording has been made or claimed as part of this assignment.**

**Rollback procedure:** stop the demo's Compose services only
(`docker compose down` from the demo checkout); the primary checkout at
`chore/deployment-panel-features` @ `0d09ce8` and its existing image remain
untouched as the fallback.

## 8. Unresolved blocker and smallest next action

**No integration blocker.** The three-file delta applied cleanly, all
targeted tests pass, and mount/DB paths cross-check against source. The one
open item is operator-owned, not code-owned:

- **Language interpretation is configured on but inert** (no
  `ANTHROPIC_API_KEY` in the real `.env`). Smallest next action if language
  is wanted for Friday: operator adds the key to their own `.env` (not this
  repo's `.env.example`, and not done by this assignment). If skipped, no
  code change is needed — deterministic commands fully cover the Friday
  target.
- **No compatible promoted-recipe store exists yet.** This is expected at
  this stage (nothing has been promoted); it is not a defect and needs no
  repair before Friday. Retrieval will demonstrably run and find nothing,
  which is correct behavior, not a demo blocker.

Next steps (separately authorized): build the image from this candidate,
tag it, run the rehearsal sequence above, and record the second screen
capture.
