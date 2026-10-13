# reachy12_scene_vision_toolkit: historical record

**Status: HISTORICAL. Reference material only. Nothing in `reachy12_scene_vision_toolkit/` is a live instruction, contract, schema, scene or compose file for this repository.**

The directory beside this file holds the original planning bundle that `reachy-1-2-sim` was built from. It is kept byte-for-byte as the record of the plan and of the reasoning behind the architecture. The live, maintained versions are the repository's own top-level files (see "Superseded material" below).

## Provenance

| item | value |
|---|---|
| source repository | `profhamilton3/reachy-tabletop-ai` (local checkout `IITG-Reachy-Project`) |
| source path | `reachy12_scene_vision_toolkit/` |
| source commit | `0a9741869a08827b2d8ec3559fe0469f3636aebd` (2026-08-27, "docs: commit toolkit and roadmaps, and correct what this work superseded"; tabletop PR #10, merge commit `c428f22653…`) |
| source tree | `fd6655236f0832c614a9291455fa40523d3b51ba`: identical at tabletop `main` `24f7ff19648163ea047b96cd3ebbb376742aba2b` when the transfer was prepared, with no change since `0a97418` |
| toolkit's own metadata | `MANIFEST.json`: `toolkit_version` `0.1-draft`, created 2026-08-11 |
| baseline the toolkit reviewed | `reachy-1-2-sim.zip` (sha256 `af64f429…`), commit `a0f02d745ce6c90e0d6e947dcdb50de82cc8e0b9` |
| transfer | Reachy-Lab cleanup Stage C, prepared 2026-10-01 (migration record `catalog/migrations/20261002T005149Z-stageC-toolkit-transfer-33e2d0cc` under `~/Reachy-Lab`) |

## Transfer inventory (byte-identical to the source commit)

All 14 files have mode `100644`. The git blob IDs equal the source commit's blob IDs, so the bytes are identical.

| file | bytes | sha256 | git blob |
|---|---|---|---|
| `AGENT_PROMPTS.md` | 5885 | `ae4d8a61b07df2ccac1c7b36b93ba2434ff59412249f82495e16876a69983d5d` | `0447b457` |
| `CLAUDE.md` | 10826 | `453d5e30eeb8fd868e43850dd1a56684c20685107f85de23332e87c822d9682c` | `f8b5bfb0` |
| `FEASIBILITY_STUDY.md` | 20574 | `55c2ed7342dc0456578e92518529f0a50b3f196365333a7ee24d6ec559240ffb` | `8de35102` |
| `MANIFEST.json` | 2177 | `ade5baf42f7db4d79b976060ae324ad85bbf26d82e8a498e9631421d9b7d3c6b` | `73861138` |
| `README.md` | 4938 | `2eef91ff09356e32dc95589c9a32b2eb1bda0f9b97b5618e5d625011d797cc49` | `68b1c528` |
| `ROADMAP.md` | 11687 | `003399a901f684f41a2373906d73b1f19067a779a52d00e53225e570827682b4` | `289f2c55` |
| `compose/docker-compose.mujoco-hybrid.example.yml` | 1661 | `5cc6a46c970786379bb1431df2623338dd0a217dce72fbf1da425a8ce63b50c3` | `ed22808c` |
| `contracts/backend_protocol.py` | 3559 | `e76a4d06d2eebed1a8978cb28b5c31c23a7e8cb5797e7a8a0604b69e07c420d5` | `aab0bc06` |
| `contracts/remote_protocol.md` | 3626 | `5d2e362e8856c095f50c3c7fdb66a59e7462b6ebe6829960029aed85a18a2edb` | `46b7d23e` |
| `docs/ACCEPTANCE_TEST_PLAN.md` | 5599 | `ef0b14552bedee3e11b77ebb244dea6edb3e680634d6b7f6bfcb95714c0c3d48` | `173aa225` |
| `docs/RESEARCH_LINKS.md` | 3684 | `a3feca7b1756fbdea51fbf0a821dd4adc1880099e13caa53927e4cd05674dc47` | `f606b845` |
| `docs/adr/0001-hybrid-native-mujoco.md` | 4500 | `c42d4446e81dec06c2f5610c518201d62b34a4c70050bacdddbaa3b55edeed3a` | `6c89bdfb` |
| `scenes/scene.schema.json` | 10635 | `e2ebb4a9ec3b49dd6b56156e6ef4a11038396ebe73c63219dd8165177273fb10` | `3c0a9fe5` |
| `scenes/tabletop.example.yaml` | 2895 | `001c857ca5f8f4dd50a58f521c3eb870eaa3611380d1a135b122b5c30d4c0610` | `e96efb95` |

## Known exception: README.md does not match MANIFEST.json (pre-existing, preserved)

**`MANIFEST.json` lists 13 files; 12 match.** `README.md` does not match:

| | sha256 | bytes |
|---|---|---|
| manifest records | `4406d219a19af14f2392e823a5967259efaa8fae7f77c96f0ffa86c05837ca7a` | 3218 |
| file | `2eef91ff09356e32dc95589c9a32b2eb1bda0f9b97b5618e5d625011d797cc49` | 4938 |

**Cause (verified):** the README's status banner, its "Status: planning bundle, largely implemented" block, was added on 2026-08-27 after the manifest was written on 2026-08-11. The manifest was not regenerated.

**The check:** removing the banner (the leading `>` lines and the blank lines after the title) reproduces exactly the README the manifest records: 3218 bytes, sha256 `4406d219…`. The manifest does not list itself.

**Both files are transferred unchanged.** Do not "fix" either file: the mismatch is part of the historical record. Recorded as exception X2 in the Reachy-Lab cleanup.

## Superseded material (state at `main` `dbc878cb1429d240b17edb8dc5b04a5cbc459be6`)

The toolkit README's banner (dated 2026-08-27) is the toolkit's own account. The table below is the byte comparison made when the transfer was prepared.

| toolkit file | live file in this repository | relation |
|---|---|---|
| `CLAUDE.md` | `CLAUDE.md` | byte-identical (added in `12490ae`, 2026-08-11); edit the live one |
| `docs/ACCEPTANCE_TEST_PLAN.md` | `docs/ACCEPTANCE_TEST_PLAN.md` | byte-identical (`12490ae`) |
| `docs/adr/0001-hybrid-native-mujoco.md` | `docs/adr/0001-hybrid-native-mujoco.md` | byte-identical (`12490ae`) |
| `FEASIBILITY_STUDY.md` | `docs/FEASIBILITY_STUDY.md` | byte-identical (`12490ae`). **The toolkit README's claim that it has "no live counterpart" is inaccurate.** |
| `contracts/remote_protocol.md` | `docs/remote_protocol.md` | byte-identical (`12490ae`) |
| `contracts/backend_protocol.py` | `docs/backend_protocol.py` | byte-identical (`12490ae`) |
| `ROADMAP.md` | `docs/ROADMAP.md` | the live roadmap has diverged (last changed `8465aa2`, 2026-09-26) |
| `scenes/scene.schema.json` | `scenes/scene.schema.json` | **drifted: do not reuse the draft.** The live schema adds `articulation`, `interactive` and `collision: "fixture"` (last changed `0784e17`) |
| `scenes/tabletop.example.yaml` | `scenes/tabletop.example.yaml`; measured lab scene `scenes/FWDCenterLabMCC.yaml` | the live copy differs (last changed `7e1b5f6`) |
| `docs/RESEARCH_LINKS.md` | `docs/RESEARCH_LINKS.md` | complementary, not a duplicate: the live file records what was verified |
| `compose/docker-compose.mujoco-hybrid.example.yml` | `compose/docker-compose.mujoco-hybrid.example.yml` | the live copy differs (last changed `01700ec`) |
| `AGENT_PROMPTS.md`, `MANIFEST.json`, `README.md` | none | exist only here |

## Historical instructions are not live instructions

`CLAUDE.md` and `AGENT_PROMPTS.md` in the bundle are **historical drafts of agent instructions**, kept as reference material. They do not govern work in this repository: the root `CLAUDE.md` does. Today the bundle's `CLAUDE.md` happens to be identical to the root one, but it will not follow later edits to the live file.

**Coding agents** that auto-load a nested `CLAUDE.md` when reading files under `reachy12_scene_vision_toolkit/` must treat it as a quoted historical document.

**The draft schema, protocol, compose file and example scene** are not loaded by any code. `scene_loader.py` reads `scenes/scene.schema.json`, not this copy. They must not be promoted to live locations.

## Pairing and merge order

**The pair:**
- **This addition** (sim PR) lands first.
- **The tabletop removal** (a tabletop-ai PR that deletes `reachy12_scene_vision_toolkit/` and leaves `reachy12_scene_vision_toolkit/RELOCATED.md`) merges only after this addition is on `main` and its 14 blobs are verified against the inventory above.

**Experiment pin (held):**
- **Merging this PR moves `main` past M′** `dbc878cb…`.
- **The frozen comparison wrapper's preflight requires** `main == origin/main == M′`.
- **This PR is therefore held until after the next four-cycle attempt,** unless the owner rules otherwise.
