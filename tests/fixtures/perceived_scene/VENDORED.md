# Vendored golden fixtures

`valid/`, `invalid/` and `README.md` are copied byte for byte from `profhamilton3/reachy-tabletop-ai`,
`tests/fixtures/perceived_scene/`, at `bc77f70666b8ac3e5e5c2923db4b5204a10710ed` (PR #48). `SHA256SUMS` (made here, at the
copy) pins them; `tests/unit/test_perceived_scene_vendored.py` checks it. Set `TABLETOP_AI_REPO` to a tabletop checkout
to also compare against upstream file by file. The validator they exercise is `native_mujoco/perceived_scene.py`.
