#!/usr/bin/env bash
# The exact entry point of the fixed 7-run, U-only, process-separated C2 diagnostic (see
# c2_process_split.py).  Written and dry-run only until the owner authorizes execution; neither the
# diagnostic nor decision D1 is authorized by the existence of this file.
#
#   run_c2_process_split.sh --pin <full 40-hex SHA of the reviewed preparation head> \
#       --out ~/goalfix-cmp-c2-process-split-<YYYY-MM-DD> --venv ~/goalfix-venv --runs 7 [--dry-run]
#
# Optional: --repo <path> (source checkout to clone; default ~/reachy-1-2-sim).
set -euo pipefail

die() { echo "REFUSED: $*" >&2; exit 2; }

PIN=""; OUT=""; VENV=""; RUNS=""; DRY=0; REPO="${HOME}/reachy-1-2-sim"
while [ $# -gt 0 ]; do
  case "$1" in
    --pin)     [ $# -ge 2 ] || die "--pin needs a value";  PIN="$2";  shift 2;;
    --out)     [ $# -ge 2 ] || die "--out needs a value";  OUT="$2";  shift 2;;
    --venv)    [ $# -ge 2 ] || die "--venv needs a value"; VENV="$2"; shift 2;;
    --runs)    [ $# -ge 2 ] || die "--runs needs a value"; RUNS="$2"; shift 2;;
    --repo)    [ $# -ge 2 ] || die "--repo needs a value"; REPO="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    -h|--help) sed -n '2,10p' "$0"; exit 0;;
    *) die "unknown argument: $1";;
  esac
done

[ -n "$PIN" ]  || die "--pin is required"
[ -n "$OUT" ]  || die "--out is required"
[ -n "$VENV" ] || die "--venv is required"
[ -n "$RUNS" ] || die "--runs is required"
OUT="${OUT/#\~/$HOME}"; VENV="${VENV/#\~/$HOME}"; REPO="${REPO/#\~/$HOME}"

[[ "$PIN" =~ ^[0-9a-f]{40}$ ]] || die "--pin must be a full 40-hex-digit SHA"
[ "$RUNS" = "7" ] || die "--runs must be exactly 7 (the scope is fixed in advance)"
case "$OUT" in /*) ;; *) die "--out must be an absolute path";; esac
PARENT="$(cd "$(dirname "$OUT")" 2>/dev/null && pwd -P)" || die "the parent directory of --out does not exist"
case "$PARENT/$(basename "$OUT")" in
  /tmp|/tmp/*|/private/tmp|/private/tmp/*) die "--out must be outside /tmp";;
esac
[ ! -e "$OUT" ] && [ ! -L "$OUT" ] || die "--out already exists"
PY="$VENV/bin/python"
[ -x "$PY" ] || die "--venv has no executable bin/python"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# The driver must never see a caller's PYTHONPATH.
unset PYTHONPATH || true
export PYTHONDONTWRITEBYTECODE=1

# Pins, tree/blob equalities, output path, free space and the exact per-run plan.  Imports no SDK,
# grpc, websockets, native_stub, mujoco_remote_backend or fake_reachy_server; starts only `git`.
# Refuses (exit 2) on any mismatch.
"$PY" "$HERE/c2_process_split.py" preflight --pin "$PIN" --out "$OUT" --repo "$REPO" \
      --runs "$RUNS" --venv "$VENV"

if [ "$DRY" = "1" ]; then
  echo "DRY RUN: nothing was cloned, imported or executed."
  exit 0
fi

# Source: a local CLONE (not a worktree) detached at the pin.
mkdir "$OUT"
git clone --local --no-checkout "$REPO" "$OUT/src"
git -C "$OUT/src" checkout --detach "$PIN"
mkdir "$OUT/tmp"

# Nothing may land in /tmp; nothing may be imported from the primary checkout.
export TMPDIR="$OUT/tmp"
export REQUIRE_REACHY_SDK=1
cd "$OUT/src"
exec env PYTHONPATH="$OUT/src/src:$OUT/src/native_mujoco:$OUT/src/scripts:$OUT/src/tests/integration:$OUT/src/tests/fixtures/goalfix_cmp:$OUT/src" \
    "$PY" "$OUT/src/tests/fixtures/goalfix_cmp/c2_process_split.py" run \
    --pin "$PIN" --out "$OUT" --venv "$VENV"
