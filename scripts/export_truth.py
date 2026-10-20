#!/usr/bin/env python3
"""Export the simulator's object truth as one `truth.v1` JSON document (#186).

    python3 scripts/export_truth.py --out truth_before.json
    python3 scripts/export_truth.py --scene scenes/FWDCenterLabSivaPool.yaml > truth.json

Connects to the native server as a short-lived READ-ONLY client (hello and
heartbeat answers only), takes the first fresh state, joins it with the scene
file's kinds, sizes and grid cells, writes the document and exits.  It cannot
change or advance the simulation.  --scene must be the scene the server runs:
an object in the state that the file does not know is an error.

Exit status: 0 written, 1 export failed (message on stderr), 2 bad arguments.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "native_mujoco"))

from scene_io import SceneLoadError, load_scene          # noqa: E402
import truth_export as te                                # noqa: E402

DEFAULT_URL = "ws://127.0.0.1:8765"
DEFAULT_SCENE = _REPO / "scenes" / "FWDCenterLabSivaPool.yaml"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default=DEFAULT_URL, help=f"native server (default {DEFAULT_URL})")
    ap.add_argument("--scene", type=pathlib.Path, default=DEFAULT_SCENE,
                    help="the scene file the server is running")
    ap.add_argument("--out", type=pathlib.Path, help="write here instead of stdout")
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for a state")
    args = ap.parse_args(argv)
    if not args.url.startswith(("ws://127.0.0.1", "ws://localhost", "ws://[::1]")):
        print("refusing a non-local --url: the sim's ports stay local by default", file=sys.stderr)
        return 2
    try:
        scene = load_scene(args.scene)
        state = asyncio.run(te.fetch_state(args.url, timeout_s=args.timeout))
        doc = te.build_truth(
            state, scene, scene_file=args.scene.name, scene_sha256=te.file_sha256(args.scene),
            exported_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    except (te.TruthError, SceneLoadError, OSError) as exc:
        print(f"truth export failed: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out} (scene {doc['scene_name']!r} revision {doc['scene_revision']}, "
              f"step {doc['sim_step']}, {len(doc['objects'])} objects)", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
