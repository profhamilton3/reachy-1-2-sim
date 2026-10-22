"""python -m tools.route_clearance --routes <arm_routes.json> --route B2_GRIP_TUNE --scene FWDCenterLabSivaPool
   --out <dir> [--cells r1c3,r2c2] [--kinematic-only] [--declare POSE:GEOM/GEOM[@DEG] ...] [--target-spot]

Writes <out>/<route>@<hash12>.json and prints the verdict. Exit status 0 on pass, 1 on fail, 2 on a bad input."""

import argparse
import sys

from . import record
from .route import RouteError


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tools.route_clearance")
    ap.add_argument("--routes", required=True, help="the tabletop route document (arm_routes.v1)")
    ap.add_argument("--route", required=True)
    ap.add_argument("--scene", required=True, help="a scene name under scenes/, e.g. FWDCenterLabSivaPool")
    ap.add_argument("--out", required=True, help="directory for the record")
    ap.add_argument("--cells", help="comma-separated cells for the occupancy map (default: all nine)")
    ap.add_argument("--kinematic-only", action="store_true", help="no physics flights (the verdict fails)")
    ap.add_argument("--declare", action="append", metavar="POSE:PAIR[@DEG]",
                    help="a declared contact, allowed only within DEG of the pose when given; replaces the route's "
                         "default list (design section 3.2)")
    ap.add_argument("--target-spot", action="store_true",
                    help="the target is the supply spot under the close pose's tip, for a route whose close pose is "
                         "over no grid cell; every grid cell is then in the occupancy map")
    args = ap.parse_args(argv)
    declared = None
    if args.declare:
        declared = []
        for item in args.declare:
            pose, sep, rest = item.partition(":")
            pair, at, bound = rest.partition("@")
            if not sep or "/" not in pair:
                ap.error("--declare wants POSE:GEOM/GEOM[@DEG], got %r" % item)
            try:
                declared.append((pose, pair, float(bound) if at else None))
            except ValueError:
                ap.error("--declare: %r is not a number of degrees" % bound)
    cells = args.cells.split(",") if args.cells else None
    try:
        rec = record.run(args.routes, args.route, args.scene, physics=not args.kinematic_only, cells=cells,
                         declared=declared, target_spot=args.target_spot, log=lambda msg: print("..", msg, file=sys.stderr, flush=True))
    except (RouteError, KeyError, FileNotFoundError) as error:
        print("FAIL:", error, file=sys.stderr)
        return 2
    path = record.write(rec, args.out)
    print("%s: %s" % (path, rec["verdict"]))
    for reason in rec["reasons"]:
        print("  -", reason)
    return 0 if rec["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
