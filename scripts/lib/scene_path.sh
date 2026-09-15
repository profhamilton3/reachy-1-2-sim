# Scene-path resolution shared by scripts/start_sim.sh and its test
# (tests/unit/test_start_sim_scene_path.sh). Pure: no docker, no lsof, no
# side effects beyond the `[ -f ... ]` existence check -- so it is safe to
# source and call from a shell test with no live server.
#
# resolve_scene_path <repo> <scene_in>
#   Sets SCENE (host path) and SCENE_FILE (container path, under
#   /opt/scenes/) in the caller's shell. Exits 1 (after printing to
#   stderr) if the resolved SCENE does not exist as a file.
resolve_scene_path() {
    local repo="$1"
    local scene_in="$2"
    case "$scene_in" in
        /*)         SCENE="$scene_in" ;;                              # absolute path
        *.yaml)     SCENE="$repo/scenes/$(basename "$scene_in")" ;;   # bare file
        *)          SCENE="$repo/scenes/${scene_in}.yaml" ;;          # short name
    esac
    if [ ! -f "$SCENE" ]; then
        echo "✖ Scene not found: $SCENE" >&2
        echo "  Available: $(ls "$repo/scenes"/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/\.yaml//' | tr '\n' ' ')" >&2
        return 1
    fi
    # A1 (2026-09-15 matrix readiness): relative to $repo/scenes, not just
    # the basename -- a board under a subdirectory (e.g. e1_boards/B4.yaml)
    # must keep that subdirectory inside the container too, or the
    # container-side scene-marker-publisher looks in the wrong place and
    # crash-loops with no board visible in RViz, even though physics,
    # bridge, /status, identity and recording are all otherwise correct
    # (observed in the E1 pilot).
    SCENE_FILE="/opt/scenes/${SCENE#"$repo/scenes/"}"
}
