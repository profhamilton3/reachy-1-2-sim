"""Unit tests for tools/goalfix_cmp/provenance.py (T6)."""
import json
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../.."))

from tools.goalfix_cmp import provenance as pv  # noqa: E402
from tools.goalfix_cmp._io import RC_INCONCLUSIVE, RC_OK, RC_STOP, read_result  # noqa: E402

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
BRIDGE_SHA_A = "8c0dad2d62dde791c8912fa723b8c2b7f190e1e8"
BRIDGE_SHA_B = "67730a1ecf646544825fb60c00129cf46de6d307"


def _image(base="sha256:base", layers=("sha256:l0", "sha256:l1", "sha256:l2", "sha256:l3"),
           created_by=("FROM ros:foxy", "RUN pip install -r requirements.txt",
                       "COPY fake_reachy_server.py /opt/", "COPY reset_watcher.py /opt/"),
           pip=None, dpkg=None, pyver="Python 3.10.12"):
    return pv.ImageBuildInfo(
        base_digest=base, layers=layers, created_by=created_by,
        pip_freeze=pip or {"numpy": "2.3.5"}, dpkg=dpkg or {"libc6": "2.35"},
        python_version=pyver)


class TestBuildParity:
    def test_matching_images_pass(self):
        assert pv.check_build_parity(_image(), _image()).ok

    def test_base_digest_difference_stops(self):
        r = pv.check_build_parity(_image(), _image(base="sha256:other"))
        assert not r.ok and "base digest" in r.reason

    def test_pip_difference_stops(self):
        r = pv.check_build_parity(_image(), _image(pip={"numpy": "2.3.6"}))
        assert not r.ok and "pip" in r.reason

    def test_dpkg_difference_stops(self):
        r = pv.check_build_parity(_image(), _image(dpkg={"libc6": "2.36"}))
        assert not r.ok and "dpkg" in r.reason

    def test_python_version_difference_stops(self):
        assert not pv.check_build_parity(_image(), _image(pyver="Python 3.12.0")).ok

    def test_divergence_before_copy_step_stops(self):
        a = _image()
        b = _image(layers=("sha256:l0-DIFFERENT", "sha256:l1", "sha256:l2", "sha256:l3"))
        r = pv.check_build_parity(a, b)
        assert not r.ok and "diverge" in r.reason

    def test_divergence_at_or_after_copy_step_passes(self):
        a = _image()
        b = _image(layers=("sha256:l0", "sha256:l1", "sha256:l2", "sha256:l3-DIFFERENT"))
        assert pv.check_build_parity(a, b).ok

    def test_marker_is_searched_in_created_by_not_in_a_digest(self):
        # A layer digest containing the marker text is NOT how the
        # divergence layer is found (T6) -- it must come from created_by.
        a = _image(layers=("sha256:l0", "sha256:fake_reachy_server.py-in-digest",
                            "sha256:l2", "sha256:l3"),
                   created_by=("FROM ros:foxy", "RUN pip install -r requirements.txt",
                               "COPY fake_reachy_server.py /opt/", "COPY reset_watcher.py /opt/"))
        b = _image(layers=("sha256:l0-DIFFERENT", "sha256:fake_reachy_server.py-in-digest",
                            "sha256:l2", "sha256:l3"),
                   created_by=a.created_by)
        # Diverges at index 0, before the real COPY step (index 2) -- must
        # STOP even though a digest string happens to contain the marker.
        r = pv.check_build_parity(a, b)
        assert not r.ok and "diverge" in r.reason


class TestManifestDiff:
    def _manifests(self):
        common = {"/opt/scripts/x.py": HASH_A, "/opt/notebooks/nb.ipynb": HASH_B}
        man_a = dict(common, **{f: HASH_A for f in pv.EXPECTED_DIFF_FILES})
        man_b = dict(common, **{f: HASH_C for f in pv.EXPECTED_DIFF_FILES})
        expected_a = {f: HASH_A for f in pv.EXPECTED_DIFF_FILES}
        expected_b = {f: HASH_C for f in pv.EXPECTED_DIFF_FILES}
        return man_a, man_b, expected_a, expected_b

    def test_exact_three_file_diff_passes(self):
        man_a, man_b, exp_a, exp_b = self._manifests()
        assert pv.check_image_manifest_diff(man_a, man_b, exp_a, exp_b).ok

    def test_wrong_a_hash_fails(self):
        man_a, man_b, exp_a, exp_b = self._manifests()
        man_a[pv.EXPECTED_DIFF_FILES[0]] = HASH_B  # A's own hash no longer matches expected
        r = pv.check_image_manifest_diff(man_a, man_b, exp_a, exp_b)
        assert not r.ok and "A's hash" in r.reason

    def test_wrong_b_hash_fails(self):
        man_a, man_b, exp_a, exp_b = self._manifests()
        man_b[pv.EXPECTED_DIFF_FILES[0]] = HASH_B
        r = pv.check_image_manifest_diff(man_a, man_b, exp_a, exp_b)
        assert not r.ok and "B's hash" in r.reason

    def test_fourth_file_differing_fails(self):
        man_a, man_b, exp_a, exp_b = self._manifests()
        man_b["/opt/scripts/x.py"] = HASH_C  # an unexpected fourth difference
        r = pv.check_image_manifest_diff(man_a, man_b, exp_a, exp_b)
        assert not r.ok and "extra" in r.reason

    def test_missing_manifest_never_passes(self):
        assert not pv.check_image_manifest_diff({}, {}, {}, {}).ok

    def test_truncated_manifest_missing_expected_file_fails(self):
        man_a, man_b, exp_a, exp_b = self._manifests()
        del man_b[pv.EXPECTED_DIFF_FILES[0]]  # truncated sha256sum output
        r = pv.check_image_manifest_diff(man_a, man_b, exp_a, exp_b)
        assert not r.ok


class TestParsePipFreeze:
    def test_plain_and_editable_and_at_forms(self):
        text = "\n".join([
            "# comment", "", "numpy==2.3.5",
            "-e git+https://example.com/repo.git@abc#egg=pkg",
            "othersitepkg @ file:///tmp/othersitepkg",
        ])
        parsed = pv.parse_pip_freeze(text)
        assert parsed["numpy"] == "2.3.5"
        assert "-e git+https://example.com/repo.git@abc#egg=pkg" in parsed
        assert "othersitepkg @ file:///tmp/othersitepkg" in parsed

    def test_at_or_dash_e_line_difference_is_detected(self):
        a = pv.parse_pip_freeze("-e git+https://example.com/repo.git@abc#egg=pkg\n")
        b = pv.parse_pip_freeze("-e git+https://example.com/repo.git@def#egg=pkg\n")
        assert a != b


class TestDockerHistoryParsing:
    def test_reversed_to_oldest_first_with_created_by(self):
        # docker history prints newest first.
        text = "\n".join([
            "id3\tCOPY reset_watcher.py /opt/",
            "id2\tCOPY fake_reachy_server.py /opt/",
            "id1\tRUN pip install -r requirements.txt",
            "id0\tFROM ros:foxy",
        ])
        ids, created_by = pv.parse_docker_history(text)
        assert ids == ("id0", "id1", "id2", "id3")
        assert created_by[2] == "COPY fake_reachy_server.py /opt/"

    def test_malformed_line_raises(self):
        with pytest.raises(pv.ProvenanceCliError):
            pv.parse_docker_history("no-tab-here")


class TestArmMap:
    def _entry(self, rep, arm, **overrides):
        base = dict(
            rep=rep, arm=arm,
            image_tag="reachy-1-2-sim:cmp-A-8c0dad2" if arm == "A" else "reachy-1-2-sim:cmp-B-67730a1",
            image_id="sha256:imgA" if arm == "A" else "sha256:imgB",
            bridge_sha=BRIDGE_SHA_A if arm == "A" else BRIDGE_SHA_B,
            opt_hashes={f: (HASH_A if arm == "A" else HASH_B) for f in pv.EXPECTED_DIFF_FILES},
        )
        base.update(overrides)
        return pv.ArmMapEntry(**base)

    def _real_map(self):
        """The plan's real arm map: 2 image tags, 6 reps each (§3, ABBA
        BAAB ABBA) -- the exact shape review §3.6/E5 says was rejected."""
        return [self._entry(i + 1, pv.ARM_MAP_ORDER[i]) for i in range(12)]

    def test_real_two_image_arm_map_passes(self):
        r = pv.validate_arm_map(self._real_map())
        assert r.ok, r.reason

    def test_pinned_bridge_sha_checked(self):
        r = pv.validate_arm_map(
            self._real_map(),
            expected_bridge_sha={"A": BRIDGE_SHA_A, "B": BRIDGE_SHA_B})
        assert r.ok, r.reason

    def test_wrong_bridge_sha_rejected(self):
        r = pv.validate_arm_map(
            self._real_map(), expected_bridge_sha={"A": "deadbeef" * 5, "B": BRIDGE_SHA_B})
        assert not r.ok

    def test_swapped_hashes_between_arms_rejected(self):
        entries = self._real_map()
        # Rep 1 (A) gets B's hash set -- A no longer shares ITS arm's set.
        entries[0] = self._entry(1, "A", opt_hashes={
            f: HASH_B for f in pv.EXPECTED_DIFF_FILES})
        r = pv.validate_arm_map(entries)
        assert not r.ok

    def test_mixed_a_tag_rejected(self):
        entries = self._real_map()
        entries[0] = self._entry(1, "A", image_tag="reachy-1-2-sim:cmp-A-DIFFERENT")
        r = pv.validate_arm_map(entries)
        assert not r.ok

    def test_missing_hash_key_rejected(self):
        entries = self._real_map()
        bad_hashes = {f: HASH_A for f in pv.EXPECTED_DIFF_FILES}
        del bad_hashes[pv.EXPECTED_DIFF_FILES[0]]
        entries[0] = self._entry(1, "A", opt_hashes=bad_hashes)
        r = pv.validate_arm_map(entries)
        assert not r.ok

    def test_seven_a_reps_rejected(self):
        entries = self._real_map()
        entries[1] = self._entry(2, "A")
        r = pv.validate_arm_map(entries)
        assert not r.ok

    def test_duplicate_rep_rejected(self):
        entries = self._real_map()
        entries[1] = self._entry(1, entries[1].arm)
        r = pv.validate_arm_map(entries)
        assert not r.ok

    def test_bad_hex_rejected(self):
        entries = self._real_map()
        entries[0] = self._entry(1, "A", opt_hashes={f: "not-a-hash" for f in pv.EXPECTED_DIFF_FILES})
        r = pv.validate_arm_map(entries)
        assert not r.ok

    def test_a_and_b_must_differ(self):
        entries = self._real_map()
        # Make every field of every B entry identical to A's.
        entries = [self._entry(i + 1, pv.ARM_MAP_ORDER[i], image_tag="same", image_id="sha256:same",
                                bridge_sha=BRIDGE_SHA_A,
                                opt_hashes={f: HASH_A for f in pv.EXPECTED_DIFF_FILES})
                   for i in range(12)]
        r = pv.validate_arm_map(entries)
        assert not r.ok


class TestCheckCycle:
    def _arm_map_and_entry(self, rep=1, arm="A"):
        entry = pv.ArmMapEntry(
            rep=rep, arm=arm, image_tag="img-1", image_id="sha256:imgA",
            bridge_sha=BRIDGE_SHA_A, opt_hashes={f: HASH_A for f in pv.EXPECTED_DIFF_FILES})
        return {rep: entry}, entry

    def _observed(self, **overrides):
        base = dict(
            running_image_id="sha256:imgA",
            opt_hashes={f: HASH_A for f in pv.EXPECTED_DIFF_FILES},
            supervisor_start_times={"bridge": 100.0}, recreate_timestamp=50.0,
            host_git_sha="M-sha", host_tree_dirty=False)
        base.update(overrides)
        return pv.ObservedCycle(**base)

    def _check(self, arm_map, observed, **kw):
        kw.setdefault("expected_host_sha", "M-sha")
        kw.setdefault("required_supervisor_programs", ["bridge"])
        return pv.check_cycle(1, arm_map, observed, **kw)

    def test_matching_cycle_passes(self):
        arm_map, _ = self._arm_map_and_entry()
        r = self._check(arm_map, self._observed())
        assert r.ok, r.reason

    def test_image_id_mismatch_stops(self):
        arm_map, _ = self._arm_map_and_entry()
        r = self._check(arm_map, self._observed(running_image_id="sha256:WRONG"))
        assert not r.ok

    def test_swapped_opt_hash_stops(self):
        arm_map, _ = self._arm_map_and_entry()
        r = self._check(arm_map, self._observed(
            opt_hashes={f: HASH_B for f in pv.EXPECTED_DIFF_FILES}))
        assert not r.ok

    def test_empty_arm_map_hashes_stops(self):
        entry = pv.ArmMapEntry(rep=1, arm="A", image_tag="img-1", image_id="sha256:imgA",
                                bridge_sha=BRIDGE_SHA_A, opt_hashes={})
        r = self._check({1: entry}, self._observed())
        assert not r.ok

    def test_empty_observed_hashes_stops(self):
        arm_map, _ = self._arm_map_and_entry()
        r = self._check(arm_map, self._observed(opt_hashes={}))
        assert not r.ok

    def test_empty_supervisor_list_stops(self):
        arm_map, _ = self._arm_map_and_entry()
        r = self._check(arm_map, self._observed(), required_supervisor_programs=[])
        assert not r.ok

    def test_stale_container_stops(self):
        arm_map, _ = self._arm_map_and_entry()
        r = self._check(arm_map, self._observed(supervisor_start_times={"bridge": 10.0}))
        assert not r.ok

    def test_dirty_tree_stops(self):
        arm_map, _ = self._arm_map_and_entry()
        r = self._check(arm_map, self._observed(host_tree_dirty=True))
        assert not r.ok

    def test_wrong_host_sha_stops(self):
        arm_map, _ = self._arm_map_and_entry()
        r = self._check(arm_map, self._observed(host_git_sha="not-M"))
        assert not r.ok


class TestCli:
    def _fake_runner(self, mapping):
        def runner(argv):
            key = tuple(argv)
            for pattern, result in mapping.items():
                if all(p in argv for p in pattern):
                    return result
            return (1, "", f"no canned output for {argv!r}")
        return runner

    def test_runner_rc_nonzero_gives_rc3(self, tmp_path):
        runner = self._fake_runner({("docker", "image", "inspect"): (1, "", "boom")})
        out = tmp_path / "out.json"
        rc = pv._cli(["build-parity", "--image-a", "a", "--image-b", "b", "--out", str(out)],
                     runner=runner)
        assert rc == RC_INCONCLUSIVE
        assert not read_result(out)["ok"]

    def test_empty_stdout_gives_rc3(self, tmp_path):
        runner = self._fake_runner({("docker", "image", "inspect"): (0, "", "")})
        out = tmp_path / "out.json"
        rc = pv._cli(["build-parity", "--image-a", "a", "--image-b", "b", "--out", str(out)],
                     runner=runner)
        assert rc == RC_INCONCLUSIVE

    def test_truncated_git_show_output_gives_rc3(self, tmp_path):
        def runner(argv):
            if "find" in argv:
                return (0, f"{HASH_A}  {pv.EXPECTED_DIFF_FILES[0]}\n", "")
            if argv[:2] == ["git", "show"]:
                return (1, "", "fatal: path does not exist -- truncated/missing expected hash")
            return (1, "", "unhandled")
        out = tmp_path / "out.json"
        rc = pv._cli(["manifest", "--container-a", "ca", "--container-b", "cb",
                      "--sha-a", "shaA", "--sha-b", "shaB", "--out", str(out)],
                     runner=runner)
        assert rc == RC_INCONCLUSIVE

    def test_arm_map_subcommand_from_file(self, tmp_path):
        entries = [
            {"rep": i + 1, "arm": pv.ARM_MAP_ORDER[i],
             "image_tag": "img-A" if pv.ARM_MAP_ORDER[i] == "A" else "img-B",
             "image_id": "sha256:A" if pv.ARM_MAP_ORDER[i] == "A" else "sha256:B",
             "bridge_sha": BRIDGE_SHA_A if pv.ARM_MAP_ORDER[i] == "A" else BRIDGE_SHA_B,
             "opt_hashes": {f: (HASH_A if pv.ARM_MAP_ORDER[i] == "A" else HASH_B)
                            for f in pv.EXPECTED_DIFF_FILES}}
            for i in range(12)
        ]
        arm_map_path = tmp_path / "arm_map.json"
        arm_map_path.write_text(json.dumps(entries))
        out = tmp_path / "out.json"
        rc = pv._cli(["arm-map", "--arm-map", str(arm_map_path), "--out", str(out)])
        assert rc == RC_OK, read_result(out)


class TestSupervisorctlUptimeParsing:
    def test_plain_hms(self):
        assert pv._parse_uptime_to_seconds("1:02:03") == 3723

    def test_with_days(self):
        assert pv._parse_uptime_to_seconds("2 days, 0:00:10") == 2 * 86400 + 10

    def test_status_line_gives_a_past_start_time(self):
        now = 1_000_000.0
        text = "bridge                           RUNNING   pid 123, uptime 0:00:30"
        times = pv._parse_supervisorctl_status(text, now=now)
        assert times["bridge"] == now - 30
