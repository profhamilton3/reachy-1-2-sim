"""Unit tests for tools/goalfix_cmp/provenance.py."""
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../.."))

from tools.goalfix_cmp import provenance as pv  # noqa: E402


def _image(base="sha256:base", layers=("l0", "l1", "fake_reachy_server.py-copy", "l3"),
           pip=None, dpkg=None, pyver="3.14.0"):
    return pv.ImageBuildInfo(
        base_digest=base, layers=layers,
        pip_freeze=pip or {"numpy": "2.3.5"}, dpkg=dpkg or {"libc6": "2.35"},
        python_version=pyver)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


class TestBuildParity:
    def test_matching_images_pass(self):
        a, b = _image(), _image()
        assert pv.check_build_parity(a, b).ok

    def test_base_digest_difference_stops(self):
        a, b = _image(), _image(base="sha256:other")
        r = pv.check_build_parity(a, b)
        assert not r.ok and "base digest" in r.reason

    def test_pip_difference_stops(self):
        a, b = _image(), _image(pip={"numpy": "2.3.6"})
        r = pv.check_build_parity(a, b)
        assert not r.ok and "pip" in r.reason

    def test_dpkg_difference_stops(self):
        a, b = _image(), _image(dpkg={"libc6": "2.36"})
        r = pv.check_build_parity(a, b)
        assert not r.ok and "dpkg" in r.reason

    def test_python_version_difference_stops(self):
        a, b = _image(), _image(pyver="3.12.0")
        r = pv.check_build_parity(a, b)
        assert not r.ok

    def test_divergence_before_pip_layer_stops(self):
        # Diverges at index 0 -- before the fake_reachy_server.py COPY layer.
        a = _image(layers=("l0", "l1", "fake_reachy_server.py-copy", "l3"))
        b = _image(layers=("l0-DIFFERENT", "l1", "fake_reachy_server.py-copy", "l3"))
        r = pv.check_build_parity(a, b)
        assert not r.ok and "diverge" in r.reason

    def test_divergence_at_or_after_marker_layer_passes_layer_check(self):
        a = _image(layers=("l0", "l1", "fake_reachy_server.py-copy", "l3"))
        b = _image(layers=("l0", "l1", "fake_reachy_server.py-copy", "l3-DIFFERENT"))
        assert pv.check_build_parity(a, b).ok


class TestManifestDiff:
    def _manifests(self):
        common = {"/opt/scripts/x.py": HASH_A, "/opt/notebooks/nb.ipynb": HASH_B}
        man_a = dict(common, **{f: HASH_A for f in pv.EXPECTED_DIFF_FILES})
        man_b = dict(common, **{f: HASH_C for f in pv.EXPECTED_DIFF_FILES})
        expected = {f: HASH_C for f in pv.EXPECTED_DIFF_FILES}
        return man_a, man_b, expected

    def test_exact_three_file_diff_passes(self):
        man_a, man_b, expected = self._manifests()
        assert pv.check_image_manifest_diff(man_a, man_b, expected).ok

    def test_swapped_hash_within_the_three_fails(self):
        man_a, man_b, expected = self._manifests()
        expected[pv.EXPECTED_DIFF_FILES[0]] = HASH_B  # doesn't match what B actually has
        r = pv.check_image_manifest_diff(man_a, man_b, expected)
        assert not r.ok

    def test_fourth_file_differing_fails(self):
        man_a, man_b, expected = self._manifests()
        man_b["/opt/scripts/x.py"] = HASH_C  # an unexpected fourth difference
        r = pv.check_image_manifest_diff(man_a, man_b, expected)
        assert not r.ok and "extra" in r.reason

    def test_missing_manifest_never_passes(self):
        r = pv.check_image_manifest_diff({}, {}, {})
        assert not r.ok

    def test_truncated_manifest_missing_expected_file_fails(self):
        man_a, man_b, expected = self._manifests()
        del man_b[pv.EXPECTED_DIFF_FILES[0]]  # truncated sha256sum output
        r = pv.check_image_manifest_diff(man_a, man_b, expected)
        assert not r.ok


class TestArmMap:
    def _valid_entries(self):
        order = pv.ARM_MAP_ORDER
        return [pv.ArmMapEntry(rep=i + 1, arm=order[i], image_tag=f"img-{i+1}",
                                opt_hashes={"a": HASH_A}) for i in range(12)]

    def test_valid_arm_map_passes(self):
        assert pv.validate_arm_map(self._valid_entries()).ok

    def test_seven_a_reps_rejected(self):
        entries = self._valid_entries()
        entries[1] = pv.ArmMapEntry(rep=2, arm="A", image_tag=entries[1].image_tag,
                                     opt_hashes=entries[1].opt_hashes)
        r = pv.validate_arm_map(entries)
        assert not r.ok

    def test_duplicate_rep_rejected(self):
        entries = self._valid_entries()
        entries[1] = pv.ArmMapEntry(rep=1, arm=entries[1].arm, image_tag="dup",
                                     opt_hashes=entries[1].opt_hashes)
        r = pv.validate_arm_map(entries)
        assert not r.ok

    def test_bad_hex_rejected(self):
        entries = self._valid_entries()
        entries[0] = pv.ArmMapEntry(rep=1, arm=entries[0].arm, image_tag=entries[0].image_tag,
                                     opt_hashes={"a": "not-a-hash"})
        r = pv.validate_arm_map(entries)
        assert not r.ok

    def test_non_distinct_tags_rejected(self):
        entries = self._valid_entries()
        entries[1] = pv.ArmMapEntry(rep=2, arm=entries[1].arm, image_tag=entries[0].image_tag,
                                     opt_hashes=entries[1].opt_hashes)
        r = pv.validate_arm_map(entries)
        assert not r.ok


class TestCheckCycle:
    def _entry(self, rep=1, arm="A"):
        return pv.ArmMapEntry(rep=rep, arm=arm, image_tag="img-1", opt_hashes={"f": HASH_A})

    def test_matching_cycle_passes(self):
        entry = self._entry()
        arm_map = {1: entry}
        observed = pv.ObservedCycle(
            running_image_id="img-1", opt_hashes={"f": HASH_A},
            supervisor_start_times={"bridge": 100.0}, recreate_timestamp=50.0,
            host_git_sha="abc", host_tree_dirty=False)
        assert pv.check_cycle(1, arm_map, observed).ok

    def test_version_mismatch_stops(self):
        arm_map = {1: self._entry()}
        observed = pv.ObservedCycle(
            running_image_id="img-WRONG", opt_hashes={"f": HASH_A},
            supervisor_start_times={"bridge": 100.0}, recreate_timestamp=50.0,
            host_git_sha="abc", host_tree_dirty=False)
        assert not pv.check_cycle(1, arm_map, observed).ok

    def test_swapped_opt_hash_stops(self):
        arm_map = {1: self._entry()}
        observed = pv.ObservedCycle(
            running_image_id="img-1", opt_hashes={"f": HASH_B},
            supervisor_start_times={"bridge": 100.0}, recreate_timestamp=50.0,
            host_git_sha="abc", host_tree_dirty=False)
        assert not pv.check_cycle(1, arm_map, observed).ok

    def test_stale_container_stops(self):
        arm_map = {1: self._entry()}
        observed = pv.ObservedCycle(
            running_image_id="img-1", opt_hashes={"f": HASH_A},
            supervisor_start_times={"bridge": 10.0}, recreate_timestamp=50.0,
            host_git_sha="abc", host_tree_dirty=False)
        assert not pv.check_cycle(1, arm_map, observed).ok

    def test_dirty_tree_stops(self):
        arm_map = {1: self._entry()}
        observed = pv.ObservedCycle(
            running_image_id="img-1", opt_hashes={"f": HASH_A},
            supervisor_start_times={"bridge": 100.0}, recreate_timestamp=50.0,
            host_git_sha="abc", host_tree_dirty=True)
        assert not pv.check_cycle(1, arm_map, observed).ok
