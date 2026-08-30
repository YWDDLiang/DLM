from pathlib import Path
import importlib.util
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "finalize_c3fd_native_sft_canary_offline.py"
SPEC = importlib.util.spec_from_file_location("finalize_native_canary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class FinalizeC3FDNativeSFTCanaryOfflineTest(unittest.TestCase):
    def test_attempt_index_supports_raw_and_refined_ids(self):
        self.assertEqual(MODULE.attempt_index({"attempt_id": "h1a2-raw-s17-0042"}), 42)
        self.assertEqual(
            MODULE.attempt_index({"attempt_id": "h1a2-ground-r1717-control-0255"}),
            255,
        )

    def test_scope_indices_require_fixed_balanced_ledger(self):
        ledger = [
            {"sample_idx": idx, "source_split": "train" if idx < 128 else "val"}
            for idx in range(256)
        ]
        scopes = MODULE.scope_indices(ledger)
        self.assertEqual(len(scopes["all"]), 256)
        self.assertEqual(len(scopes["train"]), 128)
        self.assertEqual(len(scopes["val"]), 128)

    def test_delta_and_average_maps_are_intersection_only(self):
        left = {0: 1.0, 1: 2.0}
        right = {0: 0.5, 2: 9.0}
        self.assertEqual(MODULE.delta_map(left, right), {0: -0.5})
        self.assertEqual(MODULE.average_maps(({0: 1.0, 1: 2.0}, {0: 3.0})), {0: 2.0})


if __name__ == "__main__":
    unittest.main()
