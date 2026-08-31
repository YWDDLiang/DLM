import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_c3fd_llama_prospective_sun",
    ROOT / "scripts/finalize_c3fd_llama_prospective_sun.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import prospective SUN finalizer")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ProspectiveSUNFinalizerTest(unittest.TestCase):
    def test_stream_average_uses_common_compositions(self):
        result = MODULE.average_streams([{0: 1.0, 1: 3.0}, {0: 3.0, 2: 9.0}])
        self.assertEqual(result, {0: 2.0})

    def test_bootstrap_is_deterministic(self):
        first = MODULE.bootstrap({0: -1.0, 1: 1.0}, "x")
        second = MODULE.bootstrap({0: -1.0, 1: 1.0}, "x")
        self.assertEqual(first, second)

    def test_dlm_raw_endpoint_is_explicit(self):
        source = (
            ROOT / "scripts/finalize_c3fd_llama_prospective_sun.py"
        ).read_text(encoding="utf-8")
        self.assertIn('(\"raw\", \"chgnet_energy_per_atom\", False)', source)
        self.assertIn("selection_retry_replacement_rerank", source)


if __name__ == "__main__":
    unittest.main()
