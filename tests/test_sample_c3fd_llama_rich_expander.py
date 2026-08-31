from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "src/scripts/sample_c3fd_llama_rich_expander.py"
).read_text(encoding="utf-8")


class SampleC3FDLlamaRichExpanderStaticTest(unittest.TestCase):
    def test_routes_share_manual_autoregressive_sampler(self):
        self.assertIn("def generate_suffix(", SOURCE)
        self.assertIn("past_key_values=past", SOURCE)
        self.assertIn("ROUTE_FORMULA", SOURCE)
        self.assertIn("ROUTE_SOFT_PREFIX", SOURCE)

    def test_M_loads_projector_and_F_rejects_features(self):
        self.assertIn("soft_prefix_projector_config.json", SOURCE)
        self.assertIn("F sampling must not consume M feature vectors", SOURCE)

    def test_attempts_are_retained_without_retry(self):
        self.assertIn('"retry_replacement_rerank": False', SOURCE)
        self.assertIn("all attempts are retained", SOURCE)
        self.assertIn("assemble_expanded_plan(plan, suffix)", SOURCE)


if __name__ == "__main__":
    unittest.main()
