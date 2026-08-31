from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/scripts/llama_formula_sft.py").read_text(encoding="utf-8")


class LlamaFormulaSoftPrefixStaticTest(unittest.TestCase):
    def test_projector_is_optional_and_uses_inputs_embeds(self):
        self.assertIn('parser.add_argument("--soft-prefix-length"', SOURCE)
        self.assertIn("inputs_embeds=inputs_embeds", SOURCE)
        self.assertIn("prefix_labels", SOURCE)
        self.assertIn("soft_prefix_projector.pt", SOURCE)

    def test_projector_parameters_join_optimizer_and_clipping(self):
        self.assertIn("trainable_parameters.extend(projector.parameters())", SOURCE)
        self.assertIn("clip_grad_norm_(trainable_parameters", SOURCE)

    def test_expander_prompt_is_built_from_plan_state(self):
        self.assertIn("build_expander_prompt(tokenizer, expander_plan)", SOURCE)

    def test_training_manifest_records_seed(self):
        self.assertIn('"seed": int(args.seed)', SOURCE)


if __name__ == "__main__":
    unittest.main()
