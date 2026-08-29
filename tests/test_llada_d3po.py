import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REQUIRED_DEPS = ("torch", "transformers", "peft")
MISSING_DEPS = tuple(
    name for name in REQUIRED_DEPS if importlib.util.find_spec(name) is None
)

torch = None
if not MISSING_DEPS:
    import torch

    from crystal_dlm.fixed_slot import FixedSlotConfig, Z_TO_SYMBOL
    from scripts import llada_d3po as MODULE

_TorchModuleBase = torch.nn.Module if torch is not None else object


class SlurmContractStaticTest(unittest.TestCase):
    def test_wrapper_is_one_gpu_eight_cpu_and_serial_two_seed(self):
        text = (ROOT / "slurm/65_d3po_train.sbatch").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=8", text)
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", text)
        self.assertIn("#SBATCH --time=10:00:00", text)
        self.assertIn("for seed in 81017 81018; do", text)
        self.assertNotIn("torchrun", text)
        self.assertNotRegex(text, re.compile(r"srun[^\n]*&\s*$", re.MULTILINE))

    def test_only_step348_is_referenced_as_scientific_output(self):
        text = (ROOT / "slurm/65_d3po_train.sbatch").read_text(encoding="utf-8")
        self.assertIn("checkpoints/step-348", text)
        self.assertNotIn("step-174", text)

    def test_wrapper_pins_base_and_pair_hashes(self):
        text = (ROOT / "slurm/65_d3po_train.sbatch").read_text(encoding="utf-8")
        self.assertIn("assert_sha256 \"${BASE_MODEL_SHA}\"", text)
        self.assertIn("assert_sha256 \"${PAIR_MANIFEST_SHA}\"", text)
        self.assertIn("assert_sha256 \"${TRAIN_SHA}\"", text)
        self.assertIn("assert_sha256 \"${VALIDATION_SHA}\"", text)


@unittest.skipIf(
    bool(MISSING_DEPS),
    "training dependencies are unavailable: " + ", ".join(MISSING_DEPS),
)
class D3POTrainerTest(unittest.TestCase):
    class FakeTokenizer:
        def __init__(self):
            config = FixedSlotConfig()
            tokens = []
            tokens.extend(f"<N_{value:03d}>" for value in range(1, 21))
            for prefix in ("LA", "LB", "LC"):
                tokens.extend(
                    f"<{prefix}_{value:03d}>"
                    for value in range(config.length_min_bin, config.length_max_bin + 1)
                )
            for prefix in ("AA", "AB", "AG"):
                tokens.extend(
                    f"<{prefix}_{value:03d}>"
                    for value in range(config.angle_min_bin, config.angle_max_bin + 1)
                )
            tokens.extend(
                f"<E_{Z_TO_SYMBOL[value]}>"
                for value in range(1, config.max_atomic_number + 1)
            )
            for prefix in ("X", "Y", "Z"):
                tokens.extend(
                    f"<{prefix}_{value:03d}>"
                    for value in range(config.coord_min_bin, config.coord_max_bin + 1)
                )
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.vocab = {token: index + 100 for index, token in enumerate(tokens)}

        def __len__(self):
            return max(self.vocab.values()) + 1

        def get_vocab(self):
            return dict(self.vocab)

        def convert_tokens_to_ids(self, token):
            return self.vocab.get(token)

        def __call__(self, text, add_special_tokens=False, **_kwargs):
            del add_special_tokens
            crystal_start = text.find("<N_")
            prefix_ids = [10, 11] if crystal_start != 0 else []
            crystal_text = text[crystal_start:] if crystal_start >= 0 else ""
            token_strings = re.findall(r"<[^>]+>", crystal_text)
            return {
                "input_ids": prefix_ids
                + [self.vocab[token] for token in token_strings]
            }

    class FakeAdapterModel(_TorchModuleBase):
        def __init__(self, vocab_size):
            super().__init__()
            self.adapters = torch.nn.ParameterDict(
                {
                    "policy": torch.nn.Parameter(torch.zeros(())),
                    "reference": torch.nn.Parameter(
                        torch.zeros(()), requires_grad=False
                    ),
                }
            )
            self.active_adapter = "policy"
            self.vocab_size = int(vocab_size)
            self.activation_log = []

        def set_adapter(self, adapter_name):
            self.active_adapter = str(adapter_name)
            self.activation_log.append(self.active_adapter)

        def forward(self, input_ids, attention_mask):
            del attention_mask
            value = self.adapters[self.active_adapter] * 0.0
            logits = torch.zeros(
                (*input_ids.shape, self.vocab_size),
                dtype=torch.float32,
                device=input_ids.device,
            ) + value
            return SimpleNamespace(logits=logits)

    def setUp(self):
        self.tokenizer = self.FakeTokenizer()
        self.prompt = (
            '{"N":2,"charge":"certified_neutral","counts":[1,1],'
            '"elements":["Li","O"],"family":"oxide","formula":"LiO"}'
            "\ndynamic_crystal_body:"
        )
        self.winner = (
            "<N_002><LA_010><LB_011><LC_012><AA_090><AB_091><AG_092>"
            "<E_Li><X_000><Y_001><Z_002><E_O><X_050><Y_060><Z_070>"
        )
        self.loser = (
            "<N_002><LA_013><LB_014><LC_015><AA_088><AB_089><AG_090>"
            "<E_Li><X_010><Y_011><Z_012><E_O><X_055><Y_065><Z_075>"
        )

    def row(self, *, pair_id="pair-0", pair_weight=1.0, split="train"):
        gap = 0.1
        return {
            "schema": MODULE.PAIR_SCHEMA,
            "pair_id": pair_id,
            "split": split,
            "composition_id": "Li:1|O:1",
            "chemsys": "Li-O",
            "N": 2,
            "prompt": self.prompt,
            "winner_answer": self.winner,
            "loser_answer": self.loser,
            "winner_energy_per_atom": -2.0,
            "loser_energy_per_atom": -1.9,
            "energy_gap_eV_per_atom": gap,
            "soft_target": 1.0 / (1.0 + math.exp(-gap / 0.03)),
            "pair_weight": pair_weight,
            "winner_source": "a",
            "loser_source": "b",
        }

    def write_rows(self, directory, rows, filename="train.jsonl"):
        path = Path(directory) / filename
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def test_dataset_and_collator_preserve_exact_dynamic_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_rows(directory, [self.row()])
            dataset = MODULE.D3POPairDataset(
                path, self.tokenizer, expected_split="train"
            )
            self.assertEqual(len(dataset), 1)
            item = dataset[0]
            self.assertEqual(item["num_atoms"], 2)
            self.assertEqual(item["winner_input_ids"].shape[0], 2 + 15)
            batch = MODULE.D3POPairCollator(self.tokenizer)([item])
            self.assertEqual(
                batch["winner_input_ids"].shape,
                batch["loser_input_ids"].shape,
            )
            self.assertEqual(batch["composition_ids"], ["Li:1|O:1"])
            self.assertAlmostEqual(float(batch["pair_weight"][0]), 1.0)

    def test_noncanonical_answer_is_rejected(self):
        row = self.row()
        row["winner_answer"] = " " + self.winner
        with self.assertRaisesRegex(ValueError, "canonical|whitespace"):
            MODULE.validate_pair_row(row, expected_split="train")

    def test_composition_weights_must_sum_to_one(self):
        first = self.row(pair_id="pair-0", pair_weight=0.2)
        second = self.row(pair_id="pair-1", pair_weight=0.2)
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_rows(directory, [first, second])
            with self.assertRaisesRegex(ValueError, "sum to one"):
                MODULE.D3POPairDataset(
                    path, self.tokenizer, expected_split="train"
                )

    def test_legal_support_is_position_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_rows(directory, [self.row()])
            dataset = MODULE.D3POPairDataset(
                path, self.tokenizer, expected_split="train"
            )
            batch = MODULE.D3POPairCollator(self.tokenizer)([dataset[0]])
            prompt = int(batch["prompt_lengths"][0])
            masked = torch.zeros_like(batch["winner_input_ids"], dtype=torch.bool)
            masked[0, prompt + 1] = True  # LA
            masked[0, prompt + 4] = True  # AA
            masked[0, prompt + 8] = True  # first X
            supports = MODULE.DynamicLegalSupportCache(self.tokenizer).selected(
                num_atoms=batch["num_atoms"],
                prompt_lengths=batch["prompt_lengths"],
                masked_positions=masked,
            )
            inverse_vocab = {value: key for key, value in self.tokenizer.vocab.items()}
            prefixes = [
                {inverse_vocab[token_id][1:3] for token_id in support}
                for support in supports
            ]
            self.assertEqual(prefixes, [{"LA"}, {"AA"}, {"X_"}])

    def test_step0_canary_runs_reference_then_policy_and_is_log2(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_rows(directory, [self.row()])
            dataset = MODULE.D3POPairDataset(
                path, self.tokenizer, expected_split="train"
            )
            batch = MODULE.D3POPairCollator(self.tokenizer)([dataset[0]])
            model = self.FakeAdapterModel(len(self.tokenizer))
            runtime = MODULE.AdapterRuntime(
                model=model,
                policy_parameters=(model.adapters["policy"],),
                reference_parameters=(model.adapters["reference"],),
                frozen_parameters=(model.adapters["reference"],),
            )
            report = MODULE.run_step0_canary(
                runtime,
                batch,
                MODULE.DynamicLegalSupportCache(self.tokenizer),
            )
            self.assertTrue(report["passed"])
            self.assertAlmostEqual(report["hard_label_loss"], math.log(2.0), places=6)
            self.assertEqual(model.activation_log[:2], ["reference", "policy"])

    def test_empty_shared_mask_backpropagates_exact_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_rows(directory, [self.row()])
            dataset = MODULE.D3POPairDataset(
                path, self.tokenizer, expected_split="train"
            )
            batch = MODULE.D3POPairCollator(self.tokenizer)([dataset[0]])
            model = self.FakeAdapterModel(len(self.tokenizer))
            runtime = MODULE.AdapterRuntime(
                model=model,
                policy_parameters=(model.adapters["policy"],),
                reference_parameters=(model.adapters["reference"],),
                frozen_parameters=(model.adapters["reference"],),
            )
            empty = torch.zeros_like(batch["winner_input_ids"], dtype=torch.bool)
            corruption = MODULE.shared_geometry_corruption(
                batch["winner_input_ids"],
                batch["loser_input_ids"],
                batch["prompt_lengths"],
                batch["num_atoms"],
                attention_mask=batch["attention_mask"],
                p_mask=torch.tensor([0.5]),
                shared_mask=empty,
            )
            computation = MODULE.compute_pair_loss(
                runtime,
                batch,
                MODULE.DynamicLegalSupportCache(self.tokenizer),
                generator=None,
                require_grad=True,
                corruption=corruption,
            )
            computation.output.loss.backward()
            self.assertIsNotNone(model.adapters["policy"].grad)
            self.assertEqual(float(model.adapters["policy"].grad), 0.0)

    def test_frozen_loader_args_and_hyperparameters(self):
        args = argparse.Namespace(
            model_path=Path("model"),
            checkpoint_path=Path("step-696"),
            data_dir=Path("pairs"),
        )
        helper = MODULE.build_llada_loader_args(args)
        self.assertFalse(helper.use_lora)
        self.assertTrue(helper.skip_data_vocab_resize)
        self.assertEqual(helper.representation, "dynamic_v1")
        self.assertEqual(helper.lora_rank, 8)
        self.assertEqual(helper.lora_alpha, 32)
        self.assertEqual(helper.lora_dropout, 0.0)
        self.assertEqual(MODULE.TOTAL_UPDATES, 348)
        self.assertEqual(MODULE.GRADIENT_ACCUMULATION, 16)
        self.assertEqual(MODULE.BETA, 0.1)
        self.assertEqual(MODULE.LEARNING_RATE, 5e-6)
        self.assertEqual(MODULE.ALLOWED_TRAINING_SEEDS, (81017, 81018))

    def test_weighted_sampler_consumes_pair_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_rows(directory, [self.row()])
            dataset = MODULE.D3POPairDataset(
                path, self.tokenizer, expected_split="train"
            )
            loader = MODULE.build_train_loader(
                dataset,
                MODULE.D3POPairCollator(self.tokenizer),
                seed=81017,
            )
            self.assertIsInstance(loader.sampler, torch.utils.data.WeightedRandomSampler)
            self.assertEqual(list(loader.sampler.weights), [1.0])


if __name__ == "__main__":
    unittest.main()
