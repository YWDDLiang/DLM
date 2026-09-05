import json
from pathlib import Path
import re
from types import SimpleNamespace
import tempfile
import unittest

import torch

from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from scripts import train_pmtr


class FakeTokenizer:
    def __init__(self):
        tokens = ["<PAD>", "<P>", "<N_2>", "<E_Li>", "<E_O>"]
        for axis in "ABC":
            tokens += [f"<L{axis}_{value:03d}>" for value in (20, 40, 60)]
        for axis in "ABG":
            tokens += [f"<A{axis}_{value:03d}>" for value in (60, 90, 120)]
        for axis in "XYZ":
            tokens += [f"<{axis}_{value:03d}>" for value in (0, 25, 50, 75, 100)]
        self.vocab = {token: index for index, token in enumerate(tokens)}
        self.pad_token_id = self.vocab["<PAD>"]
        self.eos_token = "<PAD>"

    def get_vocab(self):
        return dict(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def __call__(self, text, add_special_tokens=False, **_kwargs):
        if add_special_tokens:
            raise AssertionError("fake tokenizer expects no added tokens")
        ids = []
        body = str(text)
        if body.startswith("plan\n"):
            ids.append(self.vocab["<P>"])
            body = body[len("plan\n") :]
        ids.extend(self.vocab[token] for token in re.findall(r"<[^>]+>", body))
        return {"input_ids": ids}


class FakeFrozenModel(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size=8):
        super().__init__()
        self.embedding = torch.nn.Embedding(int(MASK_TOKEN_ID) + 1, hidden_size)
        self.output = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.forward_count = 0

    def get_output_embeddings(self):
        return self.output

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        self.forward_count += 1
        hidden = self.embedding(input_ids)
        return SimpleNamespace(logits=self.output(hidden))


def repair_row():
    clean = (
        "<N_2><LA_040><LB_060><LC_020><AA_090><AB_090><AG_090>"
        "<E_Li><X_000><Y_000><Z_000>"
        "<E_O><X_050><Y_050><Z_050>"
    )
    corrupted = (
        "<N_2><LA_040><LB_060><LC_020><AA_090><AB_090><AG_090>"
        "<E_Li><X_025><Y_000><Z_000>"
        "<E_O><X_050><Y_050><Z_050>"
    )
    return {
        "schema": "rollout_matched_transition_v1",
        "source_row_idx": 0,
        "prompt": "plan",
        "answer": clean,
        "source_answer": corrupted,
        "plan_state": {
            "N": 2,
            "elements": ["Li", "O"],
            "counts": [1, 1],
        },
        "num_atoms": 2,
        "species_program": ["Li", "O"],
        "species_program_source": "fake_pointer",
        # Materialized from site Y; trainer must reconstruct the site-X start.
        "forced_mask_positions": [9, 10],
        "loss_positions": [9],
        "closure": {
            "reverse_block_index": 0,
            "site_index_within_block": 0,
            "coordinate_component_index": 1,
        },
        "repair_target": {
            "kind": "site",
            "lattice_tangent": None,
            "site_slot_index": 0,
            "cartesian_site_delta_A": [-1.0, 0.0, 0.0],
        },
    }


class TrainPMTRTest(unittest.TestCase):
    def test_cli_defaults_to_full_two_epoch_head_only_training(self):
        parser = train_pmtr.build_parser()
        args = parser.parse_args(
            [
                "--data-jsonl",
                "data.jsonl",
                "--model-path",
                "base",
                "--checkpoint-path",
                "spad",
                "--output-dir",
                "out",
            ]
        )
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.expected_rows, 27_136)
        self.assertIsNone(args.limit)
        self.assertEqual(args.probe_batches, 5)

    def test_fake_model_integration_freezes_base_alternates_and_saves_final_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            output = root / "output"
            data.write_text(json.dumps(repair_row()) + "\n", encoding="utf-8")
            tokenizer = FakeTokenizer()
            model = FakeFrozenModel(len(tokenizer))
            before = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }

            def loader(_args, *, is_main=True):
                del is_main
                return tokenizer, model, 0, "fake", "fake", {}

            args = train_pmtr.build_parser().parse_args(
                [
                    "--data-jsonl",
                    str(data),
                    "--model-path",
                    "fake-base",
                    "--checkpoint-path",
                    "fake-spad",
                    "--output-dir",
                    str(output),
                    "--epochs",
                    "1",
                    "--expected-rows",
                    "1",
                    "--max-steps",
                    "2",
                    "--batch-size",
                    "1",
                    "--num-workers",
                    "0",
                    "--max-length",
                    "64",
                    "--probe-batches",
                    "2",
                    "--head-width",
                    "8",
                    "--radial-basis-count",
                    "4",
                ]
            )
            report = train_pmtr.run_training(args, model_loader=loader)
            self.assertEqual(report["optimizer_steps"], 2)
            self.assertEqual(
                report["mode_steps"],
                {"clean_identity": 1, "corrupt_repair": 1},
            )
            self.assertEqual(model.forward_count, 2)
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]))
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {train_pmtr.FINAL_STATE_NAME, train_pmtr.FINAL_CONFIG_NAME},
            )
            config = json.loads(
                (output / train_pmtr.FINAL_CONFIG_NAME).read_text(encoding="utf-8")
            )
            self.assertTrue(config["training"]["base_frozen"])
            self.assertTrue(config["training"]["full_transaction_supervision"])
            state = torch.load(output / train_pmtr.FINAL_STATE_NAME, weights_only=True)
            self.assertTrue(state)

    def test_training_sources_do_not_import_inference_time_mlip(self):
        for path in (
            Path(train_pmtr.__file__),
            Path(train_pmtr.__file__).parents[1] / "crystal_dlm" / "pmtr_training.py",
        ):
            source = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("import chgnet", source)


if __name__ == "__main__":
    unittest.main()
