import json
from pathlib import Path
import re
import tempfile
import unittest

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer, parse_dynamic_answer
from crystal_dlm.plangraph_dataset import build_plangraph_dataset
from crystal_dlm.r5_plan_state import plan_state_from_arrays

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - planning mirror may lack torch.
    torch = None

if torch is not None:
    from crystal_dlm.planned_preflight import (
        PlannedPreflightError,
        preflight_planned_data,
        verify_published_dataset,
    )


class WhitespaceTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.vocab["<unk>"])

    def add_special_tokens(self, payload):
        added = 0
        for token in payload.get("additional_special_tokens", []):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        truncation=False,
        max_length=None,
    ):
        del add_special_tokens
        tokens = re.findall(r"<[^>]+>|[^\s<]+", str(text))
        ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        if truncation and max_length is not None:
            ids = ids[: int(max_length)]
        return {"input_ids": ids}


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class PlannedPreflightTests(unittest.TestCase):
    def make_source(self, *, shift: float):
        answer, _diagnostics = arrays_to_dynamic_answer(
            lengths=[3.1 + shift, 3.1, 5.2],
            angles=[90.0, 90.0, 120.0],
            species=["Li", "O", "Li"],
            frac_coords=[
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5],
            ],
        )
        arrays = parse_dynamic_answer(answer, strict=True)
        return {
            "representation": "dynamic_v1",
            "plan_state": plan_state_from_arrays(
                arrays,
                metadata={"spacegroup.number": 194},
            ),
            "answer": answer,
        }

    def build_dataset(self, root: Path) -> Path:
        source = root / "source"
        source.mkdir()
        (source / "train.jsonl").write_text(
            "".join(
                json.dumps(self.make_source(shift=shift)) + "\n"
                for shift in (0.0, 0.1, 0.2)
            ),
            encoding="utf-8",
        )
        (source / "val.jsonl").write_text(
            json.dumps(self.make_source(shift=0.3)) + "\n",
            encoding="utf-8",
        )
        (source / "vocab_tokens.txt").write_text(
            "<N_3>\n<E_Li>\n<E_O>\n",
            encoding="utf-8",
        )
        output = root / "published"
        build_plangraph_dataset(
            split_inputs={
                "train": source / "train.jsonl",
                "val": source / "val.jsonl",
            },
            output_dir=output,
            vocab_file=source / "vocab_tokens.txt",
            project_root=root,
        )
        return output / "body"

    def test_full_denominator_d2_tokenizer_and_mask_preflight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            body = self.build_dataset(Path(temp_dir))
            report = preflight_planned_data(
                data_dir=body,
                tokenizer=WhitespaceTokenizer(),
                splits=("train", "val"),
                max_length=512,
                policy="d2",
                mask_smoke_rows=4,
                mask_smoke_batch_size=2,
            )

            self.assertTrue(report["preflight_gate_passed"])
            self.assertEqual(report["total_rows"], 4)
            self.assertEqual(report["failed_rows"], 0)
            self.assertEqual(
                report["mask_smoke"]["planned_only"]["planned_samples"],
                4,
            )
            self.assertEqual(
                report["mask_smoke"]["fixed_stateless_iid"]["planned_samples"],
                0,
            )
            self.assertGreater(
                report["mask_smoke"]["planned_only"]["future_masked_tokens"],
                0,
            )

    def test_manifest_verification_detects_body_tampering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            body = self.build_dataset(Path(temp_dir))
            with (body / "train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaises(PlannedPreflightError):
                verify_published_dataset(body)

if __name__ == "__main__":
    unittest.main()
