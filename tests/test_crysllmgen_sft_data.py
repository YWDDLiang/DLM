from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from crystal_dlm.wqcodiff.bridge import ChartCatalog, ChartSpec
from crystal_dlm.wqcodiff.crysllmgen.sft_data import (
    build_coarse_example,
    build_direct_edit_example,
    serialize_geometry_evidence,
    tokenize_sft_example,
    write_coarse_sft_jsonl,
    write_mixed_wq_sft_jsonl,
)
from crystal_dlm.wqcodiff.state import OrbitState, StratifiedState


class FakeCatalog(ChartCatalog):
    def types(self, space_group: int) -> tuple[int, ...]:
        return (0, 1)

    def get(self, space_group: int, wyckoff_type: int) -> ChartSpec:
        if wyckoff_type == 0:
            return ChartSpec(space_group, 0, "a", 1, 0, 1)
        if wyckoff_type == 1:
            return ChartSpec(space_group, 1, "b", 2, 1, 2)
        raise KeyError(wyckoff_type)


def _record(material_id: str = "mp-test") -> dict[str, object]:
    state = StratifiedState(
        space_group=225,
        lattice_system="cubic",
        lattice_chart=(1.5,),
        orbits=(
            OrbitState("o0", 0, 14, 1, 0, (), 1),
            OrbitState("o1", 1, 8, 2, 1, (0.25,), 2),
        ),
    )
    return {
        "schema": "mp20_wq_v1",
        "material_id": material_id,
        "selected": True,
        "decompositions": {
            "symprec_1e-02": {
                "state": state.to_dict(),
                "primitive_structure": {},
            }
        },
    }


class FakeTokenizer:
    @staticmethod
    def apply_chat_template(messages, *, tokenize, add_generation_prompt):
        value = ""
        for message in messages:
            if message["role"] == "system":
                value += "S:" + message["content"] + "|"
            elif message["role"] == "user":
                value += "U:" + message["content"] + "|"
            else:
                value += "A:" + message["content"] + "<EOT>"
        if add_generation_prompt:
            value += "A:"
        assert tokenize
        return list(value.encode("utf-8"))


class CrysLLMGenSFTDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = FakeCatalog()

    def test_wyckoff_example_is_deterministic_and_has_no_mask(self) -> None:
        first = build_coarse_example(
            _record(),
            representation="wyckoff",
            epoch=0,
            training_seed=11,
            catalog=self.catalog,
        )
        second = build_coarse_example(
            _record(),
            representation="wyckoff",
            epoch=0,
            training_seed=11,
            catalog=self.catalog,
        )
        self.assertEqual(first, second)
        self.assertNotIn("MASK", first["answer"])
        self.assertTrue(first["answer"].endswith(";STOP"))

    def test_canonical_ablation_is_explicitly_labelled(self) -> None:
        example = build_coarse_example(
            _record(),
            representation="wyckoff",
            epoch=0,
            training_seed=11,
            catalog=self.catalog,
            canonical_orbit_order=True,
        )
        self.assertEqual(example["order_mode"], "canonical_ablation")

    def test_writer_materializes_each_epoch_once_and_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.jsonl"
            source.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
            output = root / "sft.jsonl"
            manifest = root / "manifest.json"
            report = write_coarse_sft_jsonl(
                input_paths=[source],
                output=output,
                manifest=manifest,
                representation="wyckoff",
                epochs=3,
                training_seed=11,
                catalog=self.catalog,
            )
            self.assertEqual(report["examples"], 3)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 3)
            with self.assertRaises(FileExistsError):
                write_coarse_sft_jsonl(
                    input_paths=[source],
                    output=output,
                    manifest=manifest,
                    representation="wyckoff",
                    epochs=3,
                    training_seed=11,
                    catalog=self.catalog,
                )

    def test_official_chat_template_labels_only_the_answer(self) -> None:
        example = build_coarse_example(
            _record(),
            representation="wyckoff",
            epoch=0,
            training_seed=11,
            catalog=self.catalog,
        )
        tokenized = tokenize_sft_example(FakeTokenizer(), example, max_length=512)
        supervised = [value for value in tokenized["labels"] if value != -100]
        self.assertTrue(supervised)
        self.assertEqual(len(tokenized["input_ids"]), len(tokenized["labels"]))
        self.assertTrue(all(value == -100 for value in tokenized["labels"][:-len(supervised)]))

    def test_direct_edit_example_is_one_exact_command_with_binned_evidence(self) -> None:
        with mock.patch(
            "crystal_dlm.wqcodiff.crysllmgen.sft_data._evidence_text",
            return_value="012345ABCDE0",
        ):
            first = build_direct_edit_example(
                _record(), ordinal=7, training_seed=11, catalog=self.catalog
            )
            second = build_direct_edit_example(
                _record(), ordinal=7, training_seed=11, catalog=self.catalog
            )
        self.assertEqual(first, second)
        self.assertTrue(first["user_prompt"].startswith("P=SG=225;"))
        self.assertIn(";G=012345ABCDE0", first["user_prompt"])
        self.assertNotIn("\n", first["answer"])
        self.assertRegex(first["answer"], r"^(NOOP|BIRTH;|DEATH;|TYPE;|SPECIES;)")

    def test_geometry_evidence_is_fixed_width_without_redundant_indices(self) -> None:
        encoded = serialize_geometry_evidence(
            [(0.0, 0.1, 0.2, 0.3, 0.4, 1.0), (1.0, 0.4, 0.3, 0.2, 0.1, 0.0)]
        )
        self.assertEqual(len(encoded), 12)
        self.assertNotIn(":", encoded)
        self.assertNotIn("|", encoded)

    def test_direct_edit_curriculum_executes_every_registered_operator(self) -> None:
        with mock.patch(
            "crystal_dlm.wqcodiff.crysllmgen.sft_data._evidence_text",
            return_value="012345ABCDE0",
        ):
            examples = [
                build_direct_edit_example(
                    _record(), ordinal=ordinal, training_seed=11, catalog=self.catalog
                )
                for ordinal in range(256)
            ]
        requested = {value["requested_operator"] for value in examples}
        self.assertEqual(
            requested,
            {
                "clean",
                "deletion",
                "false_insertion",
                "wrong_wyckoff",
                "wrong_species",
                "joint",
            },
        )
        self.assertTrue(
            all(
                value["answer"].startswith(
                    ("NOOP", "BIRTH;", "DEATH;", "TYPE;", "SPECIES;")
                )
                for value in examples
            )
        )

    def test_mixed_writer_materializes_both_stages_exclusively(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.jsonl"
            source.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
            output = root / "mixed.jsonl"
            manifest = root / "mixed.manifest.json"
            with mock.patch(
                "crystal_dlm.wqcodiff.crysllmgen.sft_data._evidence_text",
                return_value="012345ABCDE0",
            ):
                report = write_mixed_wq_sft_jsonl(
                    input_paths=[source],
                    output=output,
                    manifest=manifest,
                    training_seed=11,
                    catalog=self.catalog,
                )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(report["stage_counts"], {"coarse_proposal": 1, "direct_edit": 1})
            self.assertEqual([row["stage"] for row in rows], ["coarse_proposal", "direct_edit"])
            with self.assertRaises(FileExistsError):
                write_mixed_wq_sft_jsonl(
                    input_paths=[source],
                    output=output,
                    manifest=manifest,
                    training_seed=11,
                    catalog=self.catalog,
                )


if __name__ == "__main__":
    unittest.main()
