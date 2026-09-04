import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import types
import tempfile
import unittest
from unittest.mock import patch


try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DEPENDENCY = types.ModuleType("scripts.sample_llada_dynamic_crystals")
for name in (
    "build_dynamic_lightweight_constraints",
    "graph_from_arrays",
    "import_process_one",
    "init_distributed",
    "load_model_and_tokenizer",
    "rank_path",
    "read_valid_arrays",
    "write_valid_arrays",
):
    setattr(SAMPLE_DEPENDENCY, name, lambda *args, **kwargs: None)
SPEC = importlib.util.spec_from_file_location(
    "sample_llada_r5_exact_length_basin_closure_test",
    ROOT / "src/scripts/sample_llada_r5_exact_length.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import R5 sampling CLI")
MODULE = None
if torch is not None:
    MODULE = importlib.util.module_from_spec(SPEC)
    with patch.dict(
        sys.modules,
        {"scripts.sample_llada_dynamic_crystals": SAMPLE_DEPENDENCY},
    ):
        SPEC.loader.exec_module(MODULE)


def configuration(**overrides):
    values = {
        "spad_basin_closure": True,
        "spad_basin_closure_capability_json": None,
        "generation_schedule": "spad",
        "spad_backfill": False,
        "spad_cell_closure": False,
        "pbc_min_distance_mask": True,
        "checkpoint_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def capability(checkpoint_path, adapter_sha256):
    return {
        "schema": "spad_basin_closure_capability_v1",
        "checkpoint_path": str(checkpoint_path),
        "adapter_model_sha256": adapter_sha256,
        "spad_cell_closure_trained": True,
        "spad_species_block_closure_trained": True,
        "closure_schedule_version": (
            MODULE.SPAD_BASIN_CLOSURE_SCHEDULE_VERSION
        ),
    }


@unittest.skipIf(torch is None, "torch unavailable")
class SPADBasinClosureCLITest(unittest.TestCase):
    def test_rejects_incompatible_modes(self):
        cases = (
            ({"generation_schedule": "exact-plan"}, "generation-schedule spad"),
            ({"spad_backfill": True}, "cannot be combined with --spad-backfill"),
            (
                {"spad_cell_closure": True},
                "cannot be combined with --spad-cell-closure",
            ),
        )
        for overrides, expected in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, expected):
                    MODULE.validate_spad_basin_closure_configuration(
                        configuration(**overrides)
                    )

    def test_capability_manifest_accepts_only_matching_trained_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "adapter_step_1696"
            checkpoint.mkdir()
            adapter = checkpoint / "adapter_model.safetensors"
            adapter.write_bytes(b"trained closure adapter")
            adapter_sha256 = hashlib.sha256(adapter.read_bytes()).hexdigest()
            manifest = checkpoint / "spad_basin_closure_capability.json"
            manifest.write_text(
                json.dumps(capability(checkpoint, adapter_sha256)), encoding="utf-8"
            )
            result = MODULE.validate_spad_basin_closure_configuration(
                configuration(
                    checkpoint_path=checkpoint,
                    spad_basin_closure_capability_json=manifest,
                )
            )
            self.assertEqual(result["checkpoint_path"], str(checkpoint.resolve()))
            self.assertEqual(
                result["closure_schedule_version"],
                MODULE.SPAD_BASIN_CLOSURE_SCHEDULE_VERSION,
            )

            bad_payloads = (
                {
                    **capability(checkpoint, adapter_sha256),
                    "spad_cell_closure_trained": False,
                },
                {
                    **capability(checkpoint, adapter_sha256),
                    "spad_species_block_closure_trained": 1,
                },
                {
                    **capability(checkpoint, adapter_sha256),
                    "closure_schedule_version": "wrong_schedule",
                },
                {
                    **capability(checkpoint, adapter_sha256),
                    "schema": "wrong_schema",
                },
                capability(root / "different_checkpoint", adapter_sha256),
                capability(checkpoint, "0" * 64),
            )
            for index, payload in enumerate(bad_payloads):
                manifest.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(payload=payload):
                    with self.assertRaises(ValueError):
                        MODULE.validate_spad_basin_closure_configuration(
                            configuration(
                                checkpoint_path=checkpoint,
                                spad_basin_closure_capability_json=manifest,
                            )
                        )

            manifest.write_text(
                json.dumps(capability(checkpoint, adapter_sha256)),
                encoding="utf-8",
            )
            external_manifest = root / "forged_capability.json"
            external_manifest.write_text(manifest.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "checkpoint-local"):
                MODULE.validate_spad_basin_closure_configuration(
                    configuration(
                        checkpoint_path=checkpoint,
                        spad_basin_closure_capability_json=external_manifest,
                    )
                )

    def test_calls_cell_before_reverse_species_blocks_with_distinct_seeds(self):
        plan = {"N": 4, "elements": ["O", "Na", "Cl"], "counts": [2, 1, 1]}
        program = MODULE.program_from_element_order(
            plan,
            ["Cl", "O", "Na"],
            order_source="llama_pointer",
        )
        calls = []

        def fake_cell(_model, tokens, **kwargs):
            calls.append(("cell", tokens, kwargs))
            return "cell_closed", [{"stage": "cell"}]

        def fake_blocks(_model, tokens, **kwargs):
            calls.append(("blocks", tokens, kwargs))
            return "block_closed", [[{
                "stage": "block",
                "final_geometry_supported": True,
            }]]

        with patch.object(MODULE, "revise_spad_cell", side_effect=fake_cell), patch.object(
            MODULE, "revise_spad_species_blocks", side_effect=fake_blocks
        ):
            output, cell_logs, block_logs, metadata = (
                MODULE.apply_spad_basin_closure(
                    object(),
                    "predictor_output",
                    programs=[program],
                    batch=[{"sample_idx": 7}],
                    base_seed=17,
                    prompt_length=2,
                    gen_length=23,
                    attention_mask=None,
                    temperature=0.7,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                    mask_id=126336,
                    allowed_token_ids_by_generation_pos=None,
                    lightweight_decoding_constraints={
                        "pbc_min_distance_mask": True
                    },
                )
            )

        self.assertEqual([entry[0] for entry in calls], ["cell", "blocks"])
        self.assertEqual(calls[0][1], "predictor_output")
        self.assertEqual(calls[1][1], "cell_closed")
        self.assertEqual(output, "block_closed")
        self.assertEqual(cell_logs, [{"stage": "cell"}])
        self.assertEqual(
            block_logs,
            [[{"stage": "block", "final_geometry_supported": True}]],
        )
        self.assertEqual(
            calls[1][2]["revision_blocks_by_batch"], [[[2], [1, 0], [3]]]
        )
        cell_seed = calls[0][2]["sampling_seeds_by_batch"][0]
        block_seed = calls[1][2]["sampling_seeds_by_batch"][0]
        self.assertNotEqual(cell_seed, block_seed)
        self.assertEqual(
            metadata[0]["stage_order"],
            ["predictor", "cell", "reverse_species_blocks"],
        )
        self.assertEqual(metadata[0]["species_program"], ["Cl", "O", "Na"])
        self.assertTrue(metadata[0]["final_geometry_supported"])

    def test_new_closure_seed_namespaces_do_not_collide(self):
        base_seed = 17
        sample = 7
        cell_seed = MODULE._spad_basin_closure_stage_seed(
            base_seed,
            sample,
            MODULE.SPAD_BASIN_CLOSURE_CELL_STAGE_OFFSET,
        )
        block_seed = MODULE._spad_basin_closure_stage_seed(
            base_seed,
            sample,
            MODULE.SPAD_BASIN_CLOSURE_BLOCK_STAGE_OFFSET,
        )
        next_sample_block_seed = MODULE._spad_basin_closure_stage_seed(
            base_seed,
            sample + 1,
            MODULE.SPAD_BASIN_CLOSURE_BLOCK_STAGE_OFFSET,
        )
        effective = {
            cell_seed + 10_007 * component
            for component in range(6)
        }
        effective.update(
            block_seed
            + MODULE._spad_basin_closure_block_salt(block, site, component)
            for block in range(4)
            for site in range(20)
            for component in range(3)
        )
        effective.update(
            next_sample_block_seed
            + MODULE._spad_basin_closure_block_salt(block, site, component)
            for block in range(4)
            for site in range(20)
            for component in range(3)
        )
        self.assertEqual(len(effective), 6 + 2 * 4 * 20 * 3)
        self.assertNotEqual(
            block_seed + MODULE._spad_basin_closure_block_salt(1, 0, 0),
            next_sample_block_seed
            + MODULE._spad_basin_closure_block_salt(0, 0, 0),
        )

    def test_metadata_records_failed_final_geometry(self):
        program = MODULE.program_from_element_order(
            {"N": 1, "elements": ["Na"], "counts": [1]},
            ["Na"],
            order_source="llama_pointer",
        )

        def fake_cell(_model, tokens, **kwargs):
            return tokens, [{"stage": "cell"}]

        def fake_blocks(_model, tokens, **kwargs):
            return tokens, [[{
                "stage": "block",
                "final_geometry_supported": False,
            }]]

        with patch.object(MODULE, "revise_spad_cell", side_effect=fake_cell), patch.object(
            MODULE, "revise_spad_species_blocks", side_effect=fake_blocks
        ):
            _, _, _, metadata = MODULE.apply_spad_basin_closure(
                object(),
                "tokens",
                programs=[program],
                batch=[{"sample_idx": 0}],
                base_seed=17,
                prompt_length=1,
                gen_length=11,
                attention_mask=None,
                temperature=0.7,
                cfg_scale=0.0,
                remasking="low_confidence",
                mask_id=126336,
                allowed_token_ids_by_generation_pos=None,
                lightweight_decoding_constraints={"pbc_min_distance_mask": True},
            )
        self.assertFalse(metadata[0]["final_geometry_supported"])

    def test_output_fields_keep_closure_logs_and_metadata_separate(self):
        fields = MODULE.spad_basin_closure_record_fields(
            cell_revision_log={"changed_components": 2},
            species_block_revision_log=[{"block_index": 0}],
            metadata={
                "closure_schedule_version": (
                    MODULE.SPAD_BASIN_CLOSURE_SCHEDULE_VERSION
                ),
                "stage_order": ["predictor", "cell", "reverse_species_blocks"],
            },
        )
        self.assertTrue(fields["spad_basin_closure"])
        self.assertEqual(
            fields["spad_basin_closure_cell_revision_log"]["changed_components"],
            2,
        )
        self.assertEqual(
            fields["spad_basin_closure_species_block_revision_log"][0][
                "block_index"
            ],
            0,
        )
        self.assertEqual(
            fields["spad_basin_closure_metadata"]["closure_schedule_version"],
            MODULE.SPAD_BASIN_CLOSURE_SCHEDULE_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
