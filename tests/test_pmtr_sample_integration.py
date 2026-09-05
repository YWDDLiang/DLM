import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
    "sample_llada_r5_exact_length_pmtr_test",
    ROOT / "src" / "scripts" / "sample_llada_r5_exact_length.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import R5 sampling CLI")
MODULE = importlib.util.module_from_spec(SPEC)
with patch.dict(sys.modules, {"scripts.sample_llada_dynamic_crystals": SAMPLE_DEPENDENCY}):
    SPEC.loader.exec_module(MODULE)


class PMTRSampleIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.plan = {"N": 2, "elements": ["Na", "Cl"], "counts": [1, 1]}
        self.program = MODULE.program_from_element_order(
            self.plan, ["Cl", "Na"], order_source="llama_pointer"
        )

    def call(self, transform=None, plan_vpa_projection=False):
        return MODULE.apply_spad_basin_closure(
            object(),
            "predictor_tokens",
            programs=[self.program],
            batch=[{"sample_idx": 7, "plan_state": self.plan}],
            base_seed=17,
            prompt_length=2,
            gen_length=15,
            attention_mask=None,
            temperature=0.7,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=126336,
            allowed_token_ids_by_generation_pos=None,
            lightweight_decoding_constraints={"pbc_min_distance_mask": True},
            plan_vpa_projection=plan_vpa_projection,
            transaction_logit_transform=transform,
        )

    def test_opt_in_path_passes_row_metadata_and_committed_lattice_version(self):
        transform = object()
        calls = []

        def fake_cell(_model, tokens, **kwargs):
            recorded = dict(kwargs)
            recorded["lattice_versions_by_batch"] = list(
                kwargs["lattice_versions_by_batch"]
            )
            calls.append(("cell", tokens, recorded))
            kwargs["on_lattice_commit"](0, 1)
            return "cell_closed", [{"lattice_commit_signal": True}]

        def fake_blocks(_model, tokens, **kwargs):
            calls.append(("blocks", tokens, kwargs))
            return "blocks_closed", [[{"final_geometry_supported": True}]]

        with patch.object(MODULE, "revise_spad_cell", side_effect=fake_cell), patch.object(
            MODULE, "revise_spad_species_blocks", side_effect=fake_blocks
        ):
            output, _, _, metadata = self.call(transform=transform)

        self.assertEqual(output, "blocks_closed")
        self.assertEqual(calls[1][1], "cell_closed")
        for call in calls:
            kwargs = call[2]
            self.assertIs(kwargs["transaction_logit_transform"], transform)
            self.assertEqual(kwargs["plan_metadata_by_batch"], [self.plan])
            self.assertEqual(
                kwargs["program_metadata_by_batch"],
                [{"species_order": ["Cl", "Na"], "species_program_source": "llama_pointer"}],
            )
        self.assertEqual(calls[0][2]["lattice_versions_by_batch"], [0])
        self.assertEqual(calls[1][2]["lattice_versions_by_batch"], [1])
        self.assertEqual(
            metadata[0]["pmtr"]["lattice_version_for_species_repair"], 1
        )

    def test_disabled_path_adds_no_kwargs_and_consumes_no_rng(self):
        calls = []

        def fake_cell(_model, tokens, **kwargs):
            calls.append(("cell", kwargs))
            return "cell_closed", [{}]

        def fake_blocks(_model, tokens, **kwargs):
            calls.append(("blocks", kwargs))
            return "blocks_closed", [[{"final_geometry_supported": True}]]

        torch.manual_seed(991)
        before = torch.random.get_rng_state().clone()
        with patch.object(MODULE, "revise_spad_cell", side_effect=fake_cell), patch.object(
            MODULE, "revise_spad_species_blocks", side_effect=fake_blocks
        ):
            output, _, _, metadata = self.call()
        after = torch.random.get_rng_state()

        self.assertEqual(output, "blocks_closed")
        self.assertTrue(torch.equal(before, after))
        optional = {
            "transaction_logit_transform",
            "plan_metadata_by_batch",
            "program_metadata_by_batch",
            "lattice_versions_by_batch",
            "on_lattice_commit",
        }
        self.assertTrue(all(optional.isdisjoint(kwargs) for _name, kwargs in calls))
        self.assertNotIn("pmtr", metadata[0])

    def test_checkpoint_flag_is_strictly_opt_in_to_basin_closure(self):
        args = types.SimpleNamespace(pmtr_checkpoint=None, spad_basin_closure=False)
        self.assertIsNone(MODULE.validate_pmtr_configuration(args))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pmtr.pt"
            path.write_bytes(b"checkpoint")
            args.pmtr_checkpoint = path
            with self.assertRaisesRegex(ValueError, "requires --spad-basin-closure"):
                MODULE.validate_pmtr_configuration(args)
            args.spad_basin_closure = True
            self.assertEqual(MODULE.validate_pmtr_configuration(args), path.resolve())

    def test_complete_body_entry_skips_predictor_and_reuses_closure(self):
        sentinel = object()
        expected = ("repaired", [{"cell": True}], [[{"block": True}]], [{}])
        with patch.object(
            MODULE, "apply_spad_basin_closure", return_value=expected
        ) as closure, patch.object(MODULE, "generate") as predictor:
            observed = MODULE.repair_complete_spad_tokens_with_pmtr(
                object(),
                "complete_raw_tokens",
                programs=[self.program],
                batch=[{"sample_idx": 7, "plan_state": self.plan}],
                transaction_logit_transform=sentinel,
                base_seed=17,
                prompt_length=2,
                gen_length=15,
                attention_mask=None,
                temperature=0.7,
                cfg_scale=0.0,
                remasking="low_confidence",
                mask_id=126336,
                allowed_token_ids_by_generation_pos=None,
                lightweight_decoding_constraints={"pbc_min_distance_mask": True},
            )
        self.assertEqual(observed, expected)
        predictor.assert_not_called()
        closure.assert_called_once()
        self.assertEqual(closure.call_args.args[1], "complete_raw_tokens")
        self.assertIs(
            closure.call_args.kwargs["transaction_logit_transform"], sentinel
        )

        with self.assertRaisesRegex(ValueError, "requires a logit transform"):
            MODULE.repair_complete_spad_tokens_with_pmtr(
                object(),
                torch.zeros((1, 17), dtype=torch.long),
                programs=[self.program],
                batch=[{"sample_idx": 7, "plan_state": self.plan}],
                transaction_logit_transform=None,
                base_seed=17,
                prompt_length=2,
                gen_length=15,
                attention_mask=None,
                temperature=0.7,
                cfg_scale=0.0,
                remasking="low_confidence",
                mask_id=126336,
                allowed_token_ids_by_generation_pos=None,
                lightweight_decoding_constraints={"pbc_min_distance_mask": True},
            )


if __name__ == "__main__":
    unittest.main()
