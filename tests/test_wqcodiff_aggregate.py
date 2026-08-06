from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.aggregate import (
    FinalAggregateConfig,
    _duplicate_components,
    aggregate_final,
)


class FinalAggregateTests(unittest.TestCase):
    def test_duplicate_components_join_across_method_sides(self) -> None:
        pairs = [
            (
                "p0",
                {"duplicate_cluster": {"standard": "shared"}},
                {"duplicate_cluster": {"standard": "champion-only"}},
                1.0,
            ),
            (
                "p1",
                {"duplicate_cluster": {"standard": "final-only"}},
                {"duplicate_cluster": {"standard": "shared"}},
                -1.0,
            ),
        ]
        components = _duplicate_components(pairs)
        self.assertEqual(components[0], components[1])

    def test_actual_slurm_compute_is_used_for_the_compute_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluations = root / "evaluations.jsonl"
            with evaluations.open("x", encoding="utf-8") as handle:
                for method, outcome in (("champion", False), ("final", True)):
                    for evaluator in ("chgnet", "mattersim", "mace"):
                        for stage in ("raw", "common-refiner", "relaxed"):
                            row = {
                                "schema": "wqcodiff_mlip_sun_attempt_v1",
                                "attempt_id": f"{method}-a0",
                                "pair_id": "pair-0",
                                "method": method,
                                "training_seed": 11,
                                "sampling_seed": 101,
                                "evaluator": evaluator,
                                "stage": stage,
                                "contract_hash": f"contract-{evaluator}",
                                "hull_sha256": f"hull-{evaluator}",
                                "matcher_contract_sha256": "matcher",
                                "status": "succeeded",
                                "stable_at_0p0": outcome,
                                "stable_at_0p1": outcome,
                                "unique": {"standard": True},
                                "full_novel": {"standard": True},
                                "anonymous_prototype_novel": True,
                                "protostructure_novel": True,
                                "substitution_aware_novel": True,
                                "novel_unique_standard": True,
                                "mlip_sun_at_0p0": outcome,
                                "mlip_sun_at_0p1": outcome,
                                "substitution_aware_mlip_sun_at_0p1": outcome,
                                "matcher_sensitivity_sun_at_0p1": {
                                    "strict": outcome,
                                    "standard": outcome,
                                    "lenient": outcome,
                                },
                                "duplicate_cluster": {"standard": f"{method}-dup"},
                                "orbit_count_changed": method == "final",
                                "topology_changed": method == "final",
                                "dimension_changed": False,
                                "revision_control": "geometry" if method == "final" else "none",
                                "revision_events": 2 if method == "final" else 0,
                                "revision_fills": 1 if method == "final" else 0,
                                "revision_churn": 0.25 if method == "final" else 0.0,
                                "topology_event_counts": {
                                    "orbit_birth": 1 if method == "final" else 0,
                                    "orbit_death": 0,
                                    "wyckoff_type_change": 0,
                                    "species_change": 0,
                                },
                                "material_family": "oxide",
                                "generation_calls": {"joint": 64},
                                "generation_walltime_s": 1.0,
                                "generation_flops_lower_bound": 100.0,
                            }
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
            usage_paths = []
            for index, (method, gpu_hours, peak) in enumerate(
                (("champion", 1.0, 1000.0), ("final", 2.0, 1500.0)), start=1
            ):
                path = root / f"usage-{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "schema": "wqcodiff_slurm_usage_v1",
                            "slurm_job_id": str(index),
                            "method": method,
                            "stage": "sample",
                            "gpu_hours": gpu_hours,
                            "peak_memory_mib": peak,
                        }
                    ),
                    encoding="utf-8",
                )
                usage_paths.append(str(path))
            result = aggregate_final(
                FinalAggregateConfig(
                    inputs=(str(evaluations),),
                    output=str(root / "aggregate.json"),
                    champion="champion",
                    final_method="final",
                    usage_inputs=tuple(usage_paths),
                    allow_nonpaper_counts=True,
                )
            )
            self.assertTrue(result["compute"]["compute_evidence_complete"])
            self.assertEqual(result["compute"]["actual_slurm_gpu_hour_ratio"], 2.0)
            self.assertEqual(result["compute"]["peak_memory_mib"]["final"], 1500.0)
            final_metric = next(
                row
                for row in result["metric_table"]
                if row["method"] == "final"
                and row["evaluator"] == "mattersim"
                and row["stage"] == "relaxed"
            )
            self.assertEqual(final_metric["uniqueness_standard"], 1.0)
            self.assertEqual(final_metric["species_wyckoff_protostructure_novel"], 1.0)
            final_mechanism = next(
                row
                for row in result["topology_and_revision_table"]
                if row["method"] == "final"
            )
            self.assertEqual(final_mechanism["attempts"], 1)
            self.assertEqual(final_mechanism["revision_events_total"], 2)
            self.assertEqual(
                final_mechanism["topology_event_counts_total"]["orbit_birth"], 1
            )


if __name__ == "__main__":
    unittest.main()
