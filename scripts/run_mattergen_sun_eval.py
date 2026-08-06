#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import traceback
from pathlib import Path
from typing import Literal

import ase.io
import numpy as np
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.io.ase import AseAtomsAdaptor

from mattergen.evaluation.evaluate import evaluate
from mattergen.evaluation.metrics.evaluator import MetricsEvaluator
from mattergen.evaluation.reference.reference_dataset_serializer import (
    LMDBBackedReferenceDatasetImpl,
    LMDBGZSerializer,
)
from mattergen.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher,
    DefaultOrderedStructureMatcher,
)
from mattergen.evaluation.utils.logging import logger


STRICT_STABLE_THRESHOLD = 0.0
META_STABLE_THRESHOLD = 0.1


def _is_bool_sequence(values) -> bool:
    return bool(values) and all(isinstance(value, bool) for value in values[:10])


def _bool_count(values) -> int:
    return sum(1 for value in values if bool(value))


def _quantiles(values: list[float]) -> dict[str, float | None]:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return {"mean": None, "q10": None, "q50": None, "q90": None}

    def pick(frac: float) -> float:
        index = min(len(clean) - 1, max(0, int(frac * (len(clean) - 1))))
        return clean[index]

    return {
        "mean": float(sum(clean) / len(clean)),
        "q10": pick(0.10),
        "q50": pick(0.50),
        "q90": pick(0.90),
    }


def _rate(count: int, total: int) -> float:
    return float(count) / max(1, int(total))


def compute_sun_threshold_summary(
    detailed_metrics_path: str | Path | None,
    *,
    total_submitted: int,
) -> dict[str, object] | None:
    """Compute strict/meta S.U.N from MatterGen detailed per-structure metrics.

    MatterGen's built-in ``frac_novel_unique_stable_structures`` uses its
    configured stability threshold, which defaults to 0.1 eV/atom. For this
    project we keep that as ``meta_sun`` and separately report ``strict_sun``
    using Ehull < 0.0 eV/atom.
    """

    if detailed_metrics_path is None:
        return None
    path = Path(detailed_metrics_path)
    if not path.exists():
        return None
    try:
        detailed = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"failed_to_read_detailed_metrics:{type(exc).__name__}", "message": str(exc)}

    ehull = detailed.get("energy_above_hull") or detailed.get("energy_above_hull_per_atom")
    novel_unique = detailed.get("novel_unique")
    novel = detailed.get("novel")
    unique = detailed.get("unique")
    comp_valid = detailed.get("comp_validity")
    if not isinstance(ehull, list) or not isinstance(novel_unique, list):
        return {
            "error": "missing_required_detailed_columns",
            "available_columns": sorted(str(key) for key in detailed.keys()),
        }
    n_success = min(len(ehull), len(novel_unique))
    total = int(total_submitted) if int(total_submitted) > 0 else n_success
    ehull = [float(value) for value in ehull[:n_success]]
    novel_unique_mask = [bool(value) for value in novel_unique[:n_success]]
    novel_mask = [bool(value) for value in novel[:n_success]] if isinstance(novel, list) else [False] * n_success
    unique_mask = [bool(value) for value in unique[:n_success]] if isinstance(unique, list) else [False] * n_success
    comp_mask = [bool(value) for value in comp_valid[:n_success]] if isinstance(comp_valid, list) else [True] * n_success

    strict_mask = [value < STRICT_STABLE_THRESHOLD for value in ehull]
    meta_mask = [value < META_STABLE_THRESHOLD for value in ehull]
    strict_sun_mask = [
        novel_unique_mask[idx] and strict_mask[idx] for idx in range(n_success)
    ]
    meta_sun_mask = [
        novel_unique_mask[idx] and meta_mask[idx] for idx in range(n_success)
    ]
    strict_sun_comp_mask = [
        strict_sun_mask[idx] and comp_mask[idx] for idx in range(n_success)
    ]
    meta_sun_comp_mask = [
        meta_sun_mask[idx] and comp_mask[idx] for idx in range(n_success)
    ]
    novel_unique_ehull = [
        value for value, keep in zip(ehull, novel_unique_mask) if keep
    ]
    not_novel_unique_ehull = [
        value for value, keep in zip(ehull, novel_unique_mask) if not keep
    ]

    counts = {
        "successful": n_success,
        "strict_stable": _bool_count(strict_mask),
        "meta_stable": _bool_count(meta_mask),
        "novel": _bool_count(novel_mask),
        "unique": _bool_count(unique_mask),
        "novel_unique": _bool_count(novel_unique_mask),
        "strict_sun": _bool_count(strict_sun_mask),
        "meta_sun": _bool_count(meta_sun_mask),
        "strict_sun_comp_valid": _bool_count(strict_sun_comp_mask),
        "meta_sun_comp_valid": _bool_count(meta_sun_comp_mask),
    }
    rates = {key: _rate(value, total) for key, value in counts.items()}
    rates_successful = {
        key: _rate(value, n_success)
        for key, value in counts.items()
        if key != "successful"
    }
    return {
        "thresholds": {
            "strict_stable_ehull_lt": STRICT_STABLE_THRESHOLD,
            "meta_stable_ehull_lt": META_STABLE_THRESHOLD,
        },
        "denominator": "submitted_structures",
        "total_submitted": total,
        "successful_rows": n_success,
        "counts": counts,
        "rates": rates,
        "rates_among_successful": rates_successful,
        "ehull_quantiles": {
            "all_successful": _quantiles(ehull),
            "novel_unique": _quantiles(novel_unique_ehull),
            "not_novel_unique": _quantiles(not_novel_unique_ehull),
        },
    }


def patch_lmdb_metadata_reader() -> None:
    def lmdb_get(txn, key: str):
        value = txn.get(key.encode("ascii"))
        if value is None:
            raise KeyError(key)
        return pickle.loads(value)

    def build_num_entries(self, _lmdb_path):
        result = {}
        with self.env.begin() as txn:
            chemical_systems = lmdb_get(txn, "chemical_systems")
            for chemsys in chemical_systems:
                reduced_formulas = lmdb_get(txn, f"{chemsys}.reduced_formulas")
                result[chemsys] = {}
                for reduced_formula in reduced_formulas:
                    result[chemsys][reduced_formula] = lmdb_get(
                        txn, f"{chemsys}.{reduced_formula}.length"
                    )
        return result

    LMDBBackedReferenceDatasetImpl._build_num_entries_by_chemsys_reduced_formulas = (
        build_num_entries
    )


def load_structures(path: Path):
    if path.suffix not in {".extxyz", ".xyz"}:
        raise ValueError(f"Expected .extxyz or .xyz, got {path}")
    atoms_list = ase.io.read(path, ":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    return [AseAtomsAdaptor.get_structure(atoms) for atoms in atoms_list]


def find_unsupported_structures(structures, reference):
    from smact import element_dictionary

    smact_elements = element_dictionary()
    reference_chemsys = set(reference.entries_by_chemsys.keys())
    supported = []
    unsupported_records = []
    for idx, structure in enumerate(structures):
        symbols = sorted(str(el) for el in structure.composition.elements)
        missing_reference = [sym for sym in symbols if sym not in reference_chemsys]
        missing_smact = []
        for sym in symbols:
            smact_el = smact_elements.get(sym)
            oxidation_states = getattr(smact_el, "oxidation_states", None)
            if smact_el is None or not oxidation_states:
                missing_smact.append(sym)
        if missing_reference or missing_smact:
            unsupported_records.append(
                {
                    "index": idx,
                    "formula": structure.composition.reduced_formula,
                    "missing_reference_terminals": missing_reference,
                    "missing_smact_oxidation_states": missing_smact,
                }
            )
        else:
            supported.append((idx, structure))
    return supported, unsupported_records


def safe_relax_structures(
    structures,
    *,
    structure_indices: list[int],
    device: str,
    potential_load_path: str | None,
    output_path: str,
    max_steps: int,
    fmax: float,
    max_natoms_per_batch: int,
):
    from tqdm import tqdm

    from mattersim.applications.batch_relax import BatchRelaxer, build_dataloader
    from mattersim.forcefield.potential import Potential

    class MaxStepBatchRelaxer(BatchRelaxer):
        def __init__(self, *args, max_steps: int, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_steps = max_steps
            self.step_counts: dict[int, int] = {}
            self.failed_indices: list[int] = []

        def step_batch(self):
            atoms_list = [
                opt.atoms
                for idx, opt in enumerate(self.optimizer_instances)
                if self.is_active_instance[idx]
            ]
            if not atoms_list:
                self.finished = True
                return

            dataloader = build_dataloader(
                atoms_list, batch_size=len(atoms_list), only_inference=True
            )
            energy_batch, forces_batch, stress_batch = self.potential.predict_properties(
                dataloader, include_forces=True, include_stresses=True
            )

            counter = 0
            self.finished = True
            for idx, opt in enumerate(self.optimizer_instances):
                if not self.is_active_instance[idx]:
                    continue
                structure_index = int(opt.atoms.info["structure_index"])
                opt.atoms.info["total_energy"] = energy_batch[counter]
                opt.atoms.arrays["forces"] = forces_batch[counter]
                opt.atoms.info["stress"] = stress_batch[counter]
                self.trajectories.setdefault(structure_index, []).append(opt.atoms.copy())

                opt.step()
                self.step_counts[structure_index] = (
                    self.step_counts.get(structure_index, 0) + 1
                )
                if opt.converged():
                    self.is_active_instance[idx] = False
                    self.total_converged += 1
                    if self.total_converged % 100 == 0:
                        logger.info(f"Relaxed {self.total_converged} structures.")
                elif self.step_counts[structure_index] >= self.max_steps:
                    self.is_active_instance[idx] = False
                    self.failed_indices.append(structure_index)
                    logger.warning(
                        f"Relaxation failed to converge for structure {structure_index} "
                        f"after {self.max_steps} steps; counting it as a failed job."
                    )
                else:
                    self.finished = False
                counter += 1

            self.optimizer_instances = [
                opt
                for opt, active in zip(self.optimizer_instances, self.is_active_instance)
                if active
            ]
            self.is_active_instance = [True] * len(self.optimizer_instances)

        def relax(self, atoms_list, structure_indices):
            self.trajectories = {}
            self.step_counts = {}
            self.failed_indices = []
            self.tqdmcounter = tqdm(total=len(atoms_list))
            pointer = 0
            atoms_list_ = []
            for i, atoms in enumerate(atoms_list):
                atoms_copy = atoms.copy()
                atoms_copy.info["structure_index"] = int(structure_indices[i])
                atoms_list_.append(atoms_copy)

            while pointer < len(atoms_list) or not self.finished:
                while pointer < len(atoms_list) and (
                    sum(len(opt.atoms) for opt in self.optimizer_instances)
                    + len(atoms_list[pointer])
                    <= self.max_natoms_per_batch
                ):
                    self.insert(atoms_list_[pointer])
                    self.tqdmcounter.update(1)
                    pointer += 1
                if pointer < len(atoms_list) and not self.optimizer_instances:
                    raise ValueError(
                        f"Structure {pointer} has {len(atoms_list[pointer])} atoms, "
                        f"larger than max_natoms_per_batch={self.max_natoms_per_batch}."
                    )
                self.step_batch()
            self.tqdmcounter.close()
            return self.trajectories

    atoms = [AseAtomsAdaptor.get_atoms(s) for s in structures]
    potential = Potential.from_checkpoint(
        device=device, load_path=potential_load_path, load_training_state=False
    )
    relaxer = MaxStepBatchRelaxer(
        potential=potential,
        filter="EXPCELLFILTER",
        fmax=fmax,
        max_natoms_per_batch=max_natoms_per_batch,
        max_steps=max_steps,
    )
    trajectories = relaxer.relax(atoms, structure_indices=structure_indices)
    failed_indices = set(relaxer.failed_indices)
    missing_indices = {
        idx
        for idx in structure_indices
        if idx not in trajectories or not trajectories[idx]
    }
    failed_indices |= missing_indices

    relaxed_atoms = []
    relaxed_structures = []
    successful_original_structures = []
    successful_indices = []
    total_energies = []
    for idx, structure in zip(structure_indices, structures):
        if idx in failed_indices:
            continue
        atoms_final = trajectories[idx][-1]
        relaxed_atoms.append(atoms_final)
        relaxed_structures.append(AseAtomsAdaptor.get_structure(atoms_final))
        successful_original_structures.append(structure)
        successful_indices.append(idx)
        total_energies.append(float(atoms_final.info["total_energy"]))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ase.io.write(output_path, relaxed_atoms, format="extxyz")
        logger.info(f"Relaxed converged structures saved to {output_path}")

    return (
        relaxed_structures,
        np.array(total_energies),
        successful_original_structures,
        successful_indices,
        sorted(failed_indices),
    )


def evaluate_with_safe_relax(
    *,
    structures,
    structure_indices,
    reference,
    structure_matcher,
    save_as: str,
    save_detailed_as: str,
    potential_load_path: str | None,
    device: str,
    structures_output_path: str,
    max_steps: int,
    fmax: float,
    max_natoms_per_batch: int,
    n_prefailed_jobs: int,
    metric_errors_json: str | None,
):
    relaxed_structures, energies, original_structures, _, failed_indices = (
        safe_relax_structures(
            structures,
            structure_indices=structure_indices,
            device=device,
            potential_load_path=potential_load_path,
            output_path=structures_output_path,
            max_steps=max_steps,
            fmax=fmax,
            max_natoms_per_batch=max_natoms_per_batch,
        )
    )
    evaluator = MetricsEvaluator.from_structures_and_energies(
        structures=relaxed_structures,
        energies=energies,
        original_structures=original_structures,
        reference=reference,
        structure_matcher=structure_matcher,
        energy_correction_scheme=MaterialsProject2020Compatibility(),
        n_failed_jobs=n_prefailed_jobs + len(failed_indices),
    )
    metrics, successful_metric_classes, metric_errors = compute_metrics_resilient(
        evaluator=evaluator,
        metrics=evaluator.available_metrics,
        save_as=save_as,
        pretty_print=True,
        metric_errors_json=metric_errors_json,
    )
    if save_detailed_as is not None:
        try:
            evaluator.as_dataframe(
                metrics=successful_metric_classes,
                save_as=save_detailed_as,
            )
        except Exception as exc:  # noqa: BLE001
            metric_errors["detailed_dataframe"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            logger.exception("Failed to save detailed metrics dataframe.")
            if metric_errors_json is not None:
                Path(metric_errors_json).parent.mkdir(parents=True, exist_ok=True)
                Path(metric_errors_json).write_text(
                    json.dumps(metric_errors, indent=2, sort_keys=True)
                )
    return metrics, failed_indices, metric_errors


def compute_metrics_resilient(
    *,
    evaluator: MetricsEvaluator,
    metrics,
    save_as: str | None,
    pretty_print: bool,
    metric_errors_json: str | None,
):
    metrics_dict = {}
    metric_errors = {}
    successful_metric_classes = []
    for metric_cls in metrics:
        metric = evaluator._get_metric(metric_cls)
        logger.info(f"Computing metric {metric.name}")
        try:
            metrics_dict[metric.name] = {
                "value": metric.value,
                "description": metric.description,
            }
            successful_metric_classes.append(metric_cls)
        except Exception as exc:  # noqa: BLE001
            metric_errors[metric.name] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            logger.exception(f"Skipping failed metric {metric.name}.")

    if pretty_print:
        logger.info(
            json.dumps(
                {
                    k: (round(v["value"], 4) if isinstance(v["value"], float) else v)
                    for (k, v) in metrics_dict.items()
                },
                indent=4,
            )
        )

    if save_as is not None:
        save_path = Path(save_as).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(metrics_dict, indent=4))
        logger.info(f"Saved metrics to {save_path}")
    if metric_errors_json is not None:
        errors_path = Path(metric_errors_json).resolve()
        errors_path.parent.mkdir(parents=True, exist_ok=True)
        errors_path.write_text(json.dumps(metric_errors, indent=2, sort_keys=True))

    return {k: v["value"] for k, v in metrics_dict.items()}, successful_metric_classes, metric_errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structures-path", required=True)
    parser.add_argument("--reference-dataset", required=True)
    parser.add_argument("--save-as", required=True)
    parser.add_argument("--save-detailed-as", required=True)
    parser.add_argument("--structures-output-path", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--potential-load-path", default=None)
    parser.add_argument("--relax-max-steps", type=int, default=500)
    parser.add_argument("--relax-fmax", type=float, default=0.05)
    parser.add_argument("--max-natoms-per-batch", type=int, default=512)
    parser.add_argument("--relax-failures-json", default=None)
    parser.add_argument("--unsupported-failures-json", default=None)
    parser.add_argument("--metric-errors-json", default=None)
    parser.add_argument(
        "--structure-matcher",
        choices=["ordered", "disordered"],
        default="disordered",
    )
    parser.add_argument("--no-relax", action="store_true")
    args = parser.parse_args()

    matcher: Literal["ordered", "disordered"] = args.structure_matcher
    structure_matcher = (
        DefaultDisorderedStructureMatcher()
        if matcher == "disordered"
        else DefaultOrderedStructureMatcher()
    )

    structures = load_structures(Path(args.structures_path))
    patch_lmdb_metadata_reader()
    reference = LMDBGZSerializer().deserialize(args.reference_dataset)
    supported_indexed_structures, unsupported_records = find_unsupported_structures(
        structures, reference
    )
    supported_indices = [idx for idx, _ in supported_indexed_structures]
    supported_structures = [structure for _, structure in supported_indexed_structures]

    relax_failed_indices = []
    metric_errors = {}
    if args.no_relax:
        metrics = evaluate(
            structures=supported_structures,
            relax=False,
            reference=reference,
            structure_matcher=structure_matcher,
            save_as=args.save_as,
            save_detailed_as=args.save_detailed_as,
            potential_load_path=args.potential_load_path,
            device=args.device,
            structures_output_path=args.structures_output_path,
            energy_correction_scheme=MaterialsProject2020Compatibility(),
        )
    else:
        metrics, relax_failed_indices, metric_errors = evaluate_with_safe_relax(
            structures=supported_structures,
            structure_indices=supported_indices,
            reference=reference,
            structure_matcher=structure_matcher,
            save_as=args.save_as,
            save_detailed_as=args.save_detailed_as,
            potential_load_path=args.potential_load_path,
            device=args.device,
            structures_output_path=args.structures_output_path,
            max_steps=args.relax_max_steps,
            fmax=args.relax_fmax,
            max_natoms_per_batch=args.max_natoms_per_batch,
            n_prefailed_jobs=len(unsupported_records),
            metric_errors_json=args.metric_errors_json,
        )

    if args.relax_failures_json is not None:
        Path(args.relax_failures_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.relax_failures_json).write_text(
            json.dumps(
                {
                    "n_relax_failed": len(relax_failed_indices),
                    "relax_failed_indices": relax_failed_indices,
                },
                indent=2,
                sort_keys=True,
            )
        )
    if args.unsupported_failures_json is not None:
        Path(args.unsupported_failures_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.unsupported_failures_json).write_text(
            json.dumps(
                {
                    "n_unsupported_failed": len(unsupported_records),
                    "unsupported_records": unsupported_records,
                },
                indent=2,
                sort_keys=True,
            )
        )

    sun = metrics.get("frac_novel_unique_stable_structures")
    sun_thresholds = compute_sun_threshold_summary(
        args.save_detailed_as,
        total_submitted=len(structures),
    )
    summary = {
        "structures_path": args.structures_path,
        "reference_dataset": args.reference_dataset,
        "structure_matcher": matcher,
        "relax": not args.no_relax,
        "device": args.device,
        "num_structures": len(structures),
        "num_supported_structures_before_relax": len(supported_structures),
        "n_unsupported_failed": len(unsupported_records),
        "unsupported_records": unsupported_records,
        "relax_max_steps": None if args.no_relax else args.relax_max_steps,
        "relax_fmax": None if args.no_relax else args.relax_fmax,
        "max_natoms_per_batch": None if args.no_relax else args.max_natoms_per_batch,
        "n_relax_failed": len(relax_failed_indices),
        "relax_failed_indices": relax_failed_indices,
        "metric_errors": metric_errors,
        "frac_novel_unique_stable_structures": sun,
        "sun_percent": None if sun is None else 100.0 * float(sun),
        "sun_thresholds": sun_thresholds,
        "metrics": metrics,
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
