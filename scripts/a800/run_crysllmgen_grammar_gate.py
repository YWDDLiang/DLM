#!/usr/bin/env python3
"""Run the preregistered one-million-transition WQ grammar audit on Slurm CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from crystal_dlm.wqcodiff.charts import LatticeChartCodec, PyXtalChartCatalog
from crystal_dlm.wqcodiff.crysllmgen.wq_text import (
    HEX_BINS,
    LATTICE_COMPONENT_QUANTIZERS,
    audit_synthetic_grammar_transitions,
    crystal_system_for_space_group,
    parse_wq_proposal,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_environment() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("grammar formal gate must run through Slurm")
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be exactly 1")


def _contract_checks(contract: dict[str, object]) -> list[str]:
    errors: list[str] = []
    proposal = contract.get("proposal", {})
    assert isinstance(proposal, dict)
    continuous = proposal.get("continuous_code", {})
    assert isinstance(continuous, dict)
    lattice = continuous.get("lattice_parameters", {})
    assert isinstance(lattice, dict)
    expected = {
        "log_length": {"max_abs": 20.0, "mu": 63.0},
        "triclinic_off_diagonal": {"max_abs": 256.0, "mu": 127.0},
        "monoclinic_beta_logit": {"max_abs": 30.0, "mu": 63.0},
    }
    if contract.get("schema") != "crysllmgen_wq_grammar_v1":
        errors.append("schema_mismatch")
    if int(continuous.get("bins", -1)) != HEX_BINS:
        errors.append("hex_bin_count_mismatch")
    if lattice != expected:
        errors.append("lattice_compander_mismatch")
    if proposal.get("initial_mask_semantics") is not False:
        errors.append("initial_mask_enabled")
    if proposal.get("semantic_padding_or_null_canvas") is not False:
        errors.append("semantic_padding_enabled")
    if proposal.get("species_encoding") != (
        "one_based_index_into_frozen_MP20_89_species_vocabulary"
    ):
        errors.append("species_vocabulary_encoding_mismatch")
    observed_quantizers = {
        "log_length": {
            "max_abs": LATTICE_COMPONENT_QUANTIZERS["cubic"][0][0],
            "mu": LATTICE_COMPONENT_QUANTIZERS["cubic"][0][1],
        },
        "triclinic_off_diagonal": {
            "max_abs": LATTICE_COMPONENT_QUANTIZERS["triclinic"][3][0],
            "mu": LATTICE_COMPONENT_QUANTIZERS["triclinic"][3][1],
        },
        "monoclinic_beta_logit": {
            "max_abs": LATTICE_COMPONENT_QUANTIZERS["monoclinic"][3][0],
            "mu": LATTICE_COMPONENT_QUANTIZERS["monoclinic"][3][1],
        },
    }
    if observed_quantizers != expected:
        errors.append("runtime_quantizer_mismatch")
    return errors


def _catalog_and_roundtrip(catalog: PyXtalChartCatalog) -> dict[str, object]:
    wyckoff_positions = 0
    failed_space_groups: list[dict[str, object]] = []
    for space_group in range(1, 231):
        types = tuple(int(value) for value in catalog.types(space_group))
        wyckoff_positions += len(types)
        fitting = [
            catalog.get(space_group, value)
            for value in types
            if int(catalog.get(space_group, value).primitive_multiplicity) <= 20
        ]
        if not fitting:
            failed_space_groups.append(
                {"space_group": space_group, "reason": "no_mp20_wyckoff_support"}
            )
            continue
        spec = min(
            fitting,
            key=lambda value: (
                int(value.primitive_multiplicity),
                value.wyckoff_type,
            ),
        )
        system = crystal_system_for_space_group(space_group)
        lattice_codes = ",".join("80" for _ in range(LatticeChartCodec.dimension(system)))
        free_codes = "-" if spec.dimension == 0 else ",".join("80" for _ in range(spec.dimension))
        text = (
            f"SG={space_group};Q={lattice_codes};"
            f"O=0,W={spec.wyckoff_type},E=14,U={free_codes};STOP"
        )
        try:
            recovered = parse_wq_proposal(text, catalog)
            if recovered.space_group != space_group or recovered.atom_count > 20:
                raise ValueError("roundtrip semantic mismatch")
        except Exception as exc:
            failed_space_groups.append(
                {
                    "space_group": space_group,
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
    return {
        "space_groups": 230,
        "wyckoff_positions": wyckoff_positions,
        "roundtrip_passed": 230 - len(failed_space_groups),
        "failed_space_groups": failed_space_groups,
    }


def _write_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require_environment()
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    errors = _contract_checks(contract)
    formal = contract["formal_audit"]
    catalog = PyXtalChartCatalog(hall_style="spglib")
    catalog_report = _catalog_and_roundtrip(catalog)
    transition_report = audit_synthetic_grammar_transitions(
        catalog,
        transitions=int(formal["synthetic_transitions"]),
        seed=int(formal["seed"]),
    )
    if catalog_report["space_groups"] != int(formal["space_groups_required"]):
        errors.append("space_group_coverage_mismatch")
    if catalog_report["wyckoff_positions"] != int(formal["wyckoff_positions_required"]):
        errors.append("wyckoff_position_count_mismatch")
    if catalog_report["roundtrip_passed"] != int(
        formal["parser_roundtrip_space_groups_required"]
    ):
        errors.append("parser_roundtrip_space_group_failure")
    if not transition_report["passed"]:
        errors.append("synthetic_transition_failure")
    if transition_report["illegal_generated"] != int(formal["illegal_generated_max"]):
        errors.append("illegal_generated_above_limit")
    payload = {
        "schema": "crysllmgen_wq_grammar_gate_report_v1",
        "ok": not errors,
        "errors": errors,
        "contract_path": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "runtime_source_sha256": _sha256(Path(__file__).resolve()),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "catalog": catalog_report,
        "transitions": transition_report,
        "thread_count": 1,
        "offline": True,
    }
    _write_exclusive(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["ok"]:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
