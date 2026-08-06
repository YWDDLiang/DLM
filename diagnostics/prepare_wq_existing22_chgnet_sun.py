#!/usr/bin/env python3
"""Prepare the fixed all-22 projection panel for exact CHGNet R5-C S.U.N.

The five structures that failed the already-completed CrysLLMGen structural
gate are represented as failed attempts.  They remain in the denominator but
are never passed to CHGNet as an implicit geometry-repair operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_SCHEMA = "wqcodiff_existing22_chgnet_sun_contract_v1"
METHOD = "wq_existing22_projection_user_accepted_v1"


class Existing22SunInputError(RuntimeError):
    """The frozen existing-22 evaluation input contract was violated."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Existing22SunInputError(f"{path} is not a JSON object")
    return value


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Existing22SunInputError(
                    f"{path}:{line_number} is not a JSON object"
                )
            rows.append(value)
    return rows


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _validate_lower_hex_sha(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise Existing22SunInputError(f"{label} is not a lowercase SHA256")
    return text


def _expect_file(path: Path, expected_sha256: Any, label: str) -> None:
    if not path.is_file():
        raise Existing22SunInputError(f"{label} is missing: {path}")
    expected = _validate_lower_hex_sha(expected_sha256, label)
    observed = sha256_file(path)
    if observed != expected:
        raise Existing22SunInputError(
            f"{label} changed: expected={expected}, observed={observed}"
        )


def validate_contract(
    contract: Mapping[str, Any],
    *,
    contract_path: Path | None = None,
) -> dict[str, Path]:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise Existing22SunInputError("invalid existing-22 S.U.N. contract")
    if int(contract["denominator"]["attempts"]) != 22:
        raise Existing22SunInputError("all-22 denominator changed")
    if int(contract["denominator"]["reconstructed_structures_exact"]) != 17:
        raise Existing22SunInputError("reconstructed-structure count changed")
    resources = contract["resources"]
    a800 = int(resources["a800"])
    cpus = int(resources["cpus"])
    if a800 != 1 or cpus != 8 or cpus > 8 * a800:
        raise Existing22SunInputError("CPU/A800 hard resource policy changed")
    scope = contract["scope"]
    forbidden_true = (
        "new_generation",
        "candidate_reselection",
        "geometry_modification",
        "training",
        "external_api",
        "retry_or_replacement",
        "overwrite",
    )
    if any(bool(scope[key]) for key in forbidden_true):
        raise Existing22SunInputError("evaluation-only scope changed")
    if int(scope["slurm_jobs"]) != 1 or int(scope["scientific_call_limit"]) != 1:
        raise Existing22SunInputError("single-job/single-call contract changed")

    authorization = resolve_project_path(contract["authorization"]["path"])
    _expect_file(
        authorization,
        contract["authorization"]["sha256"],
        "authorization record",
    )
    source = resolve_project_path(contract["inputs"]["directory"])
    paths = {
        "authorization": authorization,
        "source": source,
    }
    for name in (
        "claim",
        "structures",
        "attempt_metrics",
        "report",
        "terminal_acceptance",
    ):
        entry = contract["inputs"][name]
        path = source / str(entry["path"])
        _expect_file(path, entry["sha256"], f"source {name}")
        paths[name] = path
    if contract_path is not None and not contract_path.is_file():
        raise Existing22SunInputError("contract path is missing")
    return paths


def build_attempt_rows(
    contract: Mapping[str, Any],
    structures: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    denominator = contract["denominator"]
    expected_ordinals = [int(value) for value in denominator["expected_ordinals"]]
    joint_valid_ordinals = {
        int(value) for value in denominator["joint_valid_ordinals"]
    }
    failed_ordinals = {
        int(value)
        for value in denominator["frozen_structural_failure_ordinals"]
    }
    if (
        len(expected_ordinals) != 22
        or len(set(expected_ordinals)) != 22
        or joint_valid_ordinals & failed_ordinals
        or joint_valid_ordinals | failed_ordinals != set(expected_ordinals)
        or len(joint_valid_ordinals) != 17
        or len(failed_ordinals) != 5
    ):
        raise Existing22SunInputError("registered ordinal partition is invalid")
    if len(structures) != 22 or len(metrics) != 22:
        raise Existing22SunInputError("source row count changed")

    generation: list[dict[str, Any]] = []
    adapter_records: list[dict[str, Any]] = []
    attempt_ids: set[str] = set()
    for generation_ordinal, expected_ordinal in enumerate(expected_ordinals):
        structure_row = dict(structures[generation_ordinal])
        metric_row = dict(metrics[generation_ordinal])
        attempt_id = str(structure_row.get("attempt_id", ""))
        ordinal = int(structure_row.get("ordinal", -1))
        if (
            not attempt_id
            or attempt_id in attempt_ids
            or ordinal != expected_ordinal
            or metric_row.get("attempt_id") != attempt_id
            or int(metric_row.get("ordinal", -1)) != ordinal
        ):
            raise Existing22SunInputError("source attempt identity/order changed")
        attempt_ids.add(attempt_id)
        if (
            structure_row.get("status") != "succeeded"
            or not isinstance(structure_row.get("structure"), Mapping)
            or bool(structure_row.get("retry_or_replacement_used"))
            or bool(metric_row.get("retry_or_replacement_used"))
            or metric_row.get("comp_valid") is not True
            or metric_row.get("fingerprint_valid") is not True
        ):
            raise Existing22SunInputError(
                f"source rendering/composition identity changed for ordinal {ordinal}"
            )
        source_structure = dict(structure_row["structure"])
        source_structure_sha256 = stable_mapping_sha256(source_structure)
        common = {
            "schema": "wqcodiff_existing22_chgnet_sun_attempt_v1",
            "attempt_id": attempt_id,
            "method": METHOD,
            "ordinal": ordinal,
            "generation_ordinal": generation_ordinal,
            "source": "existing_fixed_topology_projection_v2_order_alignment",
            "projected_formula": metric_row.get("projected_formula"),
            "source_structure_sha256": source_structure_sha256,
            "retry_or_replacement_used": False,
        }
        if ordinal in joint_valid_ordinals:
            if (
                metric_row.get("struct_valid") is not True
                or metric_row.get("valid") is not True
                or metric_row.get("reason") not in ("", None)
            ):
                raise Existing22SunInputError(
                    f"joint-valid identity changed for ordinal {ordinal}"
                )
            attempt = {
                **common,
                "status": "succeeded",
                "reason": "",
                "structure": source_structure,
            }
        else:
            if (
                metric_row.get("struct_valid") is not False
                or metric_row.get("valid") is not False
                or metric_row.get("reason") != "structure_invalid"
            ):
                raise Existing22SunInputError(
                    f"frozen structural-failure identity changed for ordinal {ordinal}"
                )
            attempt = {
                **common,
                "status": "failed",
                "reason": "frozen_pre_chgnet_structure_invalid",
            }
        generation.append(attempt)
        adapter_records.append(
            {
                "attempt_id": attempt_id,
                "ordinal": ordinal,
                "generation_ordinal": generation_ordinal,
                "status": attempt["status"],
                "reason": attempt["reason"],
                "source_structure_sha256": source_structure_sha256,
            }
        )

    manifest = {
        "schema": "wqcodiff_existing22_chgnet_sun_adapter_manifest_v1",
        "method": METHOD,
        "attempts": 22,
        "reconstructed_structures": 17,
        "failed_placeholders": 5,
        "expected_ordinals": expected_ordinals,
        "joint_valid_ordinals": sorted(joint_valid_ordinals),
        "frozen_structural_failure_ordinals": sorted(failed_ordinals),
        "failure_handling": denominator["frozen_failure_handling"],
        "attempt_records": adapter_records,
        "new_generation": False,
        "geometry_repair_or_rescue": False,
        "retry_or_replacement_used": False,
    }
    return generation, manifest


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_exclusive(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def execute(contract_path: Path, output_directory: Path) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = load_json(contract_path)
    paths = validate_contract(contract, contract_path=contract_path)
    generation, manifest = build_attempt_rows(
        contract,
        load_jsonl(paths["structures"]),
        load_jsonl(paths["attempt_metrics"]),
    )
    output = output_directory.resolve()
    if not output.is_dir():
        raise Existing22SunInputError("output directory must already exist")
    generation_path = output / str(contract["output"]["adapter_generation"])
    manifest_path = output / str(contract["output"]["adapter_manifest"])
    if generation_path.exists() or manifest_path.exists():
        raise FileExistsError("adapter output identity already exists")
    write_jsonl_exclusive(generation_path, generation)
    manifest.update(
        {
            "contract": str(contract_path),
            "contract_sha256": sha256_file(contract_path),
            "authorization_record_sha256": sha256_file(paths["authorization"]),
            "source_structures_sha256": sha256_file(paths["structures"]),
            "source_attempt_metrics_sha256": sha256_file(
                paths["attempt_metrics"]
            ),
            "adapter_generation": str(generation_path),
            "adapter_generation_sha256": sha256_file(generation_path),
        }
    )
    write_json_exclusive(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.contract, args.output_dir)
    print("WQ_EXISTING22_SUN_ADAPTER=PASS")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
