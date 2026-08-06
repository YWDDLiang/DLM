"""Freeze the MP20-train-only novelty reference and its protostructure keys."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import ArtifactLedger, write_json_exclusive
from .metrics import matcher_contract_hash, protostructure_key


def build_novelty_reference(
    *,
    train_csv: str | Path,
    output_jsonl: str | Path,
    limit: int | None = None,
) -> dict[str, Any]:
    source = Path(train_csv).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output = ArtifactLedger(output_jsonl, key_fields=("material_id",))
    if output.records():
        raise ValueError("novelty reference is immutable and output is not empty")
    from pymatgen.core import Structure

    rows: list[dict[str, str]] = []
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]
    succeeded = 0
    failed = 0
    locked: list[tuple[str, str, str]] = []
    for row in rows:
        material_id = str(row.get("material_id") or row.get("id") or "").strip()
        if not material_id:
            raise ValueError("train novelty CSV contains a missing material_id")
        source_cif = str(row.get("cif") or row.get("cif.conv") or "")
        try:
            if not source_cif:
                raise ValueError("missing CIF column (expected cif or cif.conv)")
            structure = Structure.from_str(source_cif, fmt="cif")
            cif = structure.to(fmt="cif")
            structure_hash = hashlib.sha256(cif.encode("utf-8")).hexdigest()
            proto_key = protostructure_key(structure)
            output.append(
                {
                    "schema": "wqcodiff_novelty_reference_v1",
                    "material_id": material_id,
                    "status": "succeeded",
                    "structure_hash": structure_hash,
                    "protostructure_key": proto_key,
                    "composition": structure.composition.as_dict(),
                    "structure": structure.as_dict(),
                }
            )
            locked.append((material_id, structure_hash, proto_key))
            succeeded += 1
        except Exception as exc:
            output.append(
                {
                    "schema": "wqcodiff_novelty_reference_v1",
                    "material_id": material_id,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}:{exc}",
                    "source_cif_sha256": hashlib.sha256(
                        source_cif.encode("utf-8")
                    ).hexdigest(),
                }
            )
            failed += 1
    reference_hash = hashlib.sha256(
        "\n".join("|".join(value) for value in sorted(locked)).encode("utf-8")
    ).hexdigest()
    summary = {
        "schema": "wqcodiff_novelty_reference_summary_v1",
        "source": str(source),
        "records": len(rows),
        "succeeded": succeeded,
        "failed": failed,
        "gate_passed": failed == 0,
        "reference_sha256": reference_hash,
        "matcher_contract_sha256": matcher_contract_hash(),
        "output_jsonl": str(Path(output_jsonl).resolve()),
    }
    write_json_exclusive(Path(output_jsonl).with_suffix(".summary.json"), summary)
    return summary


def load_novelty_reference(
    path: str | Path,
) -> tuple[list[Any], frozenset[str], Mapping[str, Any]]:
    from pymatgen.core import Structure

    location = Path(path)
    summary_path = location.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != "wqcodiff_novelty_reference_summary_v1":
        raise ValueError("invalid novelty-reference summary")
    if not summary.get("gate_passed"):
        raise ValueError("novelty reference contains failed train structures")
    structures: list[Any] = []
    keys: set[str] = set()
    locked: list[tuple[str, str, str]] = []
    with location.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("schema") != "wqcodiff_novelty_reference_v1":
                raise ValueError("invalid novelty-reference record")
            if payload.get("status") != "succeeded":
                raise ValueError("failed novelty-reference record in frozen artifact")
            structures.append(Structure.from_dict(payload["structure"]))
            keys.add(str(payload["protostructure_key"]))
            locked.append(
                (
                    str(payload["material_id"]),
                    str(payload["structure_hash"]),
                    str(payload["protostructure_key"]),
                )
            )
    digest = hashlib.sha256(
        "\n".join("|".join(value) for value in sorted(locked)).encode("utf-8")
    ).hexdigest()
    if digest != summary.get("reference_sha256"):
        raise ValueError("novelty-reference hash mismatch")
    return structures, frozenset(keys), summary
