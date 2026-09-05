#!/usr/bin/env python3
"""Batch-certify deterministic MP20-train PMTR corruption proposals."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.dynamic_crystal import parse_dynamic_answer  # noqa: E402
from crystal_dlm.manifold_corruption import (  # noqa: E402
    CorruptionConfig,
    CrystalGeometry,
    canonical_request_key,
    generate_corruption_proposal,
)
from crystal_dlm.offline_pmtr_certification import (  # noqa: E402
    CertificationConfig,
    EFSMBatchEvaluator,
    certify_corruption_proposals,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            yield value


def request_key_for_source(source: Mapping[str, Any]) -> str:
    if source.get("request_key") is not None:
        return canonical_request_key(source["request_key"])
    split = str(source.get("source_split") or "train")
    return f"{split}:{int(source['source_row_idx'])}"


class CHGNetEFSMEvaluator:
    """Lazy-script adapter from ``CrystalGeometry`` to CHGNet EFSM batches."""

    def __init__(self, model: Any, structure_type: Any) -> None:
        self.model = model
        self.structure_type = structure_type

    def _structure(self, geometry: CrystalGeometry) -> Any:
        return self.structure_type(
            lattice=geometry.lattice,
            species=list(geometry.species),
            coords=geometry.frac_coords,
            coords_are_cartesian=False,
            to_unit_cell=True,
        )

    def evaluate(
        self,
        geometries: Sequence[CrystalGeometry],
        *,
        batch_size: int,
    ) -> Sequence[Mapping[str, Any] | None]:
        structures = [self._structure(geometry) for geometry in geometries]
        output: list[Mapping[str, Any] | None] = []
        for start in range(0, len(structures), int(batch_size)):
            chunk = structures[start : start + int(batch_size)]
            try:
                predicted = self.model.predict_structure(
                    chunk, task="efsm", batch_size=int(batch_size)
                )
                if isinstance(predicted, Mapping):
                    predicted = [predicted]
                values = list(predicted)
                if len(values) != len(chunk):
                    raise RuntimeError("CHGNet changed EFSM batch cardinality")
                output.extend(values)
            except Exception:  # noqa: BLE001 - isolate bad structures after batch failure.
                for structure in chunk:
                    try:
                        output.append(
                            self.model.predict_structure(structure, task="efsm")
                        )
                    except Exception:  # noqa: BLE001 - represented as unknown certificate.
                        output.append(None)
        return output


def load_chgnet_evaluator(device: str) -> EFSMBatchEvaluator:
    """Import CHGNet and pymatgen only in the offline executable path."""

    from chgnet.model.model import CHGNet
    from pymatgen.core import Structure

    model = CHGNet.load(
        use_device=str(device), check_cuda_mem=False, verbose=False
    )
    return CHGNetEFSMEvaluator(model, Structure)


def _source_proposals(
    source: Mapping[str, Any],
    *,
    seed: int,
    config: CorruptionConfig,
) -> list[Any]:
    split = str(source.get("source_split") or "")
    if split not in {"train", "mp20_train"}:
        raise ValueError(f"PMTR certification requires MP20-train, got {split!r}")
    body = str(source["answer"])
    parsed = parse_dynamic_answer(body, strict=True)
    tokens = tuple(str(token) for token in parsed["tokens"])
    if "".join(tokens) != body:
        raise ValueError("clean answer must be canonical separator-free exact 7+4N")
    clean = CrystalGeometry.from_mapping(parsed)
    request_key = request_key_for_source(source)
    return [
        generate_corruption_proposal(
            clean,
            clean_body=body,
            clean_tokens=tokens,
            request_key=request_key,
            proposal_index=proposal_index,
            seed=int(seed),
            config=config,
        )
        for proposal_index in range(int(config.max_proposals))
    ]


def _selected_sources(
    rows: Iterable[Mapping[str, Any]],
    *,
    shard_rank: int,
    shard_count: int,
    limit: int | None,
) -> Iterable[Mapping[str, Any]]:
    if int(shard_count) <= 0 or not 0 <= int(shard_rank) < int(shard_count):
        raise ValueError("shard_rank must lie in 0..shard_count-1")
    if limit is not None and int(limit) < 0:
        raise ValueError("limit must be non-negative")
    emitted = 0
    seen: set[int] = set()
    for row in rows:
        source_idx = int(row["source_row_idx"])
        if source_idx in seen:
            raise ValueError(f"duplicate source_row_idx {source_idx}")
        seen.add(source_idx)
        if source_idx % int(shard_count) != int(shard_rank):
            continue
        if limit is not None and emitted >= int(limit):
            break
        emitted += 1
        yield row


def certify_file(
    *,
    input_path: Path,
    output_dir: Path,
    evaluator: EFSMBatchEvaluator,
    seed: int,
    corruption_config: CorruptionConfig,
    certification_config: CertificationConfig,
    batch_size: int,
    shard_rank: int,
    shard_count: int,
    limit: int | None,
    source_buffer: int = 32,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not 8 <= int(batch_size) <= 16:
        raise ValueError("batch_size must be in 8..16")
    if int(source_buffer) <= 0:
        raise ValueError("source_buffer must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    certificate_path = output_dir / "certificates.jsonl"
    selected = _selected_sources(
        iter_jsonl(input_path),
        shard_rank=int(shard_rank),
        shard_count=int(shard_count),
        limit=limit,
    )
    counters: Counter[str] = Counter()
    source_rows: list[Mapping[str, Any]] = []

    def flush(handle: Any) -> None:
        if not source_rows:
            return
        proposals = [
            proposal
            for source in source_rows
            for proposal in _source_proposals(
                source, seed=int(seed), config=corruption_config
            )
        ]
        records = certify_corruption_proposals(
            proposals,
            evaluator=evaluator,
            corruption_config=corruption_config,
            certification_config=certification_config,
            batch_size=int(batch_size),
        )
        metadata = {
            request_key_for_source(source): (
                int(source["source_row_idx"]),
                str(source["source_split"]),
            )
            for source in source_rows
        }
        if len(metadata) != len(source_rows):
            raise ValueError("selected sources contain duplicate request keys")
        for record in records:
            source_idx, source_split = metadata[str(record["request_key"])]
            output = {
                **record,
                "source_row_idx": source_idx,
                "source_split": source_split,
            }
            handle.write(
                json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            counters["proposals"] += 1
            counters["certified"] += int(bool(record["certified"]))
            if record["failure"] is not None:
                counters["rejected_or_unknown"] += 1
        counters["sources"] += len(source_rows)
        source_rows.clear()

    with certificate_path.open("x", encoding="utf-8", newline="\n") as handle:
        for source in selected:
            source_rows.append(source)
            if len(source_rows) >= int(source_buffer):
                flush(handle)
        flush(handle)

    manifest = {
        "schema": "pmtr_certification_manifest_v1",
        "input": str(input_path),
        "output": certificate_path.name,
        "source_scope": "MP20-train",
        "seed": int(seed),
        "max_proposals": int(corruption_config.max_proposals),
        "batch_size": int(batch_size),
        "probe_fraction": float(certification_config.probe_fraction),
        "corruption": {
            "lattice_log_std": float(corruption_config.lattice_log_std),
            "coordinate_cartesian_std_A": float(
                corruption_config.coordinate_cartesian_std_A
            ),
            "max_logmetric_frobenius": float(
                corruption_config.max_logmetric_frobenius
            ),
            "max_atom_displacement_A": float(
                corruption_config.max_atom_displacement_A
            ),
            "max_delta_energy": float(corruption_config.max_delta_energy),
        },
        "shard": {"rank": int(shard_rank), "count": int(shard_count)},
        "limit": None if limit is None else int(limit),
        "selection": "none; downstream builder chooses first certified proposal",
        "sources": int(counters["sources"]),
        "proposals": int(counters["proposals"]),
        "certified": int(counters["certified"]),
        "rejected_or_unknown": int(counters["rejected_or_unknown"]),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(
    argv: Sequence[str] | None = None,
    *,
    evaluator: EFSMBatchEvaluator | None = None,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--max-proposals", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probe-fraction", type=float, default=0.10)
    parser.add_argument("--max-delta-energy", type=float, default=2.0)
    parser.add_argument("--lattice-log-std", type=float, default=0.055)
    parser.add_argument("--coordinate-std-A", type=float, default=0.18)
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--source-buffer", type=int, default=32)
    args = parser.parse_args(argv)

    corruption_config = CorruptionConfig(
        max_proposals=int(args.max_proposals),
        lattice_log_std=float(args.lattice_log_std),
        coordinate_cartesian_std_A=float(args.coordinate_std_A),
        max_delta_energy=float(args.max_delta_energy),
    )
    certification_config = CertificationConfig(
        probe_fraction=float(args.probe_fraction)
    )
    active_evaluator = (
        evaluator
        if evaluator is not None
        else load_chgnet_evaluator(str(args.device))
    )
    manifest = certify_file(
        input_path=args.input_jsonl.resolve(strict=True),
        output_dir=args.output_dir.resolve(),
        evaluator=active_evaluator,
        seed=int(args.seed),
        corruption_config=corruption_config,
        certification_config=certification_config,
        batch_size=int(args.batch_size),
        shard_rank=int(args.shard_rank),
        shard_count=int(args.shard_count),
        limit=args.limit,
        source_buffer=int(args.source_buffer),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
