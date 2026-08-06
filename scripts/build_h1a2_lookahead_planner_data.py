#!/usr/bin/env python3
"""Build the frozen common P-control/P* stream from original H1-A2 rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.h1a2_planner_objective import (  # noqa: E402
    LOOKAHEAD_FIELDS,
    build_lookahead_vocabs,
    plan_values,
)


SELECTION_SCHEMA = "h1a2_lookahead_planner_stream_v1"
ELEMENT_SYMBOLS = frozenset(
    """
    H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe
    Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In
    Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf
    Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm
    Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
    """.split()
)
FORMULA_TOKEN_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def formula_shape(formula: str) -> tuple[int, int]:
    """Return unique-element arity and total atoms for a flat integer formula."""

    compact = re.sub(r"\s+", "", str(formula))
    if not compact:
        raise ValueError("composition plan formula is empty")
    tokens = FORMULA_TOKEN_PATTERN.findall(compact)
    reconstructed = "".join(symbol + count for symbol, count in tokens)
    if not tokens or reconstructed != compact:
        raise ValueError(
            f"composition plan formula {compact!r} is not a flat integer-count formula"
        )
    counts: Counter[str] = Counter()
    for symbol, count_text in tokens:
        if symbol not in ELEMENT_SYMBOLS:
            raise ValueError(f"unknown element symbol {symbol!r} in formula {compact!r}")
        count = int(count_text) if count_text else 1
        if count <= 0:
            raise ValueError(
                f"element count for {symbol!r} must be positive, observed {count}"
            )
        counts[symbol] += count
    return len(counts), int(sum(counts.values()))


def read_source_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for source_row, raw_line in enumerate(handle):
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line.decode("utf-8"))
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{source_row + 1} is not a JSON object")
            if "answer" not in payload:
                raise ValueError(f"{path}:{source_row + 1} has no answer")
            values = plan_values(str(payload["answer"]).strip())
            arity, num_atoms = formula_shape(values["formula"])
            if not 1 <= num_atoms <= 20:
                raise ValueError(
                    f"{path}:{source_row + 1} has N={num_atoms}, outside [1,20]"
                )
            row = dict(payload)
            row["_selection"] = {
                "source_row": int(source_row),
                "source_line_sha256": sha256_bytes(raw_line),
                "num_atoms": num_atoms,
                "arity": arity,
                "field_values": values,
            }
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} has no non-empty rows")
    return rows


def stratum_key(row: dict[str, Any]) -> tuple[Any, ...]:
    meta = row["_selection"]
    values = meta["field_values"]
    return (
        int(meta["num_atoms"]),
        int(meta["arity"]),
        *(str(values[field]) for field in LOOKAHEAD_FIELDS),
    )


def selection_score(row: dict[str, Any], *, seed: int, split: str) -> str:
    source_sha = str(row["_selection"]["source_line_sha256"])
    return sha256_bytes(
        f"{SELECTION_SCHEMA}:{int(seed)}:{split}:{source_sha}".encode("utf-8")
    )


def proportional_quotas(
    group_sizes: dict[tuple[Any, ...], int],
    *,
    target_count: int,
) -> dict[tuple[Any, ...], int]:
    total = int(sum(group_sizes.values()))
    target = int(target_count)
    if target < 1 or target > total:
        raise ValueError(f"target_count={target} must be within [1,{total}]")
    quotas: dict[tuple[Any, ...], int] = {}
    remainders: list[tuple[float, tuple[Any, ...]]] = []
    assigned = 0
    for key in sorted(group_sizes):
        exact = target * int(group_sizes[key]) / total
        floor_value = min(int(group_sizes[key]), int(exact))
        quotas[key] = floor_value
        assigned += floor_value
        remainders.append((exact - floor_value, key))
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if assigned >= target:
            break
        if quotas[key] >= int(group_sizes[key]):
            continue
        quotas[key] += 1
        assigned += 1
    if assigned != target:
        raise RuntimeError(
            f"largest-remainder allocation produced {assigned}, expected {target}"
        )
    return quotas


def select_rows(
    rows: list[dict[str, Any]],
    *,
    target_count: int,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[stratum_key(row)].append(row)
    quotas = proportional_quotas(
        {key: len(values) for key, values in grouped.items()},
        target_count=target_count,
    )
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        candidates = sorted(
            grouped[key],
            key=lambda row: (
                selection_score(row, seed=seed, split=split),
                int(row["_selection"]["source_row"]),
            ),
        )
        selected.extend(candidates[: quotas[key]])
    selected.sort(
        key=lambda row: (
            selection_score(row, seed=seed, split=split),
            int(row["_selection"]["source_row"]),
        )
    )
    if len(selected) != int(target_count):
        raise RuntimeError(
            f"selected {len(selected)} rows, expected {int(target_count)}"
        )
    source_shas = [
        str(row["_selection"]["source_line_sha256"]) for row in selected
    ]
    if len(source_shas) != len(set(source_shas)):
        raise RuntimeError("selection contains duplicate source-line identities")
    return selected


def public_row(
    row: dict[str, Any],
    *,
    split: str,
    stream_index: int,
    seed: int,
) -> dict[str, Any]:
    output = {key: value for key, value in row.items() if key != "_selection"}
    meta = row["_selection"]
    output["v3_planner_stream"] = {
        "schema": SELECTION_SCHEMA,
        "split": split,
        "stream_index": int(stream_index),
        "source_row": int(meta["source_row"]),
        "source_line_sha256": str(meta["source_line_sha256"]),
        "selection_score": selection_score(row, seed=seed, split=split),
        "num_atoms": int(meta["num_atoms"]),
        "arity": int(meta["arity"]),
        "stratum": list(stratum_key(row)),
        "eligible_arms": ["P-control", "Pstar"],
    }
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def marginal_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counters: dict[str, Counter[str]] = {
        "num_atoms": Counter(),
        "arity": Counter(),
        **{field: Counter() for field in LOOKAHEAD_FIELDS},
    }
    for row in rows:
        meta = row["_selection"]
        counters["num_atoms"][str(meta["num_atoms"])] += 1
        counters["arity"][str(meta["arity"])] += 1
        for field in LOOKAHEAD_FIELDS:
            counters[field][str(meta["field_values"][field])] += 1
    return {
        field: dict(sorted(counter.items())) for field, counter in counters.items()
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    train_source = source_dir / "train.jsonl"
    val_source = source_dir / "val.jsonl"
    for path in (train_source, val_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    observed_train_sha = sha256_file(train_source)
    observed_val_sha = sha256_file(val_source)
    if observed_train_sha != str(args.train_sha256):
        raise ValueError(
            f"train SHA mismatch: expected {args.train_sha256}, observed {observed_train_sha}"
        )
    if observed_val_sha != str(args.val_sha256):
        raise ValueError(
            f"val SHA mismatch: expected {args.val_sha256}, observed {observed_val_sha}"
        )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    train_rows = read_source_rows(train_source)
    val_rows = read_source_rows(val_source)
    train_selected = select_rows(
        train_rows,
        target_count=int(args.train_count),
        seed=int(args.seed),
        split="train",
    )
    val_selected = select_rows(
        val_rows,
        target_count=int(args.val_count),
        seed=int(args.seed),
        split="val",
    )
    train_public = [
        public_row(row, split="train", stream_index=index, seed=int(args.seed))
        for index, row in enumerate(train_selected)
    ]
    val_public = [
        public_row(row, split="val", stream_index=index, seed=int(args.seed))
        for index, row in enumerate(val_selected)
    ]
    train_output = output_dir / "train.jsonl"
    val_output = output_dir / "val.jsonl"
    write_jsonl(train_output, train_public)
    write_jsonl(val_output, val_public)

    vocabs = build_lookahead_vocabs(
        str(row["answer"]).strip() for row in train_selected
    )
    vocab_path = output_dir / "lookahead_vocabs.json"
    vocab_path.write_text(
        json.dumps(vocabs, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": SELECTION_SCHEMA,
        "seed": int(args.seed),
        "source": {
            "train": {
                "path": str(train_source),
                "rows": len(train_rows),
                "sha256": observed_train_sha,
            },
            "val": {
                "path": str(val_source),
                "rows": len(val_rows),
                "sha256": observed_val_sha,
            },
        },
        "selection": {
            "algorithm": "full_joint_stratum_largest_remainder_then_content_hash_v1",
            "stratum_fields": [
                "num_atoms",
                "arity",
                *LOOKAHEAD_FIELDS,
            ],
            "train_rows": len(train_selected),
            "val_rows": len(val_selected),
            "train_strata": len({stratum_key(row) for row in train_rows}),
            "val_strata": len({stratum_key(row) for row in val_rows}),
            "train_selected_marginals": marginal_counts(train_selected),
            "val_selected_marginals": marginal_counts(val_selected),
            "same_order_for_arms": ["P-control", "Pstar"],
            "replacement": False,
            "validity_filter": False,
            "generated_data": False,
        },
        "output": {
            "train": {
                "path": "train.jsonl",
                "rows": len(train_public),
                "sha256": sha256_file(train_output),
            },
            "val": {
                "path": "val.jsonl",
                "rows": len(val_public),
                "sha256": sha256_file(val_output),
            },
            "lookahead_vocabs": {
                "path": "lookahead_vocabs.json",
                "sha256": sha256_file(vocab_path),
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    success = {
        "schema": SELECTION_SCHEMA,
        "manifest_sha256": sha256_file(manifest_path),
        "status": "complete",
    }
    (output_dir / "_SUCCESS").write_text(
        json.dumps(success, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--val-sha256", required=True)
    parser.add_argument("--train-count", type=int, default=3200)
    parser.add_argument("--val-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
