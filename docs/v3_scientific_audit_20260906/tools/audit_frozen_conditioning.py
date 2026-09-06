"""Audit frozen sampled conditions against original targets, without model calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

FIELDS = ("lattice_system", "spacegroup_bucket", "volume_per_atom_bin")


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def composition(row):
    plan = row["plan_state"]
    return tuple(sorted(zip(plan["elements"], map(int, plan["counts"]))))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "model_calls": 0,
              "new_generation": False, "source_files": {}, "splits": {},
              "interpretation": "Annotation agreement is not plan utility or causal evidence of harm."}
    identities, answer_hashes = {}, {}
    for split in ("train", "val"):
        original_path, prepared_path = args.original_dir / f"{split}.jsonl", args.plan_dir / f"{split}.jsonl"
        original = {row["source_row_idx"]: row for row in read_rows(original_path)}
        prepared = read_rows(prepared_path)
        if set(original) != {row["source_row_idx"] for row in prepared} or len(original) != len(prepared):
            raise ValueError(f"{split}: original/prepared source identities differ")
        statuses, reasons, goals = Counter(), Counter(), Counter()
        confusion = {field: Counter() for field in FIELDS}
        agrees = Counter()
        counters = Counter()
        compositions, targets = Counter(), defaultdict(set)
        clean_hashes = set()
        predicted_examples = []
        for row in prepared:
            source = original[row["source_row_idx"]]
            evidence = row["condition_prediction"]
            status = evidence["status"]
            statuses[status] += 1
            reasons[str(evidence.get("original_metadata_unavailable_reason"))] += 1
            goals[str(evidence.get("stability_condition"))] += 1
            counters["unchanged_clean_answer"] += row["answer"] == source["answer"]
            counters["unchanged_composition"] += composition(row) == composition(source)
            counters["valid_program_permutation"] += sorted(row["species_program"]) == sorted(row["plan_state"]["elements"])
            key = composition(row)
            compositions[key] += 1
            ah = hashlib.sha256(row["answer"].encode()).hexdigest()
            targets[key].add(ah)
            clean_hashes.add(ah)
            if status == "teacher_soft_fallback":
                continue
            counters["predicted"] += 1
            matching = []
            for field in FIELDS:
                truth, predicted = source["plan_state"][field], row["plan_state"][field]
                confusion[field][(str(truth), str(predicted))] += 1
                same = truth == predicted
                matching.append(same)
                agrees[field] += same
            counters["all_three_soft_match"] += all(matching)
            counters["any_soft_mismatch"] += not all(matching)
            if not all(matching) and len(predicted_examples) < 12:
                predicted_examples.append({"source_row_idx": row["source_row_idx"], "composition": key,
                    "original": {f: source["plan_state"][f] for f in FIELDS},
                    "prepared": {f: row["plan_state"][f] for f in FIELDS}})
        identities[split], answer_hashes[split] = set(compositions), clean_hashes
        result["source_files"][split] = {"original": str(original_path), "prepared": str(prepared_path),
                                        "original_sha256": digest(original_path), "prepared_sha256": digest(prepared_path)}
        result["splits"][split] = {"rows": len(prepared), "statuses": dict(statuses), "reasons": dict(reasons),
            "goals": dict(goals), "checks": dict(counters), "soft_agreement_predicted_only": dict(agrees),
            "confusion_predicted_only": {field: [{"original": a, "prepared": b, "count": n}
                for (a, b), n in sorted(counts.items())] for field, counts in confusion.items()},
            "unique_exact_compositions": len(compositions),
            "composition_multiplicity_histogram": dict(Counter(compositions.values())),
            "compositions_with_multiple_target_strings": sum(len(values) > 1 for values in targets.values()),
            "source_order_examples_of_soft_mismatch": predicted_examples}
    result["train_val_overlap"] = {"exact_compositions": len(identities["train"] & identities["val"]),
        "exact_clean_answer_strings": len(answer_hashes["train"] & answer_hashes["val"]),
        "note": "Answer-string equality is a limited duplicate diagnostic, not full crystallographic structure matching."}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "FROZEN_CONDITIONING_AUDIT.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "splits": {s: {k: v for k, v in item.items()
          if k not in ("confusion_predicted_only", "source_order_examples_of_soft_mismatch")}
          for s, item in result["splits"].items()}, "train_val_overlap": result["train_val_overlap"]}))


if __name__ == "__main__":
    main()
