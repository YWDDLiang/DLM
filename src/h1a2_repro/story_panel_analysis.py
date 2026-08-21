"""Pure helpers for E1 rich-Plan adherence and multiplicity analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import exp, log
from typing import Any, Mapping, Sequence


def effective_multiplicity(labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    counts = Counter(int(label) for label in labels)
    total = len(labels)
    return exp(-sum((count / total) * log(count / total) for count in counts.values()))


def summarize_story_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row.get("plan_source")), str(row.get("arm")))].append(row)
    result: dict[str, Any] = {}
    boolean_fields = (
        "parsed",
        "plan_match",
        "graph_success",
        "lattice_legal",
        "duplicate_free",
        "plan_lattice_match",
        "plan_spacegroup_match",
        "plan_volume_match",
    )
    for (source, arm), rows in sorted(groups.items()):
        payload: dict[str, Any] = {"requested": len(rows)}
        for field in boolean_fields:
            known = [bool(row[field]) for row in rows if row.get(field) is not None]
            payload[field] = {
                "known": len(known),
                "true": sum(known),
                "rate": None if not known else sum(known) / len(known),
            }
        payload["model_forward_calls"] = sum(
            int(row.get("model_forward_calls") or 0) for row in rows
        )
        result[f"{source}/{arm}"] = payload
    return result


def summarize_plan_clusters(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row.get("plan_id")), str(row.get("arm")))].append(row)
    result: dict[str, Any] = {}
    for (plan_id, arm), rows in sorted(groups.items()):
        labels = [int(row["structure_cluster"]) for row in rows if row.get("structure_cluster") is not None]
        fingerprints = {
            str(row["local_environment_fingerprint"])
            for row in rows
            if row.get("local_environment_fingerprint") is not None
        }
        result[f"{plan_id}/{arm}"] = {
            "requested": len(rows),
            "clustered": len(labels),
            "structure_clusters": len(set(labels)),
            "effective_multiplicity": effective_multiplicity(labels),
            "local_environment_fingerprints": len(fingerprints),
        }
    return result


def multiplicity_gate(plan_clusters: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    full = [payload for key, payload in plan_clusters.items() if key.endswith("/full")]
    eligible = [payload for payload in full if int(payload.get("clustered", 0)) > 0]
    single = sum(int(payload.get("structure_clusters", 0)) <= 1 for payload in eligible)
    rate = None if not eligible else single / len(eligible)
    return {
        "eligible_full_plans": len(eligible),
        "single_cluster_plans": single,
        "single_cluster_rate": rate,
        "remove_multiple_realizations_claim": rate is not None and rate >= 0.75,
    }


__all__ = [
    "effective_multiplicity",
    "multiplicity_gate",
    "summarize_plan_clusters",
    "summarize_story_records",
]
