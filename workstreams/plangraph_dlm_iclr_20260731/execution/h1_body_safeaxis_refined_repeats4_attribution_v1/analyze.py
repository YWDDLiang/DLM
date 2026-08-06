#!/usr/bin/env python3
"""Read-only paired attribution of the frozen R03G S.U.N. endpoints.

This sidecar does not recompute an endpoint. It verifies and reads the frozen
R03G 4-repeat x 2-arm x 256-attempt evidence, then decomposes paired strict and
meta discordances into:

* finite E_hull threshold crossings;
* novel-unique eligibility differences; and
* residual E_hull unknowns.

It also audits whether residual unknowns are paired by ordinal and composition,
describes the frozen E_hull distribution, and writes one report-only sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


REPEATS = (0, 1, 2, 3)
ARMS = ("control", "candidate")
ARM_KEYS = tuple(f"r{repeat}_{arm}" for repeat in REPEATS for arm in ARMS)
STATES = (
    "ineligible",
    "residual_unknown",
    "strict",
    "meta_only",
    "above_meta",
)
ENDPOINTS = ("strict_full_sun", "meta_full_sun")


class AttributionError(RuntimeError):
    """Raised when frozen attribution evidence does not satisfy its contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AttributionError(f"{path}: expected one JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AttributionError(
                    f"{path}:{line_number}: expected one JSON object"
                )
            rows.append(value)
    return rows


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def identity(path: Path) -> dict[str, Any]:
    location = path.resolve()
    return {
        "path": str(location),
        "bytes": location.stat().st_size,
        "sha256": sha256_file(location),
    }


def require_identity(
    base: Path,
    specification: Mapping[str, Any],
    label: str,
    *,
    require_within_base: bool = True,
) -> Path:
    configured = Path(str(specification["path"]))
    location = (
        configured.resolve()
        if configured.is_absolute()
        else (base / configured).resolve()
    )
    if require_within_base and base != location and base not in location.parents:
        raise AttributionError(f"{label} escapes {base}")
    if not location.is_file():
        raise FileNotFoundError(location)
    observed = sha256_file(location)
    expected = str(specification["sha256"])
    if observed != expected:
        raise AttributionError(
            f"{label} changed: expected={expected}, observed={observed}"
        )
    expected_bytes = specification.get("bytes")
    if expected_bytes is not None and location.stat().st_size != int(expected_bytes):
        raise AttributionError(f"{label} byte count changed")
    return location


def require_source_manifest(
    source_dir: Path, expected_manifest_sha256: str
) -> Path:
    manifest = source_dir / "SOURCE_SHA256.txt"
    if sha256_file(manifest) != expected_manifest_sha256:
        raise AttributionError("R03H source manifest identity changed")
    entries: list[tuple[str, str]] = []
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        pieces = line.split("  ", 1)
        if len(pieces) != 2:
            raise AttributionError(f"{manifest}:{line_number}: malformed entry")
        expected, relative = pieces
        if (
            len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise AttributionError(f"{manifest}:{line_number}: unsafe entry")
        entries.append((expected, relative))
    listed = {relative for _, relative in entries}
    observed = {
        path.relative_to(source_dir).as_posix()
        for path in source_dir.rglob("*")
        if path.is_file()
        and path.name != "SOURCE_SHA256.txt"
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    if listed != observed:
        raise AttributionError(
            "R03H source file set changed: "
            f"missing={sorted(listed - observed)}, extra={sorted(observed - listed)}"
        )
    for expected, relative in entries:
        path = source_dir / relative
        if sha256_file(path) != expected:
            raise AttributionError(f"R03H source file changed: {relative}")
    return manifest


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise AttributionError(f"{label} is not finite")
    return result


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise AttributionError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise AttributionError("quantile probability is outside [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    finite = [finite_float(value, "summary value") for value in values]
    if not finite:
        return {
            "count": 0,
            "min": None,
            "q10": None,
            "q25": None,
            "median": None,
            "mean": None,
            "q75": None,
            "q90": None,
            "max": None,
        }
    return {
        "count": len(finite),
        "min": min(finite),
        "q10": quantile(finite, 0.10),
        "q25": quantile(finite, 0.25),
        "median": quantile(finite, 0.50),
        "mean": sum(finite) / len(finite),
        "q75": quantile(finite, 0.75),
        "q90": quantile(finite, 0.90),
        "max": max(finite),
    }


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=str)}


def normalize_composition(value: Mapping[str, Any]) -> dict[str, float]:
    result = {
        str(element): finite_float(amount, f"composition[{element}]")
        for element, amount in value.items()
        if not str(element).startswith("@")
    }
    if not result or any(amount <= 0.0 for amount in result.values()):
        raise AttributionError("composition is empty or nonpositive")
    return dict(sorted(result.items()))


def classify_attempt(
    row: Mapping[str, Any],
    *,
    strict_threshold: float,
    meta_threshold: float,
    composition: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    metrics = row.get("metrics")
    hull = row.get("hull_recompute")
    if not isinstance(metrics, Mapping) or not isinstance(hull, Mapping):
        raise AttributionError("attempt lacks metrics or hull_recompute")
    applicable = hull.get("applicable") is True
    if not applicable:
        state = "ineligible"
        e_hull = None
        chemsys = None
        reason = str(hull.get("reason") or "ineligible")
    else:
        reason = None
        chemsys = str(hull["chemsys"])
        value = hull.get("recomputed_e_above_hull")
        if value is None:
            state = "residual_unknown"
            e_hull = None
        else:
            e_hull = finite_float(value, "recomputed E_hull")
            if e_hull <= strict_threshold:
                state = "strict"
            elif e_hull <= meta_threshold:
                state = "meta_only"
            else:
                state = "above_meta"
    strict = state == "strict"
    meta = state in {"strict", "meta_only"}
    if metrics.get("strict_full_sun") is not strict:
        raise AttributionError("strict endpoint differs from frozen state")
    if metrics.get("meta_full_sun") is not meta:
        raise AttributionError("meta endpoint differs from frozen state")
    if bool(metrics.get("novel_unique")) is not applicable:
        raise AttributionError("novel_unique eligibility differs from hull applicability")
    normalized = normalize_composition(composition) if composition is not None else None
    if applicable and normalized is None:
        raise AttributionError("applicable attempt lacks frozen composition")
    if normalized is not None:
        observed_chemsys = "-".join(sorted(normalized))
        if applicable and observed_chemsys != chemsys:
            raise AttributionError("composition chemical system changed")
    return {
        "attempt_id": str(row["attempt_id"]),
        "generation_ordinal": int(row["generation_ordinal"]),
        "state": state,
        "applicable_novel_unique": applicable,
        "strict_full_sun": strict,
        "meta_full_sun": meta,
        "e_above_hull": e_hull,
        "source_e_above_hull": (
            hull.get("source_e_above_hull") if applicable else None
        ),
        "chemsys": chemsys,
        "composition": normalized,
        "elements": sorted(normalized) if normalized is not None else None,
        "element_count": len(normalized) if normalized is not None else None,
        "atom_count": sum(normalized.values()) if normalized is not None else None,
        "ineligible_reason": reason,
        "generation_status": str(row.get("generation_status") or "unknown"),
        "evaluation_status": str(row.get("evaluation_status") or "unknown"),
    }


def endpoint_positive(state: str, endpoint: str) -> bool:
    if endpoint == "strict_full_sun":
        return state == "strict"
    if endpoint == "meta_full_sun":
        return state in {"strict", "meta_only"}
    raise AttributionError(f"unknown endpoint {endpoint}")


def discordance_mechanism(opposite_state: str) -> str:
    if opposite_state == "ineligible":
        return "novel_unique_eligibility"
    if opposite_state == "residual_unknown":
        return "residual_hull_unknown"
    return "finite_hull_threshold_crossing"


def recurrence_report(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_ordinal: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        by_ordinal[int(event["ordinal"])].append(event)
    recurrence = Counter(len(rows) for rows in by_ordinal.values())
    recurrent = []
    for ordinal, rows in sorted(
        by_ordinal.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        if len(rows) < 2:
            continue
        recurrent.append(
            {
                "ordinal": ordinal,
                "event_count": len(rows),
                "repeats": sorted(int(row["repeat"]) for row in rows),
                "chemsys": sorted(
                    {
                        str(row["chemsys"])
                        for row in rows
                        if row.get("chemsys") is not None
                    }
                ),
            }
        )
    return {
        "event_count": len(events),
        "distinct_ordinals": len(by_ordinal),
        "ordinal_recurrence_histogram": counter_dict(recurrence),
        "recurrent_ordinals": recurrent,
    }


def complexity_report(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    atom_counts = [
        float(event["atom_count"])
        for event in events
        if event.get("atom_count") is not None
    ]
    element_counts = Counter(
        int(event["element_count"])
        for event in events
        if event.get("element_count") is not None
    )
    chemsys = Counter(
        str(event["chemsys"])
        for event in events
        if event.get("chemsys") is not None
    )
    top_chemsys = [
        {"chemsys": key, "events": count}
        for key, count in sorted(chemsys.items(), key=lambda item: (-item[1], item[0]))
        [:20]
    ]
    return {
        "atom_count": numeric_summary(atom_counts),
        "element_count": counter_dict(element_counts),
        "top_chemsys": top_chemsys,
    }


def load_frozen_evidence(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    source_spec = config["source_r03g"]
    source_root = Path(str(source_spec["run_root"])).resolve()
    if not (source_root / str(source_spec["success_marker"])).is_file():
        raise AttributionError("R03G success marker is missing")
    source_manifest = require_identity(
        Path("/"),
        source_spec["source_manifest"],
        "R03G source manifest",
        require_within_base=False,
    )
    if sha256_file(source_manifest) != str(source_spec["source_manifest_sha256"]):
        raise AttributionError("R03G source manifest binding changed")
    terminal_path = require_identity(
        source_root, source_spec["terminal_report"], "R03G terminal report"
    )
    decision_path = require_identity(
        source_root, source_spec["decision"], "R03G decision"
    )
    input_contract_path = require_identity(
        source_root, source_spec["input_contract"], "R03G input contract"
    )
    terminal = read_json(terminal_path)
    decision = read_json(decision_path)
    input_contract = read_json(input_contract_path)
    expected_status = str(source_spec["expected_status"])
    if (
        terminal.get("ok") is not True
        or terminal.get("status") != expected_status
        or terminal.get("arm_count") != 8
        or terminal.get("repeat_count") != 4
        or terminal.get("attempts_per_arm_per_repeat") != 256
        or terminal.get("raw_attempts_total") != 2048
        or terminal.get("denominator") != "raw_all_attempts"
        or decision.get("status") != expected_status
        or decision.get("terminal_report_sha256") != sha256_file(terminal_path)
        or set(terminal.get("arms") or {}) != set(ARM_KEYS)
    ):
        raise AttributionError("R03G terminal identity or paired shape changed")
    execution = terminal.get("execution") or {}
    for name in (
        "network_used",
        "api_key_read",
        "generation_rerun",
        "refinement_rerun",
        "chgnet_rerun",
        "direct_metrics_rerun",
        "novelty_rerun",
        "sample_retry_or_replacement_used",
        "filter_or_rerank_used",
        "slurm_used",
        "gpu_used",
    ):
        if execution.get(name) is not False:
            raise AttributionError(f"R03G execution flag changed: {name}")
    if (
        terminal["snapshot_reuse"]["common_snapshot"]["sha256"]
        != str(source_spec["expected_common_snapshot_sha256"])
        or terminal["source_r03e"]["terminal"]["sha256"]
        != str(source_spec["expected_r03e_terminal_sha256"])
        or terminal["source_r03e"]["evidence_unchanged"] is not True
        or terminal["source_r03f"]["evidence_unchanged"] is not True
        or terminal.get("formal_g3") is not False
        or terminal.get("automatic_promotion") is not False
        or terminal.get("automatic_training") is not False
        or terminal.get("checkpoint_reselection") is not False
        or terminal.get("automatic_downstream") is not False
    ):
        raise AttributionError("R03G source or decision firewall changed")
    source_artifacts = (input_contract.get("source_artifacts") or {}).get("arms")
    if not isinstance(source_artifacts, Mapping) or set(source_artifacts) != set(
        ARM_KEYS
    ):
        raise AttributionError("R03G source-artifact inventory changed")

    strict_threshold = finite_float(
        config["analysis"]["strict_threshold_ev_per_atom"], "strict threshold"
    )
    meta_threshold = finite_float(
        config["analysis"]["meta_threshold_ev_per_atom"], "meta threshold"
    )
    arms: dict[str, list[dict[str, Any]]] = {}
    for arm_key in ARM_KEYS:
        terminal_arm = terminal["arms"][arm_key]
        attempts_path = require_identity(
            source_root,
            terminal_arm["attempt_results"],
            f"{arm_key} completed attempts",
        )
        vectors_path = require_identity(
            source_root,
            terminal_arm["sun_vectors"],
            f"{arm_key} frozen S.U.N. vectors",
        )
        relax_path = require_identity(
            Path("/"),
            source_artifacts[arm_key]["strict_relax_results"],
            f"{arm_key} frozen relaxation rows",
            require_within_base=False,
        )
        meta_relax_path = require_identity(
            Path("/"),
            source_artifacts[arm_key]["meta_relax_results"],
            f"{arm_key} frozen meta relaxation rows",
            require_within_base=False,
        )
        if (
            sha256_file(relax_path) != sha256_file(meta_relax_path)
            or relax_path.read_bytes() != meta_relax_path.read_bytes()
        ):
            raise AttributionError(f"{arm_key} strict/meta relaxation rows differ")
        attempts = read_jsonl(attempts_path)
        vectors = read_json(vectors_path)
        relax_rows = read_jsonl(relax_path)
        if (
            len(attempts) != 256
            or [int(row.get("generation_ordinal", -1)) for row in attempts]
            != list(range(256))
            or vectors.get("ordinals") != list(range(256))
            or len(vectors.get("strict_full_sun") or []) != 256
            or len(vectors.get("meta_full_sun") or []) != 256
            or any(row.get("retry_or_replacement_used") is not False for row in attempts)
        ):
            raise AttributionError(f"{arm_key} frozen ordinal contract changed")
        eligible = [
            row for row in attempts if (row.get("hull_recompute") or {}).get("applicable")
        ]
        if len(eligible) != len(relax_rows):
            raise AttributionError(f"{arm_key} relaxation mapping length changed")
        composition_by_ordinal: dict[int, dict[str, float]] = {}
        for local_index, (attempt, relax) in enumerate(zip(eligible, relax_rows)):
            if int(relax.get("local_index", -1)) != local_index:
                raise AttributionError(f"{arm_key} relaxation local index changed")
            composition_by_ordinal[int(attempt["generation_ordinal"])] = (
                normalize_composition(relax["composition"])
            )
        classified = []
        for ordinal, row in enumerate(attempts):
            item = classify_attempt(
                row,
                strict_threshold=strict_threshold,
                meta_threshold=meta_threshold,
                composition=composition_by_ordinal.get(ordinal),
            )
            if (
                item["strict_full_sun"]
                is not bool(vectors["strict_full_sun"][ordinal])
                or item["meta_full_sun"]
                is not bool(vectors["meta_full_sun"][ordinal])
            ):
                raise AttributionError(f"{arm_key} vector mapping changed")
            classified.append(item)
        if (
            sum(item["strict_full_sun"] for item in classified)
            != int(terminal_arm["completed_counts"]["strict_full_sun"])
            or sum(item["meta_full_sun"] for item in classified)
            != int(terminal_arm["completed_counts"]["meta_full_sun"])
        ):
            raise AttributionError(f"{arm_key} endpoint count changed")
        arms[arm_key] = classified

    source_identities = {
        "source_manifest": identity(source_manifest),
        "terminal_report": identity(terminal_path),
        "decision": identity(decision_path),
        "input_contract": identity(input_contract_path),
        "common_snapshot": dict(terminal["snapshot_reuse"]["common_snapshot"]),
        "r03e_terminal": dict(terminal["source_r03e"]["terminal"]),
    }
    return terminal, source_identities, arms


def build_pairs(
    arms: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    pairs = []
    for repeat in REPEATS:
        control = arms[f"r{repeat}_control"]
        candidate = arms[f"r{repeat}_candidate"]
        if len(control) != 256 or len(candidate) != 256:
            raise AttributionError("paired arm length changed")
        for ordinal, (left, right) in enumerate(zip(control, candidate)):
            if (
                int(left["generation_ordinal"]) != ordinal
                or int(right["generation_ordinal"]) != ordinal
            ):
                raise AttributionError("paired ordinal alignment changed")
            if (
                left["applicable_novel_unique"]
                and right["applicable_novel_unique"]
            ):
                if (
                    left["chemsys"] != right["chemsys"]
                    or left["composition"] != right["composition"]
                    or left["atom_count"] != right["atom_count"]
                    or left["element_count"] != right["element_count"]
                ):
                    raise AttributionError(
                        f"r{repeat} ordinal {ordinal}: paired composition changed"
                    )
            pair_chemsys = left.get("chemsys") or right.get("chemsys")
            atom_count = (
                left.get("atom_count")
                if left.get("atom_count") is not None
                else right.get("atom_count")
            )
            element_count = (
                left.get("element_count")
                if left.get("element_count") is not None
                else right.get("element_count")
            )
            pairs.append(
                {
                    "schema": "h1_r03h_paired_ordinal_record_v1",
                    "repeat": repeat,
                    "ordinal": ordinal,
                    "chemsys": pair_chemsys,
                    "atom_count": atom_count,
                    "element_count": element_count,
                    "control": dict(left),
                    "candidate": dict(right),
                    "state_transition": f"{left['state']}->{right['state']}",
                    "strict_transition": (
                        f"{int(bool(left['strict_full_sun']))}"
                        f"->{int(bool(right['strict_full_sun']))}"
                    ),
                    "meta_transition": (
                        f"{int(bool(left['meta_full_sun']))}"
                        f"->{int(bool(right['meta_full_sun']))}"
                    ),
                }
            )
    if len(pairs) != 1024:
        raise AttributionError("paired record count changed")
    return pairs


def build_endpoint_report(
    pairs: Sequence[Mapping[str, Any]], endpoint: str
) -> dict[str, Any]:
    candidate_only: list[dict[str, Any]] = []
    control_only: list[dict[str, Any]] = []
    both = neither = 0
    per_repeat: dict[int, Counter[str]] = {
        repeat: Counter() for repeat in REPEATS
    }
    for pair in pairs:
        control_state = str(pair["control"]["state"])
        candidate_state = str(pair["candidate"]["state"])
        control_positive = endpoint_positive(control_state, endpoint)
        candidate_positive = endpoint_positive(candidate_state, endpoint)
        repeat = int(pair["repeat"])
        if control_positive and candidate_positive:
            both += 1
            per_repeat[repeat]["both"] += 1
        elif candidate_positive:
            event = {
                "repeat": repeat,
                "ordinal": int(pair["ordinal"]),
                "chemsys": pair.get("chemsys"),
                "atom_count": pair.get("atom_count"),
                "element_count": pair.get("element_count"),
                "opposite_state": control_state,
                "mechanism": discordance_mechanism(control_state),
                "control_e_above_hull": pair["control"].get("e_above_hull"),
                "candidate_e_above_hull": pair["candidate"].get("e_above_hull"),
            }
            candidate_only.append(event)
            per_repeat[repeat]["candidate_only"] += 1
        elif control_positive:
            event = {
                "repeat": repeat,
                "ordinal": int(pair["ordinal"]),
                "chemsys": pair.get("chemsys"),
                "atom_count": pair.get("atom_count"),
                "element_count": pair.get("element_count"),
                "opposite_state": candidate_state,
                "mechanism": discordance_mechanism(candidate_state),
                "control_e_above_hull": pair["control"].get("e_above_hull"),
                "candidate_e_above_hull": pair["candidate"].get("e_above_hull"),
            }
            control_only.append(event)
            per_repeat[repeat]["control_only"] += 1
        else:
            neither += 1
            per_repeat[repeat]["neither"] += 1

    candidate_mechanisms = Counter(event["mechanism"] for event in candidate_only)
    control_mechanisms = Counter(event["mechanism"] for event in control_only)
    mechanism_names = sorted(set(candidate_mechanisms) | set(control_mechanisms))
    mechanism_net = {
        name: int(candidate_mechanisms[name] - control_mechanisms[name])
        for name in mechanism_names
    }
    control_count = both + len(control_only)
    candidate_count = both + len(candidate_only)
    return {
        "control_count": control_count,
        "candidate_count": candidate_count,
        "candidate_minus_control": candidate_count - control_count,
        "both_positive": both,
        "neither_positive": neither,
        "candidate_only": len(candidate_only),
        "control_only": len(control_only),
        "discordant": len(candidate_only) + len(control_only),
        "candidate_only_by_opposite_state": counter_dict(
            Counter(event["opposite_state"] for event in candidate_only)
        ),
        "control_only_by_opposite_state": counter_dict(
            Counter(event["opposite_state"] for event in control_only)
        ),
        "candidate_only_by_mechanism": counter_dict(candidate_mechanisms),
        "control_only_by_mechanism": counter_dict(control_mechanisms),
        "net_effect_by_mechanism": mechanism_net,
        "per_repeat": {
            str(repeat): counter_dict(per_repeat[repeat]) for repeat in REPEATS
        },
        "candidate_only_recurrence": recurrence_report(candidate_only),
        "control_only_recurrence": recurrence_report(control_only),
        "candidate_only_complexity": complexity_report(candidate_only),
        "control_only_complexity": complexity_report(control_only),
        "_candidate_only_events": candidate_only,
        "_control_only_events": control_only,
    }


def build_attribution(
    terminal: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    *,
    meta_threshold: float,
    descriptive_margin: float,
) -> dict[str, Any]:
    transition = Counter(str(pair["state_transition"]) for pair in pairs)
    state_counts = {
        arm: Counter(str(pair[arm]["state"]) for pair in pairs) for arm in ARMS
    }
    endpoint_reports = {
        endpoint: build_endpoint_report(pairs, endpoint) for endpoint in ENDPOINTS
    }
    for endpoint, report in endpoint_reports.items():
        frozen = terminal["pooled_candidate_minus_control"][endpoint]
        mcnemar = frozen["descriptive_pooled_mcnemar"]
        if (
            report["control_count"] != int(frozen["control_count"])
            or report["candidate_count"] != int(frozen["candidate_count"])
            or report["candidate_minus_control"]
            != int(frozen["candidate_minus_control_count"])
            or report["candidate_only"] != int(mcnemar["candidate_only"])
            or report["control_only"] != int(mcnemar["baseline_only"])
        ):
            raise AttributionError(f"{endpoint} attribution differs from R03G")
        report["frozen_r03g_mcnemar"] = dict(mcnemar)
        report["frozen_r03g_hierarchical_paired_bootstrap_95ci"] = list(
            frozen["hierarchical_paired_bootstrap_95ci"]
        )

    unknown_by_arm: dict[str, list[Mapping[str, Any]]] = {
        arm: [
            pair
            for pair in pairs
            if str(pair[arm]["state"]) == "residual_unknown"
        ]
        for arm in ARMS
    }
    unknown_per_repeat = {}
    for repeat in REPEATS:
        control = sorted(
            int(pair["ordinal"])
            for pair in unknown_by_arm["control"]
            if int(pair["repeat"]) == repeat
        )
        candidate = sorted(
            int(pair["ordinal"])
            for pair in unknown_by_arm["candidate"]
            if int(pair["repeat"]) == repeat
        )
        unknown_per_repeat[str(repeat)] = {
            "control_ordinals": control,
            "candidate_ordinals": candidate,
            "both_unknown": sorted(set(control) & set(candidate)),
            "control_only_unknown": sorted(set(control) - set(candidate)),
            "candidate_only_unknown": sorted(set(candidate) - set(control)),
            "identical": control == candidate,
        }
    control_unknown_chemsys = Counter(
        str(pair["control"]["chemsys"]) for pair in unknown_by_arm["control"]
    )
    candidate_unknown_chemsys = Counter(
        str(pair["candidate"]["chemsys"]) for pair in unknown_by_arm["candidate"]
    )
    distinct_unknown_chemsys = sorted(
        set(control_unknown_chemsys) | set(candidate_unknown_chemsys)
    )
    common_elements = (
        sorted(
            set.intersection(
                *(set(chemsys.split("-")) for chemsys in distinct_unknown_chemsys)
            )
        )
        if distinct_unknown_chemsys
        else []
    )
    unknown_ordinal_sets = [
        tuple(unknown_per_repeat[str(repeat)]["control_ordinals"])
        for repeat in REPEATS
    ]
    residual_unknown = {
        "control_count": len(unknown_by_arm["control"]),
        "candidate_count": len(unknown_by_arm["candidate"]),
        "paired_both_unknown": sum(
            pair["control"]["state"] == "residual_unknown"
            and pair["candidate"]["state"] == "residual_unknown"
            for pair in pairs
        ),
        "control_only_unknown": sum(
            pair["control"]["state"] == "residual_unknown"
            and pair["candidate"]["state"] != "residual_unknown"
            for pair in pairs
        ),
        "candidate_only_unknown": sum(
            pair["candidate"]["state"] == "residual_unknown"
            and pair["control"]["state"] != "residual_unknown"
            for pair in pairs
        ),
        "per_repeat": unknown_per_repeat,
        "identical_ordinals_between_arms_each_repeat": all(
            row["identical"] for row in unknown_per_repeat.values()
        ),
        "identical_ordinals_across_repeats": len(set(unknown_ordinal_sets)) == 1,
        "control_chemsys_frequency": counter_dict(control_unknown_chemsys),
        "candidate_chemsys_frequency": counter_dict(candidate_unknown_chemsys),
        "identical_chemsys_frequency_between_arms": (
            control_unknown_chemsys == candidate_unknown_chemsys
        ),
        "distinct_chemsys": distinct_unknown_chemsys,
        "common_elements_across_all_unknown_chemsys": common_elements,
        "complexity": complexity_report(unknown_by_arm["control"]),
        "strict_effect_contribution": 0,
        "meta_effect_contribution": 0,
    }
    if (
        residual_unknown["control_count"] != 36
        or residual_unknown["candidate_count"] != 36
        or residual_unknown["paired_both_unknown"] != 36
        or residual_unknown["control_only_unknown"] != 0
        or residual_unknown["candidate_only_unknown"] != 0
        or not residual_unknown["identical_ordinals_between_arms_each_repeat"]
        or not residual_unknown["identical_ordinals_across_repeats"]
        or not residual_unknown["identical_chemsys_frequency_between_arms"]
    ):
        raise AttributionError("residual unknowns are not exactly paired")

    finite_values = {
        arm: [
            float(pair[arm]["e_above_hull"])
            for pair in pairs
            if pair[arm].get("e_above_hull") is not None
        ]
        for arm in ARMS
    }
    strict_values = {
        arm: [
            float(pair[arm]["e_above_hull"])
            for pair in pairs
            if pair[arm]["state"] == "strict"
        ]
        for arm in ARMS
    }
    both_finite = [
        pair
        for pair in pairs
        if pair["control"].get("e_above_hull") is not None
        and pair["candidate"].get("e_above_hull") is not None
    ]
    deltas = [
        float(pair["candidate"]["e_above_hull"])
        - float(pair["control"]["e_above_hull"])
        for pair in both_finite
    ]
    meta_crossings = [
        pair
        for pair in both_finite
        if bool(pair["control"]["meta_full_sun"])
        != bool(pair["candidate"]["meta_full_sun"])
    ]
    meta_closest_margins = [
        min(
            abs(float(pair["control"]["e_above_hull"]) - meta_threshold),
            abs(float(pair["candidate"]["e_above_hull"]) - meta_threshold),
        )
        for pair in meta_crossings
    ]
    strict_crossings = [
        pair
        for pair in both_finite
        if bool(pair["control"]["strict_full_sun"])
        != bool(pair["candidate"]["strict_full_sun"])
    ]
    strict_loser_positive_values = [
        max(
            float(pair["control"]["e_above_hull"]),
            float(pair["candidate"]["e_above_hull"]),
        )
        for pair in strict_crossings
    ]
    finite_distribution = {
        "control": numeric_summary(finite_values["control"]),
        "candidate": numeric_summary(finite_values["candidate"]),
        "strict_boundary": {
            arm: {
                "count": len(strict_values[arm]),
                "exact_zero": sum(value == 0.0 for value in strict_values[arm]),
                "negative": sum(value < 0.0 for value in strict_values[arm]),
                "minimum": min(strict_values[arm]) if strict_values[arm] else None,
                "maximum": max(strict_values[arm]) if strict_values[arm] else None,
            }
            for arm in ARMS
        },
        "both_finite_pairs": len(both_finite),
        "paired_candidate_minus_control_e_hull": numeric_summary(deltas),
        "paired_delta_sign": {
            "candidate_lower": sum(value < 0.0 for value in deltas),
            "equal": sum(value == 0.0 for value in deltas),
            "candidate_higher": sum(value > 0.0 for value in deltas),
        },
        "finite_meta_threshold_crossings": {
            "count": len(meta_crossings),
            "candidate_gain": sum(
                not pair["control"]["meta_full_sun"]
                and pair["candidate"]["meta_full_sun"]
                for pair in meta_crossings
            ),
            "candidate_loss": sum(
                pair["control"]["meta_full_sun"]
                and not pair["candidate"]["meta_full_sun"]
                for pair in meta_crossings
            ),
            "closest_threshold_margin": numeric_summary(meta_closest_margins),
            "descriptive_margin_ev_per_atom": descriptive_margin,
            "both_arms_farther_than_margin": sum(
                abs(float(pair["control"]["e_above_hull"]) - meta_threshold)
                > descriptive_margin
                and abs(float(pair["candidate"]["e_above_hull"]) - meta_threshold)
                > descriptive_margin
                for pair in meta_crossings
            ),
        },
        "finite_strict_threshold_crossings": {
            "count": len(strict_crossings),
            "nonzero_arm_e_hull": numeric_summary(strict_loser_positive_values),
        },
    }

    marginal = {
        arm: {
            "counts": {state: int(state_counts[arm][state]) for state in STATES},
            "rates_all_attempts": {
                state: state_counts[arm][state] / 1024.0 for state in STATES
            },
            "finite_novel_unique": len(finite_values[arm]),
        }
        for arm in ARMS
    }
    marginal["candidate_minus_control_counts"] = {
        state: int(state_counts["candidate"][state] - state_counts["control"][state])
        for state in STATES
    }

    source_counts = {
        arm: {endpoint: 0 for endpoint in ENDPOINTS} for arm in ARMS
    }
    for arm_key, arm_report in terminal["arms"].items():
        arm = str(arm_report["arm"])
        for endpoint in ENDPOINTS:
            source_counts[arm][endpoint] += int(
                arm_report["source_frozen_cache_counts"][endpoint]
            )
    completed_counts = {
        arm: {
            endpoint: sum(
                endpoint_positive(str(pair[arm]["state"]), endpoint) for pair in pairs
            )
            for endpoint in ENDPOINTS
        }
        for arm in ARMS
    }
    source_effect = {
        endpoint: source_counts["candidate"][endpoint] - source_counts["control"][endpoint]
        for endpoint in ENDPOINTS
    }
    completed_effect = {
        endpoint: (
            completed_counts["candidate"][endpoint]
            - completed_counts["control"][endpoint]
        )
        for endpoint in ENDPOINTS
    }
    coverage_comparison = {
        "frozen_cache_counts": source_counts,
        "r03g_counts": completed_counts,
        "coverage_added": {
            arm: {
                endpoint: completed_counts[arm][endpoint] - source_counts[arm][endpoint]
                for endpoint in ENDPOINTS
            }
            for arm in ARMS
        },
        "frozen_cache_candidate_minus_control": source_effect,
        "r03g_candidate_minus_control": completed_effect,
        "effect_change_after_hull_coverage": {
            endpoint: completed_effect[endpoint] - source_effect[endpoint]
            for endpoint in ENDPOINTS
        },
    }
    if source_effect != {"strict_full_sun": 4, "meta_full_sun": -28}:
        raise AttributionError("frozen-cache R03E effect changed")
    if completed_effect != {"strict_full_sun": 18, "meta_full_sun": -27}:
        raise AttributionError("R03G completed effect changed")

    strict_report = endpoint_reports["strict_full_sun"]
    meta_report = endpoint_reports["meta_full_sun"]
    interpretation = {
        "primary_metric_mechanism": (
            "Safe-axis redistributes finite novel-unique outcomes away from the "
            "middle 0<E_hull<=0.1 band toward both exact-zero hull stability and "
            "E_hull>0.1. This strengthens strict S.U.N. while weakening meta S.U.N."
        ),
        "strict_effect_decomposition": (
            f"Strict +{strict_report['candidate_minus_control']} equals "
            f"{strict_report['net_effect_by_mechanism'].get('finite_hull_threshold_crossing', 0):+d} "
            "from finite-hull crossings plus "
            f"{strict_report['net_effect_by_mechanism'].get('novel_unique_eligibility', 0):+d} "
            "from novel-unique eligibility and +0 from residual unknowns."
        ),
        "meta_effect_decomposition": (
            f"Meta {meta_report['candidate_minus_control']:+d} is entirely finite-hull "
            f"threshold crossing ({meta_report['net_effect_by_mechanism'].get('finite_hull_threshold_crossing', 0):+d}); "
            f"novel-unique eligibility contributes {meta_report['net_effect_by_mechanism'].get('novel_unique_eligibility', 0):+d} "
            "and residual unknowns contribute +0."
        ),
        "unknown_conclusion": (
            "All residual unknowns are exactly paired at the same nine ordinals "
            "in every repeat and have identical chemical-system frequencies in "
            "control and candidate; they cannot explain the treatment-effect delta."
        ),
        "boundary_conclusion": (
            "Every strict-positive E_hull is exactly 0.0, not a negative floating "
            "artifact. Most finite meta discordances are not confined to a tiny "
            "0.1-threshold neighborhood."
        ),
        "causal_limit": (
            "This is a proximate endpoint attribution. It identifies which frozen "
            "metric component changes, but does not by itself prove the microscopic "
            "refinement-energy mechanism."
        ),
        "gate_conclusion": (
            "The strict signal is real and repeat-stable, but the broader meta "
            "endpoint remains adverse. R03E therefore remains scientifically stopped."
        ),
        "recommended_next_step": (
            "If another read-only diagnostic is desired, join recurrent meta-loss "
            "ordinals to frozen safe-axis schedule features before changing exactly "
            "one H1 schedule parameter. Do not tune thresholds or select samples."
        ),
    }

    for endpoint in ENDPOINTS:
        endpoint_reports[endpoint].pop("_candidate_only_events")
        endpoint_reports[endpoint].pop("_control_only_events")

    return {
        "state_definitions": {
            "ineligible": "frozen novel_unique=false; hull evaluation not applicable",
            "residual_unknown": "novel_unique=true but completed snapshot still lacks E_hull; strict/meta=false",
            "strict": "novel_unique=true and E_hull<=0.0 eV/atom",
            "meta_only": "novel_unique=true and 0.0<E_hull<=0.1 eV/atom",
            "above_meta": "novel_unique=true and E_hull>0.1 eV/atom",
        },
        "marginal_state_distribution": marginal,
        "paired_state_transition_matrix": counter_dict(transition),
        "endpoint_attribution": endpoint_reports,
        "residual_unknown_attribution": residual_unknown,
        "finite_hull_distribution": finite_distribution,
        "coverage_comparison": coverage_comparison,
        "interpretation": interpretation,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    marginal = report["marginal_state_distribution"]
    strict = report["endpoint_attribution"]["strict_full_sun"]
    meta = report["endpoint_attribution"]["meta_full_sun"]
    unknown = report["residual_unknown_attribution"]
    finite = report["finite_hull_distribution"]
    comparison = report["coverage_comparison"]
    lines = [
        "# H1-A2 R03H paired ordinal attribution",
        "",
        f"Status: `{report['status']}`",
        "",
        "This is a read-only attribution of the frozen R03G endpoints. No endpoint,",
        "threshold, sample, generation, refinement, energy, novelty, or hull value",
        "was recomputed.",
        "",
        "## Primary result",
        "",
        f"- Strict candidate-control: `{strict['candidate_minus_control']:+d}/1024` "
        f"({strict['candidate_only']} candidate-only versus "
        f"{strict['control_only']} control-only).",
        f"- Meta candidate-control: `{meta['candidate_minus_control']:+d}/1024` "
        f"({meta['candidate_only']} candidate-only versus "
        f"{meta['control_only']} control-only).",
        f"- Strict mechanism net: `{json.dumps(strict['net_effect_by_mechanism'], sort_keys=True)}`.",
        f"- Meta mechanism net: `{json.dumps(meta['net_effect_by_mechanism'], sort_keys=True)}`.",
        "",
        "## Marginal state redistribution",
        "",
        "| State | Control | Candidate | Candidate-control |",
        "|---|---:|---:|---:|",
    ]
    for state in STATES:
        lines.append(
            f"| {state} | {marginal['control']['counts'][state]} | "
            f"{marginal['candidate']['counts'][state]} | "
            f"{marginal['candidate_minus_control_counts'][state]:+d} |"
        )
    lines.extend(
        [
            "",
            "The candidate has more exact-zero strict outcomes, fewer intermediate",
            "0<E_hull<=0.1 outcomes, and more E_hull>0.1 outcomes. The proximate",
            "effect is therefore polarization of the frozen stability distribution.",
            "",
            "## Residual unknown audit",
            "",
            f"- Control/candidate unknown counts: `{unknown['control_count']}/"
            f"{unknown['candidate_count']}`.",
            f"- Both-unknown paired records: `{unknown['paired_both_unknown']}`; "
            "one-arm-only unknown records: `0/0`.",
            f"- Same unknown ordinals in every repeat: "
            f"`{unknown['identical_ordinals_across_repeats']}`.",
            f"- Distinct residual chemical systems: `{len(unknown['distinct_chemsys'])}`; "
            f"common elements: `{', '.join(unknown['common_elements_across_all_unknown_chemsys'])}`.",
            "- Residual unknown contribution to strict/meta treatment effects: `0/0`.",
            "",
            "## Frozen E_hull diagnostics",
            "",
            f"- Both-finite paired records: `{finite['both_finite_pairs']}`.",
            f"- Candidate lower/equal/higher E_hull: "
            f"`{finite['paired_delta_sign']['candidate_lower']}/"
            f"{finite['paired_delta_sign']['equal']}/"
            f"{finite['paired_delta_sign']['candidate_higher']}`.",
            f"- Paired median/mean candidate-control E_hull: "
            f"`{finite['paired_candidate_minus_control_e_hull']['median']:.8f}` / "
            f"`{finite['paired_candidate_minus_control_e_hull']['mean']:.8f}` eV/atom.",
            f"- All strict values are exactly zero: control "
            f"`{finite['strict_boundary']['control']['exact_zero']}/"
            f"{finite['strict_boundary']['control']['count']}`, candidate "
            f"`{finite['strict_boundary']['candidate']['exact_zero']}/"
            f"{finite['strict_boundary']['candidate']['count']}`.",
            f"- Finite meta crossings: "
            f"`{finite['finite_meta_threshold_crossings']['candidate_gain']}` gains "
            f"versus `{finite['finite_meta_threshold_crossings']['candidate_loss']}` losses.",
            f"- Crossings with both arms more than "
            f"{finite['finite_meta_threshold_crossings']['descriptive_margin_ev_per_atom']:.3f} "
            "eV/atom from the meta threshold: "
            f"`{finite['finite_meta_threshold_crossings']['both_arms_farther_than_margin']}/"
            f"{finite['finite_meta_threshold_crossings']['count']}`.",
            "",
            "## Coverage comparison",
            "",
            f"- Frozen-cache effect strict/meta: "
            f"`{comparison['frozen_cache_candidate_minus_control']['strict_full_sun']:+d}` / "
            f"`{comparison['frozen_cache_candidate_minus_control']['meta_full_sun']:+d}`.",
            f"- R03G effect strict/meta: "
            f"`{comparison['r03g_candidate_minus_control']['strict_full_sun']:+d}` / "
            f"`{comparison['r03g_candidate_minus_control']['meta_full_sun']:+d}`.",
            "",
            "## Interpretation",
            "",
        ]
    )
    for key in (
        "primary_metric_mechanism",
        "strict_effect_decomposition",
        "meta_effect_decomposition",
        "unknown_conclusion",
        "boundary_conclusion",
        "causal_limit",
        "gate_conclusion",
        "recommended_next_step",
    ):
        lines.append(f"- {report['interpretation'][key]}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--execution-manifest-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    run_root = args.run_root.resolve()
    config = read_json(args.config.resolve())
    if run_root != Path(str(config["run_root"])).resolve():
        raise AttributionError("fixed R03H run identity changed")
    if len(args.execution_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in args.execution_manifest_sha256
    ):
        raise AttributionError("invalid R03H source manifest SHA")
    source_manifest = require_source_manifest(
        source_dir, args.execution_manifest_sha256
    )
    if run_root.exists() and args.preflight_only:
        raise AttributionError("fixed R03H run identity already exists")
    terminal, source_identities, arms = load_frozen_evidence(config)
    pairs = build_pairs(arms)
    attribution = build_attribution(
        terminal,
        pairs,
        meta_threshold=float(config["analysis"]["meta_threshold_ev_per_atom"]),
        descriptive_margin=float(
            config["analysis"]["descriptive_meta_margin_ev_per_atom"]
        ),
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "schema": "h1_r03h_preflight_v1",
                    "status": "pass",
                    "source_manifest_sha256": sha256_file(source_manifest),
                    "source_terminal_sha256": source_identities[
                        "terminal_report"
                    ]["sha256"],
                    "paired_records": len(pairs),
                    "strict_effect": attribution["endpoint_attribution"][
                        "strict_full_sun"
                    ]["candidate_minus_control"],
                    "meta_effect": attribution["endpoint_attribution"][
                        "meta_full_sun"
                    ]["candidate_minus_control"],
                    "residual_unknown_paired_both": attribution[
                        "residual_unknown_attribution"
                    ]["paired_both_unknown"],
                    "network": False,
                    "gpu": False,
                    "slurm": False,
                },
                sort_keys=True,
            )
        )
        return

    run_root.mkdir(parents=True, exist_ok=True)
    unexpected = {
        path.name for path in run_root.iterdir() if path.name not in {"logs"}
    }
    if unexpected:
        raise AttributionError(
            f"R03H run root contains unexpected entries: {sorted(unexpected)}"
        )
    submission = {
        "schema": "h1_r03h_submission_record_v1",
        "status": "complete",
        "created_at_utc": utc_now(),
        "run_id": config["run_id"],
        "execution_manifest_sha256": args.execution_manifest_sha256,
        "location": config["execution"]["location"],
        "source_r03g_terminal_sha256": source_identities["terminal_report"][
            "sha256"
        ],
        "network_used": False,
        "api_key_read": False,
        "slurm_used": False,
        "gpu_used": False,
        "threshold_changed": False,
        "sample_selection_used": False,
        "upstream_rerun": False,
        "automatic_downstream": False,
    }
    input_contract = {
        "schema": "h1_r03h_input_contract_v1",
        "status": "pass",
        "created_at_utc": utc_now(),
        "run_id": config["run_id"],
        "execution_manifest_sha256": args.execution_manifest_sha256,
        "source_identities": source_identities,
        "repeat_count": 4,
        "arms": list(ARM_KEYS),
        "attempts_per_arm": 256,
        "paired_records": 1024,
        "ordinals": list(range(256)),
        "strict_threshold_ev_per_atom": config["analysis"][
            "strict_threshold_ev_per_atom"
        ],
        "meta_threshold_ev_per_atom": config["analysis"][
            "meta_threshold_ev_per_atom"
        ],
        "endpoint_recomputed": False,
        "threshold_changed": False,
        "sample_selection_used": False,
        "network_used": False,
        "api_key_read": False,
        "upstream_rerun": False,
    }
    submission_path = run_root / "status" / "submission_record.json"
    input_contract_path = run_root / "input_contract.json"
    pair_path = run_root / "analysis" / "paired_ordinals.jsonl"
    write_json_exclusive(submission_path, submission)
    write_json_exclusive(input_contract_path, input_contract)
    write_jsonl_exclusive(pair_path, pairs)

    report = {
        "schema": "h1_r03h_paired_ordinal_attribution_report_v1",
        "status": "complete_read_only_attribution",
        "ok": True,
        "created_at_utc": utc_now(),
        "run_id": config["run_id"],
        "execution_manifest_sha256": args.execution_manifest_sha256,
        "source_r03g": source_identities,
        "paired_shape": {
            "repeats": 4,
            "arms_per_repeat": 2,
            "attempts_per_arm": 256,
            "paired_records": 1024,
            "denominator": "raw_all_attempts",
        },
        "contract": {
            "read_only": True,
            "endpoint_recomputed": False,
            "threshold_changed": False,
            "sample_selection_used": False,
            "network_used": False,
            "api_key_read": False,
            "slurm_used": False,
            "gpu_used": False,
            "generation_rerun": False,
            "refinement_rerun": False,
            "chgnet_rerun": False,
            "direct_metrics_rerun": False,
            "novelty_rerun": False,
            "sun_rerun": False,
        },
        **attribution,
        "artifacts": {
            "input_contract": identity(input_contract_path),
            "paired_ordinals": identity(pair_path),
            "submission_record": identity(submission_path),
        },
        "decision": "attribution_complete_no_scientific_gate_change",
        "formal_g3": False,
        "automatic_promotion": False,
        "automatic_training": False,
        "checkpoint_reselection": False,
        "automatic_downstream": False,
    }
    report_path = run_root / "final" / "attribution_report.json"
    markdown_path = run_root / "final" / "attribution_report.md"
    write_json_exclusive(report_path, report)
    write_text_exclusive(markdown_path, markdown_report(report))
    claim = {
        "schema": "h1_r03h_claim_v1",
        "status": report["status"],
        "decision": report["decision"],
        "attribution_report": identity(report_path),
        "paired_ordinals": identity(pair_path),
        "residual_unknown_effect_contribution": {
            "strict_full_sun": 0,
            "meta_full_sun": 0,
        },
        "threshold_changed": False,
        "sample_selection_used": False,
        "automatic_downstream": False,
    }
    claim_path = run_root / "claim.json"
    write_json_exclusive(claim_path, claim)
    decision = {
        "schema": "h1_r03h_decision_v1",
        "status": report["status"],
        "decision": report["decision"],
        "attribution_report_sha256": sha256_file(report_path),
        "attribution_markdown_sha256": sha256_file(markdown_path),
        "claim_sha256": sha256_file(claim_path),
        "formal_g3": False,
        "automatic_promotion": False,
        "automatic_training": False,
        "checkpoint_reselection": False,
        "automatic_downstream": False,
    }
    decision_path = run_root / "final" / "decision.json"
    write_json_exclusive(decision_path, decision)
    write_text_exclusive(run_root / "final" / "_SUCCESS", "R03H_COMPLETE")
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "report_sha256": sha256_file(report_path),
                "strict_effect": report["endpoint_attribution"][
                    "strict_full_sun"
                ]["candidate_minus_control"],
                "meta_effect": report["endpoint_attribution"]["meta_full_sun"][
                    "candidate_minus_control"
                ],
                "residual_unknown_effect": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
