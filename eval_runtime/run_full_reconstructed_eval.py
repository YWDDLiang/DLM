#!/usr/bin/env python3
"""Compute exact frozen N/U and CHGNet energy for every reconstructed row.

Unlike the legacy S.U.N. wrapper, this module does not use N/U as a gate for
relaxation. Official E_hull is joined later, then S, N, and U are intersected.
"""

from __future__ import annotations

import argparse
import atexit
import importlib.util
import json
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import exact_raw_reuse
import protocol


EXACT_REUSE_READY = "exact_raw_reuse_input_ready.json"
EXACT_REUSE_POLL_SECONDS = 5
EXACT_REUSE_DIAGNOSTIC_SECONDS = 60


def _load(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _require_raw_generation(
    generation: list[dict[str, Any]], *, role: str, arm: str
) -> None:
    expected_arm = {"F": "control", "M": "candidate"}[role]
    if arm != expected_arm:
        raise ValueError(f"exact raw role {role} requires arm {expected_arm}")
    if any(
        row.get("method") != "H1-A2-DLM-RAW-BODY-NO-MODEL494"
        or row.get("diffusion_refinement_applied") is not False
        or row.get("diffusion_refinement_steps") != 0
        or row.get("refiner_noise_seed") is not None
        for row in generation
    ):
        raise ValueError("exact reuse is restricted to unrefined raw structures")


def _wait_for_file(
    path: Path, *, failure_paths: tuple[Path, ...], label: str
) -> None:
    started = time.monotonic()
    next_diagnostic = started + EXACT_REUSE_DIAGNOSTIC_SECONDS
    while not path.is_file():
        failed = next(
            (candidate for candidate in failure_paths if candidate.exists()), None
        )
        if failed is not None:
            raise RuntimeError(f"{label} failed before publishing {path.name}: {failed}")
        now = time.monotonic()
        if now >= next_diagnostic:
            print(
                json.dumps(
                    {
                        "event": "exact_raw_reuse_wait",
                        "label": label,
                        "path": str(path),
                        "waited_seconds": round(now - started, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            next_diagnostic = now + EXACT_REUSE_DIAGNOSTIC_SECONDS
        time.sleep(EXACT_REUSE_POLL_SECONDS)


def _register_exact_failure_marker(shared: Path, role: str) -> dict[str, bool]:
    """Publish peer-visible failure unless the exact-reuse run completes."""

    shared.mkdir(parents=True, exist_ok=True)
    marker = shared / f"{role}.FAILED.json"
    state = {"complete": False}

    def mark_failed() -> None:
        if state["complete"] or marker.exists():
            return
        try:
            protocol.write_json_exclusive(
                marker,
                {
                    "schema": "h1_exact_raw_reuse_failure_v1",
                    "role": role,
                    "pid": os.getpid(),
                    "reason": "process_exited_before_exact_reuse_completion",
                },
            )
        except Exception as exc:  # noqa: BLE001
            print(f"could not publish exact-reuse failure marker: {exc}", file=sys.stderr)

    atexit.register(mark_failed)
    return state


def _read_ready_manifest(
    ready_path: Path, *, expected_generation: Path, expected_role: str
) -> tuple[Path, dict[str, Any]]:
    ready = protocol.read_json(ready_path)
    if (
        ready.get("schema") != "h1_exact_raw_reuse_input_ready_v1"
        or ready.get("role") != expected_role
    ):
        raise ValueError("exact-reuse peer ready marker changed")
    manifest_identity = ready.get("input_manifest")
    generation_identity = ready.get("generation_jsonl")
    if not isinstance(manifest_identity, Mapping) or not isinstance(
        generation_identity, Mapping
    ):
        raise ValueError("exact-reuse ready marker has no input identities")
    generation = expected_generation.resolve()
    if (
        Path(str(generation_identity.get("path"))).resolve() != generation
        or generation_identity.get("sha256") != protocol.sha256_file(generation)
        or int(generation_identity.get("bytes", -1)) != generation.stat().st_size
    ):
        raise ValueError("exact-reuse peer generation identity changed")
    manifest_path = Path(str(manifest_identity.get("path"))).resolve()
    if (
        not manifest_path.is_file()
        or protocol.sha256_file(manifest_path) != manifest_identity.get("sha256")
        or manifest_path.stat().st_size != int(manifest_identity.get("bytes", -1))
    ):
        raise ValueError("exact-reuse peer input manifest identity changed")
    return manifest_path, protocol.read_json(manifest_path)


def _write_exact_result_shard(
    *,
    path: Path,
    manifest_path: Path,
    role: str,
    plan: exact_raw_reuse.ExactPairPlan,
    identities: tuple[str, ...],
    results: list[tuple[Any, Any]],
) -> None:
    if len(identities) != len(results):
        raise ValueError("exact-reuse owner result count changed")
    rows: list[dict[str, Any]] = []
    for identity, (energy_value, composition) in zip(identities, results):
        energy = None if energy_value is None else float(energy_value)
        if energy is not None and not math.isfinite(energy):
            raise ValueError("CHGNet returned a nonfinite energy")
        composition_payload = composition.as_dict()
        rows.append(
            {
                "schema": "h1_exact_raw_relax_cache_entry_v1",
                "owner": role,
                "structure_sha256": identity,
                "energy_per_atom": energy,
                "composition": composition_payload,
                "result_sha256": exact_raw_reuse.canonical_result_sha256(
                    identity, energy, composition_payload
                ),
            }
        )
    protocol.write_jsonl_exclusive(path, rows)
    protocol.write_json_exclusive(
        manifest_path,
        {
            "schema": "h1_exact_raw_relax_cache_shard_v1",
            "role": role,
            "pair_plan_sha256": plan.sha256(),
            "entries": len(rows),
            "ordered_structure_sha256": list(identities),
            "ordered_result_sha256": [row["result_sha256"] for row in rows],
            "shard": protocol.identity(path),
        },
    )


def _load_exact_result_shard(
    *,
    path: Path,
    manifest_path: Path,
    role: str,
    plan: exact_raw_reuse.ExactPairPlan,
    resumable: Any,
) -> dict[str, tuple[float | None, Any]]:
    shard_manifest = protocol.read_json(manifest_path)
    expected_identities = plan.owned_identities(role)
    if (
        shard_manifest.get("schema") != "h1_exact_raw_relax_cache_shard_v1"
        or shard_manifest.get("role") != role
        or shard_manifest.get("pair_plan_sha256") != plan.sha256()
        or shard_manifest.get("ordered_structure_sha256") != list(expected_identities)
    ):
        raise ValueError(f"exact-reuse {role} shard manifest changed")
    shard_identity = shard_manifest.get("shard")
    if (
        not isinstance(shard_identity, Mapping)
        or Path(str(shard_identity.get("path"))).resolve() != path.resolve()
        or shard_identity.get("sha256") != protocol.sha256_file(path)
        or int(shard_identity.get("bytes", -1)) != path.stat().st_size
    ):
        raise ValueError(f"exact-reuse {role} shard identity changed")
    rows = protocol.read_jsonl(path)
    expected_result_identities = shard_manifest.get("ordered_result_sha256")
    if (
        len(rows) != len(expected_identities)
        or not isinstance(expected_result_identities, list)
        or len(expected_result_identities) != len(expected_identities)
    ):
        raise ValueError(f"exact-reuse {role} shard length changed")
    loaded: dict[str, tuple[float | None, Any]] = {}
    for expected_identity, expected_result_identity, row in zip(
        expected_identities,
        expected_result_identities,
        rows,
    ):
        energy_value = row.get("energy_per_atom")
        energy = None if energy_value is None else float(energy_value)
        composition_payload = row.get("composition")
        if (
            row.get("schema") != "h1_exact_raw_relax_cache_entry_v1"
            or row.get("owner") != role
            or row.get("structure_sha256") != expected_identity
            or not isinstance(composition_payload, Mapping)
            or (energy is not None and not math.isfinite(energy))
        ):
            raise ValueError(f"exact-reuse {role} shard row changed")
        result_identity = exact_raw_reuse.canonical_result_sha256(
            expected_identity, energy, composition_payload
        )
        if (
            row.get("result_sha256") != result_identity
            or expected_result_identity != result_identity
        ):
            raise ValueError(f"exact-reuse {role} result identity changed")
        if expected_identity in loaded:
            raise ValueError("duplicate exact-reuse result identity")
        loaded[expected_identity] = (
            energy,
            resumable.decode_composition(dict(composition_payload)),
        )
    return loaded


def cleanup_multiprocessing_children(
    children: list[Any] | None = None,
    *,
    join_timeout_seconds: float = 30.0,
    terminate_timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Bound teardown after science has returned and record every intervention."""

    observed = list(multiprocessing.active_children() if children is None else children)
    diagnostics: dict[str, Any] = {
        "schema": "h1_evaluator_worker_cleanup_v1",
        "initial": [
            {"name": child.name, "pid": child.pid, "alive": bool(child.is_alive())}
            for child in observed
        ],
        "terminated_pids": [],
        "killed_pids": [],
        "surviving_pids": [],
        "errors": [],
    }

    def bounded_join(candidates: list[Any], timeout: float) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        for child in candidates:
            try:
                child.join(max(0.0, deadline - time.monotonic()))
            except Exception as exc:  # noqa: BLE001
                diagnostics["errors"].append(
                    f"join:{getattr(child, 'pid', None)}:{type(exc).__name__}:{exc}"
                )

    bounded_join(observed, join_timeout_seconds)
    lingering = [child for child in observed if child.is_alive()]
    for child in lingering:
        try:
            child.terminate()
            diagnostics["terminated_pids"].append(child.pid)
        except Exception as exc:  # noqa: BLE001
            diagnostics["errors"].append(
                f"terminate:{getattr(child, 'pid', None)}:{type(exc).__name__}:{exc}"
            )
    bounded_join(lingering, terminate_timeout_seconds)
    stubborn = [child for child in lingering if child.is_alive()]
    for child in stubborn:
        try:
            child.kill()
            diagnostics["killed_pids"].append(child.pid)
        except Exception as exc:  # noqa: BLE001
            diagnostics["errors"].append(
                f"kill:{getattr(child, 'pid', None)}:{type(exc).__name__}:{exc}"
            )
    bounded_join(stubborn, terminate_timeout_seconds)
    diagnostics["surviving_pids"] = [
        child.pid for child in stubborn if child.is_alive()
    ]
    diagnostics["initial_count"] = len(observed)
    diagnostics["clean"] = not diagnostics["surviving_pids"]
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--generation-jsonl", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--r03e-root", type=Path, required=True)
    parser.add_argument("--working-relax-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exact-raw-reuse-role", choices=exact_raw_reuse.ROLE_ORDER)
    parser.add_argument("--exact-raw-reuse-peer-generation-jsonl", type=Path)
    parser.add_argument("--exact-raw-reuse-peer-output-dir", type=Path)
    parser.add_argument("--exact-raw-reuse-dir", type=Path)
    args = parser.parse_args()

    exact_arguments = (
        args.exact_raw_reuse_role,
        args.exact_raw_reuse_peer_generation_jsonl,
        args.exact_raw_reuse_peer_output_dir,
        args.exact_raw_reuse_dir,
    )
    exact_mode = all(value is not None for value in exact_arguments)
    if any(value is not None for value in exact_arguments) and not exact_mode:
        raise ValueError("exact raw reuse arguments must be supplied together")
    exact_role = str(args.exact_raw_reuse_role) if exact_mode else None
    exact_peer_role = (
        next(role for role in exact_raw_reuse.ROLE_ORDER if role != exact_role)
        if exact_mode
        else None
    )
    exact_shared = args.exact_raw_reuse_dir.resolve() if exact_mode else None
    exact_failure_state = (
        _register_exact_failure_marker(exact_shared, exact_role)
        if exact_shared is not None and exact_role is not None
        else None
    )

    arm = protocol.validate_arm(args.arm)
    repeat = protocol.validate_repeat(args.repeat)
    config = protocol.read_json(args.config.resolve())
    protocol.validate_config(config)
    assets = config["assets"]
    frozen = config["frozen_code"]
    generation_path = args.generation_jsonl.resolve()
    generation = protocol.ordered_rows(
        protocol.read_jsonl(generation_path), ordinal_field="ordinal"
    )
    if (
        {str(row.get("arm")) for row in generation} != {arm}
        or {int(row.get("repeat", -1)) for row in generation} != {repeat}
        or any(row.get("retry_or_replacement_used") is not False for row in generation)
    ):
        raise ValueError("generation ledger contract changed")
    peer_generation_path: Path | None = None
    peer_generation: list[dict[str, Any]] | None = None
    if exact_mode:
        assert exact_role is not None and exact_peer_role is not None
        _require_raw_generation(generation, role=exact_role, arm=arm)
        peer_generation_path = args.exact_raw_reuse_peer_generation_jsonl.resolve()
        if peer_generation_path == generation_path:
            raise ValueError("exact raw reuse peer must be the other F/M cell")
        peer_generation = protocol.ordered_rows(
            protocol.read_jsonl(peer_generation_path), ordinal_field="ordinal"
        )
        peer_arm = {"F": "control", "M": "candidate"}[exact_peer_role]
        if (
            {str(row.get("arm")) for row in peer_generation} != {peer_arm}
            or {int(row.get("repeat", -1)) for row in peer_generation} != {repeat}
            or any(
                row.get("retry_or_replacement_used") is not False
                for row in peer_generation
            )
        ):
            raise ValueError("exact raw reuse peer generation contract changed")
        _require_raw_generation(peer_generation, role=exact_peer_role, arm=peer_arm)

    r03e = args.r03e_root.resolve()
    runtime = r03e / "runtime"
    a100_path = runtime / "crystal_dlm/wqcodiff/crysllmgen/a100_sun.py"
    protocol.require_file(
        a100_path,
        frozen["a100_adapter_sha256"],
        "frozen attempt-preserving A100 adapter",
    )
    sys.path.insert(0, str(runtime))
    from crystal_dlm.wqcodiff.crysllmgen import a100_sun  # noqa: PLC0415

    eval_sun_path = protocol.require_file(
        assets["eval_sun_py"], frozen["eval_sun_sha256"], "frozen eval_sun.py"
    )
    resumable_path = protocol.require_file(
        assets["eval_sun_resumable_py"],
        frozen["eval_sun_resumable_sha256"],
        "frozen eval_sun_resumable.py",
    )
    train_csv = protocol.require_file(
        assets["train_csv"], a100_sun.MP20_TRAIN_CSV_SHA256, "MP20 train CSV"
    )
    protocol.require_file(
        assets["training_index_cache"],
        a100_sun.MP20_TRAINING_INDEX_CACHE_SHA256,
        "frozen training index cache",
    )
    base_cache = protocol.require_file(
        assets["base_chgnet_relax_cache"],
        a100_sun.CHGNET_RELAX_CACHE_SHA256,
        "frozen base CHGNet cache",
    )
    protocol.require_file(
        assets["chgnet_model_asset"],
        a100_sun.CHGNET_0P3P0_SHA256,
        "CHGNet 0.3.0 model asset",
    )
    protocol.require_file(
        assets["chgnet_runtime_checkpoint"],
        a100_sun.CHGNET_0P3P0_SHA256,
        "CHGNet runtime checkpoint",
    )
    working_cache = args.working_relax_cache.resolve()
    if (
        not working_cache.is_file()
        or working_cache.stat().st_size < base_cache.stat().st_size
    ):
        raise ValueError("working CHGNet cache is absent or smaller than frozen base")
    working_cache_sha_before = protocol.sha256_file(working_cache)

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    input_pt = output / "all_attempts.pt"
    input_manifest_path = output / "input_manifest.json"
    manifest = a100_sun.prepare_a100_input(
        generation_jsonl=generation_path,
        output_pt=input_pt,
        output_manifest=input_manifest_path,
        expected_attempts=protocol.RAW_DENOMINATOR,
    )

    peer_manifest_path: Path | None = None
    peer_manifest: dict[str, Any] | None = None
    if exact_mode:
        assert exact_role is not None
        assert exact_peer_role is not None
        assert exact_shared is not None
        assert peer_generation_path is not None
        protocol.write_json_exclusive(
            output / EXACT_REUSE_READY,
            {
                "schema": "h1_exact_raw_reuse_input_ready_v1",
                "role": exact_role,
                "generation_jsonl": protocol.identity(generation_path),
                "input_manifest": protocol.identity(input_manifest_path),
            },
        )
        peer_output = args.exact_raw_reuse_peer_output_dir.resolve()
        peer_ready = peer_output / EXACT_REUSE_READY
        _wait_for_file(
            peer_ready,
            failure_paths=(
                exact_shared / f"{exact_peer_role}.FAILED.json",
                exact_shared / f"{exact_peer_role}.FAILED",
            ),
            label=f"raw-{exact_peer_role}-input",
        )
        peer_manifest_path, peer_manifest = _read_ready_manifest(
            peer_ready,
            expected_generation=peer_generation_path,
            expected_role=exact_peer_role,
        )

    eval_sun = _load(eval_sun_path, "eval_sun")
    resumable = _load(resumable_path, "eval_sun_resumable_full_reconstructed")
    structures, loader_total = eval_sun.load_generated_structures(input_pt)
    if (
        loader_total != protocol.RAW_DENOMINATOR
        or len(structures) != int(manifest["reconstructed_structures"])
    ):
        raise ValueError("frozen loader disagrees with all-attempt manifest")
    peer_structures: list[Any] | None = None
    if exact_mode:
        assert peer_manifest is not None and peer_manifest_path is not None
        peer_input_pt = peer_manifest_path.parent / "all_attempts.pt"
        if (
            Path(str(peer_manifest.get("generated_pt"))).resolve()
            != peer_input_pt.resolve()
            or peer_manifest.get("generated_pt_sha256")
            != protocol.sha256_file(peer_input_pt)
        ):
            raise ValueError("exact raw reuse peer tensor identity changed")
        peer_structures, peer_loader_total = eval_sun.load_generated_structures(
            peer_input_pt
        )
        if (
            peer_loader_total != protocol.RAW_DENOMINATOR
            or len(peer_structures)
            != int(peer_manifest["reconstructed_structures"])
        ):
            raise ValueError("frozen loader disagrees with peer all-attempt manifest")
    train_structures, train_formula_idx = eval_sun.load_training_index(train_csv)
    matcher = eval_sun.StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    novel_mask = eval_sun.compute_novelty(
        structures, train_structures, train_formula_idx, matcher
    )
    eq_class, n_unique = eval_sun.compute_uniqueness(structures, matcher)
    seen_classes: set[int] = set()
    unique_representatives: set[int] = set()
    for index in range(len(structures)):
        class_id = int(eq_class[index])
        if class_id not in seen_classes:
            seen_classes.add(class_id)
            unique_representatives.add(index)
    if len(unique_representatives) != int(n_unique):
        raise ValueError("frozen uniqueness representative count changed")

    relax_path = output / "all_reconstructed_relax_results.jsonl"
    exact_audit_path: Path | None = None
    exact_plan: exact_raw_reuse.ExactPairPlan | None = None
    exact_worker_misses = 0
    if not exact_mode:
        energies_compositions = resumable.relax_missing(
            structures,
            relax_path,
            "cuda",
            working_cache,
        )
    else:
        assert exact_role is not None
        assert exact_peer_role is not None
        assert exact_shared is not None
        assert peer_manifest is not None
        assert peer_manifest_path is not None
        assert peer_structures is not None
        assert peer_generation_path is not None
        manifests_by_role = {
            exact_role: manifest,
            exact_peer_role: peer_manifest,
        }
        manifest_paths_by_role = {
            exact_role: input_manifest_path,
            exact_peer_role: peer_manifest_path,
        }
        generation_paths_by_role = {
            exact_role: generation_path,
            exact_peer_role: peer_generation_path,
        }
        structures_by_role = {
            exact_role: structures,
            exact_peer_role: peer_structures,
        }
        views = {
            role: exact_raw_reuse.manifest_identities(manifests_by_role[role])
            for role in exact_raw_reuse.ROLE_ORDER
        }
        if any(
            len(views[role].reconstructed) != len(structures_by_role[role])
            for role in exact_raw_reuse.ROLE_ORDER
        ):
            raise ValueError("exact raw identity count disagrees with frozen loader")
        exact_plan = exact_raw_reuse.build_pair_plan(views)
        structure_by_identity: dict[str, Any] = {}
        for role in exact_raw_reuse.ROLE_ORDER:
            for identity, structure in zip(
                views[role].reconstructed, structures_by_role[role]
            ):
                structure_by_identity.setdefault(identity, structure)
        if set(structure_by_identity) != set(exact_plan.unique_identities):
            raise ValueError("exact raw pair structure coverage changed")

        owned_identities = exact_plan.owned_identities(exact_role)
        owned_structures = [structure_by_identity[value] for value in owned_identities]
        owned_relax_path = output / "owned_exact_relax_results.jsonl"
        if owned_structures:
            owned_results = resumable.relax_missing(
                owned_structures,
                owned_relax_path,
                "cuda",
                working_cache,
            )
        else:
            protocol.write_jsonl_exclusive(owned_relax_path, ())
            owned_results = []
        if len(owned_results) != len(owned_identities):
            raise ValueError("exact raw owner relaxation ledger is incomplete")
        exact_worker_misses = len(owned_identities)

        own_shard_path = exact_shared / f"{exact_role}.results.jsonl"
        own_shard_manifest_path = exact_shared / f"{exact_role}.results.json"
        _write_exact_result_shard(
            path=own_shard_path,
            manifest_path=own_shard_manifest_path,
            role=exact_role,
            plan=exact_plan,
            identities=owned_identities,
            results=owned_results,
        )
        peer_shard_manifest_path = exact_shared / f"{exact_peer_role}.results.json"
        _wait_for_file(
            peer_shard_manifest_path,
            failure_paths=(
                exact_shared / f"{exact_peer_role}.FAILED.json",
                exact_shared / f"{exact_peer_role}.FAILED",
            ),
            label=f"raw-{exact_peer_role}-relaxation-shard",
        )

        results_by_identity: dict[str, tuple[float | None, Any]] = {}
        shard_paths_by_role: dict[str, Path] = {}
        shard_manifest_paths_by_role: dict[str, Path] = {}
        for role in exact_raw_reuse.ROLE_ORDER:
            shard_path = exact_shared / f"{role}.results.jsonl"
            shard_manifest_path = exact_shared / f"{role}.results.json"
            shard_paths_by_role[role] = shard_path
            shard_manifest_paths_by_role[role] = shard_manifest_path
            shard_results = _load_exact_result_shard(
                path=shard_path,
                manifest_path=shard_manifest_path,
                role=role,
                plan=exact_plan,
                resumable=resumable,
            )
            if set(results_by_identity) & set(shard_results):
                raise ValueError("exact raw result ownership overlaps")
            results_by_identity.update(shard_results)
        if set(results_by_identity) != set(exact_plan.unique_identities):
            raise ValueError("exact raw pair result coverage changed")

        own_identities = views[exact_role].reconstructed
        energies_compositions = exact_raw_reuse.map_results_to_identities(
            own_identities, results_by_identity
        )
        for local_index, (energy, composition) in enumerate(energies_compositions):
            resumable.append_relax_result(
                relax_path, local_index, energy, composition
            )

        exact_audit_path = output / "exact_raw_reuse.json"
        own_unique_identities = set(own_identities)
        ordered_result_identities = [
            exact_raw_reuse.canonical_result_sha256(
                identity,
                None if energy is None else float(energy),
                composition.as_dict(),
            )
            for identity, (energy, composition) in zip(
                own_identities, energies_compositions
            )
        ]
        protocol.write_json_exclusive(
            exact_audit_path,
            {
                "schema": "h1_exact_raw_reuse_audit_v1",
                "role": exact_role,
                "pair_scope": "within_stream_F_M",
                "identity_definition": "sha256(canonical_lossless_structure_json)",
                "near_equivalence_reuse": False,
                "structure_matcher_used_for_reuse": False,
                "model494_refined_reuse": False,
                "pair_plan": exact_plan.canonical_payload(),
                "pair_plan_sha256": exact_plan.sha256(),
                "cache_counts": {
                    "pair_hits": exact_plan.cache_hits,
                    "pair_misses": exact_plan.cache_misses,
                    "worker_owned_misses": exact_worker_misses,
                    "cell_unique_structures": len(own_unique_identities),
                    "cell_duplicate_occurrence_hits": len(own_identities)
                    - len(own_unique_identities),
                    "cell_unique_results_from_peer": sum(
                        exact_plan.owner_for(identity) != exact_role
                        for identity in own_unique_identities
                    ),
                },
                "inputs": {
                    role: {
                        "generation_jsonl": protocol.identity(
                            generation_paths_by_role[role]
                        ),
                        "input_manifest": protocol.identity(
                            manifest_paths_by_role[role]
                        ),
                        "total_attempts": views[role].total_attempts,
                        "ordered_attempt_structure_sha256": [
                            record.get("structure_sha256")
                            if record.get("reconstructed_index") is not None
                            else None
                            for record in manifests_by_role[role]["attempt_records"]
                        ],
                    }
                    for role in exact_raw_reuse.ROLE_ORDER
                },
                "outputs": {
                    "owner_relax_results": protocol.identity(owned_relax_path),
                    "owner_shards": {
                        role: {
                            "results": protocol.identity(shard_paths_by_role[role]),
                            "manifest": protocol.identity(
                                shard_manifest_paths_by_role[role]
                            ),
                        }
                        for role in exact_raw_reuse.ROLE_ORDER
                    },
                    "cell_relax_results": protocol.identity(relax_path),
                    "ordered_reconstructed_result_sha256": ordered_result_identities,
                },
            },
        )
    if len(energies_compositions) != len(structures):
        raise ValueError("full reconstructed relaxation ledger is incomplete")

    reconstructed_by_attempt = {
        str(row["attempt_id"]): int(row["reconstructed_index"])
        for row in manifest["attempt_records"]
        if row.get("reconstructed_index") is not None
    }
    if set(reconstructed_by_attempt) - {str(row["attempt_id"]) for row in generation}:
        raise ValueError("manifest contains an unknown generation attempt")
    labels: list[dict[str, Any]] = []
    for ordinal, generation_row in enumerate(generation):
        attempt_id = str(generation_row["attempt_id"])
        reconstructed_index = reconstructed_by_attempt.get(attempt_id)
        record: dict[str, Any] = {
            "schema": "h1_full_reconstructed_preofficial_attempt_v1",
            "repeat": repeat,
            "arm": arm,
            "ordinal": ordinal,
            "attempt_id": attempt_id,
            "generation_status": generation_row["status"],
            "reconstructed": reconstructed_index is not None,
            "reconstructed_index": reconstructed_index,
            "novel": False,
            "unique_representative": False,
            "novel_unique": False,
            "chgnet_energy_per_atom": None,
            "chgnet_composition": None,
            "reduced_formula": None,
            "chemsys": None,
            "chgnet_relaxation_known": False,
            "official_hull_pending": reconstructed_index is not None,
            "retry_or_replacement_used": False,
        }
        if reconstructed_index is not None:
            energy_value, composition = energies_compositions[reconstructed_index]
            energy = None if energy_value is None else float(energy_value)
            if energy is not None and not math.isfinite(energy):
                raise ValueError("CHGNet returned a nonfinite energy")
            is_novel = bool(novel_mask[reconstructed_index])
            is_unique = reconstructed_index in unique_representatives
            record.update(
                {
                    "novel": is_novel,
                    "unique_representative": is_unique,
                    "novel_unique": is_novel and is_unique,
                    "chgnet_energy_per_atom": energy,
                    "chgnet_composition": composition.as_dict(),
                    "reduced_formula": composition.reduced_formula,
                    "chemsys": "-".join(
                        sorted(element.symbol for element in composition.elements)
                    ),
                    "chgnet_relaxation_known": energy is not None,
                }
            )
        labels.append(record)

    label_path = output / "attempt_labels_preofficial.jsonl"
    protocol.write_jsonl_exclusive(label_path, labels)
    reconstructed = len(structures)
    energy_known = sum(row["chgnet_relaxation_known"] for row in labels)
    novel = sum(row["novel"] for row in labels)
    unique = sum(row["unique_representative"] for row in labels)
    novel_unique = sum(row["novel_unique"] for row in labels)
    worker_cleanup_path = output / "worker_cleanup.json"
    worker_cleanup = cleanup_multiprocessing_children()
    protocol.write_json_exclusive(worker_cleanup_path, worker_cleanup)
    report = {
        "schema": "h1_full_reconstructed_preofficial_summary_v1",
        "status": "complete",
        "ok": True,
        "repeat": repeat,
        "arm": arm,
        "raw_attempts": protocol.RAW_DENOMINATOR,
        "generation_succeeded": sum(row["status"] == "succeeded" for row in generation),
        "reconstructed": reconstructed,
        "novel": novel,
        "unique_representatives": unique,
        "novel_unique": novel_unique,
        "chgnet_relaxation_known": energy_known,
        "chgnet_relaxation_unknown": reconstructed - energy_known,
        "official_hull_target": reconstructed,
        "stability_scope": "all_reconstructed_before_NU_intersection",
        "legacy_novel_unique_relaxation_gate_used": False,
        "target_hull_known_reconstructed": int(
            config["evaluation"]["target_hull_known_reconstructed_per_cell"]
        ),
        "target_is_nonblocking": True,
        "attempt_labels_sha256": protocol.sha256_file(label_path),
        "relax_results_sha256": protocol.sha256_file(relax_path),
        "input_manifest_sha256": protocol.sha256_file(input_manifest_path),
        "working_relax_cache_sha256_before": working_cache_sha_before,
        "working_relax_cache_sha256_after": protocol.sha256_file(working_cache),
        "base_relax_cache_sha256": protocol.sha256_file(base_cache),
        "eval_sun_sha256": protocol.sha256_file(eval_sun_path),
        "eval_sun_resumable_sha256": protocol.sha256_file(resumable_path),
        "worker_cleanup_sha256": protocol.sha256_file(worker_cleanup_path),
        "worker_cleanup_initial_count": worker_cleanup["initial_count"],
        "worker_cleanup_terminated_count": len(worker_cleanup["terminated_pids"]),
        "worker_cleanup_killed_count": len(worker_cleanup["killed_pids"]),
        "worker_cleanup_surviving_count": len(worker_cleanup["surviving_pids"]),
        "retry_replacement_repair_filter_rerank": False,
    }
    if exact_mode:
        assert exact_plan is not None and exact_audit_path is not None
        report.update(
            {
                "exact_raw_reuse": True,
                "exact_raw_reuse_pair_scope": "within_stream_F_M",
                "exact_raw_reuse_pair_cache_hits": exact_plan.cache_hits,
                "exact_raw_reuse_pair_cache_misses": exact_plan.cache_misses,
                "exact_raw_reuse_worker_owned_misses": exact_worker_misses,
                "exact_raw_reuse_plan_sha256": exact_plan.sha256(),
                "exact_raw_reuse_audit_sha256": protocol.sha256_file(
                    exact_audit_path
                ),
                "exact_raw_reuse_input_identity": protocol.identity(
                    input_manifest_path
                ),
                "exact_raw_reuse_output_identity": protocol.identity(relax_path),
                "exact_raw_reuse_structure_matcher": False,
                "exact_raw_reuse_model494_refined": False,
            }
        )
    protocol.write_json_exclusive(output / "summary.json", report)
    (output / "_SUCCESS").touch(exist_ok=False)
    if exact_failure_state is not None:
        exact_failure_state["complete"] = True
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
