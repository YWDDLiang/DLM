#!/usr/bin/env python3
"""CPU-only identity audit from an original CIF CSV to clean training sources.

No formula matching, coordinate fitting, cell reduction, or decoded-label fallback.
Run --help without importing pandas/pymatgen or loading project modules.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import warnings

API = None
STRUCTURE = None
PARSER_VERSION = None
IDENTITY_FIELDS = ("material_id", "mp_id", "metadata.material_id", "metadata.mp_id",
                   "source_metadata.material_id", "source_metadata.mp_id")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def dotted(row: dict, path: str):
    value = row
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def identities(row: dict, fields=IDENTITY_FIELDS) -> dict[str, str]:
    result = {}
    for field in fields:
        value = dotted(row, field)
        if value is not None and not isinstance(value, (dict, list, bool)) and str(value).strip():
            result[field] = str(value).strip()
    return result


def integer(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 and str(parsed) == str(value).strip() else None


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def project_root(argument: str | None) -> Path:
    candidates = [Path(argument).resolve()] if argument else [*Path(__file__).resolve().parents, Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "src/crystal_dlm/dynamic_crystal.py").is_file():
            return candidate
    raise ValueError("Cannot locate project src/crystal_dlm; supply --project-root.")


def load_api(root: str):
    source = str(Path(root) / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer, parse_dynamic_answer, structure_to_dynamic_answer
    from crystal_dlm.canonical_site_order import canonicalize_dynamic_answer_to_plan
    from crystal_dlm.fixed_slot import FixedSlotConfig
    return {"encode": arrays_to_dynamic_answer, "parse": parse_dynamic_answer,
            "structure_encode": structure_to_dynamic_answer, "canonicalize": canonicalize_dynamic_answer_to_plan,
            "config": FixedSlotConfig()}


def canonical_answer(answer: str) -> str:
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Clean target must be a nonempty string.")
    parsed = API["parse"](answer, strict=True)
    # Only the existing periodic 100->0 physical alias is collapsed.
    tokens = [token.replace("_100>", "_000>") if token.startswith(("<X_", "<Y_", "<Z_")) and token.endswith("_100>") else token
              for token in parsed["tokens"]]
    normalized = "".join(tokens)
    API["parse"](normalized, strict=True)
    return normalized


def full_answer_signature(answer: str) -> tuple[str, str]:
    """Stable element grouping only; never sort coordinates or fit geometry."""
    answer = canonical_answer(answer)
    arrays = API["parse"](answer, strict=True)
    counts = Counter(arrays["species"])
    elements = sorted(counts)
    plan = {"N": arrays["num_atoms"], "elements": elements, "counts": [counts[e] for e in elements]}
    grouped, _ = API["canonicalize"](answer, plan)
    grouped = canonical_answer(grouped)
    return text_sha256(grouped), grouped


def target_plan(row: dict, answer: str, plan_field: str) -> tuple[dict, str]:
    arrays = API["parse"](answer, strict=True)
    supplied = dotted(row, plan_field)
    if isinstance(supplied, dict):
        plan = {name: supplied.get(name) for name in ("N", "elements", "counts")}
        origin = plan_field
    else:
        elements = list(dict.fromkeys(arrays["species"]))
        counts = Counter(arrays["species"])
        plan = {"N": arrays["num_atoms"], "elements": elements, "counts": [counts[e] for e in elements]}
        origin = "derived_from_full_target_species_slots"
    if integer(plan["N"]) != arrays["num_atoms"]:
        raise ValueError("Plan N differs from clean target.")
    elements, counts = plan["elements"], plan["counts"]
    if not isinstance(elements, list) or not isinstance(counts, list) or len(elements) != len(counts) or len(set(elements)) != len(elements):
        raise ValueError("Malformed hard Plan elements/counts.")
    expanded = []
    for element, count in zip(elements, counts):
        number = integer(count)
        if not isinstance(element, str) or number is None or number < 1:
            raise ValueError("Plan counts must be positive integers.")
        expanded.extend([element] * number)
    if expanded != arrays["species"]:
        raise ValueError("Target slots do not have the Plan-expanded canonical element order.")
    return plan, origin


def jsonl_records(path: Path):
    ordinal = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                yield {"line": line_number, "blank": True}
                continue
            envelope = {"line": line_number, "ordinal": ordinal, "raw_line_sha256": hashlib.sha256(raw).hexdigest()}
            ordinal += 1
            try:
                row = json.loads(raw.decode("utf-8-sig"))
                if not isinstance(row, dict):
                    raise ValueError("JSONL record is not an object")
                envelope["row"] = row
            except Exception as exc:
                envelope["error"] = f"{type(exc).__name__}: {exc}"
            yield envelope


def prepare_training(path: Path, target_field: str, plan_field: str, id_fields: list[str], split_argument: str | None):
    records, blanks, discovered = [], [], set()
    for envelope in jsonl_records(path):
        if envelope.get("blank"):
            blanks.append(envelope["line"])
            continue
        item = {key: value for key, value in envelope.items() if key != "row"}
        item["status"] = "pending"
        row = envelope.get("row")
        if row is None:
            item["status"] = "invalid_training_json"
            records.append(item)
            continue
        values = {str(dotted(row, name)).strip() for name in ("source_split", "split", "metadata.split") if dotted(row, name) not in (None, "")}
        item["declared_splits"] = sorted(values)
        discovered.update(values)
        item["source_row_idx"] = row.get("source_row_idx")
        item["source_row_idx_is_csv_ordinal_proof"] = False
        item["stable_ids"] = identities(row, id_fields)
        item["ignored_condition_donor_index"] = dotted(row, "condition_prediction.typed_metadata_source_row_idx")
        item["csv_row_proof"] = {
            "row_idx": dotted(row, "csv_row_idx") if dotted(row, "csv_row_idx") is not None else dotted(row, "metadata.csv_row_idx"),
            "source_csv_sha256": dotted(row, "source_csv_sha256") or dotted(row, "metadata.source_csv_sha256"),
        }
        try:
            if len(values) > 1:
                raise ValueError("Conflicting original source splits.")
            if "source_row_idx" in row and integer(row["source_row_idx"]) is None:
                raise ValueError("source_row_idx must be a nonnegative integer when present.")
            clean = dotted(row, target_field)
            item["target_field"] = target_field
            item["target_answer_sha256_bytes"] = text_sha256(clean) if isinstance(clean, str) else None
            item["transition_answer_differs_from_target"] = isinstance(row.get("answer"), str) and row["answer"] != clean
            normalized = canonical_answer(clean)
            item["target_answer"] = normalized
            item["target_answer_normalized_sha256"] = text_sha256(normalized)
            item["target_alias_100_count"] = sum(clean.count(f"<{axis}_100>") for axis in "XYZ")
            item["plan"], item["plan_basis"] = target_plan(row, normalized, plan_field)
            item["signature_sha256"], item["signature_answer"] = full_answer_signature(normalized)
            hashes = {name: dotted(row, name) for name in (
                "condition_prediction.clean_answer_sha256_before", "condition_prediction.clean_answer_sha256_after") if dotted(row, name) is not None}
            item["declared_clean_answer_hashes"] = hashes
            if any(str(value).lower() != item["target_answer_sha256_bytes"] for value in hashes.values()):
                raise ValueError("Declared clean-answer byte hash differs from selected source_answer.")
        except Exception as exc:
            item["status"] = "invalid_clean_target_or_metadata"
            item["error"] = f"{type(exc).__name__}: {exc}"
        records.append(item)
    inferred = next(iter(discovered)) if len(discovered) == 1 else None
    chosen_split = split_argument or inferred
    for item in records:
        values = item.get("declared_splits", [])
        item["source_split"] = values[0] if len(values) == 1 else chosen_split
        if item["status"] == "pending" and (chosen_split is None or (values and values != [chosen_split])):
            item.update(status="source_split_mismatch_or_unproven", error="Supply one correct --split; all declared source splits must agree.")
        idx = integer(item.get("source_row_idx"))
        item["source_key"] = f"{item['source_split']}:source:{idx}" if idx is not None else f"{item['source_split']}:input_ordinal:{item['ordinal']}"
    return records, blanks, chosen_split


def worker_init(root: str):
    global API, STRUCTURE, PARSER_VERSION
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    API = load_api(root)
    from pymatgen.core import Structure
    STRUCTURE = Structure
    PARSER_VERSION = package_version("pymatgen")


def parse_csv_row(payload: dict) -> dict:
    result = {key: value for key, value in payload.items() if key != "cif"}
    result["cif_sha256_utf8"] = text_sha256(payload["cif"])
    result["pymatgen_version"] = PARSER_VERSION
    result["parse_status"] = "error"
    captured = []
    try:
        if not payload["cif"].strip():
            raise ValueError("Empty selected CIF column")
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            structure = STRUCTURE.from_str(payload["cif"], fmt="cif")
            captured = [{"category": w.category.__name__, "message": str(w.message)} for w in warning_list]
        if not structure.is_ordered:
            raise ValueError("Disordered sites cannot be silently converted to one species")
        answer, diagnostics = API["structure_encode"](structure)
        answer = canonical_answer(answer)
        signature_sha, signature_answer = full_answer_signature(answer)
        lattice = structure.lattice.matrix.tolist()
        fractional = structure.frac_coords.tolist()
        species = [str(site.specie.symbol) for site in structure]
        if not all(math.isfinite(float(v)) for row in lattice + fractional for v in row):
            raise ValueError("Nonfinite parsed lattice/fractional coordinate")
        result.update(parse_status="ok", num_atoms=int(structure.num_sites), raw_quantized_answer=answer,
                      signature_sha256=signature_sha, signature_answer=signature_answer,
                      codec_diagnostics=diagnostics.to_dict(), continuous={
                          "lattice_matrix_A": lattice, "lengths_A": list(map(float, structure.lattice.abc)),
                          "angles_degrees": list(map(float, structure.lattice.angles)),
                          "species_parser_order": species, "fractional_parser_order": fractional})
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["parser_warnings"] = captured
    return result


def index_csv(args, root: Path, output: Path):
    import pandas as pd
    header = list(pd.read_csv(args.source_csv, nrows=0).columns)
    if args.cif_column not in header:
        raise ValueError(f"Selected CIF column {args.cif_column!r} absent; no fallback to another CIF column.")
    usecols = [args.cif_column]
    if args.csv_id_column in header:
        usecols.append(args.csv_id_column)
    split_columns = [name for name in ("source_split", "split") if name in header]
    usecols.extend(split_columns)
    indexed, next_index = [], 0
    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init, initargs=(str(root),)) if args.workers > 1 else None
    if pool is None:
        worker_init(str(root))
    try:
        reader = pd.read_csv(args.source_csv, usecols=usecols, dtype=str, keep_default_na=False,
                             chunksize=args.chunk_size, on_bad_lines="error")
        with (output / "csv_index.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in reader:
                payloads = []
                for row in chunk.to_dict(orient="records"):
                    payloads.append({"csv_row_idx": next_index, "csv_row_number": next_index + 1,
                                     "csv_material_id": str(row.get(args.csv_id_column, "")).strip(),
                                     "csv_declared_splits": sorted({str(row[k]).strip() for k in split_columns if str(row[k]).strip()}),
                                     "cif": str(row[args.cif_column])})
                    next_index += 1
                parsed = pool.map(parse_csv_row, payloads) if pool else map(parse_csv_row, payloads)
                for result in parsed:
                    indexed.append(result)
                    # Continuous arrays are exported once per verified training source below.
                    compact = {key: value for key, value in result.items() if key not in ("continuous", "signature_answer", "raw_quantized_answer")}
                    handle.write(json.dumps(compact, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                print(json.dumps({"event": "csv_index_progress", "csv_rows_indexed": len(indexed)}), flush=True)
    finally:
        if pool:
            pool.shutdown(wait=True)
    return indexed, header


def load_r5(path: Path | None, split: str | None):
    records, by_hash, by_index, errors = [], defaultdict(list), defaultdict(list), []
    if path is None:
        return records, by_hash, by_index, errors
    for envelope in jsonl_records(path):
        if envelope.get("blank"):
            continue
        row = envelope.get("row")
        record = {key: value for key, value in envelope.items() if key != "row"}
        try:
            if row is None:
                raise ValueError(envelope["error"])
            declared = {str(dotted(row, name)) for name in ("source_split", "split", "metadata.split") if dotted(row, name) not in (None, "")}
            if declared and declared != {split}:
                raise ValueError("Intermediate R5 split differs")
            record["signature_sha256"], record["signature_answer"] = full_answer_signature(row["answer"])
            record["stable_ids"] = identities(row)
            record["source_row_idx"] = integer(row.get("source_row_idx"))
            record["split_declared"] = sorted(declared)
            records.append(record)
            by_hash[record["signature_sha256"]].append(record)
            if record["source_row_idx"] is not None:
                by_index[record["source_row_idx"]].append(record)
        except Exception as exc:
            errors.append({**record, "error": f"{type(exc).__name__}: {exc}"})
    return records, by_hash, by_index, errors


def choose_identity(training: dict, csv_rows: list[dict], by_hash: dict, by_id: dict,
                    csv_sha: str, r5_by_hash: dict, r5_by_index: dict) -> dict:
    """Identity decision is independent of geometry verification performed later."""
    fingerprint = training["signature_sha256"]
    same_answer = [r for r in by_hash.get(fingerprint, []) if r["signature_answer"] == training["signature_answer"]]
    info = {"full_answer_candidate_rows": [r["csv_row_idx"] for r in same_answer], "identity_method": None}
    proof = training.get("csv_row_proof", {})
    proof_index = integer(proof.get("row_idx"))
    proof_hash = str(proof.get("source_csv_sha256") or "").lower()
    if proof_index is not None or proof_hash:
        if proof_index is None or proof_hash != csv_sha or proof_index >= len(csv_rows):
            return {**info, "status": "invalid_explicit_csv_row_proof"}
        selected = csv_rows[proof_index]
        known_ids = set(training["stable_ids"].values())
        if known_ids and known_ids != {selected["csv_material_id"]}:
            return {**info, "status": "csv_row_proof_conflicts_with_stable_id"}
        return {**info, "status": "candidate", "identity_method": "explicit_csv_row_bound_to_file_sha256", "selected": selected}
    stable = set(training["stable_ids"].values())
    if len(stable) > 1:
        return {**info, "status": "conflicting_training_stable_ids"}
    if stable:
        candidates = by_id.get(next(iter(stable)), [])
        if len(candidates) != 1:
            return {**info, "status": "stable_id_missing" if not candidates else "ambiguous_csv_stable_id",
                    "id_candidate_rows": [r["csv_row_idx"] for r in candidates]}
        return {**info, "status": "candidate", "identity_method": "training_stable_material_id", "selected": candidates[0]}

    # Optional R5 is evidence only. Its bare file ordinal is never promoted to an original CSV index.
    indexed_r5 = r5_by_index.get(integer(training.get("source_row_idx")), [])
    r5_matches = indexed_r5 if indexed_r5 else r5_by_hash.get(fingerprint, [])
    r5_matches = [r for r in r5_matches if r["signature_sha256"] == fingerprint and r["signature_answer"] == training["signature_answer"]]
    info["r5_matching_lines"] = [r["line"] for r in r5_matches]
    r5_ids = {value for r in r5_matches for value in r["stable_ids"].values()}
    if len(r5_ids) == 1:
        candidates = by_id.get(next(iter(r5_ids)), [])
        if len(candidates) == 1:
            return {**info, "status": "candidate", "identity_method": "r5_explicit_source_index_and_stable_id" if indexed_r5 else "r5_full_answer_and_consistent_stable_id", "selected": candidates[0]}
    if len(same_answer) == 1:
        return {**info, "status": "candidate", "identity_method": "unique_complete_quantized_answer", "selected": same_answer[0]}
    return {**info, "status": "ambiguous_complete_quantized_answer" if same_answer else "complete_quantized_answer_not_found"}


def align_continuous(training: dict, selected: dict) -> dict:
    if selected["parse_status"] != "ok":
        raise ValueError("Identified CSV row failed CIF parsing/encoding: " + selected.get("error", "unknown"))
    canonical, diagnostics = API["canonicalize"](selected["raw_quantized_answer"], training["plan"])
    canonical = canonical_answer(canonical)
    permutation = [int(x) for x in diagnostics["site_permutation"]]
    geometry = selected["continuous"]
    if sorted(permutation) != list(range(selected["num_atoms"])):
        raise ValueError("Site permutation is not an exact bijection")
    species = [geometry["species_parser_order"][i] for i in permutation]
    fractional = [[float(v) % 1.0 for v in geometry["fractional_parser_order"][i]] for i in permutation]
    # Re-encode the original continuous values, not the decoded quantized arrays.
    encoded, codec_diagnostics = API["encode"](geometry["lengths_A"], geometry["angles_degrees"], species, fractional)
    encoded = canonical_answer(encoded)
    expected_tokens = API["parse"](training["target_answer"], strict=True)["tokens"]
    actual_tokens = API["parse"](encoded, strict=True)["tokens"]
    differences = [{"token_position": i, "target": a, "original_cif_encoded": b}
                   for i, (a, b) in enumerate(zip(expected_tokens, actual_tokens)) if a != b]
    return {"site_permutation_aligned_slot_to_parsed_cif_site": permutation,
            "permutation_method": "existing_plan_expanded_stable_within_element_v1",
            "species_program_not_used_for_site_alignment": True,
            "continuous_aligned": {"lattice_matrix_A": geometry["lattice_matrix_A"],
                                   "lengths_A": geometry["lengths_A"], "angles_degrees": geometry["angles_degrees"],
                                   "species": species, "fractional": fractional},
            "coordinate_operation": "periodic_wrap_only; no origin shift, cell reduction, coordinate sorting, or fitting",
            "reencoded_canonical_answer": encoded, "reencoded_canonical_answer_sha256": text_sha256(encoded),
            "canonicalized_original_Q_equals_target": canonical == training["target_answer"],
            "quantized_continuous_equals_target": encoded == training["target_answer"],
            "differing_token_count": len(differences), "first_differing_tokens": differences[:12],
            "continuous_codec_diagnostics": codec_diagnostics.to_dict()}


def audit(args, output: Path) -> dict:
    global API
    root = project_root(args.project_root)
    API = load_api(str(root))
    import pandas as pd
    from pymatgen.core import Structure
    source_path, training_path = Path(args.source_csv).resolve(), Path(args.training_jsonl).resolve()
    r5_path = Path(args.r5_jsonl).resolve() if args.r5_jsonl else None
    source_sha, training_sha = file_sha256(source_path), file_sha256(training_path)
    if args.source_csv_sha256 and source_sha != args.source_csv_sha256.lower():
        raise ValueError("Original CSV SHA256 differs from --source-csv-sha256")
    if args.training_sha256 and training_sha != args.training_sha256.lower():
        raise ValueError("Training JSONL SHA256 differs from --training-sha256")
    source_files = ["src/crystal_dlm/dynamic_crystal.py", "src/crystal_dlm/fixed_slot.py",
                    "src/crystal_dlm/canonical_site_order.py", "src/scripts/build_r5_exact_length_sft_data.py",
                    "scripts/build_c3fd_native_sft_data.py", "scripts/canonicalize_c3fd_native_teacher_sft.py"]
    receipt = {"source_csv": {"path": str(source_path), "sha256": source_sha, "selected_cif_column": args.cif_column,
                               "id_column": args.csv_id_column, "cif_hash_definition": "SHA256 of UTF-8 decoded selected CSV CIF field before parser"},
               "training_jsonl": {"path": str(training_path), "sha256": training_sha, "target_field": args.target_field},
               "r5_jsonl": {"path": str(r5_path), "sha256": file_sha256(r5_path)} if r5_path else None,
               "parser": {"call": "pymatgen.core.Structure.from_str(cif, fmt='cif')", "signature": str(inspect.signature(Structure.from_str)),
                          "pymatgen": package_version("pymatgen"), "pandas": pd.__version__, "numpy": package_version("numpy"), "python": sys.version},
               "project_root": str(root), "project_source_sha256": {path: file_sha256(root / path) for path in source_files},
               "native_codec": asdict(API["config"]), "audit_tool_sha256": file_sha256(Path(__file__)),
               "no_formula_matching": True, "decoded_target_fallback": False, "bare_source_index_is_not_csv_proof": True}
    dump_json(output / "INPUT_RECEIPTS.json", receipt)
    training, blanks, split = prepare_training(training_path, args.target_field, args.plan_field,
                                              args.training_id_field or list(IDENTITY_FIELDS), args.split)
    csv_rows, columns = index_csv(args, root, output)
    by_hash, by_id = defaultdict(list), defaultdict(list)
    for row in csv_rows:
        if row["parse_status"] == "ok":
            by_hash[row["signature_sha256"]].append(row)
        if row["csv_material_id"]:
            by_id[row["csv_material_id"]].append(row)
    r5_records, r5_hash, r5_index, r5_errors = load_r5(r5_path, split)
    if r5_path:
        with (output / "r5_index.jsonl").open("w", encoding="utf-8") as handle:
            for row in r5_records + r5_errors:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    source_keys = Counter(item["source_key"] for item in training)
    results = []
    for item in training:
        result = {key: value for key, value in item.items() if key not in ("signature_answer", "target_answer", "plan")}
        result.update(schema="continuous_source_identity_row_v1", identity_verified=False,
                      source_csv_sha256=source_sha, training_jsonl_sha256=training_sha,
                      selected_cif_column=args.cif_column, parser_version=receipt["parser"]["pymatgen"],
                      original_split_binding=split, decoded_target_used=False)
        result["target_canonical_answer"] = item.get("target_answer")
        if item["status"] != "pending":
            results.append(result)
            continue
        decision = choose_identity(item, csv_rows, by_hash, by_id, source_sha, r5_hash, r5_index)
        selected = decision.pop("selected", None)
        result.update(decision)
        hint = integer(item.get("source_row_idx"))
        result["bare_source_row_hint_matches_selected_csv_row"] = hint == selected["csv_row_idx"] if selected else None
        if selected is not None:
            result.update(csv_row_idx=selected["csv_row_idx"], csv_row_number=selected["csv_row_number"],
                          material_id=selected["csv_material_id"] or None, cif_sha256_utf8=selected["cif_sha256_utf8"],
                          parser_warnings=selected["parser_warnings"], csv_parse_status=selected["parse_status"],
                          original_cif_codec_diagnostics=selected.get("codec_diagnostics"))
            try:
                if selected["csv_declared_splits"] and selected["csv_declared_splits"] != [split]:
                    raise ValueError("Selected CSV row declares a different source split")
                result.update(align_continuous(item, selected))
                if not result["canonicalized_original_Q_equals_target"] or not result["quantized_continuous_equals_target"]:
                    raise ValueError("Q(original continuous CIF aligned by the existing stable site permutation) differs from canonical source_answer.")
                result.update(status="verified", identity_verified=True)
            except Exception as exc:
                result.update(status="identified_source_geometry_or_split_mismatch", error=f"{type(exc).__name__}: {exc}")
        if source_keys[item["source_key"]] > 1 and not args.allow_repeated_source_records:
            result.update(status="duplicate_training_source_key", identity_verified=False,
                          source_occurrences=source_keys[item["source_key"]])
        results.append(result)
    reuse = defaultdict(set)
    for row in results:
        if row["identity_verified"]:
            reuse[row["csv_row_idx"]].add(row["source_key"])
    reused = {idx: sorted(keys) for idx, keys in reuse.items() if len(keys) > 1}
    for row in results:
        if row.get("csv_row_idx") in reused:
            row.update(status="csv_row_reused_by_distinct_training_sources", identity_verified=False,
                       conflicting_training_source_keys=reused[row["csv_row_idx"]])
        if not row["identity_verified"] and "continuous_aligned" in row:
            row["unverified_candidate_continuous"] = row.pop("continuous_aligned")
    with (output / "source_identity.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    duplicate_groups = [{"signature_sha256": digest, "csv_rows": [{"row_idx": r["csv_row_idx"], "material_id": r["csv_material_id"], "cif_sha256_utf8": r["cif_sha256_utf8"]} for r in group]}
                        for digest, group in by_hash.items() if len(group) > 1]
    dump_json(output / "DUPLICATE_FULL_ANSWERS.json", duplicate_groups)
    unique_sources = len(source_keys)
    count_ok = args.expected_sources is None or unique_sources == args.expected_sources
    verified = sum(row["identity_verified"] for row in results)
    verified_csv_rows = {row["csv_row_idx"] for row in results if row["identity_verified"]}
    successful = bool(results) and verified == len(results) and count_ok and not r5_errors
    counts = {"training_records": len(results), "unique_training_sources": unique_sources,
              "expected_training_sources": args.expected_sources, "source_count_matches_expected": count_ok,
              "blank_jsonl_lines": len(blanks), "csv_rows": len(csv_rows),
              "csv_parse_or_encode_errors": sum(row["parse_status"] != "ok" for row in csv_rows),
              "verified_training_records": verified, "unverified_training_records": len(results) - verified,
              "unique_matched_csv_rows": len(verified_csv_rows),
              "csv_full_answer_duplicate_groups": len(duplicate_groups), "r5_records": len(r5_records), "r5_errors": len(r5_errors)}
    summary = {"schema": "continuous_source_identity_audit_v1", "status": "pass" if successful else "issues",
               "audit_complete": True, "original_cif_identity_gate_passed": successful, "split": split,
               "counts": counts, "statuses": dict(Counter(row["status"] for row in results)),
               "identity_methods": dict(Counter(row.get("identity_method") for row in results if row["identity_verified"])),
               "csv_columns": columns, "blank_line_numbers": blanks, "input_receipts": receipt,
               "all_input_source_records_reported": len(results) == len(training),
               "site_permutation_direction": "aligned training slot -> parsed original CIF site index",
               "signature_definition": "Complete native Q answer, periodic 100->0 alias canonicalized, stable alphabetical element grouping only; includes every lattice and coordinate token.",
               "r5_identity_rule": "Only explicit source index or matching complete-answer evidence plus consistent stable material ID. Bare intermediate file ordinal is not original CSV proof.",
               "remaining_csv_rows_not_assigned": [r["csv_row_idx"] for r in csv_rows if r["csv_row_idx"] not in verified_csv_rows],
               "model_calls": 0, "GPU_calls": 0, "formula_matching": False, "decoded_targets_used": False,
               "scope": "Identity and numerical data alignment only; no energy/force/stability certificate and no filtered training dataset."}
    dump_json(output / "SUMMARY.json", summary)
    (output / "_AUDIT_COMPLETE").write_text("complete\n", encoding="utf-8")
    (output / ("_AUDIT_SUCCESS" if successful else "_AUDIT_ISSUES")).write_text(summary["status"] + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--training-jsonl", required=True)
    parser.add_argument("--output-dir", required=True, help="Fresh output directory; never overwrite an existing nonempty audit.")
    parser.add_argument("--project-root", help="Project containing src/crystal_dlm; otherwise locate from script/CWD ancestors.")
    parser.add_argument("--split", help="Original split binding (normally train or val); inferred only if all training declarations agree.")
    parser.add_argument("--expected-sources", type=int, help="Optional expected unique source count; no dataset size is hardcoded.")
    parser.add_argument("--source-csv-sha256", help="Optional known original CSV digest to verify.")
    parser.add_argument("--training-sha256", help="Optional known prepared JSONL digest to verify.")
    parser.add_argument("--cif-column", default="cif", help="Use this exact column only; never fall back to cif.conv.")
    parser.add_argument("--csv-id-column", default="material_id")
    parser.add_argument("--target-field", default="source_answer", help="Clean target dotted field; default source_answer, never automatic answer fallback.")
    parser.add_argument("--plan-field", default="plan_state")
    parser.add_argument("--training-id-field", action="append", help="Explicit stable material-ID dotted field; repeat if needed. Typed conditioning donor metadata is never searched.")
    parser.add_argument("--r5-jsonl", help="Optional original intermediate R5 records carrying stable IDs/explicit source indices.")
    parser.add_argument("--allow-repeated-source-records", action="store_true", help="Explicitly allow repeated views of the same source; records remain separate.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_size < 1 or (args.expected_sources is not None and args.expected_sources < 1):
        parser.error("workers, chunk-size and expected-sources must be positive")
    output = Path(args.output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("output-dir must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        summary = audit(args, output)
        print(json.dumps({"event": "audit_finished", "output_dir": str(output), "status": summary["status"],
                          "elapsed_seconds": time.monotonic() - started, **summary["counts"]}, ensure_ascii=False), flush=True)
        return 0 if summary["original_cif_identity_gate_passed"] else 2
    except Exception as exc:
        error = {"schema": "continuous_source_identity_audit_v1", "status": "error", "audit_complete": False,
                 "original_cif_identity_gate_passed": False, "error": f"{type(exc).__name__}: {exc}",
                 "elapsed_seconds": time.monotonic() - started, "partial_outputs_are_not_a_completed_identity_audit": True}
        dump_json(output / "SUMMARY.json", error)
        (output / "_AUDIT_ERROR").write_text(error["error"] + "\n", encoding="utf-8")
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
