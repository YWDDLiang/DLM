#!/usr/bin/env python3
"""Preserve evaluation requests while exporting CIFs/refiner inputs/endpoints."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from crystal_dlm.dynamic_crystal import arrays_to_structure, parse_dynamic_answer
from crystal_dlm.programmed_path_data import read_jsonl


def load_refined_payload(path):
    import torch
    spec = importlib.util.spec_from_file_location("state_refined_assembler", ROOT / "scripts/assemble_grounding_repeat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._refined_structures(torch.load(path, map_location="cpu", weights_only=False), invalid_as_failure=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paths-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--crysllmgen-dir", type=Path)
    p.add_argument("--refined-pt", type=Path)
    p.add_argument("--refiner-seed", type=int, default=20260905)
    args = p.parse_args()
    rows = read_jsonl(args.paths_jsonl)
    if any(r.get("source_split") != "evaluation" for r in rows):
        raise ValueError("these artifacts are evaluation-only, not training labels")
    ordinals = [int(r.get("evaluation_ordinal", r["sample_idx"])) for r in rows]
    if sorted(ordinals) != list(range(len(rows))):
        raise ValueError("evaluation requests must have a complete fixed ordinal ledger")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "cifs").mkdir()
    graphs, output, errors = [], [], Counter()
    refined, refined_failures = load_refined_payload(args.refined_pt) if args.refined_pt else ({}, {})
    if args.refined_pt and not set(refined).union(refined_failures) <= set(ordinals):
        raise ValueError("refiner returned an out-of-ledger request")
    process_one = graph_from_arrays = None
    if args.crysllmgen_dir:
        from scripts.sample_llada_dynamic_crystals import import_process_one, graph_from_arrays
        process_one = import_process_one(args.crysllmgen_dir)
    from pymatgen.core import Structure
    for ordinal, record in zip(ordinals, rows):
        result = dict(record, evaluation_ordinal=ordinal, endpoint="tau800" if args.refined_pt else "native",
                      parseable=False, artifact_error=None, native_execution_success=record.get("native_execution_success", record["success"]))
        if args.refined_pt:
            result.pop("structure", None)
            result.pop("cif_path", None)
        try:
            arrays = parse_dynamic_answer(record["body"], strict=True)
            native = arrays_to_structure(arrays)
            if args.refined_pt:
                if ordinal not in refined:
                    raise ValueError(refined_failures.get(ordinal, "refinement_missing_or_native_parser_failed"))
                structure = Structure.from_dict(refined[ordinal])
                if structure.num_sites != native.num_sites or structure.composition != native.composition:
                    raise ValueError("refinement changed exact composition")
            else:
                structure = native
            cif = structure.to(fmt="cif")
            parsed = Structure.from_str(cif, fmt="cif")  # Independent parse; no energy or distance selection.
            if parsed.num_sites != structure.num_sites or parsed.composition != structure.composition:
                raise ValueError("CIF roundtrip changed exact composition")
            # Evaluate every available CIF, including a complete endpoint left
            # by a failed runtime. Keep the runtime outcome in its own field.
            result.update(structure=structure.as_dict(), parseable=True, success=True,
                          cif_path=str(args.output_dir / "cifs" / f"{ordinal:06d}.cif"))
            Path(result["cif_path"]).write_text(cif, encoding="utf-8")
            if args.refined_pt:
                result["success"] = True
                result["diffusion_refinement_steps"] = 800
            if process_one is not None:
                try:
                    graph, _ = graph_from_arrays(arrays, process_one)
                    import numpy as np
                    if int(np.asarray(graph["n_atom"]).reshape(-1)[0]) != native.num_sites or Counter(np.asarray(graph["a_type"]).reshape(-1).tolist()) != Counter(native.atomic_numbers):
                        raise ValueError("graph preparation changed the exact composition")
                    graph.update(sample_idx=ordinal, refiner_seed=(args.refiner_seed + int(record["sample_idx"])) % (2**32))
                    graphs.append(graph)
                except Exception as error:
                    # A graph failure cannot remove a parseable CIF from the request ledger.
                    result["refiner_graph_error"] = f"{type(error).__name__}: {error}"
                    errors["refiner_graph_failure"] += 1
        except Exception as error:
            result.update(success=False, artifact_error=f"{type(error).__name__}: {error}")
            errors["refined_invalid_or_missing" if args.refined_pt else "native_parser_failure"] += 1
        output.append(result)
    with (args.output_dir / "paths.jsonl").open("x", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.crysllmgen_dir:
        import torch
        torch.save(graphs, args.output_dir / "proposal_graphs.pt")
    report = {"requests": len(rows), "parseable": sum(r["parseable"] for r in output),
              "successful": sum(r["success"] for r in output), "refiner_graphs": len(graphs), "errors": dict(errors),
              "endpoint": "tau800" if args.refined_pt else "native", "continuous_coordinates_preserved": bool(args.refined_pt),
              "source": str(args.paths_jsonl), "refined_payload": str(args.refined_pt) if args.refined_pt else None}
    (args.output_dir / "ARTIFACT_FINAL.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
