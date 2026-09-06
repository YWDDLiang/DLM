#!/usr/bin/env python3
"""Eight fixed original singleton prefixes; two frozen models; no new trajectories."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

SCHEMA = "k4k8_equal_same_state_probability_mixture_v1"
TEMPERATURE = 0.7


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def mix_vectors(left, right):
    """Original raw-dtype masks precede FP64 probability mixing."""
    import torch
    if left.ndim != 1 or left.shape != right.shape:
        raise ValueError("component vectors must have the same full vocabulary")
    masks = []
    for vector in (left, right):
        if not vector.is_floating_point() or bool(torch.isnan(vector).any()) or bool(torch.isposinf(vector).any()):
            raise ValueError("invalid component logits; no fallback component")
        masks.append(torch.isfinite(vector) & (vector > torch.finfo(vector.dtype).min))
    if not torch.equal(masks[0], masks[1]) or not bool(masks[0].any()):
        raise ValueError("unequal or empty canonical support; no union/fallback")
    legal = masks[0]
    logp4 = torch.log_softmax(left[legal].double() / TEMPERATURE, -1)
    logp8 = torch.log_softmax(right[legal].double() / TEMPERATURE, -1)
    logq = torch.logaddexp(logp4, logp8) - math.log(2.0)
    if not bool(torch.isfinite(logp4).all() & torch.isfinite(logp8).all() & torch.isfinite(logq).all()):
        raise ValueError("nonfinite component/mixed log probabilities")
    output = torch.full(left.shape, torch.finfo(torch.float64).min, dtype=torch.float64, device=left.device)
    output[legal] = TEMPERATURE * logq
    p4, p8, q = logp4.exp(), logp8.exp(), logq.exp()
    kl4 = float((p4 * (logp4 - logq)).sum())
    kl8 = float((p8 * (logp8 - logq)).sum())
    normalization_error = max(abs(float(p.sum()) - 1.0) for p in (p4, p8, q))
    dominance_slack = min(float((logq - logp + math.log(2.0)).min()) for logp in (logp4, logp8))
    direct_error = float((q - (p4 + p8) / 2).abs().max())
    if normalization_error > 1e-12 or direct_error > 1e-12 or dominance_slack < -1e-12:
        raise RuntimeError("mixture normalization/arithmetic/dominance check failed")
    if min(kl4, kl8) < -1e-12 or max(kl4, kl8) > math.log(2.0) + 1e-12:
        raise RuntimeError("component-to-mixture local KL bound failed")
    return output, legal, {"legal_count": int(legal.sum()), "normalization_error": normalization_error,
        "arithmetic_probability_error": direct_error, "log_dominance_slack": dominance_slack,
        "kl_k4_to_mixture": kl4, "kl_k8_to_mixture": kl8, "js_nats": (kl4 + kl8) / 2,
        "tv_components": float((p4 - p8).abs().sum() / 2),
        "maximum_component_probability_difference": float((p4 - p8).abs().max()),
        "entropy_k4": float(-(p4 * logp4).sum()), "entropy_k8": float(-(p8 * logp8).sum()),
        "entropy_mixture": float(-(q * logq).sum()), "output_dtype": str(output.dtype),
        "input_dtypes": [str(left.dtype), str(right.dtype)], "support_identical": True,
        "mask_inferred_before_cast": True}


def original_draw(vector, seed, salt):
    from crystal_dlm.spad_generation import _transaction_candidate_tokens
    result = _transaction_candidate_tokens(vector[None, None], active_absolute_positions={0: 0},
        temperature=TEMPERATURE, remasking="low_confidence", sampling_seeds_by_batch=[seed], salt=salt)
    return int(result[0, 0])


def fixed_states(manifest):
    from crystal_dlm.programmed_path_runtime import replay_scalar_states
    selected = []
    # Selection depends only on original ordinal, phase and scalar type.
    rules = [(0, "construct", "cell"), (0, "construct", "Z"),
             (1, "cooperative", "Z"), (2, "closure", "Z")]
    origins = [{int(r["condition_ordinal"]): r for r in manifest["methods"][name]["first_four"]}
               for name in ("k4", "k8")]
    for ordinal in (0, 1, 2):
        a, b = origins[0][ordinal], origins[1][ordinal]
        if any(a.get(k) != b.get(k) for k in ("prompt_token_ids", "species_program", "sampling_seed")) or a["trace"]["initial_body"] != b["trace"]["initial_body"]:
            raise ValueError("the original component conditions, program or seed differ")
    for origin in ("k4", "k8"):
        rows = {int(r["condition_ordinal"]): r for r in manifest["methods"][origin]["first_four"]}
        if sorted(rows) != [0, 1, 2, 3]:
            raise ValueError("frozen package is not the original first four requests")
        for ordinal, phase, kind in rules:
            record = rows[ordinal]
            if int(record["sampling_batch_size"]) != 1 or record["trace"]["temperature"] != TEMPERATURE:
                raise ValueError("selected prefix requires original singleton batch and T=.7")
            if record["trace"].get("short_contact_policy") or record["trace"].get("mixture_policy"):
                raise ValueError("prefix origin must be the unchanged legacy policy")
            events = [e for e in record["trace"]["events"] if e["op"] == "draw"]
            states = list(replay_scalar_states(record["trace"]))
            if len(events) != len(states):
                raise ValueError("scalar replay did not preserve original draw ledger")
            match = None
            for state, event in zip(states, events, strict=True):
                pos = state["position"]
                scalar_match = 1 <= pos <= 6 if kind == "cell" else pos >= 10 and (pos - 10) % 4 == 0
                if state["phase"] == phase and scalar_match:
                    match = (state, event)
                    break
            if match is None:
                raise ValueError("fixed prefix absent; do not select a replacement")
            state, event = match
            selected.append({"origin": origin, "record": record, "state": state, "salt": int(event["salt"]),
                "key": f"{origin}:{ordinal}:{phase}:{kind}:{state['decision_index']}",
                "condition_ordinal": ordinal, "kind": kind})
    if len(selected) != 8:
        raise AssertionError("the bounded probe has exactly eight selected states")
    return selected


def self_test():
    import torch
    from crystal_dlm.probability_mixture_sampling import mix_post_hard_vectors
    passed = []
    left = torch.tensor([2., -1., 0., torch.finfo(torch.float32).min])
    same, legal, metrics = mix_vectors(left, left)
    assert torch.allclose(torch.softmax(same / TEMPERATURE, -1), torch.softmax(left.double() / TEMPERATURE, -1), atol=1e-15, rtol=1e-14)
    passed.append("identical components reproduce their normalized probabilities")
    right = torch.tensor([-2., 0.5, 1., torch.finfo(torch.float32).min])
    mixed, _, metrics = mix_vectors(left, right)
    arithmetic = (torch.softmax(left.double() / TEMPERATURE, -1) + torch.softmax(right.double() / TEMPERATURE, -1)) / 2
    geometric = torch.softmax((left.double() + right.double()) / (2 * TEMPERATURE), -1)
    assert torch.allclose(torch.softmax(mixed / TEMPERATURE, -1), arithmetic, atol=1e-15, rtol=1e-14)
    assert float((arithmetic - geometric).abs().max()) > .01
    passed.append("arithmetic probability mixture differs from averaged logits")
    bf = torch.tensor([2., -1., torch.finfo(torch.bfloat16).min], dtype=torch.bfloat16)
    fp = torch.tensor([1., 0., torch.finfo(torch.float32).min], dtype=torch.float32)
    out, mask, _ = mix_vectors(bf, fp)
    assert mask.tolist() == [True, True, False] and out[-1] == torch.finfo(torch.float64).min
    passed.append("original BF16 and FP32 sentinels stay outside support")
    for bad in (torch.tensor([1., 0., 2.]), torch.tensor([1., float('nan'), torch.finfo(torch.float32).min]),
                torch.full((3,), torch.finfo(torch.float32).min)):
        try:
            mix_vectors(fp, bad)
        except ValueError:
            continue
        raise AssertionError("component failure incorrectly fell back")
    passed.append("unequal support, NaN and empty component reject without fallback")
    common = 0
    generator = torch.Generator().manual_seed(571)
    for index in range(100):
        a, b = torch.randn(9, generator=generator), torch.randn(9, generator=generator)
        q, _, _ = mix_vectors(a, b)
        draws = [original_draw(v, 6000 + index, 100000000 + 10007 * index) for v in (a, b, q)]
        if draws[0] == draws[1]:
            common += 1
            assert draws[2] == draws[0]
    assert common > 0
    passed.append("actual original same-seed Gumbel keeps a shared component winner")
    a, b = torch.tensor([.60, .39, .01]), torch.tensor([.01, .39, .60])
    assert int(a.argmax()) == 0 and int(b.argmax()) == 2 and int(((a + b) / 2).argmax()) == 1
    passed.append("mixture may choose a third action when component winners differ")
    token = original_draw(mixed, 8732798746829386280, 100000000)
    actual = float(torch.log_softmax(mixed / TEMPERATURE, -1)[token])
    expected = math.log(float(arithmetic[token]))
    assert abs(actual - expected) < 1e-12
    passed.append("original Gumbel and selected-action trace use the same mixed law")
    production, _ = mix_post_hard_vectors(left, right, temperature=TEMPERATURE)
    assert production.dtype == torch.float64 and torch.equal(production[~legal], mixed[~legal])
    assert float((production[legal] - mixed[legal]).abs().max()) < 1e-12
    production_bf, _ = mix_post_hard_vectors(bf, fp, temperature=TEMPERATURE)
    assert torch.equal(production_bf, out)
    passed.append("production helper agrees with independent formula and original-dtype masks")
    return {"status": "PASS", "checks": passed, "checks_passed": len(passed),
        "actual_gumbel_random_cases": 100, "shared_component_winners": common,
        "real_model_forwards": 0, "GPU_tested": False, "schema": SCHEMA}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True, help="Frozen code checkout containing src/")
    p.add_argument("--input-manifest", type=Path)
    p.add_argument("--candidate-spec", type=Path, default=Path(__file__).with_name("candidate_spec.json"))
    p.add_argument("--model-path")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    sys.path.insert(0, str(args.source_root.resolve() / "src"))
    import torch
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.self_test:
        report = self_test()
        report.update(probe_sha256=digest(__file__), python=sys.version,
            torch_version=torch.__version__, torch_cuda_build=torch.version.cuda,
            production_helper_sha256=digest(args.source_root / "src/crystal_dlm/probability_mixture_sampling.py"))
        if args.input_manifest:
            frozen = json.loads(args.input_manifest.read_text())
            states = fixed_states(frozen)
            report["fixed_prefix_selection"] = [{"key": v["key"], "sampling_batch_size": v["record"]["sampling_batch_size"],
                "num_atoms": v["record"]["num_atoms"], "salt": v["salt"],
                "state_sha256": value_digest(v["state"])} for v in states]
            report["input_manifest_sha256"] = digest(args.input_manifest)
        (args.output_dir / "CPU_CHECKS.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return
    if not args.input_manifest or not args.model_path or not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("real probe needs frozen manifest, base model and allocated CUDA device")
    from crystal_dlm.programmed_path_data import compile_condition, load_path_model, read_jsonl
    from crystal_dlm.short_contact_sampling import validate_legacy_contact_policy
    from crystal_dlm.sampling_layout import sampling_batches
    from scripts.sample_state_programmed_paths import make_sampler
    from scripts.sample_llada_dynamic_crystals import build_dynamic_lightweight_constraints
    from crystal_dlm.probability_mixture_sampling import mix_post_hard_vectors
    manifest = json.loads(args.input_manifest.read_text())
    spec = json.loads(args.candidate_spec.read_text())
    expected_spec = {"schema": SCHEMA, "components": ["k4", "k8"], "weights": [.5, .5],
        "temperature": TEMPERATURE, "phases": ["construct", "cooperative", "closure"],
        "short_contact_enabled": False, "weight_search": False, "case_routing": False,
        "inference_energy_evaluations": 0, "new_training": False,
        "probe_selected_states": 8, "probe_actual_model_forwards": 16,
        "probe_selection_uses_outcomes": False, "probe_difference_is_stability_gate": False,
        "input_prefix_manifest_sha256": digest(args.input_manifest)}
    if any(spec.get(key) != value for key, value in expected_spec.items()):
        raise ValueError("candidate specification differs from the single frozen mixture")
    selected = fixed_states(manifest)
    # Source receipts bind actual immutable ledgers; no output labels select states.
    all_records = {}
    checkpoint_bindings = {}
    for name in ("k4", "k8"):
        entry = manifest["methods"][name]
        source = Path(entry["source_path"])
        if digest(source) != manifest["source_receipts"][str(source)] or not (source.parent / "_SUCCESS").is_file():
            raise ValueError("actual original source ledger is changed or incomplete")
        rows = read_jsonl(source)
        lookup = {int(r["condition_ordinal"]): r for r in rows}
        if len(lookup) != len(rows) or any(lookup[int(r["condition_ordinal"])] != r for r in entry["first_four"]):
            raise ValueError("manifest prefix differs from original full ledger")
        final = validate_legacy_contact_policy(entry["checkpoint"])
        reference = spec["immutable_policy_references"][name]
        if (reference["checkpoint_path"] != entry["checkpoint"] or reference["original_paths"] != str(source)
                or reference["original_paths_sha256"] != digest(source)
                or reference["train_final_sha256"] != digest(Path(entry["checkpoint"]).parent.parent / "TRAIN_FINAL.json")):
            raise ValueError("actual component metadata differs from candidate policy receipts")
        checkpoint_bindings[name] = {"checkpoint": entry["checkpoint"],
            "train_final_sha256": digest(Path(entry["checkpoint"]).parent.parent / "TRAIN_FINAL.json"),
            "train_final_policy_path": final["policy_path"], "source_sha256": digest(source)}
        all_records[name] = rows
    torch.cuda.set_device(torch.device(args.device))
    torch.set_num_threads(2)
    device = torch.device(args.device)
    captures = {name: [] for name in ("k4", "k8")}
    abi = None
    started = time.monotonic()
    actual_calls = 0
    load_seconds = {}
    forward_seconds = {}
    peak_allocated = 0.
    state_records = []
    for choice in selected:
        r, s = choice["record"], choice["state"]
        state_records.append({"key": choice["key"], "origin": choice["origin"],
            "trajectory_id": r["trajectory_id"], "condition_ordinal": choice["condition_ordinal"],
            "program": r["species_program"], "num_atoms": r["num_atoms"], "sampling_seed": r["sampling_seed"],
            "salt": choice["salt"], "sampling_batch_size_in_original": r["sampling_batch_size"],
            "sampling_layout_world_size_in_original": r.get("sampling_layout_world_size"),
            "batch_ordinals": [choice["condition_ordinal"]], "prompt_token_ids": r["prompt_token_ids"],
            **s})
    for component in ("k4", "k8"):
        load_started = time.monotonic()
        model, tokenizer = load_path_model(args.model_path, manifest["methods"][component]["checkpoint"], device, trainable=False)
        if hasattr(model, "raw_initialization"):
            raise ValueError("requires frozen legacy state-conditioned K4/K8")
        load_seconds[component] = time.monotonic() - load_started
        signature = {"vocab_sha256": value_digest(tokenizer.get_vocab()),
                     "output_vocab_size": int(model.get_output_embeddings().weight.shape[0])}
        if abi is None:
            abi = signature
        elif signature != abi:
            raise ValueError("component tokenizer/output ABI differs")
        constraints = build_dynamic_lightweight_constraints(tokenizer, duplicate_coordinate_mask=True,
            lattice_volume_mask=True, min_lattice_rad=1e-4, canonicalize_periodic_alias=True,
            pbc_min_distance_mask=True, pbc_min_distance_A=.5, pbc_image_radius=2)
        # Verify the original effective modulo layout, without inventing stored metadata.
        for origin in ("k4", "k8"):
            compiled_items = []
            for r in sorted(all_records[origin], key=lambda v: v["condition_ordinal"]):
                c = compile_condition(r, tokenizer, mask_id=126336, purpose="evaluation")
                if c["prompt_token_ids"] != r["prompt_token_ids"] or c["initial_body"] != r["trace"]["initial_body"]:
                    raise ValueError("component compiler differs from original prompt/scaffold")
                compiled_items.append((int(r["condition_ordinal"]), int(r["candidate_index"]), c))
            membership = {}
            for batch in sampling_batches(compiled_items, batch_size=4, rank=0, world_size=1, layout_world_size=2):
                if any(int(item[2]["record"]["sampling_batch_size"]) != len(batch) for item in batch):
                    raise ValueError("effective original layout cannot be reconstructed")
                for item in batch:
                    membership[item[0]] = [entry[0] for entry in batch]
            if any(membership[i] != [i] for i in (0, 1, 2)):
                raise ValueError("fixed selected prefixes were not original complete singleton batches")
        calls = [0]
        def count(_module, _arguments):
            calls[0] += 1
        handle = model.register_forward_pre_hook(count)
        torch.cuda.reset_peak_memory_stats(device)
        neural_started = time.monotonic()
        try:
            for choice in selected:
                r, s = choice["record"], choice["state"]
                c = compile_condition(r, tokenizer, mask_id=126336, purpose="evaluation")
                sampler = make_sampler(model, tokenizer, constraints, [c], [r["sampling_seed"]], TEMPERATURE)
                x = torch.tensor([c["prompt_token_ids"] + s["input_body"]], device=device)
                old = torch.tensor([c["prompt_token_ids"] + s["old_body"]], device=device)
                sampler._prepare(x)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, unavailable = sampler.processed_logits(x, old, {0: s["position"]},
                        {0: s["transaction_positions"]}, torch.ones_like(x), phase=s["phase"])
                if unavailable:
                    raise RuntimeError(f"component {component} unavailable at {choice['key']}; no fallback")
                vector = logits[0, len(c["prompt_token_ids"]) + s["position"]].detach().cpu().clone()
                captures[component].append(vector)
                if choice["origin"] == component:
                    observed = float(torch.log_softmax(vector.double() / TEMPERATURE, -1)[s["target_token"]])
                    if abs(observed - s["recorded_log_probability"]) > 1e-6:
                        raise RuntimeError("original component prefix log probability did not replay within 1e-6")
                    if original_draw(vector.to(device), int(r["sampling_seed"]), choice["salt"]) != s["target_token"]:
                        raise RuntimeError("original component same-seed Gumbel token did not replay")
                del sampler, logits, x, old
        finally:
            handle.remove()
        torch.cuda.synchronize(device)
        forward_seconds[component] = time.monotonic() - neural_started
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device) / 2 ** 30)
        if calls[0] != 8:
            raise RuntimeError("bounded probe did not perform exactly eight forwards per component")
        actual_calls += calls[0]
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"component_complete": component, "actual_forwards": calls[0],
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
    metrics = []
    for index, choice in enumerate(selected):
        a, b = captures["k4"][index], captures["k8"][index]
        q, legal, result = mix_vectors(a, b)
        production, production_diagnostic = mix_post_hard_vectors(a.to(device), b.to(device), temperature=TEMPERATURE)
        production = production.detach().cpu()
        if production.dtype != torch.float64 or not torch.equal(production[~legal], q[~legal]):
            raise RuntimeError("production helper changed output dtype or forbidden support")
        production_error = float((production[legal] - q[legal]).abs().max())
        if production_error > 1e-12:
            raise RuntimeError("production helper differs from independent same-state mixture formula")
        r, s = choice["record"], choice["state"]
        seed, salt = int(r["sampling_seed"]), choice["salt"]
        # The original row-local generator lives on CUDA. Equal CPU seeds do
        # not reproduce its random array, so every real counterfactual stays
        # on the original device even though the saved vectors live on CPU.
        winners = [original_draw(v.to(device), seed, salt) for v in (a, b, production)]
        if not all(bool(legal[v]) for v in winners):
            raise RuntimeError("original Gumbel left common canonical support")
        shared = winners[0] == winners[1]
        if shared and winners[2] != winners[0]:
            raise RuntimeError("mixture changed a common same-Gumbel component winner")
        origin_vector = a if choice["origin"] == "k4" else b
        origin_error = abs(float(torch.log_softmax(origin_vector.double() / TEMPERATURE, -1)[s["target_token"]]) - s["recorded_log_probability"])
        qlogp = torch.log_softmax(production / TEMPERATURE, -1)
        mixture_selected_logp = float(qlogp[winners[2]])
        recomputed = torch.logaddexp(torch.log_softmax(a[legal].double() / TEMPERATURE, -1),
            torch.log_softmax(b[legal].double() / TEMPERATURE, -1)) - math.log(2)
        mapped = (legal.nonzero(as_tuple=True)[0] == winners[2]).nonzero(as_tuple=True)[0].item()
        logp_error = abs(mixture_selected_logp - float(recomputed[mapped]))
        if logp_error > 1e-12:
            raise RuntimeError("selected-action trace log probability differs from q")
        metrics.append({"key": choice["key"], **result, "origin_logp_error": origin_error,
            "production_cuda_raw_logit_error": production_error,
            "k4_winner": winners[0], "k8_winner": winners[1], "mixture_winner": winners[2],
            "shared_component_winner": shared, "shared_winner_preserved": winners[2] == winners[0] if shared else None,
            "mixture_selected_log_probability": mixture_selected_logp, "trace_logp_error": logp_error,
            "legal_ids": legal.nonzero(as_tuple=True)[0].tolist(),
            "k4_legal_logits": a[legal].double().tolist(), "k8_legal_logits": b[legal].double().tolist(),
            "mixture_legal_raw_logits": production[legal].tolist()})
    (args.output_dir / "STATE_BINDINGS.json").write_text(json.dumps(state_records, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "PREFIX_METRICS.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    torch.save(captures, args.output_dir / "full_original_dtype_vectors.pt")
    source_files = ["src/crystal_dlm/programmed_path_runtime.py", "src/crystal_dlm/programmed_path_data.py",
        "src/crystal_dlm/spad_generation.py", "src/crystal_dlm/sampling_layout.py",
        "src/crystal_dlm/probability_mixture_sampling.py",
        "src/scripts/sample_state_programmed_paths.py", "src/scripts/sample_llada_dynamic_crystals.py"]
    report = {"status": "PASS", "scope": "numerical_distribution_probe_only_no_stability_claim",
        "schema": SCHEMA, "weights": [.5, .5], "temperature": TEMPERATURE,
        "state_count": 8, "actual_model_forwards": actual_calls, "new_trajectories": 0,
        "energy_evaluations": 0, "optimizer_steps": 0, "contact_enabled": False,
        "state_selection": "each origin: ordinal0 first construct cell and Z; ordinal1 first cooperative Z; ordinal2 first closure Z",
        "input_manifest_sha256": digest(args.input_manifest), "candidate_spec_sha256": digest(args.candidate_spec),
        "checkpoint_bindings": checkpoint_bindings,
        "source_sha256": {path: digest(args.source_root / path) for path in source_files},
        "probe_sha256": digest(__file__), "token_abi": abi,
        "maximum_origin_logp_error": max(r["origin_logp_error"] for r in metrics),
        "maximum_production_cuda_raw_logit_error": max(r["production_cuda_raw_logit_error"] for r in metrics),
        "maximum_tv_components": max(r["tv_components"] for r in metrics),
        "distribution_difference_above_numerical_floor_1e_8": any(r["tv_components"] > 1e-8 for r in metrics),
        "distribution_difference_is_stability_gate": False,
        "component_winners_differ": sum(r["k4_winner"] != r["k8_winner"] for r in metrics),
        "mixture_winner_differs_from_k4": sum(r["mixture_winner"] != r["k4_winner"] for r in metrics),
        "mixture_winner_differs_from_k8": sum(r["mixture_winner"] != r["k8_winner"] for r in metrics),
        "shared_component_winners": sum(r["shared_component_winner"] for r in metrics),
        "shared_winner_violations": 0, "gumbel_device": str(device), "load_seconds": load_seconds,
        "forward_and_replay_seconds": forward_seconds, "peak_allocated_GiB": peak_allocated,
        "elapsed_seconds": time.monotonic() - started,
        "sampling_layout_world_size_in_original": None, "reconstructed_effective_layout": 2,
        "all_selected_original_batch_memberships": "singleton, checked against full original ledgers"}
    (args.output_dir / "PROBE.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
