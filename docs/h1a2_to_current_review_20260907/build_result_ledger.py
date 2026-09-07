"""Build a read-only, source-bound historical result ledger; never rescore models."""
from pathlib import Path
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
CURRENT = HERE.parents[1]
OLD = CURRENT.parent / "diffsion_language_model_meets_diffusion"
EVIDENCE = HERE / "evidence"
EVIDENCE.mkdir(exist_ok=True)
sources, rows, pairs, row_checks = {}, [], [], {}


def source(name, path):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    dest = EVIDENCE / (name + path.suffix)
    if dest.exists():
        assert dest.read_bytes() == raw, name
    else:
        dest.write_bytes(raw)
    sources[name] = {"original_path": str(path), "local_file": str(dest.relative_to(HERE)),
                     "sha256": digest, "bytes": len(raw)}
    return json.loads(raw) if path.suffix == ".json" else raw.decode("utf-8")


def add(name, source_id, pointer, group, n, strict, meta, *, denominator="all_requested",
        endpoint="tau800", evidence="machine_terminal", note="", **extra):
    assert type(n) is int and type(strict) is int and type(meta) is int
    assert 0 <= strict <= meta <= n and n > 0
    rows.append({"method": name, "source_id": source_id, "source_pointer": pointer,
                 "comparison_group": group, "denominator": denominator, "requests": n,
                 "endpoint": endpoint, "strict_sun": strict, "meta_sun": meta,
                 "strict_percent": strict * 100 / n, "meta_percent": meta * 100 / n,
                 "joint_10_50_point_met": strict * 10 >= n and meta * 2 >= n,
                 "evidence": evidence, "note": note, **extra})


base = OLD / "workstreams/final_method_development_20260808/evidence"
exact = source("h1a2_exact1200_terminal", base / "h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v6_final/terminal_report.json")
old = exact["historical_old_protocol_recomputation"]
add("H1-A2 historical compatibility", "h1a2_exact1200_terminal", "/historical_old_protocol_recomputation",
    "H1A2_frozen_survivor1000", 1000, old["strict"]["full_sun_numerator_over_1000"], old["meta"]["full_sun_numerator_over_1000"],
    denominator="historical_first1000_survivors", note="The historical word raw denotes unadjusted rate; structures already passed model494 tau800.")
for arm, name in (("historical", "H1-A2 same frozen1000, fresh official"), ("control", "H1-A2 exact Plan1200"), ("candidate", "R03 exact Plan1200")):
    c = exact["cells"][arm]["counts"]
    add(name, "h1a2_exact1200_terminal", "/cells/" + arm, "H1A2_frozen_survivor1000" if arm == "historical" else "H1A2_R03_exact1200",
        c["raw_attempts"], c["strict_sun"], c["meta_sun"],
        denominator="historical_first1000_survivors" if arm == "historical" else "all_requested",
        note="Evaluator-only view of existing frozen1000." if arm == "historical" else "Same archived Plan IDs and per-ordinal seed contract within this pair.",
        reconstructed=c["reconstructed"], novel_unique=c["novel_unique"], hull_unknown=c["hull_unknown_reconstructed"],
        strict_stable=c["strict_stable_all_hull_known"], meta_stable=c["meta_stable_all_hull_known"])
pairs.append({"comparison_group": "H1A2_R03_exact1200", "source_id": "h1a2_exact1200_terminal",
              "reported_pairing": exact["paired_exact_mcnemar"]})
for arm in ("control", "candidate"):
    c = exact["historical_style_survivor_prefix1000"][arm]
    add(("H1-A2" if arm == "control" else "R03") + " exact1200 survivor-prefix view", "h1a2_exact1200_terminal",
        "/historical_style_survivor_prefix1000/" + arm, "H1A2_R03_exact1200", c["denominator"], c["strict_sun"], c["meta_sun"],
        denominator="first1000_body_successes_compatibility", note="Same generation as all1200; not a new run or independent estimate.")

archive = source("h1a2_r03_archive256_terminal", base / "h1_r03_h1a2_archived_first256_official_ehull_completion_v1/terminal_report.json")
assert sources["h1a2_r03_archive256_terminal"]["sha256"] == "63128c86b9ba0f1688c3f8db543512e6c19d5a68bbabd6dc538f10c1b79f85b0"
for arm, name in (("control", "H1-A2 archived256 official completion"), ("candidate", "R03 archived256 official completion")):
    v = archive["arms"][arm]
    add(name, "h1a2_r03_archive256_terminal", "/arms/" + arm, "archived_reproduction256_official_completion",
        v["denominators"]["all_attempts"], v["official"]["strict_full_sun"], v["official"]["meta_full_sun"],
        note="Same archived generation/refinement/energy and N/U; only official hull completion. One cohort, not independent confirmation.",
        reconstructed=v["denominators"]["reconstructed_exact_legacy"], novel_unique=v["denominators"]["novel_unique"],
        hull_unknown=v["denominators"]["hull_unknown"])
    raw_attempts = source("archive256_" + arm + "_attempts", base / "h1_r03_h1a2_archived_first256_official_ehull_completion_v1/official_results/arms" / arm / "attempt_results_official.jsonl")
    attempts = [json.loads(line) for line in raw_attempts.splitlines() if line.strip()]
    assert len(attempts) == 256 and {x["generation_ordinal"] for x in attempts} == set(range(256))
    assert len({x["attempt_id"] for x in attempts}) == 256
    for field in ("strict_full_sun", "meta_full_sun"):
        assert all(type(x["metrics"][field]) is bool for x in attempts)
        assert sum(x["metrics"][field] for x in attempts) == v["official"][field]
    row_checks[arm] = {x["generation_ordinal"]: x for x in attempts}
pairs.append({"comparison_group": "archived_reproduction256_official_completion", "source_id": "h1a2_r03_archive256_terminal",
              "reported_pairing": archive["paired_exact_mcnemar"]})
for metric in ("strict_full_sun", "meta_full_sun"):
    control_only = sum(row_checks["control"][i]["metrics"][metric] and not row_checks["candidate"][i]["metrics"][metric] for i in range(256))
    candidate_only = sum(row_checks["candidate"][i]["metrics"][metric] and not row_checks["control"][i]["metrics"][metric] for i in range(256))
    assert control_only == archive["paired_exact_mcnemar"][metric]["control_only"]
    assert candidate_only == archive["paired_exact_mcnemar"][metric]["candidate_only"]
pairs[-1]["independently_recounted_attempt_rows"] = 512

clean = source("r03_four_process_official_terminal", base / "h1_sun_official_gga_u_skip_unknown_reeval_v2/terminal_report.json")
cells = [c for c in clean["cell_reports"] if c["cell"]["panel"] == "R03_REFINED256_CURRENT_REPLAY"]
assert len(cells) == 4 and {c["cell"]["repeat"] for c in cells} == set(range(4))
for view in ("old", "clean"):
    add("R03 four-process " + view + " cache", "r03_four_process_official_terminal", "/cell_reports[panel=R03_REFINED256_CURRENT_REPLAY]/" + view,
        "R03_original_first256_process_repeat_panel", sum(c["denominators"]["all_attempts"] for c in cells),
        sum(c[view]["strict_full_sun"] for c in cells), sum(c[view]["meta_full_sun"] for c in cells),
        denominator="4_process_repeats_of_same256", note="Four CUDA realizations of the same Plan cohort, not four independent Planner samples; old/clean are views of the same structures.",
        per_repeat=[{"repeat": c["cell"]["repeat"], "strict": c[view]["strict_full_sun"], "meta": c[view]["meta_full_sun"]} for c in cells])

screens = CURRENT / "results/remote_screens"
grounding = source("grounding_fixed1000", screens / "GROUNDING_FIXED1000_FINAL.json")
for c in grounding["cells"]:
    add("Grounding " + c["arm"], "grounding_fixed1000", "/cells[arm=" + c["arm"] + "]", "grounding_fixed1000",
        c["requested"], c["strict"], c["meta"], reconstructed=c["reconstructed"], novel_unique=c["novel_unique"],
        strict_stable=c["strict_stable"], meta_stable=c["meta_stable"], note="Counterfactual Plan grounding versus matched teacher-CE control.")
pairs.append({"comparison_group": "grounding_fixed1000", "source_id": "grounding_fixed1000",
              "strict": grounding["strict_mcnemar"], "meta": grounding["meta_mcnemar"]})
sufficient = source("sufficient_dlm1000", screens / "DLM_SUFFICIENT_RAW1000_FINAL.json")
for c in sufficient["pooled"]:
    add("Sufficient DLM total epoch" + str(c["epoch"]), "sufficient_dlm1000", "/pooled[epoch=" + str(c["epoch"]) + "]",
        "sufficient_same_plan1000", c["requested"], c["strict_sun"], c["meta_sun"], reconstructed=c["reconstructed"],
        novel_unique=c["novel_unique"], strict_stable=c["strict_stable"], meta_stable=c["meta_stable"],
        note="raw1000 in the filename denotes requested attempts, not an unrefined endpoint; records explicitly include refined outputs.")
sgtc = source("sgtc_l7_official1000", screens / "sgtc_l7_official_final_20260829_v2/SGTC_L7_OFFICIAL_FINAL.json")
for c in sgtc["cells"]:
    add("SGTC L7 " + c["arm"], "sgtc_l7_official1000", "/cells[arm=" + c["arm"] + "]", "sgtc_l7_seed18_1000",
        c["requested"], c["strict_sun"], c["meta_sun"], reconstructed=c["reconstructed"], novel_unique=c["novel_unique"],
        strict_stable=c["strict_stable"], meta_stable=c["meta_stable"])
condition = source("condition_schedule_l6", screens / "DLM_CONDITION_SCHEDULE_L6_FINAL.json")
for c in condition["pooled_repeat_sum"]:
    add("L6 " + c["arm"], "condition_schedule_l6", "/pooled_repeat_sum[arm=" + c["arm"] + "]", "L6_fixed_two_seed_panel",
        c["requested"], c["strict_sun"], c["meta_sun"], denominator="reported_two_seed_256_sum", reconstructed=c["reconstructed"],
        novel_unique=c["novel_unique"], strict_stable=c["strict_stable"], meta_stable=c["meta_stable"])
conversion = source("refiner_conversion_l6", screens / "DLM_REFINER_EFFECT_L6_DIAGNOSTIC.json")
for c in conversion["pooled_repeat_sum"]:
    if c["stage"] == "raw":
        add("L6 " + c["source_arm"] + " raw", "refiner_conversion_l6", "/pooled_repeat_sum[source_arm=" + c["source_arm"] + ",stage=raw]",
            "L6_fixed_two_seed_panel", c["requested"], c["strict_sun"], c["meta_sun"], endpoint="raw_before_model494",
            denominator="reported_two_seed_256_sum", reconstructed=c["reconstructed"], novel_unique=c["novel_unique"])
tau = source("tau_calibration_l6", screens / "DLM_REFINER_TAU_L6_FINAL.json")
for c in tau["pooled_repeat_sum"]:
    if c["tau"] in (200, 500):
        add("L6 full_axis tau" + str(c["tau"]), "tau_calibration_l6", "/pooled_repeat_sum[tau=" + str(c["tau"]) + "]",
            "L6_fixed_two_seed_panel", c["requested"], c["strict_sun"], c["meta_sun"], endpoint="tau" + str(c["tau"]),
            denominator="reported_two_seed_256_sum", reconstructed=c["reconstructed"], novel_unique=c["novel_unique"])

latest = source("september7_completed_snapshot", CURRENT / "docs/v3_scientific_audit_20260906/evidence_v3_failure_20260907/SUN_SNAPSHOT_AFTER_40088.json")
for c in latest["development"]:
    assert c["complete"]
    for endpoint in ("raw", "tau800"):
        d = c[endpoint]["counts"]
        add(c["method"], "september7_completed_snapshot", "/development[method=" + c["method"] + "] / " + endpoint,
            "september_fixed_development256_commonR", d["requests"], d["strict_sun"], d["meta_sun"], endpoint=endpoint,
            reconstructed=d["reconstructed"], novel_unique=d["novel_unique"], strict_stable=d["strict_stable"], meta_stable=d["meta_stable"],
            verified_strict_sun=d["verified_strict_sun"], verified_meta_sun=d["verified_meta_sun"])
for c in latest["independent_original_k8"]:
    for endpoint in ("raw", "tau800"):
        d = c[endpoint]["counts"]
        add("K8 independent " + str(c["requests"]), "september7_completed_snapshot", "/independent_original_k8[requests=" + str(c["requests"]) + "] / " + endpoint,
            "september_independent1200", d["requests"], d["strict_sun"], d["meta_sun"], endpoint=endpoint,
            denominator="parser_only_prefix1000" if c["requests"] == 1000 else "all_requested",
            note="Same underlying1200 generation; prefix1000 is a reporting view, not another cohort.")

composition = source("c3fd_v25_composition2000", screens / "C3FD_V25_REQUESTED1000_FINAL.json")
source("aggregate_headline_boundary", CURRENT / "docs/v3_scientific_audit_20260906/historical/docs/STORY_REVIEW_INTERNAL.md")
source("potential_closure_report", CURRENT / "docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/09_EFFICIENCY_FIRST_POTENTIAL_CLOSURE_PLAN.md")
result = {"scope": "Historical evidence ledger; no cross-cohort ranking, no model/evaluator run, no outcome selection", "sources": sources,
          "rows": rows, "paired_reports": pairs,
          "composition_only_evidence": {"source_id": "c3fd_v25_composition2000", "pooled": composition["pooled"], "structural_SUN_not_measured_here": True},
          "unbound_aggregate_claim": {"claimed_strict_per1000": 105, "claimed_meta_per1000": 488,
              "source_id": "aggregate_headline_boundary", "status": "documented_aggregate_headline_not_bound_to_one1000_cohort", "included_as_measured_row": False},
          "additional_reported_pilot": {"method": "Potential-Closure stream17", "source_id": "potential_closure_report", "requests": 256,
              "potential_raw": [10, 58], "potential_tau800": [20, 125], "matched_control_tau800": [17, 115],
              "raw_energy_mean_delta_eV_atom": -0.236, "reported_cluster_interval": [-0.599, 0.131],
              "status": "completed_pilot_recorded_in_execution_worklog; primary_terminal_not_copied_in_this_review", "joint_target_met": False,
              "not_automatically_current_best": "different stream/endpoint contract; retain promising paired direction without merging with current256"},
          "no_new_GPU_or_MLIP": True, "no_historical_file_modified": True}
(HERE / "HISTORICAL_RESULT_LEDGER.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
keys = list(dict.fromkeys(k for row in rows for k in row))
with (HERE / "HISTORICAL_RESULT_LEDGER.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
lines = ["# Historical measured-result ledger", "", "Each comparison group is a different estimand. Counts are Strict/Meta; no across-group ranking is performed.", "",
         "| Group | Method | Endpoint | Denominator | Strict/Meta | Percent |", "|---|---|---|---:|---:|---:|"]
for row in rows:
    lines.append(f"| {row['comparison_group']} | {row['method']} | {row['endpoint']} | {row['requests']} ({row['denominator']}) | {row['strict_sun']}/{row['meta_sun']} | {row['strict_percent']:.2f}%/{row['meta_percent']:.2f}% |")
lines += ["", "Original source capsules, source JSON pointers, paired results and independence caveats are in HISTORICAL_RESULT_LEDGER.json. The 105/488 aggregate claim is not treated as one measured1000 cohort.", ""]
(HERE / "HISTORICAL_RESULT_LEDGER.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"rows": len(rows), "source_capsules": len(sources), "joint_point_hits": [{k: r[k] for k in ("method", "requests", "strict_sun", "meta_sun", "denominator", "comparison_group")} for r in rows if r["joint_10_50_point_met"]]}))
