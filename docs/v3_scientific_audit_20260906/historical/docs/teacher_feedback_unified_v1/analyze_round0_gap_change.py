import json, math, pathlib, statistics
import sys
ROOT=pathlib.Path(sys.argv[1])
# This script writes a new immutable diagnostic directory, never evaluation inputs.
def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
def comp(p, reduced=False):
    counts=[int(n) for n in p["counts"]]
    g=math.gcd(*counts) if reduced else 1
    return tuple(sorted((e,n//g) for e,n in zip(p["elements"],counts)))
def fmt(c):
    return "".join(e+(str(n) if n!=1 else "") for e,n in c)
a={r["sample_idx"]:r for r in rows(ROOT/"runs/spad_state_reference_verification_39893/native-evaluation/attempt_results.jsonl")}
b={r["sample_idx"]:r for r in rows(ROOT/"runs/spad_state_eval_method_39910/evaluation/attempt_results.jsonl")}
ap={r["sample_idx"]:r for r in rows(ROOT/"runs/spad_state_eval_reference_39884/native/paths.jsonl")}
bp={r["sample_idx"]:r for r in rows(ROOT/"runs/spad_state_eval_method_39910/native/paths.jsonl")}
train=rows(ROOT/"runs/spad_state_train_conditions_39857/conditions/plans_for_dlm.jsonl")
diag=json.loads((ROOT/"runs/spad_state_path_train_39892/BASIN_DIAGNOSTIC.json").read_text())
dr={str(r["group_id"]):r for r in diag["teacher_reweighting"]["conditions"]}
train_exact={}
train_reduced={}
for r in train:
    train_exact.setdefault(comp(r["plan_state"]),[]).append(r)
    train_reduced.setdefault(comp(r["plan_state"],True),[]).append(r)
out=[]
for i in sorted(a):
    ra,rb=a[i],b[i]
    if not (ra["terminal_verified"] and rb["terminal_verified"]):
        continue
    da=rb["gap_eV_atom"]-ra["gap_eV_atom"]
    de=rb["raw_energy_eV_atom"]-ra["raw_energy_eV_atom"]
    dt=rb["terminal_energy_eV_atom"]-ra["terminal_energy_eV_atom"]
    assert abs(da-(de-dt))<1e-8
    exact=train_exact.get(comp(bp[i]["plan_state"]),[])
    matched=[dr[str(r["group_id"])] for r in exact if str(r["group_id"]) in dr]
    ps=bp[i].get("trace_summary") or {}
    out.append({"sample_idx":i,"composition":fmt(comp(bp[i]["plan_state"])),"N":bp[i]["num_atoms"],
       "delta_A":da,"delta_raw":de,"delta_B":dt,
       "reference_A":ra["gap_eV_atom"],"method_A":rb["gap_eV_atom"],
       "reference_raw":ra["raw_energy_eV_atom"],"method_raw":rb["raw_energy_eV_atom"],
       "reference_terminal":ra["terminal_energy_eV_atom"],"method_terminal":rb["terminal_energy_eV_atom"],
       "reference_steps":ra["actual_relaxation_steps"],"method_steps":rb["actual_relaxation_steps"],
       "same_final_tokens":ap[i].get("final_body_token_ids")==bp[i].get("final_body_token_ids"),
       "cooperative_accepted":ps.get("cooperative_accepted"),
       "cooperative_changed":ps.get("committed_changed_scalars_by_phase",{}).get("cooperative",0),
       "rollback_reasons":ps.get("rollback_reasons"),
       "teacher_exact_composition_groups":[r["group_id"] for r in exact],
       "teacher_same_reduced_composition_groups":[r["group_id"] for r in train_reduced.get(comp(bp[i]["plan_state"],True),[])],
       "teacher_energy_diagnostics":matched})
n=len(out); pos=sorted((r for r in out if r["delta_A"]>0),key=lambda r:r["delta_A"],reverse=True)
neg=sorted((r for r in out if r["delta_A"]<0),key=lambda r:r["delta_A"])
sp=sum(r["delta_A"] for r in pos); sn=sum(r["delta_A"] for r in neg)
summary={"pairs":n,"mean_delta_A":statistics.fmean(r["delta_A"] for r in out),
 "mean_delta_raw":statistics.fmean(r["delta_raw"] for r in out),
 "mean_delta_terminal":statistics.fmean(r["delta_B"] for r in out),
 "median_delta_A":statistics.median(r["delta_A"] for r in out),
 "positive_sum":sp,"negative_sum":sn,"net_sum":sp+sn,
 "top1_positive_share":pos[0]["delta_A"]/sp if sp else None,
 "top3_positive_share":sum(r["delta_A"] for r in pos[:3])/sp if sp else None,
 "above_1meV":sum(r["delta_A"]>.001 for r in out),
 "below_minus1meV":sum(r["delta_A"]<-.001 for r in out),
 "within_1meV":sum(abs(r["delta_A"])<=.001 for r in out),
 "same_tokens":sum(r["same_final_tokens"] for r in out),
 "exact_training_composition_overlap":sum(bool(r["teacher_exact_composition_groups"]) for r in out),
 "reduced_training_composition_overlap":sum(bool(r["teacher_same_reduced_composition_groups"]) for r in out),
 "exact_training_composition_with_verified_teacher":sum(bool(r["teacher_energy_diagnostics"]) for r in out),
 "exact_training_composition_with_multiple_verified":sum(any(d["verified_paths"]>1 for d in r["teacher_energy_diagnostics"]) for r in out),
 "exact_training_composition_teacher_both_improve":sum(any(d["delta_A_eV_atom"] < -1e-9 and d["delta_B_eV_atom"] < -1e-9 for d in r["teacher_energy_diagnostics"]) for r in out),
 "scope":"posthoc descriptive diagnosis; no metric, teacher, checkpoint, or threshold changed"}
dest=ROOT/"runs/spad_state_round0_eval_39910/gap-diagnostic-v1"
dest.mkdir(exist_ok=False)
(dest/"diagnostic.json").write_text(json.dumps({"summary":summary,"pairs":out,"top_increases":pos[:6],"top_decreases":neg[:6]},indent=2)+"\n")
print("GAP_SUMMARY "+json.dumps(summary))
for r in pos[:6]:
    print("GAP_INCREASE "+json.dumps(r))
for r in neg[:3]:
    print("GAP_DECREASE "+json.dumps(r))

