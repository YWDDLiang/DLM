#!/usr/bin/env python3
"""Read-only, SHA-bound comparisons of twelve complete legacy phase outcomes."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

METHODS=("k4","k8");PHASES=("construct","cooperative","closure");ENDPOINTS=("native","tau800")
BOOLS=("reconstructed","native_execution_success","endpoint_execution_success","novel","unique_representative","novel_unique",
       "terminal_verified","strict_stable","meta_stable","strict_sun","meta_sun","verified_strict_stable","verified_meta_stable",
       "verified_strict_sun","verified_meta_sun")
SUNS=("strict_sun","meta_sun","verified_strict_sun","verified_meta_sun")


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def hull_known(row):return row["official_hull_status"]=="known" and finite(row.get("hull_energy_eV_atom"))
def hull_unknown(row):return row["official_hull_status"] in {"official_cache_unresolved","official_cache_not_covered"}
def hull_input_unavailable(row):return row["official_hull_status"] in {"input_not_reconstructed","not_reconstructed"}
def hull_other_unavailable(row):return not hull_known(row) and not hull_unknown(row) and not hull_input_unavailable(row)
def energy_known(row):return hull_known(row) and finite(row.get("terminal_energy_eV_atom")) and finite(row.get("e_above_hull_eV_atom"))
def verified_known(row):return row["terminal_verified"] and row["terminal_status"]=="verified" and energy_known(row)
def generation_failure(row):return not row["endpoint_execution_success"] or not row["reconstructed"] or row["terminal_status"]=="generation_failure"


def load_bound(entry,base,jsonl):
    path=(base/entry["file"]).resolve()
    if digest(path)!=entry["sha256"]:raise ValueError(f"SHA mismatch: {entry['file']}")
    with path.open(encoding="utf-8") as stream:
        value=[json.loads(line) for line in stream if line.strip()] if jsonl else json.load(stream)
    return value,{"file":str(path),"sha256":entry["sha256"],"source_path":entry["source_path"]}


def validate_rows(rows,evaluation,label):
    indexed={int(r["sample_idx"]):r for r in rows}
    if len(rows)!=256 or len(indexed)!=256 or set(indexed)!=set(range(256)) or len({r["trajectory_id"] for r in rows})!=256:
        raise ValueError(f"{label}: expected the complete unique256 ledger")
    if evaluation["counts"].get("requests")!=256:raise ValueError(f"{label}: evaluation denominator differs")
    for row in rows:
        if type(row.get("sample_idx")) is not int or not isinstance(row.get("group_id"),str) or not isinstance(row.get("trajectory_id"),str):
            raise ValueError(f"{label}: source identities have invalid types")
        if any(type(row.get(key)) is not bool for key in BOOLS):raise ValueError(f"{label}: a required outcome is not an explicit bool")
        if row["terminal_verified"]!=(row["terminal_status"]=="verified"):
            raise ValueError(f"{label}: verified flag conflicts with terminal status")
        if row["novel_unique"]!=(row["novel"] and row["unique_representative"]):raise ValueError(f"{label}: N/U flags conflict")
        for family in ("strict","meta"):
            stable=row[family+"_stable"]
            if row[family+"_sun"]!=(stable and row["novel_unique"]):raise ValueError(f"{label}: SUN flags conflict")
            if row["verified_"+family+"_stable"]!=(stable and row["terminal_verified"]):raise ValueError(f"{label}: verified S flags conflict")
            if row["verified_"+family+"_sun"]!=(row[family+"_sun"] and row["terminal_verified"]):raise ValueError(f"{label}: verified SUN flags conflict")
    for key in BOOLS:
        if key not in evaluation["counts"] or sum(r[key] for r in rows)!=evaluation["counts"][key]:
            raise ValueError(f"{label}: original evaluation count differs at {key}")
    return indexed


def identity(row):return {key:row[key] for key in ("sample_idx","group_id","trajectory_id")}


def boolean_pair(a,b,key):
    groups={"both":[],"lost":[],"gained":[],"neither":[]}
    for index in range(256):
        left,right=a[index][key],b[index][key]
        name="both" if left and right else "lost" if left else "gained" if right else "neither"
        groups[name].append(index)
    return {"counts":{k:len(v) for k,v in groups.items()},"ids":groups}


def evidence_class(a,b,family):
    """Disjoint evidence buckets, not a claim of a unique causal mechanism."""
    if generation_failure(a) or generation_failure(b):return "generation_or_parse_unavailable"
    if hull_unknown(a) or hull_unknown(b):return "unknown_hull"
    if hull_input_unavailable(a) or hull_input_unavailable(b):return "input_not_reconstructed_hull"
    if not hull_known(a) or not hull_known(b):return "other_hull_unavailable"
    if not energy_known(a) or not energy_known(b):return "terminal_energy_unavailable"
    if not (verified_known(a) and verified_known(b)):
        return "verification_changed" if a["terminal_verified"]!=b["terminal_verified"] else "not_both_verified"
    threshold=a[family+"_stable"]!=b[family+"_stable"]
    nu=a["novel_unique"]!=b["novel_unique"]
    return "verified_threshold_and_NU" if threshold and nu else "verified_threshold_only" if threshold else "NU_only" if nu else "no_threshold_or_NU_change"


def pair_row(method,endpoint,left_phase,right_phase,a,b,geometry=None):
    both=verified_known(a) and verified_known(b)
    value={"method":method,"endpoint":endpoint,"from_phase":left_phase,"to_phase":right_phase,**identity(a),
           "from_status":a["terminal_status"],"to_status":b["terminal_status"],"both_verified_known":both,
           "verification_changed":a["terminal_verified"]!=b["terminal_verified"],"NU_changed":a["novel_unique"]!=b["novel_unique"],
           "novel_changed":a["novel"]!=b["novel"],"unique_changed":a["unique_representative"]!=b["unique_representative"],
           "generation_or_parse_unavailable":generation_failure(a) or generation_failure(b),
           "raw_geometry_invalid":a["terminal_status"]=="invalid_raw" or b["terminal_status"]=="invalid_raw",
           "unknown_hull":hull_unknown(a) or hull_unknown(b),
           "hull_unavailable":not hull_known(a) or not hull_known(b),
           "hull_input_not_reconstructed":hull_input_unavailable(a) or hull_input_unavailable(b),
           "hull_other_unavailable":hull_other_unavailable(a) or hull_other_unavailable(b),
           "terminal_energy_unavailable":not energy_known(a) or not energy_known(b),
           "from_e_above_hull":a.get("e_above_hull_eV_atom"),"to_e_above_hull":b.get("e_above_hull_eV_atom"),
           "trusted_e_above_hull_delta":b["e_above_hull_eV_atom"]-a["e_above_hull_eV_atom"] if both else None,
           "upstream_exact_tokens_unchanged":None}
    for family in ("strict","meta"):
        value[family+"_stability_flag_changed"]=a[family+"_stable"]!=b[family+"_stable"]
        value[family+"_verified_threshold_flip"]=both and a[family+"_stable"]!=b[family+"_stable"]
        value[family+"_threshold_flag_change_not_verified"]=not both and a[family+"_stable"]!=b[family+"_stable"]
        value[family+"_evidence_class"]=evidence_class(a,b,family)
    for key in SUNS:
        value[key+"_change"]="lost" if a[key] and not b[key] else "gained" if b[key] and not a[key] else "both" if a[key] else "neither"
    if geometry is not None:
        transition=geometry["transitions"][left_phase+"_to_"+right_phase]
        if transition["available"]:value["upstream_exact_tokens_unchanged"]=transition["exact_tokens_unchanged"]
    value["same_upstream_tokens_outcome_changed"]=value["upstream_exact_tokens_unchanged"] is True and any(value[k+"_change"] in {"lost","gained"} for k in SUNS)
    s_or_v=value["strict_stability_flag_changed"] or value["meta_stability_flag_changed"] or value["verification_changed"]
    value["same_upstream_S_or_verified_changed"]=value["upstream_exact_tokens_unchanged"] is True and s_or_v
    value["same_upstream_only_NU_changed"]=value["upstream_exact_tokens_unchanged"] is True and value["NU_changed"] and not s_or_v
    return value


def load_aux_geometry(manifest_path,reference):
    directory=manifest_path.parent.parent
    receipt=directory/"PHASE_GEOMETRY_CASE_RECEIPTS_40086.json"
    if not receipt.is_file():return {},{"status":"not_supplied"}
    binding=json.loads(receipt.read_text(encoding="utf-8"));result={};sources={}
    for method in METHODS:
        name=f"PHASE_GEOMETRY_{method.upper()}_CASES_40086.jsonl";path=directory/name
        if not path.is_file():return {},{"status":"not_supplied"}
        if digest(path)!=binding[name]["sha256"]:raise ValueError("auxiliary phase geometry SHA differs")
        rows=[json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
        indexed={int(r["sample_idx"]):r for r in rows}
        if len(rows)!=256 or set(indexed)!=set(range(256)):raise ValueError("auxiliary geometry population differs")
        for index,row in indexed.items():
            if not row["metadata_complete"] or identity(row)!=identity(reference[index]):raise ValueError("auxiliary phase geometry identity differs")
        result[method]=indexed;sources[name]=binding[name]
    return result,{"status":"bound_auxiliary_only","receipt_sha256":digest(receipt),"sources":sources}


def compare_manifest(path):
    manifest=json.loads(path.read_text(encoding="utf-8"));datasets={};reports={};sources={};reference=None
    for method in METHODS:
        for phase in PHASES:
            for endpoint in ENDPOINTS:
                label=f"{method}/{phase}/{endpoint}";entry=manifest["methods"][method][phase][endpoint]
                rows,attempt_receipt=load_bound(entry["attempts"],path.parent,True)
                evaluation,eval_receipt=load_bound(entry["evaluation"],path.parent,False)
                if evaluation["endpoint"]!=endpoint:raise ValueError(f"{label}: endpoint label differs")
                dataset=validate_rows(rows,evaluation,label)
                if reference is None:reference=dataset
                if any(identity(dataset[i])!=identity(reference[i]) for i in range(256)):
                    raise ValueError(f"{label}: cross-phase source identity differs")
                datasets[(method,phase,endpoint)]=dataset;reports[label]=evaluation
                sources[label]={"attempts":attempt_receipt,"evaluation":eval_receipt}
    protocols={}
    for key in ("terminal_protocol","verification_protocol","frozen_nu_source_sha256","novelty_uniqueness_endpoint"):
        values=[report.get(key) for report in reports.values()]
        if any(v is None for v in values) or any(v!=values[0] for v in values[1:]):raise ValueError(f"declared comparison protocol differs or is missing: {key}")
        protocols[key]=values[0]
    geometry,geometry_receipt=load_aux_geometry(path,reference)
    summary={"status":"PASS","requests_per_cell":256,"same_request_population":True,"protocols":protocols,
             "sources":sources,"geometry":geometry_receipt,"stages":{},"adjacent":{},"model_forwards":0,"MLIP_calls":0,
             "optimizer_steps":0,"NU_recomputed":False,"repeated_runs_causal_isolation":False}
    all_rows=[];changes=[];common={}
    for method in METHODS:
        for endpoint in ENDPOINTS:
            key=f"{method}/{endpoint}";phase_data=[datasets[(method,p,endpoint)] for p in PHASES]
            common_verified=[i for i in range(256) if all(d[i]["terminal_verified"] for d in phase_data)]
            common_known=[i for i in range(256) if all(verified_known(d[i]) for d in phase_data)]
            common[key]={"all_three_terminal_verified":[identity(reference[i]) for i in common_verified],
                "all_three_verified_known":[identity(reference[i]) for i in common_known],
                "all_three_verified_known_above_Meta":[{**identity(reference[i]),
                    "e_above_hull_by_phase":{p:datasets[(method,p,endpoint)][i]["e_above_hull_eV_atom"] for p in PHASES},
                    "meta_stable_by_phase":{p:datasets[(method,p,endpoint)][i]["meta_stable"] for p in PHASES}}
                    for i in common_known if all(not d[i]["meta_stable"] for d in phase_data)],
                "common_verified_known_above_Meta_by_phase":{p:[{**identity(reference[i]),"e_above_hull_eV_atom":datasets[(method,p,endpoint)][i]["e_above_hull_eV_atom"]}
                    for i in common_known if not datasets[(method,p,endpoint)][i]["meta_stable"]] for p in PHASES},
                **{metric:[identity(reference[i]) for i in range(256) if all(d[i][metric] for d in phase_data)] for metric in ("verified_strict_sun","verified_meta_sun")}}
            for phase,dataset in zip(PHASES,phase_data):
                label=f"{method}/{phase}/{endpoint}";rows=list(dataset.values());counts=reports[label]["counts"]
                summary["stages"][label]={"counts":counts,"status_counts":dict(Counter(r["terminal_status"] for r in rows)),
                    "unknown_hull":sum(hull_unknown(r) for r in rows),
                    "hull_input_not_reconstructed":sum(hull_input_unavailable(r) for r in rows),
                    "hull_other_unavailable":sum(hull_other_unavailable(r) for r in rows),
                    "hull_status_counts":dict(Counter(r["official_hull_status"] for r in rows)),
                    "generation_or_parse_unavailable":sum(generation_failure(r) for r in rows),
                    "verified_known_above_Meta":sum(verified_known(r) and not r["meta_stable"] for r in rows),
                    "stable_but_NU_lost":{f:sum(r[f+"_stable"] and not r["novel_unique"] for r in rows) for f in ("strict","meta")},
                    "NU_but_not_stable_flag":{f:sum(r["novel_unique"] and not r[f+"_stable"] for r in rows) for f in ("strict","meta")}}
                for i in range(256):
                    all_rows.append({"method":method,"phase":phase,"endpoint":endpoint,**dataset[i],
                                     "verified_known":verified_known(dataset[i]),"trusted_above_Meta":verified_known(dataset[i]) and not dataset[i]["meta_stable"]})
            for left,right in zip(PHASES,PHASES[1:]):
                a,b=datasets[(method,left,endpoint)],datasets[(method,right,endpoint)]
                pair_values=[pair_row(method,endpoint,left,right,a[i],b[i],geometry.get(method,{}).get(i)) for i in range(256)]
                changes.extend(pair_values)
                pairs={metric:boolean_pair(a,b,metric) for metric in (*SUNS,"strict_stable","meta_stable","novel_unique","terminal_verified")}
                groups={metric:{direction:dict(Counter(r[("strict" if "strict" in metric else "meta")+"_evidence_class"]
                                for r in pair_values if r[metric+"_change"]==direction)) for direction in ("lost","gained")} for metric in SUNS}
                summary["adjacent"][f"{key}/{left}_to_{right}"]={"pairs":pairs,"SUN_change_evidence_buckets":groups,
                    "both_verified_known":sum(r["both_verified_known"] for r in pair_values),
                    "verification_changed":sum(r["verification_changed"] for r in pair_values),"NU_changed":sum(r["NU_changed"] for r in pair_values),
                    "same_upstream_tokens_outcome_changed_ids":[r["sample_idx"] for r in pair_values if r["same_upstream_tokens_outcome_changed"]],
                    "same_upstream_S_or_verified_changed_ids":[r["sample_idx"] for r in pair_values if r["same_upstream_S_or_verified_changed"]],
                    "same_upstream_only_NU_changed_ids":[r["sample_idx"] for r in pair_values if r["same_upstream_only_NU_changed"]]}
    return summary,all_rows,changes,common


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def chinese_report(summary,common):
    lines=["# 原 K4/K8 三阶段配对结果","","所有格均为同一256请求；沿用原S/N-U/SUN与verified标记，没有重算N/U。", "",
           "| 模型／终点 | construct Strict/Meta | cooperative Strict/Meta | closure Strict/Meta |","|---|---:|---:|---:|"]
    for method in METHODS:
        for endpoint in ENDPOINTS:
            cells=[]
            for phase in PHASES:
                c=summary["stages"][f"{method}/{phase}/{endpoint}"]["counts"]
                cells.append(f"{c['strict_sun']}/{c['meta_sun']}（verified {c['verified_strict_sun']}/{c['verified_meta_sun']}）")
            lines.append(f"| {method}/{endpoint} | "+" | ".join(cells)+" |")
    lines += ["","共同三阶段均verified及均verified+known却原Meta标记均为False的请求，完整ID及实际e_above_hull见 COMMON_VERIFIED_IDS.json。"]
    for key,value in common.items():
        lines.append(f"- {key}：三阶段均verified {len(value['all_three_terminal_verified'])}；均verified+known {len(value['all_three_verified_known'])}；按原Meta标记三阶段均未达标 {len(value['all_three_verified_known_above_Meta'])}。")
    lines += ["","相邻阶段的both/lost/gained/neither与完整ID见 SUMMARY.json；逐请求变化见 adjacent_changes.csv/JSONL。",
        "变化的互斥证据桶按生成/解析不可用、未知hull、缺终态能量、verified资格变化/不足、双方verified+known后的阈值与N-U变化依次分类；它不是单一因果归因。",
        "not_converged、invalid_terminal及unknown不会被计作可信高能。NU但未标为稳定的计数包含未验证或不可用情况，不能改称高能失败。",
        "同一原始几何可能在重复共同R或独立refiner/R执行中产生S或verified差异；但相同token的SUN变化并非都属于重复误差。整个cohort中其它结构变化也可能改变U代表，造成仅N-U变化。本表分别列出相同上游token时S/verified变化与仅N-U变化；tau800还包含refiner端点变化，不能单独归因于DLM阶段。原有refiner站点/晶格表示与重复性边界继续适用。",
        "阶段几何只作辅助对照，不填补未verified的能量证据，不用于逐请求选择阶段，也不由几何变化宣布SUN改善。",""]
    return "\n".join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input-manifest",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    try:
        summary,all_rows,changes,common=compare_manifest(args.input_manifest)
        summary["input_manifest_sha256"]=digest(args.input_manifest)
        for name,rows in (("all_outcomes",all_rows),("adjacent_changes",changes)):
            (args.output_dir/f"{name}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in rows),encoding="utf-8")
            write_csv(args.output_dir/f"{name}.csv",rows)
        (args.output_dir/"COMMON_VERIFIED_IDS.json").write_text(json.dumps(common,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        (args.output_dir/"REPORT.md").write_text(chinese_report(summary,common),encoding="utf-8")
        (args.output_dir/"SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        (args.output_dir/"_SUCCESS").touch();print(json.dumps({"status":"PASS","cells":12,"outcome_rows":len(all_rows),"adjacent_rows":len(changes)}),flush=True)
    except Exception as error:
        (args.output_dir/"FAILURE.json").write_text(json.dumps({"status":"FAIL","error":f"{type(error).__name__}: {error}"},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        (args.output_dir/"_FAILED").touch();raise


if __name__=="__main__":main()
