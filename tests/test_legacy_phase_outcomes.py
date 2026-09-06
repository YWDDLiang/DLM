from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("legacy_outcomes_test",ROOT/"scripts/compare_legacy_phase_outcomes.py")
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


def row(index,phase):
    energy=-.02 if index in (0,2,3,4,5,6) else .2
    verified=True;status="verified";nu=True;known=True;generated=True
    if index==1:verified=False;status="not_converged";energy=.5
    if phase!="construct":
        if index==2:energy=.2
        if index==3:nu=False
        if index in (4,7):verified=False;status="not_converged"
        if index==5:known=False
        if index==6:generated=False;known=False;verified=False;status="generation_failure";energy=None;nu=False
    above=energy if known and energy is not None else None
    result={"sample_idx":index,"group_id":f"eval:{index}","trajectory_id":f"eval:{index}:0:0",
            "reconstructed":generated,"native_execution_success":generated,"endpoint_execution_success":generated,
            "novel":nu,"unique_representative":True,"novel_unique":nu,"terminal_verified":verified,"terminal_status":status,
            "terminal_energy_eV_atom":energy,"hull_energy_eV_atom":0. if known else None,
            "official_hull_status":"known" if known else "input_not_reconstructed" if not generated else "official_cache_unresolved","e_above_hull_eV_atom":above}
    for family,limit in (("strict",0.),("meta",.1)):
        stable=above is not None and above<=limit
        result.update({family+"_stable":stable,family+"_sun":stable and nu,
                       "verified_"+family+"_stable":stable and verified,"verified_"+family+"_sun":stable and nu and verified})
    return result


def fixture_manifest(directory):
    manifest={"methods":{}}
    for method in MODULE.METHODS:
        manifest["methods"][method]={}
        for phase in MODULE.PHASES:
            manifest["methods"][method][phase]={}
            for endpoint in MODULE.ENDPOINTS:
                rows=[row(i,phase) for i in range(256)]
                report={"counts":{**{k:sum(r[k] for r in rows) for k in MODULE.BOOLS},"requests":256},
                        "endpoint":endpoint,"terminal_protocol":{"fixed":"R"},"verification_protocol":{"fixed":"checks"},
                        "frozen_nu_source_sha256":"unchanged","novelty_uniqueness_endpoint":"input_before_R"}
                entry={}
                for kind,value in (("attempts",rows),("evaluation",report)):
                    name=f"{method}_{phase}_{endpoint}_{kind}.json"
                    content="".join(json.dumps(r)+"\n" for r in value) if kind=="attempts" else json.dumps(value)
                    path=directory/name;path.write_text(content,encoding="utf-8")
                    entry[kind]={"file":name,"sha256":MODULE.digest(path),"source_path":"/frozen/"+name}
                manifest["methods"][method][phase][endpoint]=entry
    path=directory/"INPUTS.json";path.write_text(json.dumps(manifest),encoding="utf-8")
    return path,manifest


def test_twelve_cells_preserve_counts_and_classify_verified_hull_NU_and_failures(tmp_path,monkeypatch):
    path,_=fixture_manifest(tmp_path)
    summary,all_rows,changes,common=MODULE.compare_manifest(path)
    assert len(all_rows)==3072 and len(changes)==2048
    key="k4/native/construct_to_cooperative"
    assert summary["adjacent"][key]["pairs"]["strict_sun"]["ids"]["lost"]==[2,3,5,6]
    values={r["sample_idx"]:r for r in changes if r["method"]=="k4" and r["endpoint"]=="native" and r["from_phase"]=="construct"}
    assert values[2]["strict_evidence_class"]=="verified_threshold_only"
    assert values[3]["strict_evidence_class"]=="NU_only"
    assert values[4]["strict_evidence_class"]=="verification_changed"
    assert values[5]["strict_evidence_class"]=="unknown_hull"
    assert values[6]["strict_evidence_class"]=="generation_or_parse_unavailable"
    assert summary["stages"]["k4/cooperative/native"]["unknown_hull"]==1
    assert summary["stages"]["k4/cooperative/native"]["hull_input_not_reconstructed"]==1
    auxiliary={"transitions":{"construct_to_cooperative":{"available":True,"exact_tokens_unchanged":True}}}
    nu=MODULE.pair_row("k4","native","construct","cooperative",row(3,"construct"),row(3,"cooperative"),auxiliary)
    energy=MODULE.pair_row("k4","native","construct","cooperative",row(2,"construct"),row(2,"cooperative"),auxiliary)
    assert nu["same_upstream_only_NU_changed"] and not nu["same_upstream_S_or_verified_changed"]
    assert energy["same_upstream_S_or_verified_changed"] and not energy["same_upstream_only_NU_changed"]
    high={r["sample_idx"] for r in common["k4/native"]["all_three_verified_known_above_Meta"]}
    assert 1 not in high and 7 not in high and 8 in high
    assert all(not r["trusted_above_Meta"] for r in all_rows if r["sample_idx"]==1)
    output=tmp_path/"analysis";monkeypatch.setattr(sys,"argv",["compare","--input-manifest",str(path),"--output-dir",str(output)])
    MODULE.main()
    with (output/"all_outcomes.csv").open(encoding="utf-8-sig",newline="") as stream:
        assert len(list(csv.DictReader(stream)))==3072
    assert (output/"_SUCCESS").is_file() and "不能" in (output/"REPORT.md").read_text(encoding="utf-8")


def test_SHA_count_and_cross_phase_identity_errors_are_not_ignored(tmp_path):
    path,manifest=fixture_manifest(tmp_path)
    entry=manifest["methods"]["k8"]["closure"]["tau800"]["attempts"]
    source=tmp_path/entry["file"];original=source.read_text()
    source.write_text(original+"\n")
    with pytest.raises(ValueError,match="SHA mismatch"):MODULE.compare_manifest(path)
    source.write_text(original)
    rows=[json.loads(line) for line in original.splitlines()]
    rows[0]["group_id"]="another source"
    source.write_text("".join(json.dumps(r)+"\n" for r in rows));entry["sha256"]=MODULE.digest(source);path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="identity differs"):MODULE.compare_manifest(path)
    rows=[row(i,"closure") for i in range(256)]
    evaluation={"counts":{**{k:sum(r[k] for r in rows) for k in MODULE.BOOLS},"requests":256}}
    evaluation["counts"]["meta_sun"]+=1
    with pytest.raises(ValueError,match="count differs"):MODULE.validate_rows(rows,evaluation,"fixture")
