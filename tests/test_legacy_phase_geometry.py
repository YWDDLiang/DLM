from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from pymatgen.core import Lattice,Structure

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from crystal_dlm.programmed_path_data import trace_terminal_body
from test_legacy_construction_extraction import environment,recorded

SPEC=importlib.util.spec_from_file_location("legacy_phase_geometry_test",ROOT/"scripts/audit_legacy_phase_geometry.py")
AUDIT=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(AUDIT)


def full_trace(tok,index=0):
    row=recorded(tok,index,"success");events=row["trace"]["events"]
    target=row["final_body_token_ids"].copy()
    for pos,family in enumerate(("LA","LB","LC"),1):target[pos]=tok.vocab[f"<{family}_030>"]
    target[14]=tok.vocab["<Z_025>"]
    positions=list(range(1,7))+[8,9,10,12,13,14]
    events.append({"op":"begin","phase":"cooperative","kind":"cell_sites","positions":positions})
    for step,pos in enumerate(positions):
        events.append({"op":"draw","phase":"cooperative","position":pos,"token":target[pos],"salt":200000000+10007*step,"log_probability":-.1})
    events.append({"op":"end","phase":"cooperative"})
    positions=[12,13,14,8,9,10]
    events.append({"op":"begin","phase":"closure","kind":"species_block","positions":positions})
    for step,pos in enumerate(positions):
        events.append({"op":"draw","phase":"closure","position":pos,"token":tok.vocab["<Z_050>"] if pos==14 else target[pos],
                       "salt":300000000+step,"log_probability":-.2})
    events.append({"op":"rollback","positions":positions,"reason":"block_final_support"})
    events.append({"op":"end","phase":"closure"})
    row["final_body_token_ids"]=trace_terminal_body(row["trace"])
    return row


def test_three_endpoints_apply_rollback_to_correct_phase_snapshot(environment):
    tok,support=environment;row=full_trace(tok);saved=deepcopy(row)
    out=AUDIT.analyze_one(row,tok,support,"k4")
    assert out["metadata_complete"] and all(s["valid"] for s in out["stages"].values())
    a,b,c=[out["stages"][name] for name in AUDIT.PHASES]
    assert a["geometry"]["volume_per_atom_A3"]==pytest.approx(32.)
    assert b["geometry"]["volume_per_atom_A3"]==pytest.approx(13.5)
    assert c["token_ids"]==b["token_ids"] and c["token_ids"][14]==tok.vocab["<Z_025>"]
    assert c["counts"]["rollback_events"]==1 and c["counts"]["transactions_with_rollback"]==1
    assert out["transitions"]["cooperative_to_closure"]["exact_tokens_unchanged"]
    assert not out["transitions"]["construct_to_cooperative"]["exact_tokens_unchanged"]
    assert row==saved


def test_construct_failure_has_no_invented_later_endpoint(environment):
    tok,support=environment;out=AUDIT.analyze_one(recorded(tok,case="construct_failure"),tok,support,"k8")
    assert not out["stages"]["construct"]["valid"]
    for phase in ("cooperative","closure"):
        assert not out["stages"][phase]["executed"] and out["stages"][phase]["geometry"] is None
        assert out["stages"][phase]["token_ids"] is None


def test_missing_phase_metadata_fails_audit_and_retains_request(environment):
    tok,support=environment;good=full_trace(tok,0);bad=full_trace(tok,1)
    bad["trace"]["events"].pop()
    rows,summary=AUDIT.analyze_rows([good,bad],tok,support,"k4")
    assert len(rows)==2 and summary["status"]=="FAIL" and summary["metadata_errors"]==1
    assert rows[1]["stages"] is None and "unfinished" in rows[1]["error"]
    assert summary["stages"]["closure"]["unknown_due_metadata_error"]==1


def test_full_periodic_minimum_includes_self_image_outside_125_original_images():
    s=Structure(Lattice([[4.,0.,0.],[12.1,.01,0.],[0.,0.,4.]]),["He"],[[0.,0.,0.]])
    value=AUDIT.periodic_minimum(s)
    assert value["distinct_atom_minimum_A"] is None
    assert value["minimum_A"]==pytest.approx(np.sqrt(.1**2+.01**2),abs=1e-9)
