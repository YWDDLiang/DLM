from copy import deepcopy
import math

import pytest

from crystal_dlm.dynamic_crystal import parse_dynamic_answer
from crystal_dlm.programmed_path_data import compile_condition, path_seed
from crystal_dlm.self_repair_data import attach_fixed_repair_roots, install_repair_initial_body, repair_net_change
from test_programmed_path_data import Tokenizer


def example():
    tok = Tokenizer()
    row = {"group_id": "23", "source_split": "train", "source_row_idx": 23, "prompt": "native:\n",
           "plan_state": {"N": 3, "elements": ["Fe", "O"], "counts": [1, 2]},
           "species_program": ["Fe", "O"], "species_program_source": "test_pointer"}
    text = "<N_003><LA_040><LB_040><LC_040><AA_090><AB_090><AG_090>"
    text += "".join(f"<E_{s}><X_{x:03d}><Y_000><Z_000>" for s, x in zip(("O", "O", "Fe"), (0, 30, 60)))
    ids = [tok.vocab[t] for t in parse_dynamic_answer(text, strict=True)["tokens"]]
    parent = {**row, "candidate_index": 0, "trajectory_id": "23:0:0", "checkpoint": "old-policy",
              "success": True, "body": text, "final_body_token_ids": ids}
    return tok, row, parent


def test_fixed_roots_do_not_replace_failures_or_read_outcome_values():
    tok, row, parent = example()
    original = deepcopy(parent)
    alternative = {**parent, "candidate_index": 1, "trajectory_id": "23:0:1", "raw_energy": -999.}
    selected = attach_fixed_repair_roots([(7, row)], [parent, alternative])
    ordinal, root = selected[0]
    assert ordinal == 7 and root["repair_parent_trajectory_id"] == parent["trajectory_id"]
    assert "raw_energy" not in root and parent == original
    parent["success"] = False
    assert not attach_fixed_repair_roots([(7, row)], [parent, alternative])[0][1]["repair_parent_success"]
    with pytest.raises(ValueError, match="replacement"):
        attach_fixed_repair_roots([(7, row)], [alternative])
    with pytest.raises(ValueError, match="train-only"):
        attach_fixed_repair_roots([(7, row)], [{**parent, "source_split": "evaluation"}])


def test_complete_root_replaces_mask_canvas_and_new_seed_namespace():
    tok, row, parent = example()
    root = attach_fixed_repair_roots([(0, row)], [parent])[0][1]
    compiled = compile_condition(root, tok, mask_id=99999)
    assert 99999 in compiled["initial_body"]
    install_repair_initial_body(compiled, tok)
    assert compiled["initial_body"] == parent["final_body_token_ids"]
    assert compiled["initial_body"] is not root["repair_initial_body"]
    parent_seeds = {path_seed(23, row["group_id"], r, j) for r in (0, 1) for j in range(8)}
    child_seeds = {path_seed(23, root["group_id"], 2, j) for j in range(4)}
    assert len(child_seeds) == 4 and not parent_seeds & child_seeds
    root["repair_initial_body_text"] = parent["body"].replace("<E_Fe>", "<E_O>")
    with pytest.raises(ValueError):
        install_repair_initial_body(compile_condition(root, tok, mask_id=99999), tok)


def test_net_change_does_not_confuse_rollback_or_typed_slot_changes():
    _, _, parent = example()
    body = parent["final_body_token_ids"]
    assert repair_net_change(body, body)["final_equals_initial"]
    final = body.copy(); final[1] += 1; final[8] += 1
    assert repair_net_change(body, final) == {"final_equals_initial": False,
                                            "net_changed_cell_scalars": 1,
                                            "net_changed_coordinate_scalars": 1}
    final[7] += 1
    with pytest.raises(ValueError, match="N/E"):
        repair_net_change(body, final)


def load_diagnostic():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts/diagnose_self_repair_support.py"
    spec = importlib.util.spec_from_file_location("self_repair_diagnostic_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_convex_opportunity_does_not_require_an_individual_winner_or_fake_noop():
    d = load_diagnostic()
    mixed = d.convex_support([(.4, -.8), (-.8, .4)], .001)
    assert mixed["improvement_feasible_without_KL"]
    assert math.isclose(mixed["max_common_gain_without_KL"], .2)
    assert not d.convex_support([(.2, .3), (.4, .1)], .001)["nonregression_feasible_without_KL"]
    assert d.convex_support([(0., 0.)], .001)["nonregression_feasible_without_KL"]
    assert not d.convex_support([(0., 0.)], .001)["improvement_feasible_without_KL"]
    assert d.convex_support([(-.3, 0.)], .001)["improvement_feasible_without_KL"]


def test_report_preserves_unknown_baselines_and_fixed_request_denominator():
    d = load_diagnostic()
    paths, labels, parents = [], [], []
    for root_index in range(2):
        parent_id = f"p{root_index}"
        parents.append({"trajectory_id": parent_id, "verified": root_index == 0, "status": "verified" if root_index == 0 else "not_converged",
                        "raw_energy": 6. if root_index == 0 else None, "terminal_energy": 5. if root_index == 0 else None,
                        "gap": 1. if root_index == 0 else None})
        for j in range(4):
            initial = list(range(11)); final = initial.copy(); final[1] += 2
            path = {"trajectory_id": f"{parent_id}:2:{j}", "group_id": parent_id, "candidate_index": j,
                    "source_split": "train", "path_mode": "self_repair_support_check", "success": True,
                    "repair_parent_trajectory_id": parent_id, "repair_initial_body": initial,
                    "final_body_token_ids": final, "sampling_seed": root_index * 4 + j,
                    "trace": {"initial_body": initial, "mask_id": 99, "events": [
                        {"op": "begin", "phase": "cooperative", "positions": [1]},
                        {"op": "draw", "phase": "cooperative", "position": 1, "token": final[1], "log_probability": -1.},
                        {"op": "end", "phase": "cooperative"}]}}
            paths.append(path)
            da, db = ((.4, -.8), (-.8, .4), (0., 0.), (0., 0.))[j]
            valid = j < 2 or root_index == 1
            labels.append({"trajectory_id": path["trajectory_id"], "group_id": parent_id, "verified": valid,
                           "status": "verified" if valid else "not_converged", "gap": 1 + da,
                           "terminal_energy": 5 + db, "raw_energy": 6 + da + db})
    result = d.summarize(paths, labels, parents, expected_roots=2)
    assert result["summary"]["requests"] == 8
    assert result["summary"]["comparable_roots"] == 1
    assert result["summary"]["roots_with_single_joint_improvement"] == 0
    assert result["summary"]["roots_convex_improvement_feasible_without_KL"] == 1
    assert result["summary"]["verification_four_cells"]["parent_0_child_1"] == 4
    labels[0]["gap"] += .5
    with pytest.raises(ValueError, match="delta energy"):
        d.summarize(paths, labels, parents, expected_roots=2)
