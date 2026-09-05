"""Fixed, outcome-blind self-generated starting states for repair support checks."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from crystal_dlm.programmed_path_data import validate_completed_body


def attach_fixed_repair_roots(selected_conditions, parent_paths):
    roots = {}
    for parent in parent_paths:
        if parent.get("source_split") != "train":
            raise ValueError("self-repair roots must be train-only self-generated paths")
        if int(parent["candidate_index"]) != 0:
            continue
        key = str(parent["group_id"])
        if key in roots:
            raise ValueError("duplicate fixed root occurrence")
        roots[key] = parent
    output = []
    for ordinal, condition in selected_conditions:
        parent = roots.get(str(condition["group_id"]))
        if parent is None:
            raise ValueError("missing fixed candidate-zero root; replacement is not allowed")
        for key in ("plan_state", "species_program", "species_program_source", "source_row_idx"):
            if condition[key] != parent[key]:
                raise ValueError(f"repair root condition mismatch: {key}")
        row = deepcopy(condition)
        row.update(group_id=f"self-repair-v1:{parent['trajectory_id']}",
                   repair_parent_group_id=str(parent["group_id"]),
                   repair_parent_trajectory_id=parent["trajectory_id"],
                   repair_parent_checkpoint=parent["checkpoint"],
                   repair_parent_success=bool(parent["success"]),
                   repair_initial_body=list(parent["final_body_token_ids"]),
                   repair_initial_body_sha256=hashlib.sha256(json.dumps(parent["final_body_token_ids"], separators=(",", ":")).encode()).hexdigest(),
                   repair_initial_body_text=parent["body"],
                   root_selection="fixed_candidate_index_zero_no_outcomes")
        output.append((ordinal, row))
    return output


def install_repair_initial_body(compiled, tokenizer):
    """Check the codec/program identity before replacing the normal MASK start."""
    row = compiled["record"]
    initial = list(row["repair_initial_body"])
    if len(initial) != len(compiled["initial_body"]):
        raise ValueError("repair root changed the exact native body length")
    if row["repair_parent_success"]:
        parsed = validate_completed_body(row["repair_initial_body_text"], compiled)
        encoded = [int(tokenizer.get_vocab()[token]) for token in parsed["tokens"]]
        if encoded != initial:
            raise ValueError("repair root text/token identity differs")
    # Failed roots retain their original body and produce failed attempts later.
    compiled["initial_body"] = initial
    return compiled


def repair_net_change(initial, final):
    if len(initial) != len(final):
        raise ValueError("repair endpoint changed body length")
    changed = [i for i, (a, b) in enumerate(zip(initial, final)) if a != b]
    if any(i == 0 or i >= 7 and (i - 7) % 4 == 0 for i in changed):
        raise ValueError("repair changed fixed N/E slots")
    return {"final_equals_initial": not changed,
            "net_changed_cell_scalars": sum(1 <= i <= 6 for i in changed),
            "net_changed_coordinate_scalars": sum(i >= 8 and (i - 8) % 4 < 3 for i in changed)}
