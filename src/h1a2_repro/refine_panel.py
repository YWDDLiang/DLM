"""Select the preregistered E2 refinement panel without outcome filtering."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .story_panel import stable_seed


def index_by_task(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if not task_id:
            raise ValueError(f"{label} row has no task_id")
        if task_id in result:
            raise ValueError(f"duplicate task_id {task_id!r} in {label}")
        result[task_id] = dict(row)
    return result


def select_refinement_panel(
    graphs: Sequence[Any],
    accepted_rows: Sequence[Mapping[str, Any]],
    all_tasks: Sequence[Mapping[str, Any]],
    contract_task_ids: Sequence[str],
    *,
    seed: int = 20260822,
) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(graphs) != len(accepted_rows):
        raise ValueError(f"proposal graphs {len(graphs)} != accepted task rows {len(accepted_rows)}")
    task_index = index_by_task(all_tasks, label="E1 tasks")
    accepted_index: dict[str, tuple[Any, dict[str, Any]]] = {}
    for graph, row in zip(graphs, accepted_rows):
        task_id = str(row.get("task_id", ""))
        if not task_id:
            raise ValueError("accepted task row has no task_id")
        if task_id in accepted_index:
            raise ValueError(f"duplicate accepted task_id {task_id!r}")
        accepted_index[task_id] = (graph, dict(row))

    selected_graphs: list[Any] = []
    selected_metadata: list[dict[str, Any]] = []
    attempt_ledger: list[dict[str, Any]] = []
    seen: set[str] = set()
    for contract_index, task_id in enumerate(contract_task_ids):
        if task_id in seen:
            raise ValueError(f"duplicate task_id {task_id!r} in E2 contract")
        seen.add(task_id)
        if task_id not in task_index:
            raise ValueError(f"E2 contract task {task_id!r} is absent from E1 tasks")
        task = dict(task_index[task_id])
        ledger_row = {
            key: task.get(key)
            for key in (
                "task_id",
                "pair_id",
                "plan_id",
                "plan_source",
                "arm",
                "replicate",
                "scientific_seed",
                "shuffle_fields",
                "shuffle_donor_plan_id",
            )
        }
        ledger_row.update({"contract_index": contract_index, "body_success": task_id in accepted_index})
        if task_id in accepted_index:
            graph, accepted = accepted_index[task_id]
            refiner_seed = stable_seed(
                int(seed),
                str(task.get("pair_id")),
                str(task.get("plan_source")),
                f"refine-rep-{task.get('replicate')}",
            )
            metadata = dict(accepted)
            metadata.update({"contract_index": contract_index, "refiner_seed": refiner_seed})
            selected_graphs.append(graph)
            selected_metadata.append(metadata)
            ledger_row.update({"refine_eligible": True, "refiner_seed": refiner_seed})
        else:
            ledger_row.update({"refine_eligible": False, "refiner_seed": None})
        attempt_ledger.append(ledger_row)

    report = {
        "schema": "h1a2_story_e2_selection_v1",
        "requested_attempts": len(contract_task_ids),
        "body_successes": len(selected_graphs),
        "body_failures": len(contract_task_ids) - len(selected_graphs),
        "selected_graphs": len(selected_graphs),
        "root_seed": int(seed),
        "selection_uses_outcomes": False,
    }
    return selected_graphs, selected_metadata, attempt_ledger, report


__all__ = ["index_by_task", "select_refinement_panel"]
