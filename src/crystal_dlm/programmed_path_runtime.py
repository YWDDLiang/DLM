"""Programmed crystal paths with joint transactions and replayable attempts.

This is opt-in.  The historical SPAD entry points and their default behavior
remain unchanged.  Sampling and likelihood replay share the same mask helper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterator, Sequence

import torch

from crystal_dlm.llada_generation import (
    _apply_lightweight_decoding_masks,
    _apply_schema_masks,
    _lattice_matrix_from_token_ids,
    _prepare_atom_count_grammar,
)
from crystal_dlm.periodic_geometry_ops import minimum_image_distances
from crystal_dlm.spad_generation import _transaction_candidate_tokens
from crystal_dlm.spad_program import (
    SpeciesProgram, coordinate_positions, reverse_species_block_revision_slots,
    spad_predictor_position_groups,
)
from crystal_dlm.state_conditioned_model import context_from_programs


@dataclass
class ProgrammedPathTrace:
    initial_body: list[int]
    mask_id: int
    temperature: float
    element_order: list[str]
    events: list[dict[str, Any]] = field(default_factory=list)
    success: bool = True
    failure: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "programmed_crystal_attempt_path_v1",
            "initial_body": self.initial_body, "mask_id": self.mask_id,
            "temperature": self.temperature, "element_order": self.element_order,
            "events": self.events, "success": self.success, "failure": self.failure,
        }


def complete_geometry_supported(body: torch.Tensor, constraints: dict) -> bool:
    """Common bounded-image support, including one-site periodic self images."""
    lattice = _lattice_matrix_from_token_ids(body, prompt_length=0, constraints=constraints)
    if lattice is None or not bool(torch.isfinite(lattice).all()):
        return False
    count = int(constraints["count_token_to_n"].get(int(body[0]), 0))
    if count < 1 or len(body) != 7 + 4 * count:
        return False
    coords = []
    maps = constraints["coord_token_to_bin"]
    period = float(constraints.get("coord_period", 100))
    for site in range(count):
        row = [maps[axis].get(int(body[8 + 4 * site + k])) for k, axis in enumerate("XYZ")]
        if any(value is None for value in row):
            return False
        coords.append([float(value) / period for value in row])
    threshold = float(constraints.get("pbc_min_distance_A", 0.5))
    radius = int(constraints.get("pbc_image_radius", 2))
    shell = torch.arange(-radius, radius + 1, dtype=lattice.dtype, device=body.device)
    shifts = torch.cartesian_prod(shell, shell, shell)
    shifts = shifts[(shifts != 0).any(-1)]
    if float(torch.linalg.vector_norm(shifts @ lattice, dim=-1).min()) < threshold:
        return False
    if count > 1:
        fractions = torch.tensor(coords, dtype=lattice.dtype, device=body.device)
        distances = minimum_image_distances(
            fractions[:, None] - fractions[None, :], lattice, image_radius=radius
        )
        distances.fill_diagonal_(torch.inf)
        if float(distances.min()) < threshold:
            return False
    return True


def cooperative_slots(
    body: torch.Tensor, program: SpeciesProgram, constraints: dict,
) -> tuple[int, ...]:
    """Freeze a program-rooted, geometry-aware region before staging starts."""
    count = program.num_atoms
    size = min(count, max(2, math.ceil(count / 2)))
    lattice = _lattice_matrix_from_token_ids(body, prompt_length=0, constraints=constraints)
    if lattice is None:
        raise ValueError("region selection requires a complete supported cell")
    maps = constraints["coord_token_to_bin"]
    period = float(constraints.get("coord_period", 100))
    fractions = torch.tensor([
        [maps[axis][int(body[p])] / period for axis, p in zip("XYZ", coordinate_positions(i))]
        for i in range(count)
    ], dtype=lattice.dtype, device=body.device)
    distances = minimum_image_distances(
        fractions[:, None] - fractions[None, :], lattice,
        image_radius=int(constraints.get("pbc_image_radius", 2)),
    ).detach().cpu()
    rank = {slot: r for r, entry in enumerate(program.entries) for slot in entry.slot_indices}
    selected = [program.entries[0].anchor_slot]
    while len(selected) < size:
        remaining = [i for i in range(count) if i not in selected]
        used_species = {rank[i] for i in selected}
        unseen = [i for i in remaining if rank[i] not in used_species]
        if unseen:
            priority = min(rank[i] for i in unseen)
            pool = [i for i in unseen if rank[i] == priority]
        else:
            pool = remaining
        chosen = min(pool, key=lambda i: (min(float(distances[i, j]) for j in selected), rank[i], i))
        selected.append(chosen)
    return tuple(selected)


class ProgrammedPathSampler:
    """Batch equal-N canvases; keep one independent attempted trace per row."""

    def __init__(
        self, model: Any, *, prompt_length: int, gen_length: int, mask_id: int,
        programs: Sequence[SpeciesProgram], allowed_token_ids: list[list[int]],
        atom_count_grammar: dict | None, constraints: dict, temperature: float = 0.7,
        sampling_seeds: Sequence[int],
    ) -> None:
        self.model = model
        self.prompt_length = int(prompt_length)
        self.gen_length = int(gen_length)
        self.mask_id = int(mask_id)
        self.programs = list(programs)
        self.allowed_ids = allowed_token_ids
        self.grammar = atom_count_grammar
        self.constraints = constraints
        self.temperature = float(temperature)
        self.seeds = [int(x) for x in sampling_seeds]
        if len(self.programs) != len(self.seeds) or not self.programs:
            raise ValueError("one program and independent seed are required per row")
        self.num_sites = self.programs[0].num_atoms
        if any(p.num_atoms != self.num_sites for p in self.programs):
            raise ValueError("batch by atom count before sampling")
        if self.gen_length != 7 + 4 * self.num_sites or len(allowed_token_ids) != self.gen_length:
            raise ValueError("sampler requires canonical exact 7+4N")
        if not 0 <= self.temperature:
            raise ValueError("temperature must be nonnegative")
        self.traces: list[ProgrammedPathTrace] = []
        self._allowed: torch.Tensor | None = None
        self._prepared_grammar = None

    def _prepare(self, x: torch.Tensor):
        vocabulary = self.model.get_output_embeddings().weight.shape[0]
        self._allowed = torch.zeros(self.gen_length, vocabulary, dtype=torch.bool, device=x.device)
        for pos, ids in enumerate(self.allowed_ids):
            self._allowed[pos, list(ids)] = True
        self._prepared_grammar = (
            None if self.grammar is None
            else _prepare_atom_count_grammar(self.grammar, vocabulary, x.device)
        )

    def processed_logits(
        self, x: torch.Tensor, old: torch.Tensor, positions: dict[int, int],
        transaction_positions: dict[int, Sequence[int]], attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, set[int]]:
        context = context_from_programs(
            old, prompt_length=self.prompt_length, num_sites=self.num_sites,
            programs=self.programs, active_positions=transaction_positions,
        )
        raw = self.model(x, attention_mask=attention_mask, geometry_context=context).logits
        logits = raw.clone()
        _apply_schema_masks(
            logits, x, self.prompt_length, self.gen_length, self._allowed, self._prepared_grammar
        )
        active = torch.zeros(x.shape[0], self.gen_length, dtype=torch.bool, device=x.device)
        for row, pos in positions.items():
            active[row, pos] = True
        report = _apply_lightweight_decoding_masks(
            logits, x, self.prompt_length, self.gen_length, self.constraints,
            active, self.mask_id,
        )
        unavailable = {row for row, pos in report["pbc_no_legal_completion"] if positions.get(row) == pos}
        minimum = torch.finfo(logits.dtype).min
        for row, pos in positions.items():
            values = logits[row, self.prompt_length + pos]
            legal = torch.isfinite(values) & (values > minimum)
            if not bool(legal.any()) or bool(torch.isnan(values).any()) or bool(torch.isposinf(values).any()):
                unavailable.add(row)
        return logits, unavailable

    def _draw(
        self, x: torch.Tensor, old: torch.Tensor, positions: dict[int, int],
        transaction_positions: dict[int, Sequence[int]], attention_mask: torch.Tensor,
        *, phase: str, salt: int,
    ) -> set[int]:
        if not positions:
            return set()
        logits, unavailable = self.processed_logits(x, old, positions, transaction_positions, attention_mask)
        active = {row: self.prompt_length + pos for row, pos in positions.items() if row not in unavailable}
        selected = _transaction_candidate_tokens(
            logits, active_absolute_positions=active, temperature=self.temperature,
            remasking="low_confidence", sampling_seeds_by_batch=self.seeds, salt=salt,
        )
        for row, pos in positions.items():
            if row in unavailable:
                self.traces[row].events.append({"op": "no_support", "position": pos, "phase": phase})
                continue
            absolute = self.prompt_length + pos
            token = int(selected[row, absolute])
            logp = 0.0 if self.temperature == 0 else float(
                torch.log_softmax(logits[row, absolute].double() / self.temperature, -1)[token]
            )
            self.traces[row].events.append({
                "op": "draw", "position": pos, "token": token,
                "log_probability": logp, "phase": phase, "salt": int(salt),
            })
            x[row, absolute] = token
        return unavailable

    def _begin(self, x, rows: dict[int, Sequence[int]], *, phase: str, kind: str):
        snapshot = x.clone()
        for row, positions in rows.items():
            positions = list(positions)
            self.traces[row].events.append({
                "op": "begin", "phase": phase, "kind": kind, "positions": positions,
            })
            x[row, [self.prompt_length + p for p in positions]] = self.mask_id
        return snapshot

    def _restore(self, x, old, row: int, positions: Sequence[int], reason: str):
        absolute = [self.prompt_length + p for p in positions]
        x[row, absolute] = old[row, absolute]
        self.traces[row].events.append({"op": "rollback", "positions": list(positions), "reason": reason})

    @torch.no_grad()
    def run(
        self, initial_tokens: torch.Tensor, attention_mask: torch.Tensor,
        *, construct: bool = True, cooperative: bool = True, closure: bool = True,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        x = initial_tokens.clone()
        if x.shape != attention_mask.shape or x.shape != (len(self.programs), self.prompt_length + self.gen_length):
            raise ValueError("initial tokens and full attention must match the batch canvas")
        self._prepare(x)
        self.traces = [ProgrammedPathTrace(
            x[row, self.prompt_length:].tolist(), self.mask_id, self.temperature,
            list(program.element_order),
        ) for row, program in enumerate(self.programs)]
        alive = set(range(x.shape[0]))
        if construct:
            schedules = [[g[0] for g in spad_predictor_position_groups(p)[1:]] for p in self.programs]
            for row in alive:
                self.traces[row].events.append({"op": "begin", "phase": "construct", "kind": "construct", "positions": []})
            for step in range(6 + 3 * self.num_sites):
                pos = {row: schedules[row][step] for row in alive}
                unavailable = self._draw(
                    x, x.clone(), pos, {row: [p] for row, p in pos.items()}, attention_mask,
                    phase="construct", salt=100_000_000 + 10_007 * step,
                )
                for row in unavailable:
                    self.traces[row].success = False
                    self.traces[row].failure = "construct_empty_support"
                alive -= unavailable
            for row in range(x.shape[0]):
                self.traces[row].events.append({"op": "end", "phase": "construct"})
        for row in list(alive):
            if not complete_geometry_supported(x[row, self.prompt_length:], self.constraints):
                self.traces[row].success = False
                self.traces[row].failure = "unsupported_predictor"
                alive.remove(row)
        if cooperative and alive:
            regions = {row: cooperative_slots(x[row, self.prompt_length:], self.programs[row], self.constraints) for row in alive}
            transactions = {row: (*range(1, 7), *(p for site in slots for p in coordinate_positions(site))) for row, slots in regions.items()}
            old = self._begin(x, transactions, phase="cooperative", kind="cell_sites")
            working = set(alive)
            for step in range(max(map(len, transactions.values()))):
                pos = {row: transactions[row][step] for row in working if step < len(transactions[row])}
                bad = self._draw(x, old, pos, transactions, attention_mask,
                                 phase="cooperative", salt=200_000_000 + 10_007 * step)
                for row in bad:
                    self._restore(x, old, row, transactions[row], "cooperative_empty_support")
                working -= bad
            for row in working:
                if not complete_geometry_supported(x[row, self.prompt_length:], self.constraints):
                    self._restore(x, old, row, transactions[row], "cooperative_final_support")
            for row in alive:
                self.traces[row].events.append({"op": "end", "phase": "cooperative"})
        if closure and alive:
            blocks = {row: reverse_species_block_revision_slots(self.programs[row]) for row in alive}
            for block_index in range(max(map(len, blocks.values()))):
                slots_by_row = {row: blocks[row][block_index] for row in alive if block_index < len(blocks[row])}
                transactions = {row: tuple(p for site in sites for p in coordinate_positions(site)) for row, sites in slots_by_row.items()}
                old = self._begin(x, transactions, phase="closure", kind="species_block")
                for site_order in range(max(map(len, slots_by_row.values()))):
                    working = {row for row, slots in slots_by_row.items() if site_order < len(slots)}
                    for component in range(3):
                        pos = {row: coordinate_positions(slots_by_row[row][site_order])[component] for row in working}
                        bad = self._draw(
                            x, old, pos, transactions, attention_mask, phase="closure",
                            salt=300_000_000 + block_index * 1_000_000 + site_order * 10_000 + component,
                        )
                        for row in bad:
                            self._restore(x, old, row, coordinate_positions(slots_by_row[row][site_order]), "site_empty_support")
                        working -= bad
                for row in slots_by_row:
                    # Restoring a later site can conflict with an earlier new site.
                    # Preserve the full block's old feasible state in that case.
                    if not complete_geometry_supported(x[row, self.prompt_length:], self.constraints):
                        self._restore(x, old, row, transactions[row], "block_final_support")
                    self.traces[row].events.append({"op": "end", "phase": "closure"})
        for row in alive:
            if not complete_geometry_supported(x[row, self.prompt_length:], self.constraints):
                raise RuntimeError("whole-transaction rollback failed to preserve supported input")
        return x, [trace.to_dict() for trace in self.traces]


def replay_scalar_states(trace: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Rebuild exact conditioning before every sampled scalar, including rejection."""
    body = list(trace["initial_body"])
    snapshot: list[int] | None = None
    positions: list[int] = []
    phase = ""
    kind = ""
    decision = 0
    for event in trace["events"]:
        op = event["op"]
        if op == "begin":
            snapshot = body.copy()
            positions = list(event["positions"])
            phase, kind = event["phase"], event["kind"]
            for pos in positions:
                body[pos] = int(trace["mask_id"])
        elif op == "draw":
            pos = int(event["position"])
            old = body.copy() if kind == "construct" else snapshot.copy()
            yield {
                "decision_index": decision, "input_body": body.copy(), "old_body": old,
                "transaction_positions": [pos] if kind == "construct" else positions.copy(),
                "position": pos, "target_token": int(event["token"]), "phase": phase,
                "recorded_log_probability": float(event["log_probability"]),
                "temperature": float(trace["temperature"]),
            }
            body[pos] = int(event["token"])
            decision += 1
        elif op == "rollback":
            if snapshot is None:
                raise ValueError("rollback without an active snapshot")
            for pos in event["positions"]:
                body[int(pos)] = snapshot[int(pos)]
        elif op == "end":
            snapshot = None
            positions = []
        elif op != "no_support":
            raise ValueError(f"unknown path event {op}")
