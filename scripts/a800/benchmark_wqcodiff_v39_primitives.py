#!/usr/bin/env python3
"""Benchmark v39 semantics-preserving CPU/D3PM fast paths."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from crystal_dlm.wqcodiff.bridge import ChartCatalog, ChartSpec, TargetStratumBridge
from crystal_dlm.wqcodiff.events import TopologyEventType
from crystal_dlm.wqcodiff.kernel import TopologyEventKernel
from crystal_dlm.wqcodiff.runtime import geometry_signals_from_graph
from crystal_dlm.wqcodiff.state import OrbitState, StratifiedState
from crystal_dlm.wqcodiff.vocabulary import MP20_ATOMIC_NUMBERS


THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class _SyntheticCatalog(ChartCatalog):
    def __init__(self) -> None:
        self.specs = {
            (1, value): ChartSpec(
                1,
                value,
                chr(ord("a") + value) if value < 26 else "A",
                1 + value % 2,
                value % 4,
            )
            for value in range(27)
        }

    def get(self, space_group: int, wyckoff_type: int) -> ChartSpec:
        return self.specs[(space_group, wyckoff_type)]

    def types(self, space_group: int):
        return tuple(range(27))


def _median_seconds(function: Callable[[], object], repeats: int, rounds: int) -> float:
    timings: list[float] = []
    function()
    gc.disable()
    try:
        for _ in range(rounds):
            started = time.perf_counter()
            for _ in range(repeats):
                function()
            timings.append((time.perf_counter() - started) / repeats)
    finally:
        gc.enable()
    return statistics.median(timings)


def _legacy_geometry(
    coordinates: np.ndarray,
    lattice: np.ndarray,
    mapping: np.ndarray,
    orbit_count: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    from crystal_dlm.wqcodiff.charts import periodic_cartesian_distance

    minimum = np.full(len(coordinates), np.inf, dtype=np.float64)
    neighbors = np.zeros(len(coordinates), dtype=np.int64)
    for first in range(len(coordinates)):
        for second in range(first + 1, len(coordinates)):
            distance = periodic_cartesian_distance(
                coordinates[first], coordinates[second], lattice
            )
            minimum[first] = min(minimum[first], distance)
            minimum[second] = min(minimum[second], distance)
            if distance <= 3.0:
                neighbors[first] += 1
                neighbors[second] += 1
    if len(coordinates) == 1:
        minimum[0] = 10.0
    collision = np.zeros(orbit_count, dtype=np.float64)
    coordination = np.zeros(orbit_count, dtype=np.float64)
    for orbit_index in range(orbit_count):
        selected = mapping == orbit_index
        min_distance = float(np.min(minimum[selected]))
        collision[orbit_index] = max(
            0.0, min(1.0, (0.8 - min_distance) / 0.8)
        )
        counts = neighbors[selected]
        coordination[orbit_index] = float(
            np.mean((counts < 1) | (counts > 12))
        )
    volume_per_atom = abs(float(np.linalg.det(lattice))) / len(coordinates)
    condition = float(np.linalg.cond(lattice))
    strain = min(
        1.0,
        abs(math.log(max(volume_per_atom, 1.0e-12) / 20.0)) / 4.0
        + max(0.0, math.log(max(condition, 1.0)) - math.log(8.0)) / 4.0,
    )
    return collision, coordination, float(strain)


def _event_benchmark(repeats: int, rounds: int) -> dict[str, object]:
    catalog = _SyntheticCatalog()
    state = StratifiedState(
        space_group=1,
        lattice_system="triclinic",
        lattice_chart=(1.5, 1.6, 1.7, 0.0, 0.0, 0.0),
        orbits=tuple(
            OrbitState(
                f"o{index}",
                index,
                MP20_ATOMIC_NUMBERS[index],
                1 + index % 2,
                index % 4,
                tuple(0.1 + 0.03 * axis for axis in range(index % 4)),
            )
            for index in range(8)
        ),
    )
    kernel = TopologyEventKernel(
        catalog=catalog,
        bridge=TargetStratumBridge(catalog),
        species=MP20_ATOMIC_NUMBERS,
    )

    def legacy():
        return tuple(
            event
            for event in kernel.legal_events(state)
            if event.event_type is TopologyEventType.DEATH
        )

    def optimized():
        return kernel.legal_events(
            state, event_types={TopologyEventType.DEATH}
        )

    if legacy() != optimized():
        raise RuntimeError("event support equivalence failed")
    old_seconds = _median_seconds(legacy, repeats, rounds)
    new_seconds = _median_seconds(optimized, repeats, rounds)
    return {
        "legacy_seconds_per_call": old_seconds,
        "v39_seconds_per_call": new_seconds,
        "speedup": old_seconds / new_seconds,
        "full_support_size": len(kernel.legal_events(state)),
        "filtered_support_size": len(optimized()),
        "exact_equal": True,
    }


def _geometry_benchmark(repeats: int, rounds: int) -> dict[str, object]:
    rng = np.random.default_rng(20260719)
    coordinates = rng.random((20, 3), dtype=np.float64)
    lattice = np.asarray(
        [[4.9, 0.0, 0.0], [0.7, 5.4, 0.0], [-0.3, 0.8, 6.2]],
        dtype=np.float64,
    )
    mapping = np.repeat(np.arange(10, dtype=np.int64), 2)

    def legacy():
        return _legacy_geometry(coordinates, lattice, mapping, 10)

    def optimized():
        return geometry_signals_from_graph(coordinates, lattice, mapping, 10)

    old_values = legacy()
    new_values = optimized()
    max_error = max(
        float(np.max(np.abs(old_values[0] - new_values[0]))),
        float(np.max(np.abs(old_values[1] - new_values[1]))),
        abs(old_values[2] - new_values[2]),
    )
    if max_error > 1.0e-12:
        raise RuntimeError(f"geometry equivalence failed: {max_error}")
    old_seconds = _median_seconds(legacy, repeats, rounds)
    new_seconds = _median_seconds(optimized, repeats, rounds)
    return {
        "legacy_seconds_per_call": old_seconds,
        "v39_seconds_per_call": new_seconds,
        "speedup": old_seconds / new_seconds,
        "max_abs_error": max_error,
        "exact_gate_pass": True,
    }


def _d3pm_benchmark(repeats: int, rounds: int) -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch_not_installed"}

    from crystal_dlm.wqcodiff.sampling import _alpha, _d3pm_posterior_draw

    logits = torch.linspace(-2.0, 2.0, 89)

    def reference(logit, current, generator):
        candidates = list(range(logit.numel()))
        candidate_tensor = torch.tensor(
            candidates, dtype=torch.long, device=logit.device
        )
        clean = torch.softmax(logit.float()[candidate_tensor] / 0.83, dim=-1)
        alpha_t = _alpha(0.91)
        alpha_s = _alpha(0.84)
        transition = min(max(alpha_t / max(alpha_s, 1.0e-12), 0.0), 1.0)
        uniform = 1.0 / len(candidates)
        prior_s = alpha_s * clean + (1.0 - alpha_s) * uniform
        likelihood = torch.full_like(prior_s, (1.0 - transition) * uniform)
        likelihood[candidates.index(current)] += transition
        posterior = prior_s * likelihood
        posterior = posterior / posterior.sum().clamp_min(1.0e-12)
        selected = int(torch.multinomial(posterior, 1, generator=generator).item())
        return candidates[selected]

    old_equivalence = torch.Generator(device="cpu").manual_seed(47)
    new_equivalence = torch.Generator(device="cpu").manual_seed(47)
    old_draws = [reference(logits, 40, old_equivalence) for _ in range(1024)]
    new_draws = [
        _d3pm_posterior_draw(
            logits,
            40,
            current_time=0.91,
            next_time=0.84,
            generator=new_equivalence,
            temperature=0.83,
        )
        for _ in range(1024)
    ]
    if old_draws != new_draws:
        raise RuntimeError("D3PM draw-for-draw equivalence failed")

    old_generator = torch.Generator(device="cpu").manual_seed(83)
    new_generator = torch.Generator(device="cpu").manual_seed(83)

    def old_call():
        return reference(logits, 40, old_generator)

    def new_call():
        return _d3pm_posterior_draw(
            logits,
            40,
            current_time=0.91,
            next_time=0.84,
            generator=new_generator,
            temperature=0.83,
        )

    old_seconds = _median_seconds(old_call, repeats, rounds)
    new_seconds = _median_seconds(new_call, repeats, rounds)
    return {
        "available": True,
        "legacy_seconds_per_call": old_seconds,
        "v39_seconds_per_call": new_seconds,
        "speedup": old_seconds / new_seconds,
        "draw_for_draw_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.repeats <= 0 or args.rounds < 3:
        raise ValueError("repeats must be positive and rounds must be at least three")
    thread_env = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    if set(thread_env.values()) != {"1"}:
        raise RuntimeError(f"thread contract violated: {thread_env}")
    random.seed(20260719)
    result = {
        "schema": "wqcodiff_v39_primitive_benchmark_v1",
        "python": sys.version,
        "platform": platform.platform(),
        "thread_env": thread_env,
        "repeats": args.repeats,
        "rounds": args.rounds,
        "events": _event_benchmark(args.repeats, args.rounds),
        "geometry": _geometry_benchmark(args.repeats, args.rounds),
        "d3pm": _d3pm_benchmark(args.repeats, args.rounds),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
