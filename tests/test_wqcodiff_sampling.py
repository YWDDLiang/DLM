from __future__ import annotations

import importlib.util
import random
import types
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "Torch sampling tests run locally/on Slurm")
class D3PMPosteriorTests(unittest.TestCase):
    def test_unrestricted_fast_path_matches_registered_reference_draw_for_draw(self) -> None:
        import torch

        from crystal_dlm.wqcodiff.sampling import _alpha, _d3pm_posterior_draw

        def registered_reference(
            logits,
            current,
            *,
            current_time,
            next_time,
            generator,
            temperature,
        ):
            candidates = list(range(logits.numel()))
            candidate_tensor = torch.tensor(
                candidates, dtype=torch.long, device=logits.device
            )
            clean = torch.softmax(
                logits.float()[candidate_tensor] / temperature, dim=-1
            )
            alpha_t = _alpha(current_time)
            alpha_s = _alpha(next_time)
            transition = min(
                max(alpha_t / max(alpha_s, 1.0e-12), 0.0), 1.0
            )
            uniform = 1.0 / len(candidates)
            prior_s = alpha_s * clean + (1.0 - alpha_s) * uniform
            likelihood = torch.full_like(
                prior_s, (1.0 - transition) * uniform
            )
            likelihood[candidates.index(current)] += transition
            posterior = prior_s * likelihood
            posterior = posterior / posterior.sum().clamp_min(1.0e-12)
            selected = int(
                torch.multinomial(posterior, 1, generator=generator).item()
            )
            return candidates[selected]

        logits = torch.tensor([-2.0, 0.1, 1.7, 0.5, -0.4])
        old_generator = torch.Generator(device="cpu").manual_seed(314159)
        new_generator = torch.Generator(device="cpu").manual_seed(314159)
        old_draws = [
            registered_reference(
                logits,
                2,
                current_time=0.91,
                next_time=0.84,
                generator=old_generator,
                temperature=0.83,
            )
            for _ in range(1024)
        ]
        new_draws = [
            _d3pm_posterior_draw(
                logits,
                2,
                current_time=0.91,
                next_time=0.84,
                generator=new_generator,
                temperature=0.83,
            )
            for _ in range(1024)
        ]
        self.assertEqual(new_draws, old_draws)

    def test_uniform_kernel_posterior_is_seed_reproducible_and_respects_legal_support(self) -> None:
        import torch

        from crystal_dlm.wqcodiff.sampling import _d3pm_posterior_draw

        logits = torch.tensor([-3.0, 0.5, 1.0, 4.0, -2.0])

        def draws(seed: int) -> list[int]:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            return [
                _d3pm_posterior_draw(
                    logits,
                    current=2,
                    current_time=0.8,
                    next_time=0.6,
                    generator=generator,
                    temperature=1.0,
                    legal=(1, 3),
                )
                for _ in range(256)
            ]

        first = draws(17)
        self.assertEqual(first, draws(17))
        self.assertTrue(set(first) <= {1, 3})
        self.assertIn(3, first)

    def test_empty_legal_support_is_rejected_before_sampling(self) -> None:
        import torch

        from crystal_dlm.wqcodiff.sampling import _d3pm_posterior_draw

        with self.assertRaisesRegex(ValueError, "empty legal support"):
            _d3pm_posterior_draw(
                torch.zeros(3),
                current=0,
                current_time=0.8,
                next_time=0.7,
                generator=torch.Generator(),
                temperature=1.0,
                legal=(),
            )

    def test_explicit_topology_event_budget_is_one_per_reverse_step(self) -> None:
        import torch

        from crystal_dlm.wqcodiff.kernel import TransitionError
        from crystal_dlm.wqcodiff.sampling import (
            _AttemptContext,
            _reserve_topology_event_slot,
        )

        context = _AttemptContext(
            "attempt-budget",
            random.Random(11),
            torch.Generator(device="cpu"),
            {"joint": 0, "bridge": 0, "projection": 0},
            [],
        )
        context.reverse_step = 4
        _reserve_topology_event_slot(context)
        with self.assertRaisesRegex(TransitionError, "more than one topology event"):
            _reserve_topology_event_slot(context)
        context.reverse_step = 5
        _reserve_topology_event_slot(context)


@unittest.skipUnless(TORCH_AVAILABLE, "Torch sampling tests run locally/on Slurm")
class EventLogitTransferTests(unittest.TestCase):
    def test_bulk_cpu_heads_match_the_per_candidate_reference_exactly(self) -> None:
        import torch

        from crystal_dlm.wqcodiff.events import TopologyEvent, TopologyEventType
        from crystal_dlm.wqcodiff.model import WQModelOutput, WQVariant
        from crystal_dlm.wqcodiff.sampling import (
            ATOMIC_NUMBER_TO_CLASS,
            EVENT_CLASS,
            _event_logits,
            _log_softmax,
        )
        from crystal_dlm.wqcodiff.state import OrbitState, StratifiedState

        state = StratifiedState(
            space_group=1,
            lattice_system="triclinic",
            lattice_chart=(1.5, 1.6, 1.7, 0.0, 0.0, 0.0),
            orbits=(
                OrbitState("o0", 0, 6, 1, 3, (0.1, 0.2, 0.3)),
                OrbitState("o1", 1, 8, 1, 2, (0.4, 0.5)),
            ),
            attempt_id="event-logit-equivalence",
            timestep=0.7,
        )
        events = (
            TopologyEvent(TopologyEventType.NONE),
            TopologyEvent(
                TopologyEventType.BIRTH,
                target_wyckoff_type=2,
                target_species=6,
                new_orbit_id="born",
            ),
            TopologyEvent(TopologyEventType.DEATH, orbit_id="o0"),
            TopologyEvent(
                TopologyEventType.WYCKOFF_CHANGE,
                orbit_id="o1",
                target_wyckoff_type=2,
                new_orbit_id="o1",
            ),
            TopologyEvent(
                TopologyEventType.SPECIES_CHANGE,
                orbit_id="o0",
                target_species=8,
            ),
        )

        class Kernel:
            @staticmethod
            def legal_events(_state):
                return events

        generator = torch.Generator(device="cpu")
        generator.manual_seed(29)
        output = WQModelOutput(
            space_group_logits=torch.randn(1, 230, generator=generator),
            species_logits=torch.randn(2, 89, generator=generator),
            wyckoff_logits=torch.randn(2, 27, generator=generator),
            event_logits=torch.randn(1, 5, generator=generator),
            event_orbit_logits=torch.randn(2, generator=generator),
            birth_species_logits=torch.randn(1, 89, generator=generator),
            birth_wyckoff_logits=torch.randn(1, 27, generator=generator),
            birth_coordinate_mean=torch.zeros(1, 3),
            birth_coordinate_log_scale=torch.zeros(1, 3),
            revision_logits=torch.zeros(2, 3),
            atom_coordinate_score=torch.zeros(2, 3),
            lattice_score=torch.zeros(1, 6),
            bridge_mean=torch.zeros(2, 3),
            bridge_log_scale=torch.zeros(2, 3),
            orbit_features=torch.zeros(2, 256),
        )
        config = types.SimpleNamespace(
            temperature=0.83,
            disc_once_tau=0.5,
            backbone_calls=16,
            variant=WQVariant.STRAT_GEO,
        )

        def per_candidate_reference():
            type_log = _log_softmax(output.event_logits[0], config.temperature)
            pointer_log = _log_softmax(
                output.event_orbit_logits, config.temperature
            )
            birth_species = _log_softmax(
                output.birth_species_logits[0], config.temperature
            )
            birth_wyckoff = _log_softmax(
                output.birth_wyckoff_logits[0], config.temperature
            )
            orbit_species = _log_softmax(
                output.species_logits, config.temperature
            )
            orbit_wyckoff = _log_softmax(
                output.wyckoff_logits, config.temperature
            )
            orbit_index = {
                orbit.orbit_id: index for index, orbit in enumerate(state.orbits)
            }
            result = []
            for event in events:
                kind = event.event_type
                value = type_log[EVENT_CLASS[kind]]
                if kind is TopologyEventType.BIRTH:
                    value = (
                        value
                        + birth_species[
                            ATOMIC_NUMBER_TO_CLASS[int(event.target_species)]
                        ]
                        + birth_wyckoff[int(event.target_wyckoff_type)]
                    )
                elif kind is TopologyEventType.DEATH:
                    value = value + pointer_log[orbit_index[str(event.orbit_id)]]
                elif kind is TopologyEventType.WYCKOFF_CHANGE:
                    index = orbit_index[str(event.orbit_id)]
                    value = (
                        value
                        + pointer_log[index]
                        + orbit_wyckoff[index, int(event.target_wyckoff_type)]
                    )
                elif kind is TopologyEventType.SPECIES_CHANGE:
                    index = orbit_index[str(event.orbit_id)]
                    target_class = ATOMIC_NUMBER_TO_CLASS[
                        int(event.target_species)
                    ]
                    value = (
                        value
                        + pointer_log[index]
                        + orbit_species[index, target_class]
                    )
                result.append((event, float(value.detach().cpu())))
            return result

        self.assertEqual(
            _event_logits(
                state,
                output,
                Kernel(),
                config,
                step=2,
                midpoint=8,
                recovery=True,
            ),
            per_candidate_reference(),
        )


if __name__ == "__main__":
    unittest.main()
