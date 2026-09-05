import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.optimize import brentq

from crystal_dlm.basin_path_objective import (
    solve_basin_path_teacher,
    weighted_sampled_path_nll,
)


def _group(group_id, labels):
    return {
        "group_id": group_id,
        "candidates": [
            {"trajectory_id": str(i), "verified": True,
             "raw_energy": a + b, "terminal_energy": b}
            for i, (a, b) in enumerate(labels)
        ],
    }


def _weights(result, index=0):
    return np.array([row["weight"] for row in result["groups"][index]["candidates"]])


class BasinPathObjectiveTest(unittest.TestCase):
    def test_two_objectives_decrease_and_match_binary_optimum(self):
        groups = [_group("one", [(0.0, 0.0), (0.2, 0.2)]),
                  _group("two", [(0.0, -100.0), (0.2, -99.8)])]
        before = copy.deepcopy(groups)
        result = solve_basin_path_teacher(groups)
        summary = result["summary"]
        p = brentq(lambda p: p * np.log(2 * p) + (1 - p) * np.log(2 * (1 - p)) - 0.2,
                   0.5, 0.99999)
        self.assertEqual(groups, before)
        self.assertEqual(summary["solver_status"], "optimal")
        self.assertAlmostEqual(summary["rho_max"], 2 * (p - 0.5), places=7)
        self.assertAlmostEqual(summary["target_gain"], 0.1 * (p - 0.5), places=8)
        for name in ("mean_delta_gap", "mean_delta_terminal"):
            self.assertLessEqual(summary[name], -summary["target_gain"] + 1e-10)
        self.assertLessEqual(summary["mean_kl"], 0.1 + 1e-10)
        self.assertLessEqual(summary["max_common_mean_kl"], 0.2 + 1e-10)
        self.assertLess(summary["rho_duality_gap"], 1e-7)
        self.assertLess(summary["primal_residual"], 1e-9)
        np.testing.assert_allclose(_weights(result), [0.5 + (p - 0.5) / 2,
                                                     0.5 - (p - 0.5) / 2], atol=1e-8)

        # Neither group separately admits two improvements; their fixed-weight
        # average does. Do not require a per-group winner or drop either group.
        result = solve_basin_path_teacher([
            _group("tradeoff-a", [(0.1, 0.5), (0.3, 0.4)]),
            _group("tradeoff-b", [(0.5, 0.1), (0.4, 0.3)]),
        ])
        self.assertEqual(result["summary"]["solver_status"], "optimal")
        self.assertGreater(result["summary"]["target_gain"], 0)
        self.assertLess(result["summary"]["mean_delta_gap"], 0)
        self.assertLess(result["summary"]["mean_delta_terminal"], 0)

        # Nonfinite optimizer output must not become an "optimal" certificate.
        with patch("crystal_dlm.basin_path_objective.minimize",
                   return_value=SimpleNamespace(x=np.array([np.nan, np.nan]))):
            result = solve_basin_path_teacher(groups)
        summary = result["summary"]
        self.assertEqual(summary["solver_status"], "feasible_projection_not_converged")
        self.assertIsNone(summary["projection_duality_gap"])
        self.assertTrue(np.isfinite(_weights(result)).all())
        self.assertLessEqual(summary["mean_delta_gap"], -summary["target_gain"] + 1e-10)
        self.assertLessEqual(summary["mean_delta_terminal"], -summary["target_gain"] + 1e-10)

    def test_conflicting_or_one_constant_objective_returns_uniform(self):
        for labels in ([(0.0, 1.0), (1.0, 0.0)],
                       [(0.0, 0.0), (0.0, 1.0)],
                       [(0.0, 3.0), (1.0, 0.0)]):
            with self.subTest(labels=labels):
                result = solve_basin_path_teacher([_group("conflict", labels)])
                np.testing.assert_array_equal(_weights(result), [0.5, 0.5])
                self.assertEqual(result["summary"]["solver_status"], "uniform_no_common_gain")
                self.assertEqual(result["summary"]["rho_max"], 0.0)
                self.assertEqual(result["summary"]["target_gain"], 0.0)
                self.assertEqual(result["summary"]["mean_kl"], 0.0)

    def test_singletons_constant_labels_and_zero_budget(self):
        result = solve_basin_path_teacher([
            _group("single", [(0.1, -7.0)]),
            _group("constant", [(2.0, -10.0)] * 3),
        ])
        np.testing.assert_array_equal(_weights(result, 0), [1.0])
        np.testing.assert_array_equal(_weights(result, 1), [1 / 3] * 3)
        self.assertEqual(result["summary"]["solver_status"], "uniform_constant_labels")
        self.assertAlmostEqual(result["summary"]["ESS"], 3.0)
        variable = [_group("variable", [(0.0, 0.0), (1.0, 1.0)])]
        for kwargs, status in (({"kappa": 0}, "uniform_zero_kl_budget"),
                               ({"retained_fraction": 0}, "uniform_zero_target")):
            result = solve_basin_path_teacher(variable, **kwargs)
            np.testing.assert_array_equal(_weights(result), [0.5, 0.5])
            self.assertEqual(result["summary"]["solver_status"], status)
        for kwargs in ({"kappa": -0.1}, {"energy_scale": 0},
                       {"retained_fraction": 1.1}, {"kappa": float("nan")}):
            with self.assertRaises(ValueError):
                solve_basin_path_teacher(variable, **kwargs)

    def test_missing_verification_and_energies_never_get_labels_or_weight(self):
        group = _group("mixed", [(0, 0), (1, 1), (-100, -100)])
        group["candidates"][2]["verified"] = False
        for raw, terminal in ((None, 1), (1, None), (float("nan"), 1),
                              (1, float("inf"))):
            group["candidates"].append({"trajectory_id": "unknown", "verified": True,
                                        "raw_energy": raw, "terminal_energy": terminal})
        empty = _group("unverified", [(0, 0)])
        empty["candidates"][0]["verified"] = False
        result = solve_basin_path_teacher([group, empty, _group("empty", [])])
        np.testing.assert_allclose(_weights(result)[:2], _weights(
            solve_basin_path_teacher([_group("valid", [(0, 0), (1, 1)])])))
        np.testing.assert_array_equal(_weights(result)[2:], np.zeros(5))
        np.testing.assert_array_equal(_weights(result, 1), [0])
        self.assertIsNone(result["groups"][0]["candidates"][3]["raw_energy"])
        self.assertIsNone(result["groups"][0]["candidates"][4]["terminal_energy"])
        self.assertEqual(result["summary"]["total_groups"], 3)
        self.assertEqual(result["summary"]["validated_groups"], 1)
        self.assertEqual(result["summary"]["validated_candidates"], 2)
        for groups in ([], [empty], [_group("empty", [])]):
            result = solve_basin_path_teacher(groups)
            self.assertEqual(result["summary"]["solver_status"], "no_verified_groups")
            self.assertIsNone(result["summary"]["mean_delta_gap"])
            self.assertIsNone(result["summary"]["mean_delta_terminal"])
            self.assertEqual(result["summary"]["ESS"], 0)

    def test_occurrence_multiplicity_and_zero_temperature_boundary(self):
        group = _group("duplicates", [(0, 0), (1, 1), (1, 1)])
        group["candidates"][2]["trajectory_id"] = "1"
        result = solve_basin_path_teacher([group], kappa=2)
        self.assertEqual([c["trajectory_id"] for c in result["groups"][0]["candidates"]],
                         ["0", "1", "1"])
        np.testing.assert_allclose(_weights(result), [2 / 3, 1 / 6, 1 / 6], atol=1e-8)
        self.assertAlmostEqual(result["summary"]["rho_max"], (2 / 3) / 0.1, places=7)
        self.assertEqual(result["summary"]["solver_status"], "optimal")
        self.assertAlmostEqual(result["summary"]["ESS"], 2.0, places=7)
        # At eta=0 the best common-gain face can require a nonuniform mixture
        # of tied, conflicting minimizers, not a single argmin or uniform ties.
        result = solve_basin_path_teacher([
            _group("tied-face", [(0, 4), (3, 0), (4, 4)])], kappa=2)
        self.assertEqual(result["summary"]["solver_status"], "optimal")
        self.assertLess(result["summary"]["rho_duality_gap"], 2e-6)
        self.assertLess(result["summary"]["primal_residual"], 1e-8)

    def test_optional_torch_ht_loss_and_zero_weight_gradients(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch is not installed")
        log_probs = torch.tensor([[-2.0, -4.0], [-float("inf"), float("nan")]],
                                 dtype=torch.float64, requires_grad=True, device="cpu")
        weights = torch.tensor([[0.5], [0.0]], requires_grad=True, device="cpu")
        loss = weighted_sampled_path_nll(log_probs, weights, [[0.5, 0.25], [0, float("nan")]], 2)
        self.assertAlmostEqual(loss.item(), 5.0)
        loss.backward()
        np.testing.assert_allclose(log_probs.grad.numpy(), [[-0.5, -1], [0, 0]])
        self.assertIsNone(weights.grad)
        zero_log_probs = torch.tensor([-float("inf"), float("nan")],
                                      requires_grad=True, device="cpu")
        zero_loss = weighted_sampled_path_nll(zero_log_probs, [0, 0], [0, 0], 1)
        self.assertEqual(zero_loss.item(), 0)
        zero_loss.backward()
        self.assertTrue(bool((zero_log_probs.grad == 0).all()))
        # Enumerate a uniformly sampled single state: its expected HT estimate
        # is the full path sum, not a per-token/path-length average.
        full = torch.tensor([-1.0, -2.0, -3.0], device="cpu")
        estimates = [weighted_sampled_path_nll(full[i:i + 1], 1, 1 / 3, 1)
                     for i in range(3)]
        self.assertAlmostEqual(torch.stack(estimates).mean().item(), -full.sum().item())
        with self.assertRaises(ValueError):
            weighted_sampled_path_nll(full, 1, 0, 1)


if __name__ == "__main__":
    unittest.main()
