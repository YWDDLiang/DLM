"""Independent CPU checks for P33 numerical interfaces, without a model.

Run in a torch environment:
  PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
      -s tests -p test_mixed_geometry_diffusion.py -v

The small analytic/NumPy reference tests also run without torch.  Tensor tests
are explicitly skipped there; passing references does not validate production
tensor execution.  No second full sampling implementation is supplied.
"""

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from crystal_dlm import mixed_geometry_diffusion as mg
else:
    mg = None


def reference_density_and_u(x, mean, sigma, radius=24):
    lifted = np.asarray(x - mean + np.arange(-radius, radius + 1), dtype=np.float64)
    logits = -0.5 * np.square(lifted / sigma)
    shift = float(logits.max())
    weights = np.exp(logits - shift)
    log_density = shift + math.log(float(weights.sum())) - math.log(math.sqrt(2 * math.pi) * sigma)
    u = -float((weights * lifted / sigma).sum() / weights.sum())
    return log_density, u


class IndependentReferenceTests(unittest.TestCase):
    def test_wrapped_score_agrees_with_density_derivative_and_differs_from_mic(self):
        x, mean, sigma = 0.49, 0.0, 0.2
        _, u = reference_density_and_u(x, mean, sigma)
        step = 1e-6
        derivative = (reference_density_and_u(x + step, mean, sigma)[0]
                      - reference_density_and_u(x - step, mean, sigma)[0]) / (2 * step)
        self.assertAlmostEqual(u, sigma * derivative, places=8)
        self.assertAlmostEqual(u / sigma, -1.3044125223203686, places=10)
        self.assertGreater(abs(u / sigma - (-x / sigma**2)), 10)
        self.assertAlmostEqual(reference_density_and_u(0.5, 0, sigma)[1], 0, places=14)

    def test_point_readout_and_gaussian_readout_have_different_meaning(self):
        epsilon = 0.002
        point, noisy_point = 0.0049, 0.0052
        u_point = -(noisy_point - point) / epsilon
        self.assertAlmostEqual(noisy_point + epsilon * u_point, point, places=15)
        # With standard Gaussian data, the exact probability-flow v is zero.
        # The P33 clean-mean readout still shrinks its terminal variance by a^2.
        alpha = math.cos(math.pi * epsilon / 2)
        reference_samples = np.array([-1.0, 1.0])
        self.assertLess(float(np.var(alpha * reference_samples)), 1)
        self.assertAlmostEqual(float(np.var(alpha * reference_samples)), alpha**2, places=15)

    def test_symmetric_bimodal_readout_is_not_a_posterior_mode_draw(self):
        epsilon = 0.002
        observation = 0.5
        left_logp, left_u = reference_density_and_u(observation, 0.25, epsilon)
        right_logp, right_u = reference_density_and_u(observation, 0.75, epsilon)
        self.assertAlmostEqual(left_logp, right_logp, places=10)
        mixture_u = 0.5 * (left_u + right_u)
        output = (observation + epsilon * mixture_u) % 1
        self.assertAlmostEqual(output, 0.5, places=15)
        self.assertGreater(min(abs(output - mode) for mode in (0.25, 0.75)), 0.2)

    def test_chart_volume_units_and_atom_count_from_diagonal_closed_form(self):
        logs = np.array([0.2, 0.5, -0.1])
        lattice = np.diag(np.exp(logs))
        count, unit = 4, 2.0
        metric_eigenvalues = np.linalg.eigvalsh(lattice @ lattice.T / unit**2)
        trace_half_log = 0.5 * float(np.log(metric_eigenvalues).sum())
        expected = math.log(np.linalg.det(lattice) / (count * unit**3)) / math.sqrt(3)
        self.assertAlmostEqual((trace_half_log - math.log(count)) / math.sqrt(3), expected, places=14)


@unittest.skipUnless(torch is not None, "torch unavailable: tensor implementation not executed")
class MixedGeometryTensorTests(unittest.TestCase):
    def tensor(self, value, *, requires_grad=False):
        return torch.tensor(value, dtype=torch.float64, requires_grad=requires_grad)

    def state(self, z=None, fractional=None, time=1.0, mask=None):
        z = self.tensor([0.2, -0.1, 0.5, 0.0, -0.3, 0.7]) if z is None else z
        fractional = self.tensor([[0.91, 0.08, 0.31]]) if fractional is None else fractional
        mask = torch.ones(fractional.shape[:-1], dtype=torch.bool) if mask is None else mask
        return mg.GeometryState(z, fractional, self.tensor(time), mask)

    def test_basis_chart_roundtrip_preserves_row_metric_and_rotation(self):
        basis = mg.half_log_spd_basis()
        torch.testing.assert_close(torch.einsum("iab,jab->ij", basis, basis), torch.eye(6, dtype=torch.float64), atol=5e-16, rtol=0)
        lattice = self.tensor([[3.0, 0, 0], [0.7, 4.0, 0], [-0.4, 0.2, 5.0]], requires_grad=True)
        rotation = self.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        chart = mg.lattice_to_chart(lattice, 4)
        rotated = mg.lattice_to_chart(lattice.detach() @ rotation, 4)
        torch.testing.assert_close(chart, rotated, atol=2e-14, rtol=0)
        decoded = mg.chart_to_lattice(chart, 4)
        torch.testing.assert_close(decoded @ decoded.T, lattice.detach() @ lattice.detach().T, atol=2e-12, rtol=1e-13)
        self.assertFalse(chart.requires_grad)
        self.assertFalse(decoded.requires_grad)
        torch.testing.assert_close(decoded, decoded.tril(), atol=0, rtol=0)

    def test_chart_units_and_per_atom_volume_are_analytical(self):
        lattice = self.tensor([[2, 0, 0], [0, 3, 0], [0, 0, 5]])
        chart = mg.lattice_to_chart(lattice, 3)
        self.assertAlmostEqual(float(chart[5]), math.log(10) / math.sqrt(3), places=14)
        torch.testing.assert_close(mg.lattice_to_chart(7 * lattice, 3, reference_length=7), chart, atol=1e-14, rtol=0)
        torch.testing.assert_close(mg.lattice_to_chart(2 * lattice, 24), chart, atol=1e-14, rtol=0)
        changed_count = mg.lattice_to_chart(lattice, 24)
        self.assertAlmostEqual(float(changed_count[5] - chart[5]), -math.log(8) / math.sqrt(3), places=14)

    def test_normalizer_uses_population_std_floor_train_gate_and_json(self):
        identity = torch.eye(3, dtype=torch.float64)
        train = torch.stack((identity, math.e * identity))
        normalizer = mg.LatticeNormalizer.fit(train, self.tensor([1, 1]))
        self.assertAlmostEqual(normalizer.mean[5], math.sqrt(3) / 2, places=14)
        self.assertAlmostEqual(normalizer.raw_std[5], math.sqrt(3) / 2, places=14)
        self.assertEqual(normalizer.floor_hits, (True, True, True, True, True, False))
        torch.testing.assert_close(normalizer.encode(train, 1)[:, 5], self.tensor([-1, 1]), atol=2e-14, rtol=0)
        torch.testing.assert_close(normalizer.decode(normalizer.encode(train, 1), 1), train, atol=2e-14, rtol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normalizer.json"
            normalizer.save(path)
            self.assertEqual(mg.LatticeNormalizer.load(path), normalizer)
        with self.assertRaises(ValueError):
            mg.LatticeNormalizer.fit(train, 1, split="val")

    def test_chart_rejects_invalid_counts_orientation_and_overflow_without_clips(self):
        identity = torch.eye(3, dtype=torch.float64)
        for count in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                mg.lattice_to_chart(identity, count)
        with self.assertRaises(ValueError):
            mg.lattice_to_chart(self.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]), 1)
        with self.assertRaises(FloatingPointError):
            mg.chart_to_lattice(self.tensor([0, 0, 0, 0, 0, 1e6]), 1)

    def test_vp_endpoints_and_clean_noise_inverse(self):
        time = self.tensor([0, 0.002, 0.4, 1])
        alpha, sigma = mg.vp_coefficients(time)
        self.assertEqual(float(alpha[-1]), 0)
        self.assertEqual(float(sigma[0]), 0)
        clean = self.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]).expand(4, -1)
        noise = self.tensor([[1, -1, 2, -2, 3, -3]]).expand(4, -1)
        noisy = alpha[:, None] * clean + sigma[:, None] * noise
        target = mg.lattice_v_target(clean, noise, time)
        torch.testing.assert_close(alpha[:, None] * noisy - sigma[:, None] * target, clean, atol=1e-15, rtol=0)
        torch.testing.assert_close(sigma[:, None] * noisy + alpha[:, None] * target, noise, atol=1e-15, rtol=0)

    def test_corruption_uses_fractional_units_and_excludes_padding(self):
        clean_z = torch.zeros((2, 6), dtype=torch.float64, requires_grad=True)
        fractional = self.tensor([[[0.99, 0.2, 0.4], [float("nan")]*3], [[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]]])
        mask = torch.tensor([[True, False], [True, True]])
        noise_f = torch.ones_like(fractional)
        noise_f[0, 1] = float("nan")
        noisy = mg.corrupt_geometry(clean_z, fractional, self.tensor([0.02, 0.1]), atom_mask=mask,
                                    lattice_noise=torch.zeros_like(clean_z), coordinate_noise=noise_f,
                                    state_dtype=torch.float64, target_dtype=torch.float64)
        self.assertAlmostEqual(float(noisy.state.fractional[0, 0, 0]), 0.01, places=14)
        torch.testing.assert_close(noisy.state.fractional[0, 1], torch.zeros(3, dtype=torch.float64), atol=0, rtol=0)
        torch.testing.assert_close(noisy.u_target[0, 1], torch.zeros(3, dtype=torch.float64), atol=0, rtol=0)
        self.assertFalse(noisy.state.z.requires_grad)
        self.assertFalse(noisy.v_target.requires_grad)
        self.assertFalse(noisy.u_target.requires_grad)
        # Same fractional noise has different Cartesian covariance in a long cell.
        fractional_variance = 0.02**2
        lattice = self.tensor([[2, 0, 0], [0, 5, 0], [0, 0, 9]])
        covariance = fractional_variance * lattice.T @ lattice
        self.assertGreater(float(covariance[2, 2]), 10 * float(covariance[0, 0]))

    def test_wrapped_target_matches_independent_density_derivative(self):
        for x, mean, sigma in ((0.49, 0, 0.2), (0.5, 0, 0.2), (0.97, 0.01, 0.04), (0.31, 0.83, 1.0)):
            with self.subTest(x=x, sigma=sigma):
                noisy, clean = self.tensor([[x]*3]), self.tensor([[mean]*3])
                output = mg.wrapped_normal_u_target(noisy, clean, sigma)
                _, expected = reference_density_and_u(x, mean, sigma)
                torch.testing.assert_close(output, torch.full_like(output, expected), atol=2e-12, rtol=1e-12)
                translated = mg.wrapped_normal_u_target(noisy + 4, clean - 3, sigma)
                torch.testing.assert_close(output, translated, atol=3e-12, rtol=1e-12)

    def test_image_tail_bound_and_expanded_window_across_time_and_boundary(self):
        bounds = mg.wrapped_image_tail_bounds()
        self.assertLess(bounds["absolute_u_error"], 1e-11)
        cases = [(x, t) for x in (0, 0.001, 0.49, 0.5, 0.999) for t in (0.002, 0.05, 0.2, 1)]
        for x, t in cases:
            noisy = self.tensor([[x]*3])
            clean = self.tensor([[0.999]*3])
            small = mg.wrapped_normal_u_target(noisy, clean, t, image_radius=8)
            large = mg.wrapped_normal_u_target(noisy, clean, t, image_radius=20)
            self.assertLessEqual(float((small-large).abs().max()), bounds["absolute_u_error"] + 1e-13)
        uniform = mg.wrapped_normal_u_target(self.tensor([[0.12, 0.31, 0.7]]), self.tensor([[0.75, 0.83, 0.8]]), 1)
        self.assertLess(float(uniform.abs().max()), 4e-8)

    def test_risk_is_source_balanced_masked_and_differentiable_only_in_predictions(self):
        v = self.tensor([[1]*6, [3]*6], requires_grad=True)
        u = self.tensor([[[2]*3, [float("nan")]*3], [[4]*3, [4]*3]], requires_grad=True)
        target_v = torch.zeros_like(v, requires_grad=True)
        target_u = torch.zeros_like(u, requires_grad=True)
        mask = torch.tensor([[True, False], [True, True]])
        # Fractional weights must normalize by their actual sum, not clamp to 1.
        risk = mg.geometry_denoising_risk(v, u, target_v, target_u, mask, source_weights=self.tensor([0.1, 0.3]))
        self.assertEqual(risk.total.dtype, torch.float32)
        self.assertAlmostEqual(float(risk.total), 10, places=5)
        torch.testing.assert_close(risk.per_example, torch.tensor([2.5, 12.5]), atol=0, rtol=0)
        risk.total.backward()
        self.assertIsNone(target_v.grad)
        self.assertIsNone(target_u.grad)
        self.assertTrue(bool(torch.isfinite(v.grad).all()))
        self.assertTrue(bool(torch.isfinite(u.grad).all()))
        torch.testing.assert_close(u.grad[0, 1], torch.zeros(3, dtype=torch.float64), atol=0, rtol=0)
        self.assertAlmostEqual(float(v.grad[0, 0]), 1/24, places=7)

    def test_zero_source_weight_gives_zero_graph_contribution(self):
        v = torch.ones((1, 6), requires_grad=True)
        u = torch.ones((1, 1, 3), requires_grad=True)
        risk = mg.geometry_denoising_risk(v, u, torch.zeros_like(v), torch.zeros_like(u),
                                         torch.ones((1, 1), dtype=torch.bool), source_weights=torch.zeros(1))
        self.assertEqual(float(risk.total), 0)
        risk.total.backward()
        self.assertEqual(float(v.grad.abs().sum() + u.grad.abs().sum()), 0)

    def test_p33_point_oracle_has_fresh_epsilon_readout_and_exact_local_recovery(self):
        point_z = self.tensor([0.3, -0.2, 0.1, 0.5, -0.4, 0.2])
        point_f = self.tensor([[0.97, 0.02, 0.33]])
        initial = self.state()
        calls = []

        def field(state):
            calls.append((float(state.t), state.z.clone(), state.fractional.clone()))
            alpha, sigma = mg.vp_coefficients(state.t)
            v = (alpha[..., None] * state.z - point_z) / sigma[..., None]
            u = mg.wrapped_normal_u_target(state.fractional, point_f, state.t)
            return v, u

        result = mg.sample_geometry(field, initial)
        self.assertEqual(result.nfe, 33)
        self.assertEqual(len(calls), result.nfe)
        self.assertAlmostEqual(calls[-1][0], 0.002, places=15)
        self.assertGreater(calls[-2][0], calls[-1][0])
        torch.testing.assert_close(calls[-1][1], result.pre_readout_state.z, atol=0, rtol=0)
        torch.testing.assert_close(result.z, point_z, atol=2e-14, rtol=0)
        wrapped_error = (result.fractional-point_f + 0.5).remainder(1)-0.5
        self.assertLess(float(wrapped_error.abs().max()), 2e-12)
        self.assertEqual(result.z.dtype, torch.float64)

    def test_gaussian_oracle_preserves_flow_but_readout_has_known_shrink(self):
        initial = self.state()
        result = mg.sample_geometry(lambda state: (torch.zeros_like(state.z), torch.zeros_like(state.fractional)), initial)
        torch.testing.assert_close(result.pre_readout_state.z, initial.z, atol=0, rtol=0)
        alpha = math.cos(math.pi * 0.002 / 2)
        torch.testing.assert_close(result.z, alpha * initial.z, atol=1e-15, rtol=0)
        torch.testing.assert_close(result.fractional, initial.fractional, atol=0, rtol=0)
        self.assertLess(alpha**2, 1)

    def test_terminal_readout_can_land_between_symmetric_modes(self):
        epsilon = 0.002
        state = self.state(z=torch.zeros(6, dtype=torch.float64), fractional=self.tensor([[0.5]*3]), time=epsilon)
        _, left_u = reference_density_and_u(0.5, 0.25, epsilon)
        _, right_u = reference_density_and_u(0.5, 0.75, epsilon)
        v = torch.zeros_like(state.z)  # symmetric lattice modes +/-1 at z_t=0
        u = torch.full_like(state.fractional, 0.5 * (left_u + right_u))
        z, fractional = mg.terminal_readout(state, v, u)
        self.assertEqual(float(z.abs().max()), 0)
        torch.testing.assert_close(fractional, self.tensor([[0.5]*3]), atol=1e-14, rtol=0)
        self.assertGreater(float((fractional-0.25).abs().min()), 0.2)

    def test_arbitrary_batch_axes_mask_and_configured_nfe(self):
        mask = torch.ones((2, 3, 7), dtype=torch.bool)
        mask[..., -2:] = False
        prior = mg.sample_prior(mask, generator=torch.Generator().manual_seed(51))
        self.assertEqual(prior.z.shape, (2, 3, 6))
        self.assertEqual(prior.fractional.shape, (2, 3, 7, 3))
        config = mg.MixedGeometryConfig(epsilon=0.01, euler_steps=5)
        result = mg.sample_geometry(lambda state: (torch.zeros_like(state.z), torch.zeros_like(state.fractional)), prior, config=config)
        self.assertEqual(result.nfe, 6)
        self.assertEqual(result.time_grid[-1], 0.01)
        self.assertEqual(float(result.fractional[..., -2:, :].abs().sum()), 0)
        times = mg.sample_log_uniform_times((2, 3), generator=torch.Generator().manual_seed(7), config=config)
        self.assertTrue(bool(((times >= 0.01) & (times <= 1)).all()))

    def test_reverse_direction_and_failures_report_real_attempted_nfe(self):
        state = self.state()
        step = mg.probability_flow_step(state, torch.ones_like(state.z), torch.ones_like(state.fractional), 0.9)
        torch.testing.assert_close(step.z-state.z, torch.full_like(state.z, -0.1*math.pi/2), atol=1e-15, rtol=0)
        torch.testing.assert_close(step.fractional, (state.fractional+0.1).remainder(1), atol=1e-15, rtol=0)
        with self.assertRaises(ValueError):
            mg.probability_flow_step(state, torch.zeros_like(state.z), torch.zeros_like(state.fractional), 1)

        calls = []
        def bad_at_readout(current):
            calls.append(float(current.t))
            v = torch.zeros_like(current.z)
            if len(calls) == 33:
                v[0] = float("nan")
            return v, torch.zeros_like(current.fractional)

        with self.assertRaises(mg.GeometrySamplingError) as context:
            mg.sample_geometry(bad_at_readout, state)
        self.assertEqual(context.exception.nfe, 33)
        self.assertEqual(len(calls), 33)
        self.assertAlmostEqual(context.exception.time, 0.002, places=15)


if __name__ == "__main__":
    unittest.main()
