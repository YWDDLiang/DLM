import math

import numpy as np
import torch

from crystal_dlm import periodic_v2_corruption as corruption
from crystal_dlm.periodic_base_training_data import prepare_periodic_base_source
from crystal_dlm.programmed_path_runtime import complete_geometry_supported
from test_periodic_base_training_data import source
from test_state_programmed_runtime import TinyTokenizer, constraints


def test_row_log_spd_volume_shape_and_affine_cartesian_noise():
    lattice = np.array([[3., 0., 0.], [.8, 4., 0.], [.4, -.6, 5.]])
    fractional = np.array([[.95, .01, .4], [.2, .4, .8]])
    basis = corruption.traceless_shape_basis()
    np.testing.assert_allclose(np.einsum("iab,jab->ij", basis, basis), np.eye(5), atol=3e-16)
    np.testing.assert_allclose(np.trace(basis, axis1=1, axis2=2), 0., atol=1e-16)
    log_volume, shape = corruption.log_spd_volume_shape(lattice)
    assert abs(log_volume - math.log(np.linalg.det(lattice))) < 2e-15
    assert abs(np.trace(shape)) < 2e-15

    class KnownNormals:
        def normal(self, size=None):
            if size is None:
                return 1.
            if size == 5:
                return np.array([.2, -.1, .3, -.2, .15])
            return np.array([[1., -.7, .3], [-.4, .5, -1.]])

    noisy_lattice, noisy_fractional, info = corruption.sample_log_spd_corruption(
        lattice, fractional, KnownNormals(), sigma_v=.2, sigma_shape=.15, sigma_F=.1,
    )
    assert np.all(np.diag(noisy_lattice) > 0)
    np.testing.assert_allclose(noisy_lattice, np.tril(noisy_lattice), atol=0.)
    assert abs(np.linalg.det(noisy_lattice) / np.linalg.det(lattice) - math.exp(.2)) < 5e-15
    delta_R = .1 * KnownNormals().normal(size=fractional.shape)
    unwrapped = fractional + np.linalg.solve(noisy_lattice.T, delta_R.T).T
    np.testing.assert_allclose(noisy_fractional, np.mod(unwrapped, 1.), atol=1e-15)
    np.testing.assert_allclose(unwrapped @ noisy_lattice, fractional @ noisy_lattice + delta_R, atol=1e-15)
    assert info["sampled_log_volume_delta"] == .2
    assert np.linalg.norm(unwrapped @ noisy_lattice - fractional @ lattice) > np.linalg.norm(delta_R)


def test_support_is_checked_after_quantization_and_only_integer_old_is_returned(monkeypatch):
    tokenizer = TinyTokenizer()
    support = constraints(tokenizer)
    prepared = prepare_periodic_base_source(source(count=1), tokenizer, support)

    def proposal(*args, **kwargs):
        # Continuous 0.49 violates the 0.5 A self-image threshold, but rounds to
        # the supported native 0.5 A token. Admission must use integer geometry.
        return np.diag([.49, 4., 5.]), np.array([[.999, .101, .201]]), {}

    monkeypatch.setattr(corruption, "sample_log_spd_corruption", proposal)
    old, sigmas, info = corruption.corrupt_periodic_v2_source(
        prepared, np.random.default_rng(1), tokenizer.vocab, support,
    )
    assert old[1] == tokenizer.vocab["<LA_005>"]
    assert old[8] == tokenizer.vocab["<X_000>"]
    assert all(isinstance(token, int) for token in old)
    assert complete_geometry_supported(torch.tensor(old), support)
    assert info["old_state_admitted"] and info["attempt_count"] == 1
    assert not info["fallback_to_clean"]
    assert sigmas == corruption.V2_NOISE_GRID[info["level_index"]]


def test_eight_rejections_retain_clean_source_and_explicit_fallback(monkeypatch):
    tokenizer = TinyTokenizer()
    support = constraints(tokenizer)
    prepared = prepare_periodic_base_source(source(count=1), tokenizer, support)
    calls = []

    def clipped_proposal(*args, **kwargs):
        calls.append(1)
        return np.diag([50.2, 4., 5.]), np.zeros((1, 3)), {}

    monkeypatch.setattr(corruption, "sample_log_spd_corruption", clipped_proposal)
    old, sigmas, info = corruption.corrupt_periodic_v2_source(
        prepared, np.random.default_rng(7), tokenizer.vocab, support,
    )
    assert len(calls) == info["attempt_count"] == info["rejected_attempts"] == 8
    assert info["rejection_counts"] == {"quantization_clipped": 8}
    assert info["fallback_to_clean"] and info["quantized_unchanged"]
    assert info["old_state_admitted"]
    assert old == list(prepared.clean_tokens) and sigmas == (0., 0., 0.)


def test_accepted_zero_token_change_keeps_declared_noise_scales(monkeypatch):
    tokenizer = TinyTokenizer()
    support = constraints(tokenizer)
    prepared = prepare_periodic_base_source(source(count=1), tokenizer, support)
    lattice = corruption.lattice_matrix_from_parameters(prepared.arrays["lengths"], prepared.arrays["angles"])
    monkeypatch.setattr(corruption, "sample_log_spd_corruption", lambda *args, **kwargs:
                        (lattice, np.asarray(prepared.arrays["frac_coords"]), {}))
    old, sigmas, info = corruption.corrupt_periodic_v2_source(
        prepared, np.random.default_rng(5), tokenizer.vocab, support,
    )
    assert old == list(prepared.clean_tokens)
    assert info["quantized_unchanged"] and not info["fallback_to_clean"]
    assert sigmas == corruption.V2_NOISE_GRID[info["level_index"]]
