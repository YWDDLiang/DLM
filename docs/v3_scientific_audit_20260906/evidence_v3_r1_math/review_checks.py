"""Independent R1 counterexamples: NumPy/stdlib only, no model or MLIP calls."""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess

import numpy as np


OUT = Path(__file__).resolve().parent
AUDIT = OUT.parent
ROOT = AUDIT.parents[1]
EXECUTION_SHA = "2e904c260bafb6750c9a5dbb0d920c4ff8c3a868"


def gaussian_prior_kl(rate: float, squared_clean_norm: float) -> float:
    signal = math.exp(-rate)
    return 0.5 * (signal * squared_clean_norm + 6 * (-signal - math.log1p(-signal)))


def rate_for_bound(squared_clean_norm: float, bound: float) -> float:
    lower, upper = 1e-8, 100.0
    for _ in range(100):
        mid = (lower + upper) / 2
        if gaussian_prior_kl(mid, squared_clean_norm) > bound:
            lower = mid
        else:
            upper = mid
    return upper


def vp_counterexample(squared_clean_norm: float) -> dict:
    steps = 64
    rate = rate_for_bound(squared_clean_norm, 1e-6)
    alpha_bar = np.exp(-rate * np.square(np.arange(steps + 1) / steps))
    step_alpha = alpha_bar[1:] / alpha_bar[:-1]
    beta = 1 - step_alpha
    beta_tilde = beta * (1 - alpha_bar[:-1]) / (1 - alpha_bar[1:])
    oracle_variance, zero_epsilon_variance = 1.0, 1.0
    for k in range(steps - 1, -1, -1):
        # Data N(0,1) => exact E[epsilon | z_k] = sqrt(1-alpha_bar_k)*z_k.
        # The candidate mean becomes sqrt(step_alpha_k)*z_k, but its fixed
        # beta_tilde variance is smaller than the true beta reverse variance.
        oracle_variance = step_alpha[k] * oracle_variance + beta_tilde[k]
        zero_epsilon_variance = zero_epsilon_variance / step_alpha[k] + beta_tilde[k]
    assert 0 < oracle_variance < 1
    assert zero_epsilon_variance >= 1 / alpha_bar[-1]
    z, predicted_epsilon = 0.71, -0.23
    mean_errors = []
    for k in range(1, steps + 1):
        a, b, btilde = step_alpha[k-1], beta[k-1], beta_tilde[k-1]
        x0 = (z - math.sqrt(1-alpha_bar[k])*predicted_epsilon) / math.sqrt(alpha_bar[k])
        via_x0 = (math.sqrt(alpha_bar[k-1])*b*x0/(1-alpha_bar[k])
                  + math.sqrt(a)*(1-alpha_bar[k-1])*z/(1-alpha_bar[k]))
        stable = (z - b*predicted_epsilon/math.sqrt(1-alpha_bar[k])) / math.sqrt(a)
        mean_errors.append(abs(via_x0-stable))
    assert max(mean_errors) < 1e-12
    return {
        "illustrative_max_squared_chart_norm_not_measured_data": squared_clean_norm,
        "lambda_for_conditional_KL_bound_1e_6": rate,
        "terminal_alpha_bar": float(alpha_bar[-1]),
        "terminal_conditional_KL_bound": gaussian_prior_kl(rate, squared_clean_norm),
        "largest_forward_beta": float(beta.max()),
        "data_and_prior_standard_Gaussian_true_final_variance": 1.0,
        "candidate_reverse_final_variance_with_oracle_epsilon": float(oracle_variance),
        "zero_epsilon_head_final_std_not_a_trained_model": math.sqrt(zero_epsilon_variance),
        "x0_based_mean_vs_simplified_mean_max_error": max(mean_errors),
        "boundary": "The rates are illustrative. The perfect-score variance defect is a finite reverse approximation error, not a normalization failure or a SUN forecast.",
    }


def gaussian_upper_tail(value: float) -> float:
    return 0.5 * math.erfc(value / math.sqrt(2))


def clock_counterexamples() -> list[dict]:
    steps = 64
    base_sigma = np.concatenate(([0.0], np.geomspace(1e-3, 1.0, steps)))
    result = []
    for gamma in (1.0, 1.25, 1.5):
        sigma = base_sigma ** gamma
        variance_increments = np.diff(sigma*sigma)
        gain = variance_increments / sigma[1:]
        relative_increment = variance_increments[1:] / np.square(sigma[2:])
        assert np.min(variance_increments) > 0
        assert sigma[-1] == 1.0
        assert np.max(abs(gain * sigma[1:] - variance_increments)) < 1e-15
        # Near f0=.0049, original nearest .01 codec bin is [-.005,.005).
        # At the candidate's final step, a locally exact point-source score
        # cancels input noise but the stated sampler adds sigma_1*xi again.
        # Other torus images have negligible probability at these scales.
        upper = (0.005 - 0.0049) / sigma[1]
        lower = (0.0049 - (-0.005)) / sigma[1]
        probability_different_bin = gaussian_upper_tail(upper) + gaussian_upper_tail(lower)
        result.append({
            "gamma": gamma,
            "sigma_at_k1": float(sigma[1]),
            "sigma_at_k64": float(sigma[-1]),
            "relative_variance_increment_at_k_ge_2": float(relative_increment[0]),
            "max_normalized_score_update_gain": float(gain.max()),
            "time_indices_strictly_below_fractional_sigma_0p005": int((sigma[1:] < 0.005).sum()),
            "time_indices_strictly_below_fractional_sigma_0p001": int((sigma[1:] < 0.001 - 1e-15).sum()),
            "point_source_last_step_Q_bin_change_probability_f0_0p0049": probability_different_bin,
        })
    return result


def semantic_counterexamples() -> dict:
    # c=0 -> g=0,s=0; c=1 -> g=1,s=1. Global semantic MI is positive,
    # yet fixed-c conditional MI is zero and either g=c or g=s fits all data.
    rows = [{"c": c, "g": c, "s": c, "probability": 0.5} for c in (0, 1)]
    ignored = [r["c"] for r in rows]
    followed = [r["s"] for r in rows]
    assert ignored == followed
    return {
        "unique_composition_toy": {
            "rows": rows,
            "I_G_S_nats": math.log(2),
            "I_G_S_given_C_nats": 0.0,
            "two_perfect_fit_models": ["G=C (ignores S)", "G=S (uses S)"],
            "intervention_C0_S1_predictions": [0, 1],
        },
        "frozen_independent_prior_toy": {
            "attainable_triplets": [[0, 0, 0], [1, 1, 1]],
            "individually_calibrated_Bernoulli_marginals": [0.5, 0.5, 0.5],
            "independent_prior_mass_with_nonempty_attribute_preimage": 2/8,
            "maximum_requested_triplet_compliance": 2/8,
            "boundary": "This is a semantic support counterexample, not a statement that 75% of actual project Plans are impossible or that SUN is bounded by 25%.",
        },
    }


def rollback_counterexample() -> dict:
    c_flip = np.array([[0., 1.], [1., 0.]])
    intended = np.eye(2)
    # Existing full-cell rollback restores U and records a rollback event,
    # while whole-generation success can remain True. Checking only that flag
    # erroneously accepts U; the outer-event-aware kernel restores G instead.
    naive_success_only = c_flip.copy()
    assert np.max(abs(intended.sum(1)-1)) == 0
    assert np.max(abs(naive_success_only.sum(1)-1)) == 0
    return {
        "inner_trace_fixture": {
            "success": True,
            "failure": None,
            "events": [{"op": "rollback", "reason": "full_cell_empty_support"}],
        },
        "noise_kernel": c_flip.tolist(),
        "event_aware_outer_rollback_kernel": intended.tolist(),
        "wrong_success_flag_only_outer_kernel": naive_success_only.tolist(),
        "boundary": "Both are normalized kernels; they are different algorithms. This fixture demonstrates the actual status-interface trap without running a model.",
    }


def snapshot_sources() -> list[dict]:
    records = []
    for name in ("V3_DLM_CANDIDATES_AND_ATTACKS.md", "v3_candidate_cpu_checks.py", "v3_candidate_cpu_checks.json"):
        source = AUDIT / name
        raw = source.read_bytes()
        destination = OUT / "reviewed" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        records.append({"source": name, "sha256": sha256(raw).hexdigest(), "snapshot": str(destination.relative_to(OUT))})
    for source in ("src/crystal_dlm/programmed_path_runtime.py",
                   "src/crystal_dlm/periodic_v2_corruption.py",
                   "src/crystal_dlm/periodic_v2_training_data.py",
                   "src/crystal_dlm/periodic_repair_initialization.py",
                   "src/crystal_dlm/periodic_repair_model.py",
                   "src/scripts/train_periodic_dlm_v2.py"):
        raw = subprocess.check_output(["git", "show", EXECUTION_SHA+":"+source], cwd=ROOT)
        destination = OUT / "frozen" / (source+".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        records.append({"source": source, "git_commit": EXECUTION_SHA, "sha256": sha256(raw).hexdigest(), "snapshot": str(destination.relative_to(OUT))})
    return records


def main() -> None:
    result = {
        "scope": "Independent R1 mathematical counterexamples; no checkpoint, GPU, MLIP, source CIF, or performance measurement.",
        "C_clock_and_endpoint": clock_counterexamples(),
        "C_VP_reverse": [vp_counterexample(6.0), vp_counterexample(100.0)],
        "A_identifiability_and_prior": semantic_counterexamples(),
        "B_outer_rollback": rollback_counterexample(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "review_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    manifest = snapshot_sources()
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
