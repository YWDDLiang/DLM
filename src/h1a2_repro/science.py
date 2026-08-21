"""Paper-facing scientific defaults.

Only seeds supported by recovered evidence are numeric. Missing historical
training seeds stay ``None`` until the A800 audit is complete.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedDefaults:
    release_python_hash: int = 0
    planner_train: int = 17
    planner_sample_base: int = 17
    quick_planner_sample: int = 17029
    quick_ledger_training_label: int = 17
    quick_ledger_sampling_root: int = 17029
    dlm_data: int = 20260515
    dlm_train_global: int | None = None
    dlm_inference: int | None = None
    diffusion_upstream_default: int = 1234
    diffusion_train_confirmed: int | None = None
    diffusion_inference: int | None = None


@dataclass(frozen=True)
class InferenceDefaults:
    planner_attempts: int = 1200
    planner_world_size: int = 2
    planner_batch_size: int = 4
    planner_temperature: float = 0.9
    planner_top_p: float = 0.95
    planner_top_k: int = 50
    planner_max_new_tokens: int = 96
    quick_attempts: int = 256
    quick_repeats: int = 4
    refined_target: int = 1000
    diffusion_steps: int = 800


SEEDS = SeedDefaults()
INFERENCE = InferenceDefaults()

PUBLIC_RESULT = {
    "method": "H1-A2 method family",
    "primary_view": "paper_sun_main_table",
    "entries": 1000,
    "strict_sun": {"numerator": 105, "denominator": 1000, "rate": 0.105},
    "meta_sun": {"numerator": 488, "denominator": 1000, "rate": 0.488},
    "exact_all_attempt_view": {
        "description": "exact parsed-Plan replay; all requested attempts",
        "entries": 1200,
        "strict_sun": {"numerator": 103, "denominator": 1200, "rate": 103 / 1200},
        "meta_sun": {"numerator": 553, "denominator": 1200, "rate": 553 / 1200},
        "hull_known": 1132,
        "hull_unknown": 32,
        "hull_known_strict_sun": {"numerator": 103, "denominator": 1132, "rate": 103 / 1132},
        "hull_known_meta_sun": {"numerator": 553, "denominator": 1132, "rate": 553 / 1132},
    },
    "historical_compatibility_view": {
        "description": "historical frozen first-1000 refined artifact",
        "entries": 1000,
        "strict_sun": {"numerator": 94, "denominator": 1000, "rate": 0.094},
        "meta_sun": {"numerator": 474, "denominator": 1000, "rate": 0.474},
    },
}
