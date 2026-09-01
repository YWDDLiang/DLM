from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_seed22_source_is_outcome_blind() -> None:
    text = (ROOT / "slurm/123_c3fd_final_seed22_source.sbatch").read_text()
    assert "sampling_seed\t22" in text
    assert "--seed 22" in text
    assert "requested\t1000" in text
    assert "selection_or_outcomes\tfalse" in text


def test_final_cohort_excludes_all_existing_cohorts() -> None:
    text = (ROOT / "slurm/124_freeze_g2_final_prospective.sbatch").read_text()
    assert "method_frozen_before_cohort\tG2_periodic_relation" in text
    assert "--exclude-cohort-root \"${ROOT}/cohorts\"" in text
    assert "--planner-sampling-seed 22" in text
    assert "outcomes_read'] is False" in text
