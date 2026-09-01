from pathlib import Path


def test_final_fused_planner_is_seed22_outcome_blind() -> None:
    text = (Path(__file__).resolve().parents[1] / "slurm/125_sample_final_fused_planner.sbatch").read_text()
    assert "source_seed\t22" in text and "sampling_seed\t22" in text
    assert "--seed 22" in text
    assert "--expected-seed 22" in text
    assert "H1A2_CODE_COMMIT" in text
    assert "outcomes_read\tfalse" in text
    assert "frozen_job39051" in text
    assert "retry_filter_replacement_rerank_best_of_n\tfalse" in text
