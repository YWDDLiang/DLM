from pathlib import Path


def test_final_generation_has_only_base_and_g2_matched() -> None:
    text=(Path(__file__).resolve().parents[1]/'slurm/126_g2_final_generation.sbatch').read_text()
    assert 'arms\tBASE,G2' in text
    assert 'dlm_seed\t91117' in text and 'refiner_seed\t101117' in text
    assert '--temperature 0.7 --seed 91117' in text
    assert '--periodic-relation-rank 64' in text
    assert 'retry_rerank_replacement_best_of_n\tfalse' in text
    assert 'G1' not in text
