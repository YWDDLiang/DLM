from pathlib import Path


def test_final_offline_has_four_fixed_cells() -> None:
    text=(Path(__file__).resolve().parents[1]/'slurm/127_g2_final_offline_eval.sbatch').read_text()
    assert 'arms\tBASE,G2' in text and 'endpoints\traw,refined' in text
    for call in ('BASE raw control','BASE refined control','G2 raw candidate','G2 refined candidate'):
        assert call in text
    assert 'chgnet\tall_four_cells' in text
    assert 'H1_ACTIVE_DENOMINATOR=256' in text
    assert 'official_query\tfalse' in text
