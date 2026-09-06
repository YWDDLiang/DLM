import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "scripts" / "check_periodic_raw_posttrain_admission.py"
spec = importlib.util.spec_from_file_location("raw_posttrain_admission", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
EXPECTED_CURRENT_K4, THRESHOLDS = module.EXPECTED_CURRENT_K4, module.THRESHOLDS


def test_gate_is_fixed_before_results_and_close_to_current_k4():
    assert EXPECTED_CURRENT_K4 == {
        "requests": 256, "reconstructed": 254,
        "strict_sun": 7, "meta_sun": 57,
    }
    assert THRESHOLDS == {
        "requests": 256, "reconstructed": 252,
        "strict_sun": 6, "meta_sun": 55,
    }
    assert all(THRESHOLDS[key] <= EXPECTED_CURRENT_K4[key] for key in THRESHOLDS)
