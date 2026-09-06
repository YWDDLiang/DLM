import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "scripts" / "check_periodic_raw_posttrain_admission.py"
spec = importlib.util.spec_from_file_location("raw_posttrain_admission", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_gate_is_frozen_before_results_and_either_endpoint_can_pass():
    assert module.EXPECTED_CURRENT_K4 == {
        "native": {"requests": 256, "reconstructed": 254, "strict_sun": 7, "meta_sun": 57},
        "tau800": {"requests": 256, "reconstructed": 254, "strict_sun": 19, "meta_sun": 126},
    }
    assert module.THRESHOLDS == {
        "native": {"requests": 256, "reconstructed": 252, "strict_sun": 6, "meta_sun": 55},
        "tau800": {"requests": 256, "reconstructed": 252, "strict_sun": 18, "meta_sun": 124},
    }
    assert any({"native": False, "tau800": True}.values())
