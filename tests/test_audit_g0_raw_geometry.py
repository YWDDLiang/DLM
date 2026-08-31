from scripts.audit_g0_raw_geometry import _attempt_index, _terminal_class


def test_attempt_index_prefers_explicit_identity() -> None:
    assert _attempt_index({"sample_idx": 7, "attempt_id": "attempt-0003"}, 2) == 7
    assert _attempt_index({"attempt_id": "attempt-0042"}, 2) == 42
    assert _attempt_index({}, 9) == 9


def test_terminal_class_uses_registered_precedence() -> None:
    assert _terminal_class({"composition": True, "parse": True}) == "parse"
    assert _terminal_class({"lattice": True, "pbc_min_distance_lt_0p5A": True}) == "lattice"
    assert _terminal_class({"pbc_min_distance_lt_0p5A": True}) == "pbc_min_distance_lt_0p5A"
    assert _terminal_class({}) == "pass"
