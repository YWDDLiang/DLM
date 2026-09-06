from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analyze_programmed_path_regressions",
                                             ROOT / "scripts" / "analyze_programmed_path_regressions.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(count=3, endpoint="native"):
    reference, method, reference_paths, method_paths = [], [], [], []
    body = "<N_002><LA_040><LB_040><LC_040><AA_090><AB_090><AG_090>" \
           "<E_H><X_000><Y_000><Z_000><E_H><X_050><Y_050><Z_050>"
    for index in range(count):
        path = {
            "sample_idx": index, "group_id": f"eval:{index}", "source_split": "evaluation",
            "prompt": "fixed Plan prompt", "plan_state": {"N": 2, "elements": ["H"], "counts": [2]},
            "species_program": ["H"], "species_program_source": "frozen_pointer",
            "num_atoms": 2, "sampling_seed": 100 + index, "sampling_batch_size": 4,
            "candidate_index": 0, "endpoint": endpoint, "success": True, "body": body,
        }
        a = dict.fromkeys(MODULE.BINARY_FIELDS, True)
        a.update(sample_idx=index, group_id=path["group_id"], trajectory_id=f"reference:{index}",
                 terminal_status="verified", official_hull_status="known", chemsys="H",
                 hull_energy_eV_atom=0., raw_energy_eV_atom=5., terminal_energy_eV_atom=0., gap_eV_atom=5.,
                 raw={"force_max_eV_A": 1., "stress_max_GPa": 2.},
                 terminal={"force_max_eV_A": .05, "stress_max_GPa": .1}, actual_relaxation_steps=30)
        b = deepcopy(a)
        b["trajectory_id"] = f"method:{index}"
        reference.append(a)
        method.append(b)
        reference_paths.append(dict(path, trajectory_id=a["trajectory_id"]))
        method_paths.append(dict(path, trajectory_id=b["trajectory_id"]))
    manifest = {
        "endpoint": endpoint, "counts": {"requests": count}, "cohort_role": "fixed_development",
        "terminal_protocol": {"model": "CHGNet-0.3.0", "optimizer": "FIRE", "fmax": .1,
                              "stress_tolerance_GPa": .5, "max_steps": 500},
        "verification_protocol": {"energy_tolerance_eV_atom": .001},
        "frozen_nu_source_sha256": "frozen", "official_cache": "/fixed/hull", "policy_stage": "reference",
    }
    method_manifest = deepcopy(manifest)
    method_manifest["policy_stage"] = "final"
    return reference, method, reference_paths, method_paths, manifest, method_manifest


def test_missing_and_new_failures_keep_full_binary_denominator_and_evidence():
    values = fixture()
    reference, method, _, method_paths, _, _ = values
    method[1].update(terminal_verified=False, terminal_status="generation_failure", reconstructed=False,
                     native_execution_success=False, endpoint_execution_success=False,
                     raw_energy_eV_atom=None, terminal_energy_eV_atom=None, gap_eV_atom=None,
                     raw=None, terminal=None, actual_relaxation_steps=None, hull_energy_eV_atom=None,
                     official_hull_status="input_not_reconstructed", parser_error="missing geometry")
    method[1].update({field: False for field in ("novel", "unique_representative", "novel_unique", "strict_stable", "meta_stable",
                                               "strict_sun", "meta_sun", "verified_strict_sun", "verified_meta_sun")})
    method_paths[1].update(success=False, parseable=False, artifact_error="native parser failed",
                           trace={"failure": "construct_empty_support"})
    reference[2]["terminal_verified"] = False
    reference[2]["terminal_status"] = "not_converged"
    report = MODULE.analyze_regressions(*values)
    binary = report["binary_all_requests"]["terminal_verified"]
    assert report["requests"] == binary["requests"] == 3
    assert (binary["both"], binary["reference_only"], binary["method_only"], binary["neither"]) == (1, 1, 1, 0)
    finite = report["finite_physical_intersections"]["A_gap_eV_atom"]
    verified = report["common_verified_intersections"]["A_gap_eV_atom"]
    assert finite["n"] == 2 and finite["excluded_requests"] == 1
    assert verified["n"] == 1 and verified["excluded_requests"] == 2
    assert finite["delta"]["mean"] == 0.
    assert finite["nonfinite_or_missing_within_eligible"] == {"method_only": 1}
    lost = report["execution_and_verification_transitions"]["native_execution_success"]["lost"]
    assert len(lost) == 1 and lost[0]["sample_idx"] == 1
    assert lost[0]["method_evidence"]["path_trace_failure"] == "construct_empty_support"
    assert "optimizer_converged" in lost[0]["method_evidence"]["unrecorded_label_detail_fields"]
    assert report["hull_and_energy_missingness"]["method"]["hull_nonfinite_or_missing"] == 1


def test_outlier_tail_contribution_and_fixed_symmetric_trim_are_explicit():
    values = fixture(11)
    for index, row in enumerate(values[1]):
        delta = 100. if index == 10 else 1.
        row["terminal_energy_eV_atom"] = -.2
        row["raw_energy_eV_atom"] = 5. + delta - .2
        row["gap_eV_atom"] = 5. + delta
    report = MODULE.analyze_regressions(*values)
    metric = report["common_verified_intersections"]["A_gap_eV_atom"]
    assert metric["n"] == 11 and metric["delta"]["mean"] == 10.
    assert metric["delta"]["median"] == metric["delta"]["p10"] == metric["delta"]["p90"] == 1.
    assert metric["symmetric_trimmed_delta"] == {
        "fraction_each_tail": .1, "removed_per_tail": 1, "retained_n": 9, "mean": 1.,
        "selection": "rank by paired delta, remove floor(0.10*n) at each end",
    }
    top1 = metric["absolute_tail_removals"]["1"]
    assert top1["mean_delta_without_tail"] == 1.
    assert top1["removed_signed_delta_sum"] == 100.
    assert top1["removed_contribution_to_original_mean"] == pytest.approx(100 / 11)
    assert top1["removed_sample_idx"] == [10]
    assert metric["largest_increases"][0]["sample_idx"] == 10
    assert metric["largest_decreases"] == []
    assert any(case["sample_idx"] == 10 and case["method_input_geometry"]["status"] == "available"
               for case in report["A_B_tail_cases"])
    decomposition = report["energy_decomposition"]["common_verified_complete_energy_pairs"]
    assert decomposition["identity_verified"]
    assert decomposition["mean_delta_A"] == pytest.approx(
        decomposition["mean_delta_raw_energy"] - decomposition["mean_delta_eR_equals_delta_B"])
    assert report["binary_all_requests"]["strict_sun"]["requests"] == 11


def test_energy_decomposition_uses_one_common_finite_subset():
    values = fixture(3)
    values[1][0].update(raw_energy_eV_atom=7., terminal_energy_eV_atom=-1., gap_eV_atom=8.)
    values[1][1]["raw_energy_eV_atom"] = None
    values[1][2]["terminal_verified"] = False
    values[1][2]["terminal_status"] = "not_converged"
    report = MODULE.analyze_regressions(*values)
    verified = report["energy_decomposition"]["common_verified_complete_energy_pairs"]
    assert verified["n"] == 1 and verified["excluded_requests"] == 2
    assert verified["mean_delta_A"] == 3.
    assert verified["mean_delta_raw_energy"] == 2.
    assert verified["mean_delta_eR_equals_delta_B"] == -1.
    assert report["common_verified_intersections"]["A_gap_eV_atom"]["n"] == 2


def test_actual_triclinic_metric_includes_periodic_pair_and_short_self_images():
    lattice = np.array([[4., 0., 0.], [3.9, .2, 0.], [0., 0., 4.]])
    geometry = MODULE.geometry_from_arrays(lattice, [[0., 0., 0.], [.49, .49, 0.]], source="fixture")
    expected_pair = np.linalg.norm(np.array([.49, -.51, 0.]) @ lattice)
    expected_self = np.linalg.norm(lattice[0] - lattice[1])
    assert geometry["minimum_distinct_pair_distance_A"] == pytest.approx(expected_pair)
    assert geometry["minimum_self_image_distance_A"] == pytest.approx(expected_self)
    assert geometry["minimum_distinct_pair_distance_A"] < .11
    assert geometry["minimum_self_image_distance_A"] < .23
    assert geometry["volume_per_atom_A3"] == pytest.approx(abs(np.linalg.det(lattice)) / 2)
    assert geometry["lattice_condition_number_2"] == pytest.approx(np.linalg.cond(lattice))
    assert geometry["periodic_image_radius"] == 2
    single = MODULE.geometry_from_arrays(lattice, [[0., 0., 0.]], source="fixture")
    assert single["minimum_distinct_pair_distance_A"] is None
    assert single["minimum_any_periodic_distance_A"] == pytest.approx(expected_self)


def test_tau_geometry_cannot_be_replaced_by_native_body():
    values = fixture(1, endpoint="tau800")
    path = values[2][0]
    result = MODULE.stored_geometry(path, endpoint="tau800")
    assert result["status"] == "unavailable" and "not a refined substitute" in result["reason"]
    upstream = MODULE.stored_geometry(path, endpoint="native", native_body_only=True)
    assert upstream["status"] == "available" and upstream["volume_per_atom_A3"] == pytest.approx(32.)
    path["structure"] = {"lattice": {"matrix": (np.eye(3) * 8).tolist()},
                         "sites": [{"abc": [0, 0, 0]}, {"abc": [.5, .5, .5]}]}
    result = MODULE.stored_geometry(path, endpoint="tau800")
    assert result["volume_per_atom_A3"] == pytest.approx(256.)
    assert result["source"] == "stored_endpoint_structure_dict"


@pytest.mark.parametrize("field,value", [("terminal_protocol", {"model": "other"}),
                                         ("verification_protocol", {"different": True}),
                                         ("frozen_nu_source_sha256", "other"), ("official_cache", "/other")])
def test_protocol_mismatch_is_rejected(field, value):
    values = fixture()
    values[5][field] = value
    with pytest.raises(ValueError, match="protocols differ"):
        MODULE.analyze_regressions(*values)


def test_hull_and_attempt_path_identity_mismatches_are_rejected():
    values = fixture()
    values[1][0]["hull_energy_eV_atom"] = .01
    with pytest.raises(ValueError, match="hull references differ"):
        MODULE.analyze_regressions(*values)
    values = fixture()
    values[1][0]["trajectory_id"] = "unrelated"
    with pytest.raises(ValueError, match="attempt/path identity"):
        MODULE.analyze_regressions(*values)
    values = fixture()
    values[3][0]["sampling_seed"] += 1
    with pytest.raises(ValueError, match="paired condition differs"):
        MODULE.analyze_regressions(*values)


def test_empty_numerical_intersection_and_small_tail_deletion_are_not_imputed():
    values = fixture(1)
    values[1][0].update(raw_energy_eV_atom=None, gap_eV_atom=float("nan"), terminal_energy_eV_atom=None,
                       terminal_verified=False, terminal_status="evaluation_error")
    report = MODULE.analyze_regressions(*values)
    metric = report["finite_physical_intersections"]["A_gap_eV_atom"]
    assert metric["n"] == 0 and metric["excluded_requests"] == 1
    assert metric["delta"]["mean"] is None and metric["symmetric_trimmed_delta"]["mean"] is None
    assert all(row["mean_delta_without_tail"] is None for row in metric["absolute_tail_removals"].values())
    assert report["binary_all_requests"]["terminal_verified"]["reference_only"] == 1
    json.dumps(report, allow_nan=False)


def test_exact_mcnemar_uses_discordant_pairs_only():
    assert MODULE.exact_mcnemar(0, 0) == 1.
    assert MODULE.exact_mcnemar(1, 1) == 1.
    assert MODULE.exact_mcnemar(3, 0) == .25
    assert MODULE.exact_mcnemar(0, 5) == .0625


def test_cli_writes_immutable_diagnostic_artifacts_only(tmp_path, monkeypatch):
    values = fixture(2)
    values[1][0].update(raw_energy_eV_atom=6., gap_eV_atom=6.)
    reference_dir, method_dir = tmp_path / "reference", tmp_path / "method"
    for directory, rows, manifest in ((reference_dir, values[0], values[4]), (method_dir, values[1], values[5])):
        directory.mkdir()
        (directory / "_SUCCESS").touch()
        (directory / "attempt_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        (directory / "EVALUATION_FINAL.json").write_text(json.dumps(manifest), encoding="utf-8")
    reference_path, method_path = tmp_path / "reference_paths.jsonl", tmp_path / "method_paths.jsonl"
    reference_path.write_text("".join(json.dumps(row) + "\n" for row in values[2]), encoding="utf-8")
    method_path.write_text("".join(json.dumps(row) + "\n" for row in values[3]), encoding="utf-8")
    output = tmp_path / "analysis"
    monkeypatch.setattr(sys, "argv", ["analyze", "--reference-eval-dir", str(reference_dir), "--method-eval-dir", str(method_dir),
                                     "--reference-paths-jsonl", str(reference_path), "--method-paths-jsonl", str(method_path),
                                     "--output-dir", str(output)])
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    MODULE.main()
    assert (output / "REGRESSION_ANALYSIS.json").is_file() and (output / "analysis.md").is_file()
    assert (output / "_SUCCESS").is_file()
    assert all(path.read_bytes() == contents for path, contents in before.items())
    report = json.loads((output / "REGRESSION_ANALYSIS.json").read_text(encoding="utf-8"))
    assert report["provenance"]["input_sha256_before"] == report["provenance"]["input_sha256_after"]
    assert "Full-request counts" in (output / "analysis.md").read_text(encoding="utf-8")
