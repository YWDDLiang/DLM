"""Continuous artifact/ledger integration with real CIFs and no model or MLIP."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPORT = load_script("continuous_artifact_exporter", "export_programmed_path_artifacts.py")
REGRESSION = load_script("continuous_artifact_regression", "analyze_programmed_path_regressions.py")
EVALUATE = load_script("continuous_artifact_evaluator_reader", "evaluate_programmed_paths.py")
BODY = "<N_002><LA_040><LB_040><LC_040><AA_090><AB_090><AG_090>" \
       "<E_H><X_000><Y_000><Z_000><E_H><X_050><Y_050><Z_050>"


@pytest.fixture
def pmg():
    return pytest.importorskip("pymatgen.core")


def path_record(structure, *, ordinal=0, sample_idx=None, success=True):
    elements = list(dict.fromkeys(str(site.specie) for site in structure))
    counts = [sum(str(site.specie) == element for site in structure) for element in elements]
    return {"source_split": "evaluation", "evaluation_ordinal": ordinal,
            "sample_idx": ordinal if sample_idx is None else sample_idx,
            "trajectory_id": f"continuous:{ordinal}", "group_id": f"request:{ordinal}",
            "success": success, "structure": structure.as_dict(), "body": BODY,
            "num_atoms": structure.num_sites,
            "plan_state": {"N": structure.num_sites, "elements": elements, "counts": counts}}


def run_export(tmp_path, monkeypatch, records, *, native_source=None, refined=None, processor=None, name="artifacts"):
    source = tmp_path / f"{name}.input.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    before = source.read_bytes()
    output = tmp_path / name
    argv = ["export", "--paths-jsonl", str(source), "--output-dir", str(output), "--refiner-seed", "100"]
    if native_source is not None:
        argv += ["--native-source", native_source]
    if refined is not None:
        argv += ["--refined-pt", str(tmp_path / "fixture.pt")]
        monkeypatch.setattr(EXPORT, "load_refined_payload", lambda _path: refined)
    if processor is not None:
        def forbidden_array_graph(*_args, **_kwargs):
            raise AssertionError("continuous graph input must not use the quantized body or bin guard")
        monkeypatch.setitem(sys.modules, "scripts.sample_llada_dynamic_crystals",
                            SimpleNamespace(import_process_one=lambda _path: processor, graph_from_arrays=forbidden_array_graph))
        argv += ["--crysllmgen-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    EXPORT.main()
    assert source.read_bytes() == before
    rows = [json.loads(line) for line in (output / "paths.jsonl").read_text(encoding="utf-8").splitlines()]
    report = json.loads((output / "ARTIFACT_FINAL.json").read_text(encoding="utf-8"))
    graphs = torch.load(output / "proposal_graphs.pt", weights_only=False) if processor is not None else None
    return rows, report, graphs


def graph_processor(pmg, observed, *, wrong_composition=False):
    def process_one(cif, *arguments):
        assert arguments == (True, False, "crystalnn", False, .01)
        parsed = pmg.Structure.from_str(cif, fmt="cif")
        observed.append(parsed)
        atom_types = np.array(parsed.atomic_numbers)
        if wrong_composition:
            atom_types[:] = 1
        graph = {"n_atom": np.array([parsed.num_sites]), "a_type": atom_types,
                 "fixture_fractional": parsed.frac_coords.copy()}
        return (parsed.frac_coords, atom_types, parsed.lattice.abc, parsed.lattice.angles,
                parsed.num_sites, np.empty((0, 2)), np.empty((0, 3)), graph)
    return process_one


def test_structure_mode_preserves_float_and_bypasses_q_duplicate_guard(tmp_path, monkeypatch, pmg):
    native = pmg.Structure(pmg.Lattice.cubic(49.8765432198765), ["Na", "Cl"],
                           [[.095123456789]*3, [.104876543211]*3])
    # These different, supported float sites occupy the same .01 token bin.
    assert np.array_equal(np.rint(native.frac_coords[0]*100), np.rint(native.frac_coords[1]*100))
    assert native.distance_matrix[0, 1] > .5
    observed = []
    monkeypatch.setattr(EXPORT, "parse_dynamic_answer", lambda *_a, **_k: pytest.fail("structure mode parsed a Q body"))
    rows, report, graphs = run_export(tmp_path, monkeypatch, [path_record(native)], native_source="structure",
                                      processor=graph_processor(pmg, observed))
    saved = pmg.Structure.from_dict(rows[0]["structure"])
    np.testing.assert_allclose(saved.frac_coords, native.frac_coords, rtol=0, atol=1e-15)
    np.testing.assert_allclose(saved.lattice.matrix, native.lattice.matrix, rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.sort(observed[0].frac_coords, axis=0), np.sort(native.frac_coords, axis=0), rtol=0, atol=1e-14)
    assert observed[0].lattice.a == pytest.approx(native.lattice.a, abs=1e-13)
    assert report["continuous_coordinates_preserved"] and report["native_source"] == "structure"
    assert report["requests"] == report["parseable"] == report["refiner_graphs"] == 1
    assert graphs[0]["sample_idx"] == 0 and graphs[0]["refiner_seed"] == 100
    assert "refiner_graph_error" not in rows[0]


@pytest.mark.parametrize("plan", [{"N": 3, "elements": ["Na", "Cl"], "counts": [1, 1]},
                                  {"N": 2, "elements": ["H"], "counts": [2]}])
def test_structure_mode_rejects_changed_plan_composition(tmp_path, monkeypatch, pmg, plan):
    native = pmg.Structure(pmg.Lattice.cubic(4.1), ["Na", "Cl"], [[.1]*3, [.6]*3])
    row = path_record(native)
    row["plan_state"] = plan
    observed = []
    rows, report, graphs = run_export(tmp_path, monkeypatch, [row], native_source="structure",
                                      processor=graph_processor(pmg, observed))
    assert report["requests"] == 1 and report["parseable"] == 0 and graphs == [] and observed == []
    assert rows[0]["body"] is None and rows[0]["structure"] is None and rows[0]["native_structure"] is None
    assert "Plan composition" in rows[0]["artifact_error"]


def test_graph_composition_failure_keeps_native_request(tmp_path, monkeypatch, pmg):
    native = pmg.Structure(pmg.Lattice.cubic(4.1), ["Na", "Cl"], [[.1]*3, [.6]*3])
    rows, report, graphs = run_export(tmp_path, monkeypatch, [path_record(native)], native_source="structure",
                                      processor=graph_processor(pmg, [], wrong_composition=True))
    assert rows[0]["success"] and rows[0]["parseable"] and graphs == []
    assert "exact composition" in rows[0]["refiner_graph_error"]
    assert report["requests"] == 1 and report["errors"] == {"refiner_graph_failure": 1}


def test_tau_keeps_true_continuous_native_upstream(tmp_path, monkeypatch, pmg):
    native = pmg.Structure(pmg.Lattice.cubic(4.123456789), ["H", "H"], [[.123456789]*3, [.612345678]*3])
    refined = pmg.Structure(pmg.Lattice.cubic(5.234567891), ["H", "H"], [[.223456789]*3, [.712345678]*3])
    observed = []
    rows, report, _ = run_export(tmp_path, monkeypatch, [path_record(native)], native_source="structure",
                                 refined=({0: refined.as_dict()}, {}), processor=graph_processor(pmg, observed))
    row = rows[0]
    np.testing.assert_allclose(pmg.Structure.from_dict(row["structure"]).frac_coords, refined.frac_coords, rtol=0, atol=1e-15)
    np.testing.assert_allclose(pmg.Structure.from_dict(row["native_structure"]).frac_coords, native.frac_coords, rtol=0, atol=1e-15)
    assert observed[0].lattice.a == pytest.approx(native.lattice.a)  # Graphs refer to the native input.
    assert row["endpoint"] == report["endpoint"] == "tau800" and row["diffusion_refinement_steps"] == 800
    upstream = REGRESSION.stored_geometry(row, endpoint="native", native_upstream=True)
    endpoint = REGRESSION.stored_geometry(row, endpoint="tau800")
    assert upstream["source"] == "stored_native_structure_dict"
    assert upstream["volume_per_atom_A3"] == pytest.approx(native.volume / 2)
    assert endpoint["volume_per_atom_A3"] == pytest.approx(refined.volume / 2)
    assert upstream["volume_per_atom_A3"] != pytest.approx(32.)  # The diagnostic body is a 4-A cube.


def test_missing_tau_output_retains_upstream_but_clears_endpoint(tmp_path, monkeypatch, pmg):
    native = pmg.Structure(pmg.Lattice.cubic(4.123456789), ["H", "H"], [[.1234]*3, [.6234]*3])
    rows, report, _ = run_export(tmp_path, monkeypatch, [path_record(native)], native_source="structure", refined=({}, {}))
    row = rows[0]
    assert report["requests"] == 1 and not row["success"] and not row["parseable"]
    assert row["body"] is None and row["structure"] is None
    assert row["diagnostic_candidate"]["body"] == BODY
    assert REGRESSION.stored_geometry(row, endpoint="native", native_upstream=True)["status"] == "available"
    with pytest.raises(ValueError, match="refinement_missing"):
        EVALUATE.read_input_structure(row)


def test_failed_continuous_previews_do_not_consume_graph_or_ledger_indices(tmp_path, monkeypatch, pmg):
    native = pmg.Structure(pmg.Lattice.cubic(4.123), ["H", "H"], [[.1234]*3, [.6234]*3])
    failed = path_record(native, ordinal=0, sample_idx=10, success=False)
    failed.update(failure="nonfinite_sampled_chart", native_structure=native.as_dict())
    good = path_record(native, ordinal=1, sample_idx=11)
    empty = path_record(native, ordinal=2, sample_idx=12, success=False)
    empty.update(body=None, structure=None, failure="sample_failed", diagnostic_candidate={"structure": native.as_dict()})
    observed = []
    rows, report, graphs = run_export(tmp_path, monkeypatch, [failed, good, empty], native_source="structure",
                                      processor=graph_processor(pmg, observed))
    assert report["requests"] == 3 and report["successful"] == report["refiner_graphs"] == 1
    assert [row["sample_idx"] for row in rows] == [10, 11, 12]
    assert [row["evaluation_ordinal"] for row in rows] == [0, 1, 2]
    assert [graph["sample_idx"] for graph in graphs] == [1] and graphs[0]["refiner_seed"] == 111
    assert len(observed) == 1
    for row in (rows[0], rows[2]):
        assert row["body"] is None and row["structure"] is None and row["native_structure"] is None
        with pytest.raises(ValueError):
            EVALUATE.read_input_structure(row)
    assert rows[0]["diagnostic_candidate"]["structure"] == json.loads(json.dumps(failed["structure"]))


def test_structure_mode_missing_structure_cannot_use_valid_body(tmp_path, monkeypatch, pmg):
    native = pmg.Structure(pmg.Lattice.cubic(4.), ["H", "H"], [[0.]*3, [.5]*3])
    record = path_record(native)
    record["structure"] = None
    rows, report, _ = run_export(tmp_path, monkeypatch, [record], native_source="structure")
    assert report["requests"] == 1 and report["parseable"] == 0
    assert rows[0]["structure"] is None and rows[0]["body"] is None


def test_declared_q_clip_failure_is_not_rescued_by_a_complete_preview(tmp_path, monkeypatch, pmg):
    native = pmg.Structure(pmg.Lattice.cubic(4.), ["H", "H"], [[0.]*3, [.5]*3])
    row = path_record(native, success=False)
    row.update(parseable=False, failure="secondary_Q_length_clips", quantization={"length_clips": 1, "angle_clips": 0})
    monkeypatch.setattr(EXPORT, "parse_dynamic_answer", lambda *_a, **_k: pytest.fail("a declared failure was parsed"))
    rows, report, _ = run_export(tmp_path, monkeypatch, [row])  # Default body mode.
    assert report["successful"] == 0 and rows[0]["body"] is None and rows[0]["structure"] is None
    assert rows[0]["diagnostic_candidate"]["body"] == BODY
    with pytest.raises(ValueError):
        EVALUATE.read_input_structure(rows[0])


def test_default_body_mode_keeps_legacy_available_endpoint(tmp_path, monkeypatch, pmg):
    unrelated_preview = pmg.Structure(pmg.Lattice.cubic(6.789), ["H", "H"], [[.1]*3, [.6]*3])
    row = path_record(unrelated_preview, success=False)
    rows, report, _ = run_export(tmp_path, monkeypatch, [row])
    saved = pmg.Structure.from_dict(rows[0]["structure"])
    assert saved.lattice.a == pytest.approx(4.)
    assert rows[0]["success"] and rows[0]["parseable"] and not rows[0]["native_execution_success"]
    assert report["native_source"] == "body" and not report["continuous_coordinates_preserved"]


@pytest.mark.parametrize("native", [None, {"malformed": True}])
def test_explicit_missing_native_upstream_never_falls_back_to_q(native):
    path = {"endpoint": "tau800", "body": BODY, "native_structure": native}
    assert REGRESSION.stored_geometry(path, endpoint="native", native_upstream=True)["status"] == "unavailable"
    del path["native_structure"]
    legacy = REGRESSION.stored_geometry(path, endpoint="native", native_upstream=True)
    assert legacy["status"] == "available" and legacy["volume_per_atom_A3"] == pytest.approx(32.)


def test_regression_tail_reads_continuous_native_not_diagnostic_body():
    from test_programmed_path_regressions import fixture
    values = fixture(1, endpoint="tau800")
    for paths, native_length, refined_length in ((values[2], 4.123, 7.), (values[3], 5.234, 8.)):
        paths[0]["native_structure"] = {"lattice": {"matrix": (np.eye(3)*native_length).tolist()},
                                          "sites": [{"abc": [.1234]*3}, {"abc": [.6234]*3}]}
        paths[0]["structure"] = {"lattice": {"matrix": (np.eye(3)*refined_length).tolist()},
                                 "sites": [{"abc": [.1]*3}, {"abc": [.6]*3}]}
    values[1][0].update(raw_energy_eV_atom=6., gap_eV_atom=6.)
    report = REGRESSION.analyze_regressions(*values)
    upstream = report["A_B_tail_cases"][0]["upstream_native_geometry"]
    assert upstream["reference"]["volume_per_atom_A3"] == pytest.approx(4.123**3/2)
    assert upstream["method"]["volume_per_atom_A3"] == pytest.approx(5.234**3/2)
