import hashlib
import json
from pathlib import Path

import pytest

from crystal_dlm.terminal_energy_consistency import TERMINAL_VERIFICATION_PROTOCOL
from scripts.build_programmed_path_teacher import main, uniform_reference_settings


def test_reference_settings_require_documented_explicit_membership(tmp_path):
    assert uniform_reference_settings([]) == ({}, None)
    with pytest.raises(ValueError, match="requires"):
        uniform_reference_settings(["13087"])
    with pytest.raises(ValueError, match="requires"):
        uniform_reference_settings(["13087"], " ")
    with pytest.raises(ValueError, match="documentation requires"):
        uniform_reference_settings([], "uncertain")
    with pytest.raises(ValueError, match="unique"):
        uniform_reference_settings(["13087", "13087"], "uncertain")
    path = tmp_path / "reference_uncertainty.json"
    record = {"group_ids": ["13087"], "reason": "Unresolved training-label energy scale.",
              "evidence": {"geometry_audit": "no short-image violation found"},
              "selection_scope": "training labels only", "physical_error_proven": False}
    path.write_text(json.dumps(record), encoding="utf-8")
    groups, provenance = uniform_reference_settings(["13087"], manifest_path=path)
    assert groups == {"13087": record["reason"]}
    assert provenance["manifest"] == record
    assert provenance["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance["manifest_path"] == str(path.resolve())
    assert provenance["labels_and_verification_unchanged"]
    assert provenance["fixed_groups_retain_path_supervision"]
    assert not provenance["physical_disproof_claimed"]
    with pytest.raises(ValueError, match="exactly match"):
        uniform_reference_settings(["other"], manifest_path=path)
    with pytest.raises(ValueError, match="reasons differ"):
        uniform_reference_settings(["13087"], "another reason", path)
    record.pop("reason")
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty reason"):
        uniform_reference_settings(["13087"], manifest_path=path)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_builder_cli_preserves_all_requests_and_constraint_provenance(tmp_path):
    path_dir, label_dir = tmp_path / "paths", tmp_path / "labels"
    path_dir.mkdir()
    label_dir.mkdir()
    for directory in (path_dir, label_dir):
        (directory / "_SUCCESS").touch()
    protocol = {"model": "CHGNet-0.3.0", "optimizer": "FIRE", "relax_cell": True,
                "ase_filter": "FrechetCellFilter", "fmax": .1, "stress_tolerance_GPa": .5,
                "max_steps": 500, "scalar_pressure": 0., "constant_volume": False,
                "hydrostatic_strain": False}
    (label_dir / "LABEL_FINAL.json").write_text(json.dumps({
        "purpose": "train", "verification_protocol": TERMINAL_VERIFICATION_PROTOCOL,
        "protocol": protocol,
    }), encoding="utf-8")
    paths, labels = [], []
    for source, group_id in enumerate(("13087", "other")):
        for candidate in range(4):
            identity = f"{group_id}:0:{candidate}"
            paths.append({"trajectory_id": identity, "group_id": group_id,
                          "source_row_idx": source, "source_split": "train", "success": True,
                          "checkpoint": "unchanged-reference", "collection_round": 0,
                          "candidate_index": candidate, "final_body_token_ids": [0],
                          "trace": {"initial_body": [0], "events": []}})
            verified = candidate < 2
            a = .2 * candidate
            b = (-83.8574 if candidate else -10.) if source == 0 else .2 * candidate
            labels.append({"trajectory_id": identity, "group_id": group_id,
                           "source_row_idx": source, "source_split": "train",
                           "verified": verified, "status": "verified" if verified else "not_converged",
                           "raw_energy": a + b, "terminal_energy": b,
                           "versions": {"model": "CHGNet-0.3.0", "fixture": True}})
    path_file, label_file = path_dir / "paths.jsonl", label_dir / "labels.jsonl"
    _write_jsonl(path_file, paths)
    _write_jsonl(label_file, labels)
    output = tmp_path / "constrained_teacher"
    reason = "Unresolved training-label scale; protocol verification is retained."
    common = ["--paths-jsonl", str(path_file), "--labels-jsonl", str(label_file),
              "--expected-conditions", "2", "--candidates", "4", "--diagnostic-only"]
    main([*common, "--output-dir", str(output), "--uniform-reference-group", "13087",
          "--uniform-reference-reason", reason])
    teacher = json.loads((output / "teacher.json").read_text(encoding="utf-8"))
    assert (output / "_SUCCESS").is_file()
    summary = teacher["summary"]
    assert summary["total_candidates"] == 8
    assert summary["total_groups"] == summary["validated_groups"] == 2
    assert summary["validated_candidates"] == 4
    assert summary["uniform_reference_group_ids"] == ["13087"]
    assert summary["fixed_reference_residual"] == 0
    assert not summary["trainable_teacher"]  # Explicit diagnostic mode is unchanged.
    fixed = teacher["groups"][0]
    assert [row["weight"] for row in fixed["candidates"]] == [.5, .5, 0., 0.]
    assert [row["verified"] for row in fixed["candidates"]] == [True, True, False, False]
    assert [row["terminal_energy"] for row in fixed["candidates"]] == [row["terminal_energy"] for row in labels[:4]]
    assert [row["status"] for row in fixed["candidates"]] == [row["status"] for row in labels[:4]]
    provenance = teacher["provenance"]
    assert provenance["checkpoint"] == "unchanged-reference"
    assert provenance["uniform_reference_constraint"]["reason"] == reason
    assert provenance["source_sha256"][str(path_file)] == hashlib.sha256(path_file.read_bytes()).hexdigest()
    assert provenance["source_sha256"][str(label_file)] == hashlib.sha256(label_file.read_bytes()).hexdigest()
    baseline_output = tmp_path / "baseline_teacher"
    main([*common, "--output-dir", str(baseline_output)])
    baseline = json.loads((baseline_output / "teacher.json").read_text(encoding="utf-8"))
    assert "uniform_reference_constraint" not in baseline["provenance"]
    assert "uniform_reference_group_ids" not in baseline["summary"]
