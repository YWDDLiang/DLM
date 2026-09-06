import json
from copy import deepcopy

import pytest
import torch

from crystal_dlm.periodic_v2_initialization import PeriodicV2DLM, validate_v2_final_checkpoint
from crystal_dlm.periodic_v2_model import PeriodicV2Config
from crystal_dlm.programmed_path_data import load_path_model
from test_periodic_raw_initialization import case


def test_partial_epoch_and_unfinished_final_checkpoint_are_not_sampling_policies(tmp_path):
    train = tmp_path / "train"
    checkpoint = train / "checkpoints" / "step-2262"
    checkpoint.mkdir(parents=True)
    (checkpoint / "periodic_v2_config.json").write_text("{}")
    marker = checkpoint / "CHECKPOINT_FINAL.json"
    marker.write_text(json.dumps({"eligible_policy": False}))
    with pytest.raises(ValueError, match="completed registered final"):
        load_path_model("unused", checkpoint, "cpu")
    marker.write_text(json.dumps({"eligible_policy": True, "completed_step": 4524, "completed_epochs": 2}))
    with pytest.raises(ValueError, match="completed registered final"):
        validate_v2_final_checkpoint(checkpoint)
    (train / "_SUCCESS").touch()
    (train / "TRAIN_FINAL.json").write_text(json.dumps({
        "eligible_policy": True, "method": "periodic_dlm_v2_from_original_llada",
        "policy_path": str(checkpoint), "updates": 4524}))
    validate_v2_final_checkpoint(checkpoint)
    # A different intermediate path remains ineligible even after the run ends.
    other = train / "checkpoints" / "step-100"
    other.mkdir()
    (other / "CHECKPOINT_FINAL.json").write_text(json.dumps({"eligible_policy": True, "completed_step": 4524, "completed_epochs": 2}))
    with pytest.raises(ValueError, match="endpoint differ"):
        validate_v2_final_checkpoint(other)


def test_all_v2_modules_and_new_token_rows_survive_compact_save_rebuild(tmp_path):
    tokenizer, raw, current, context, _, _ = case()
    original_base = deepcopy(raw.base_model)
    config = PeriodicV2Config(16, heads=4, width=12)
    model = PeriodicV2DLM(raw.base_model, tokenizer, raw.state_config, raw.repair_config,
                          raw.raw_initialization, config).eval()
    with torch.no_grad():
        model.geometry_attention.legacy.edge[-1].weight.normal_(0, .01)
        for name in ("site_to_site", "cell_to_site", "site_to_cell"):
            getattr(model.geometry_attention, name).weight.normal_(0, .01)
        model.state_conditioner.cell_projection.weight.normal_(0, .001)
        model.numeric_adapter.projections[0].weight.normal_(0, .001)
        model.new_token_rows.input_delta.normal_(0, .001)
        model.new_token_rows.output_delta.normal_(0, .001)
        model.base_model.lora_B["default"].weight.normal_(0, .001)
        expected = model(current, geometry_context=context).logits
    model.save_pretrained(tmp_path)
    restored = PeriodicV2DLM(original_base, tokenizer, raw.state_config, raw.repair_config,
                             raw.raw_initialization, PeriodicV2Config(**json.loads(
                                 (tmp_path / "periodic_v2_config.json").read_text()))).eval()
    restored.base_model.load_state_dict(torch.load(tmp_path / "toy_adapter.pt", weights_only=True), strict=False)
    restored.load_state_conditioner(tmp_path)
    restored.load_repair(tmp_path)
    with torch.no_grad():
        actual = restored(current, geometry_context=context).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
