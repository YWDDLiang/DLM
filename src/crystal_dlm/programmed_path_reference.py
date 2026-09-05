"""The single closure reference, using the retained cell and species routines."""
from __future__ import annotations

import torch
from torch import nn

from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.programmed_path_runtime import complete_geometry_supported
from crystal_dlm.spad_generation import revise_spad_cell, revise_spad_species_blocks
from crystal_dlm.spad_program import reverse_species_block_revision_slots


class UnconditionedReferenceModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base_model = base

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def forward(self, input_ids, attention_mask=None, *, geometry_context=None):
        return self.base_model(input_ids, attention_mask=attention_mask)


@torch.no_grad()
def close_reference(model, tokens, traces, *, programs, seeds, prompt_length,
                    attention_mask, allowed, constraints, temperature, mask_id=MASK_TOKEN_ID):
    """Actual old cell->reverse-species closure, with paired RNG namespaces.

    Only RNG addressing changes: old transactions, masks and acceptance rules
    are retained. A shared final validator also checks periodic self images.
    Old revision logs are evaluation metadata, not complete training traces.
    """
    result = tokens.clone()
    alive = [i for i, t in enumerate(traces) if t["success"]]
    ledgers = [None] * len(traces)
    if not alive:
        return result, ledgers
    sub = tokens[alive]
    common = dict(prompt_length=prompt_length, gen_length=tokens.shape[1] - prompt_length,
                  attention_mask=attention_mask[alive], temperature=temperature, cfg_scale=0.,
                  remasking="low_confidence", mask_id=mask_id,
                  allowed_token_ids_by_generation_pos=allowed, atom_count_grammar=None,
                  lightweight_decoding_constraints=constraints)
    # These offsets match the new sampler's shared cell and closure draws.
    cell, cell_logs = revise_spad_cell(model, sub, **common, strict_geometry_fallback=True,
                                      sampling_seeds_by_batch=[seeds[i] + 200_000_000 for i in alive])
    blocks, block_logs = revise_spad_species_blocks(
        model, cell, **common,
        revision_blocks_by_batch=[reverse_species_block_revision_slots(programs[i]) for i in alive],
        sampling_seeds_by_batch=[seeds[i] + 300_000_000 for i in alive],
        sampling_salt_fn=lambda block, site, component: block * 1_000_000 + site * 10_000 + component,
    )
    for index, row in enumerate(alive):
        result[row] = blocks[index]
        supported = complete_geometry_supported(result[row, prompt_length:], constraints)
        if not supported:
            traces[row]["success"] = False
            traces[row]["failure"] = "reference_common_final_support"
        ledgers[row] = {"cell": cell_logs[index], "species_blocks": block_logs[index],
                        "stage_order": ["predictor", "cell", "reverse_species_blocks"],
                        "common_final_support": supported, "inference_mlip": False}
    return result, ledgers
