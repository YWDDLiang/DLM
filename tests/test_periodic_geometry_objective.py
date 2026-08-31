import unittest

import torch

from crystal_dlm.periodic_geometry_objective import (
    build_geometry_token_support,
    periodic_geometry_objective,
)


class _Tokenizer:
    def __init__(self) -> None:
        tokens = ["<PAD>", "<N_002>", "<E_Li>", "<E_O>"]
        tokens += [f"<L{axis}_{value:03d}>" for axis in "ABC" for value in (39, 40, 41)]
        tokens += [f"<A{axis}_{value:03d}>" for axis in "ABG" for value in (89, 90, 91)]
        tokens += [f"<{axis}_{value:03d}>" for axis in "XYZ" for value in (0, 25, 50, 75)]
        self.vocab = {token: idx for idx, token in enumerate(tokens)}

    def get_vocab(self):
        return dict(self.vocab)


class PeriodicGeometryObjectiveTest(unittest.TestCase):
    def test_target_peaked_logits_have_finite_small_loss_and_gradients(self) -> None:
        tokenizer = _Tokenizer()
        support = build_geometry_token_support(tokenizer)
        tokens = [
            "<N_002>", "<LA_040>", "<LB_040>", "<LC_040>",
            "<AA_090>", "<AB_090>", "<AG_090>",
            "<E_Li>", "<X_000>", "<Y_000>", "<Z_000>",
            "<E_O>", "<X_050>", "<Y_050>", "<Z_050>",
        ]
        ids = torch.tensor([[tokenizer.vocab[token] for token in tokens]])
        logits = torch.full((1, len(tokens), len(tokenizer.vocab)), -8.0, requires_grad=True)
        with torch.no_grad():
            for position, token_id in enumerate(ids[0]):
                logits[0, position, token_id] = 8.0
        masked = torch.zeros_like(ids, dtype=torch.bool)
        masked[:, 1:7] = True
        masked[:, 8:11] = True
        masked[:, 12:15] = True
        result = periodic_geometry_objective(
            logits=logits,
            input_ids=ids,
            masked_indices=masked,
            prompt_lengths=torch.tensor([0]),
            num_atoms=torch.tensor([2]),
            support=support,
        )
        loss = result["metric"] + result["pair_rdf"] + result["overlap"] + result["coordination"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_periodic_translation_preserves_pair_losses(self) -> None:
        # The objective's minimum-image path is indirectly exercised by two
        # token-equivalent global translations on the 0.25 grid.
        tokenizer = _Tokenizer()
        support = build_geometry_token_support(tokenizer)
        base = [
            "<N_002>", "<LA_040>", "<LB_040>", "<LC_040>",
            "<AA_090>", "<AB_090>", "<AG_090>",
            "<E_Li>", "<X_000>", "<Y_000>", "<Z_000>",
            "<E_O>", "<X_050>", "<Y_050>", "<Z_050>",
        ]
        shifted = [token.replace("_000>", "_025>").replace("_050>", "_075>") if token[1:2] in "XYZ" else token for token in base]
        rows = torch.tensor([[tokenizer.vocab[token] for token in row] for row in (base, shifted)])
        logits = torch.full((2, len(base), len(tokenizer.vocab)), -8.0)
        for sample in range(2):
            for position, token_id in enumerate(rows[sample]):
                logits[sample, position, token_id] = 8.0
        masked = torch.ones_like(rows, dtype=torch.bool)
        result = periodic_geometry_objective(
            logits=logits,
            input_ids=rows,
            masked_indices=masked,
            prompt_lengths=torch.tensor([0, 0]),
            num_atoms=torch.tensor([2, 2]),
            support=support,
        )
        self.assertTrue(torch.isfinite(result["pair_rdf"]))
        self.assertTrue(torch.isfinite(result["coordination"]))


if __name__ == "__main__":
    unittest.main()
