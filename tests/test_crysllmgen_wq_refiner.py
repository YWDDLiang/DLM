from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover - dependency-light local source checks
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch is required")
class CrysLLMGenWQRefinerTests(unittest.TestCase):
    def _model_and_batch(self):
        from crystal_dlm.wqcodiff.crysllmgen.wq_refiner import CrysLLMGenWQRefiner
        from crystal_dlm.wqcodiff.model import WQTensorBatch

        class Time(nn.Module):
            def forward(self, values):
                return torch.zeros((len(values), 256), dtype=torch.float32)

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 512)
                self.final_layer_norm = nn.LayerNorm(512)
                self.coord = nn.Linear(512, 3, bias=False)
                self.lattice = nn.Linear(512, 9, bias=False)

            def forward(self, time, atom_types, coords, lattices, counts, batch):
                del time, coords, lattices, counts
                features = self.final_layer_norm(self.embedding(atom_types - 1))
                graph = torch.zeros(
                    (int(batch.max()) + 1, 512), dtype=features.dtype
                )
                graph.index_add_(0, batch, features)
                denominator = torch.bincount(batch).clamp_min(1).to(features.dtype)
                graph = graph / denominator[:, None]
                return self.lattice(graph).reshape(-1, 3, 3), self.coord(features)

        model = CrysLLMGenWQRefiner(decoder=Decoder(), time_embedding=Time())
        batch = WQTensorBatch(
            atom_species=torch.tensor([14, 8, 8]),
            frac_coords=torch.rand(3, 3),
            lattices=torch.eye(3).unsqueeze(0),
            atom_batch=torch.zeros(3, dtype=torch.long),
            atom_to_orbit=torch.tensor([0, 1, 1]),
            orbit_species=torch.tensor([14, 8]),
            orbit_wyckoff=torch.tensor([1, 2]),
            orbit_batch=torch.zeros(2, dtype=torch.long),
            space_group=torch.tensor([225]),
            time=torch.tensor([0.5]),
            geometry_evidence=torch.zeros(2, 6),
        )
        return model, batch

    def test_parent_geometry_and_orbit_heads_have_registered_shapes(self) -> None:
        model, batch = self._model_and_batch()
        output = model(batch)
        self.assertEqual(tuple(output.atom_coordinate_score.shape), (3, 3))
        self.assertEqual(tuple(output.lattice_score.shape), (1, 6))
        self.assertEqual(tuple(output.revision_logits.shape), (2, 3))
        self.assertEqual(tuple(output.event_logits.shape), (1, 5))
        self.assertGreater(model.inherited_parameter_count(), 0)

    def test_refiner_rejects_mask_species_and_can_freeze_parent_only(self) -> None:
        model, batch = self._model_and_batch()
        model.set_inherited_backbone_trainable(False)
        self.assertFalse(any(value.requires_grad for value in model.decoder.parameters()))
        self.assertTrue(any(value.requires_grad for value in model.revision_head.parameters()))
        batch.atom_species[0] = 0
        with self.assertRaisesRegex(ValueError, "MASK"):
            model(batch)


if __name__ == "__main__":
    unittest.main()
