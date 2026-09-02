import unittest

from crystal_dlm.species_program_pointer import maximum_contact_tree_order


class ContactTreeTeacherTest(unittest.TestCase):
    def test_contact_teacher_is_deterministic_and_uses_scaffold_connectivity(self):
        species = ["O", "O", "Na", "Cl"]
        distances = [
            [0.0, 2.8, 1.8, 3.2],
            [2.8, 0.0, 1.9, 3.1],
            [1.8, 1.9, 0.0, 2.4],
            [3.2, 3.1, 2.4, 0.0],
        ]
        order = maximum_contact_tree_order(species, distances)
        self.assertEqual(set(order), {"O", "Na", "Cl"})
        self.assertEqual(len(order), 3)
        self.assertEqual(order, maximum_contact_tree_order(species, distances))
        self.assertEqual(order[0], "Na")

    def test_contact_teacher_validates_shape(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            maximum_contact_tree_order(["Na", "Cl"], [[0.0]])


try:
    import torch
    from crystal_dlm.species_program_pointer import (
        PlanConditionedSpeciesPointer,
        SpeciesPointerConfig,
        species_pointer_loss,
    )
except ModuleNotFoundError:
    torch = None
    PlanConditionedSpeciesPointer = None


@unittest.skipIf(PlanConditionedSpeciesPointer is None, "torch unavailable")
class PointerHeadTest(unittest.TestCase):
    def test_teacher_forced_pointer_is_finite_and_composition_preserving(self):
        torch.manual_seed(0)
        model = PlanConditionedSpeciesPointer(
            SpeciesPointerConfig(
                llama_hidden_size=16,
                pointer_size=8,
                num_lattice_systems=3,
                num_spacegroup_buckets=4,
                num_volume_per_atom_bins=5,
            )
        )
        hidden = torch.randn(2, 16)
        atomic = torch.tensor([[8, 11, 17], [6, 8, 0]])
        counts = torch.tensor([[3, 1, 2], [1, 2, 0]])
        valid = torch.tensor([[True, True, True], [True, True, False]])
        soft = torch.tensor([[1, 2, 3], [0, 1, 2]])
        teacher = torch.tensor([[1, 0, 2], [1, 0, 0]])
        logits = model.permutation_logits(
            hidden,
            atomic,
            counts,
            valid,
            soft,
            teacher_order=teacher,
        )
        loss = species_pointer_loss(logits, teacher, valid)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        decoded = model.decode(hidden, atomic, counts, valid, soft)
        self.assertEqual(set(decoded[0].tolist()), {0, 1, 2})
        self.assertEqual(set(decoded[1, :2].tolist()), {0, 1})

    def test_pointer_rejects_duplicate_teacher_prefix(self):
        model = PlanConditionedSpeciesPointer(
            SpeciesPointerConfig(llama_hidden_size=4, pointer_size=4)
        )
        with self.assertRaisesRegex(ValueError, "unavailable"):
            model.permutation_logits(
                torch.zeros(1, 4),
                torch.tensor([[8, 11]]),
                torch.tensor([[1, 1]]),
                torch.tensor([[True, True]]),
                torch.tensor([[0, 0, 0]]),
                teacher_order=torch.tensor([[0, 0]]),
            )


if __name__ == "__main__":
    unittest.main()
