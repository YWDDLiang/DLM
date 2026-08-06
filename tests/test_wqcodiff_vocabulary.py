from __future__ import annotations

import unittest

from crystal_dlm.wqcodiff.vocabulary import (
    MP20_ATOMIC_NUMBERS,
    atomic_number_to_input_id,
    crystal_system_from_space_group,
    target_to_atomic_number,
)


class VocabularyTests(unittest.TestCase):
    def test_species_vocabulary_is_exactly_89_and_roundtrips(self) -> None:
        self.assertEqual(len(MP20_ATOMIC_NUMBERS), 89)
        for atomic_number in MP20_ATOMIC_NUMBERS:
            input_id = atomic_number_to_input_id(atomic_number)
            self.assertEqual(target_to_atomic_number(input_id - 1), atomic_number)

    def test_space_group_system_boundaries(self) -> None:
        expected = {
            1: "triclinic",
            2: "triclinic",
            3: "monoclinic",
            15: "monoclinic",
            16: "orthorhombic",
            74: "orthorhombic",
            75: "tetragonal",
            142: "tetragonal",
            143: "trigonal",
            167: "trigonal",
            168: "hexagonal",
            194: "hexagonal",
            195: "cubic",
            230: "cubic",
        }
        self.assertEqual(
            {value: crystal_system_from_space_group(value) for value in expected},
            expected,
        )


if __name__ == "__main__":
    unittest.main()
