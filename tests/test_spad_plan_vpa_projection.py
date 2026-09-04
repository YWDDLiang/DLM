import unittest


try:
    import torch
    from crystal_dlm.spad_generation import project_spad_cell_to_plan_vpa
except ModuleNotFoundError:
    torch = None
    project_spad_cell_to_plan_vpa = None


@unittest.skipIf(torch is None, "torch unavailable")
class SPADPlanVPAProjectionTest(unittest.TestCase):
    @staticmethod
    def _length_token(axis: str, length_bin: int) -> int:
        return {"LA": 1000, "LB": 2000, "LC": 3000}[axis] + int(
            length_bin
        )

    @staticmethod
    def _coord_token(axis: str, coord_bin: int) -> int:
        return {"X": 5000, "Y": 5200, "Z": 5400}[axis] + int(coord_bin)

    @classmethod
    def _constraints(cls, num_atoms: int) -> dict:
        return {
            "representation": "dynamic_v1",
            "body_offset": 0,
            "max_atoms": int(num_atoms),
            "coord_period": 100,
            "count_token_to_n": {10: int(num_atoms)},
            "length_step": 0.1,
            "length_token_to_bin": {
                axis: {
                    cls._length_token(axis, value): value
                    for value in range(1, 61)
                }
                for axis in ("LA", "LB", "LC")
            },
            "angle_token_to_bin": {
                "AA": {4090: 90},
                "AB": {4190: 90},
                "AG": {4211: 11, 4290: 90},
            },
            "coord_token_to_bin": {
                axis: {
                    cls._coord_token(axis, value): value
                    for value in range(101)
                }
                for axis in ("X", "Y", "Z")
            },
            "pbc_min_distance_mask": True,
            "pbc_min_distance_A": 0.5,
            "pbc_image_radius": 2,
        }

    @classmethod
    def _tokens(
        cls,
        *,
        length_bins: tuple[int, int, int],
        coordinates: list[tuple[int, int, int]],
        gamma: int = 90,
    ) -> torch.Tensor:
        suffix = [
            10,
            cls._length_token("LA", length_bins[0]),
            cls._length_token("LB", length_bins[1]),
            cls._length_token("LC", length_bins[2]),
            4090,
            4190,
            4211 if gamma == 11 else 4290,
        ]
        for slot, (x_bin, y_bin, z_bin) in enumerate(coordinates):
            suffix.extend(
                [
                    20 + slot,
                    cls._coord_token("X", x_bin),
                    cls._coord_token("Y", y_bin),
                    cls._coord_token("Z", z_bin),
                ]
            )
        return torch.tensor([[99, *suffix]], dtype=torch.long)

    @classmethod
    def _project(cls, tokens: torch.Tensor, plan_vpa: str, num_atoms: int):
        return project_spad_cell_to_plan_vpa(
            tokens,
            volume_per_atom_bins_by_batch=[plan_vpa],
            prompt_length=1,
            gen_length=int(tokens.shape[1]) - 1,
            lightweight_decoding_constraints=cls._constraints(num_atoms),
            enabled=True,
        )

    def test_disabled_path_returns_exact_original_without_logs(self):
        tokens = self._tokens(
            length_bins=(31, 31, 31),
            coordinates=[(0, 0, 0), (50, 50, 50)],
        )
        output, logs = project_spad_cell_to_plan_vpa(
            tokens,
            volume_per_atom_bins_by_batch=["volpa_015_019"],
            prompt_length=1,
            gen_length=int(tokens.shape[1]) - 1,
            lightweight_decoding_constraints=self._constraints(2),
            enabled=False,
        )
        self.assertIs(output, tokens)
        self.assertEqual(logs, [])

    def test_in_bin_cell_is_an_exact_noop(self):
        tokens = self._tokens(
            length_bins=(31, 31, 31),
            coordinates=[(0, 0, 0), (50, 50, 50)],
        )
        output, logs = self._project(tokens, "volpa_010_014", 2)
        self.assertTrue(torch.equal(output, tokens))
        self.assertFalse(logs[0]["applied"])
        self.assertEqual(logs[0]["reason"], "already_in_plan_vpa_interval")
        self.assertAlmostEqual(logs[0]["before_vpa"], 14.8955, places=9)
        self.assertEqual(logs[0]["after_vpa"], logs[0]["before_vpa"])

    def test_near_bin_cell_is_projected_to_legal_quantized_lengths(self):
        tokens = self._tokens(
            length_bins=(31, 31, 31),
            coordinates=[(0, 0, 0), (50, 50, 50)],
        )
        output, logs = self._project(tokens, "volpa_015_019", 2)
        self.assertTrue(logs[0]["applied"])
        self.assertGreaterEqual(logs[0]["after_vpa"], 15.0)
        self.assertLess(logs[0]["after_vpa"], 20.0)
        self.assertEqual(
            output[0, 2:5].tolist(),
            [
                self._length_token("LA", 31),
                self._length_token("LB", 31),
                self._length_token("LC", 32),
            ],
        )

    def test_projection_over_five_percent_is_rejected(self):
        tokens = self._tokens(
            length_bins=(30, 30, 30),
            coordinates=[(0, 0, 0), (50, 50, 50)],
        )
        output, logs = self._project(tokens, "volpa_020_024", 2)
        self.assertTrue(torch.equal(output, tokens))
        self.assertEqual(
            logs[0]["reason"], "scale_exceeds_five_percent_trust_region"
        )

    def test_atom_pair_collision_rolls_back_exact_cell(self):
        tokens = self._tokens(
            length_bins=(32, 32, 32),
            coordinates=[(0, 0, 0), (16, 0, 0)],
        )
        output, logs = self._project(tokens, "volpa_010_014", 2)
        self.assertTrue(torch.equal(output, tokens))
        self.assertEqual(logs[0]["reason"], "projection_geometry_rollback")
        self.assertEqual(logs[0]["geometry_validation"], "atom_pair_collision")
        self.assertLess(logs[0]["minimum_125_image_distance_A"], 0.5)
        self.assertEqual(logs[0]["after_vpa"], logs[0]["before_vpa"])

    def test_one_atom_nonzero_self_image_is_checked(self):
        tokens = self._tokens(
            length_bins=(27, 27, 15),
            coordinates=[(0, 0, 0)],
            gamma=11,
        )
        output, logs = self._project(tokens, "volpa_001_001", 1)
        self.assertTrue(torch.equal(output, tokens))
        self.assertEqual(logs[0]["reason"], "projection_geometry_rollback")
        self.assertEqual(
            logs[0]["geometry_validation"], "nonzero_self_image_collision"
        )
        self.assertLess(logs[0]["minimum_125_image_distance_A"], 0.5)

    def test_quantized_tie_breaking_is_deterministic(self):
        tokens = self._tokens(
            length_bins=(31, 31, 31),
            coordinates=[(0, 0, 0), (50, 50, 50)],
        )
        first, first_logs = self._project(tokens, "volpa_015_019", 2)
        second, second_logs = self._project(tokens, "volpa_015_019", 2)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_logs, second_logs)
        self.assertEqual(
            first_logs[0]["new_length_token_ids"],
            [
                self._length_token("LA", 31),
                self._length_token("LB", 31),
                self._length_token("LC", 32),
            ],
        )

    def test_projection_keeps_every_non_length_token_unchanged(self):
        tokens = self._tokens(
            length_bins=(31, 31, 31),
            coordinates=[(0, 0, 0), (50, 50, 50)],
        )
        output, logs = self._project(tokens, "volpa_015_019", 2)
        self.assertTrue(logs[0]["applied"])
        non_length = [
            index for index in range(tokens.shape[1]) if index not in (2, 3, 4)
        ]
        self.assertTrue(torch.equal(output[0, non_length], tokens[0, non_length]))


if __name__ == "__main__":
    unittest.main()
