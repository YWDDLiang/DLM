import unittest


try:
    import torch
    from crystal_dlm.llada_generation import _apply_lightweight_decoding_masks
except ModuleNotFoundError:
    torch = None
    _apply_lightweight_decoding_masks = None


@unittest.skipIf(torch is None, "torch unavailable")
class SPADGeometryDecodingTest(unittest.TestCase):
    def test_periodic_000_100_logits_are_merged_into_canonical_000(self):
        logits = torch.zeros((1, 12, 32), dtype=torch.float32)
        x = torch.zeros((1, 12), dtype=torch.long)
        active = torch.zeros((1, 11), dtype=torch.bool)
        active[0, 10] = True
        constraints = {
            "canonicalize_periodic_alias": True,
            "coordinate_alias_token_ids": {"Z": (10, 11)},
            "max_atoms": 1,
        }
        _apply_lightweight_decoding_masks(
            logits,
            x,
            prompt_length=1,
            gen_length=11,
            constraints=constraints,
            active_generation_mask=active,
            mask_id=31,
        )
        self.assertAlmostEqual(float(logits[0, 11, 10]), 0.69314718, places=5)
        self.assertEqual(float(logits[0, 11, 11]), torch.finfo(torch.float32).min)

    def test_z_commit_masks_every_candidate_below_half_angstrom_mic(self):
        prompt_length = 1
        gen_length = 15
        mask_id = 499
        logits = torch.zeros((1, prompt_length + gen_length, 500), dtype=torch.float32)
        x = torch.zeros((1, prompt_length + gen_length), dtype=torch.long)
        x[0, 0] = 9
        generation = x[0, prompt_length:]
        generation[0] = 1  # N=2
        generation[1:4] = 2  # a=b=c=1.0 A
        generation[4:7] = 3  # alpha=beta=gamma=90 degrees
        generation[7] = 4
        generation[8] = 100
        generation[9] = 210
        generation[10] = 320
        generation[11] = 5
        generation[12] = 100
        generation[13] = 210
        generation[14] = mask_id
        active = torch.zeros((1, gen_length), dtype=torch.bool)
        active[0, 14] = True
        coord_maps = {
            "X": {100 + value: value for value in range(101)},
            "Y": {210 + value: value for value in range(101)},
            "Z": {320 + value: value for value in range(101)},
        }
        constraints = {
            "pbc_min_distance_mask": True,
            "pbc_min_distance_A": 0.5,
            "pbc_image_radius": 2,
            "max_atoms": 2,
            "coord_period": 100,
            "count_token_to_n": {1: 2},
            "length_token_to_bin": {
                "LA": {2: 10},
                "LB": {2: 10},
                "LC": {2: 10},
            },
            "length_step": 0.1,
            "angle_token_to_bin": {
                "AA": {3: 90},
                "AB": {3: 90},
                "AG": {3: 90},
            },
            "coord_token_to_bin": coord_maps,
            "coord_bin_to_token_id": {
                "Z": {value: 320 + value for value in range(101)}
            },
        }
        _apply_lightweight_decoding_masks(
            logits,
            x,
            prompt_length=prompt_length,
            gen_length=gen_length,
            constraints=constraints,
            active_generation_mask=active,
            mask_id=mask_id,
        )
        minimum = torch.finfo(torch.float32).min
        self.assertEqual(float(logits[0, prompt_length + 14, 320]), minimum)
        self.assertEqual(float(logits[0, prompt_length + 14, 369]), minimum)
        self.assertEqual(float(logits[0, prompt_length + 14, 370]), 0.0)
        self.assertEqual(float(logits[0, prompt_length + 14, 371]), minimum)
        self.assertEqual(float(logits[0, prompt_length + 14, 420]), minimum)


if __name__ == "__main__":
    unittest.main()
