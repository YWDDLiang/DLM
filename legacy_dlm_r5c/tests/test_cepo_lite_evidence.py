import unittest

from scripts.build_cepo_lite_evidence import credit_positions, token_weights


class CepoLiteEvidenceTest(unittest.TestCase):
    def test_credit_positions_focus_on_count_and_active_elements(self):
        tokens = ["<N_002>"] + ["<LA_010>"] * 6
        tokens.extend(["<S00>", "<E_Li>", "<X_000>", "<Y_000>", "<Z_000>"])
        tokens.extend(["<S01>", "<E_O>", "<X_050>", "<Y_050>", "<Z_050>"])
        tokens.extend(["<S02>", "<EMPTY>", "<X_PAD>", "<Y_PAD>", "<Z_PAD>"])
        self.assertIn(0, credit_positions(tokens))
        self.assertIn(8, credit_positions(tokens))
        self.assertIn(13, credit_positions(tokens))
        self.assertIn(18, credit_positions(tokens))
        self.assertNotIn(9, credit_positions(tokens))

    def test_token_weights_contrast_positive_and_negative(self):
        row = {"response": "<N_002><LA_010><LB_010><LC_010><AA_090><AB_090><AG_090><S00><E_Li><X_000><Y_000><Z_000><S01><E_O><X_050><Y_050><Z_050>"}
        pos = dict(row)
        neg = {"response": row["response"].replace("<E_O>", "<E_Li>")}
        weights = token_weights(row, pos, neg, lambda_weight=0.3, clip_eps=0.2)
        self.assertAlmostEqual(weights[13], 1.2)


if __name__ == "__main__":
    unittest.main()
