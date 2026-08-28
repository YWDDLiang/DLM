from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.ccfd import FormulaToken
from crystal_dlm.composition_pair_prior import CompositionPairPrior, ValenceNode


class CompositionPairPriorTest(unittest.TestCase):
    def token(self, element, oxidation):
        return FormulaToken.from_symbol(element, oxidation, 1)

    def test_seen_chemistry_pair_scores_above_unseen_known_pair(self):
        na = self.token("Na", 1)
        k = self.token("K", 1)
        chloride = self.token("Cl", -1)
        fluoride = self.token("F", -1)
        prior = CompositionPairPrior.fit(
            [
                (na, chloride),
                (na, chloride),
                (na, fluoride),
                (k, chloride),
            ],
            alpha=0.5,
        )
        self.assertGreater(
            prior.context_score(chloride, (na,)),
            prior.context_score(k, (na,)),
        )

    def test_counts_each_node_once_per_composition(self):
        fe2 = self.token("Fe", 2)
        oxygen = self.token("O", -2)
        prior = CompositionPairPrior.fit([(fe2, fe2, oxygen)])
        self.assertEqual(prior.composition_count, 1)
        self.assertEqual(prior.node_counts[ValenceNode.from_token(fe2)], 1)
        self.assertEqual(len(prior.pair_counts), 1)

    def test_prior_is_soft_metadata_not_a_legality_decision(self):
        prior = CompositionPairPrior.fit(
            [(self.token("Na", 1), self.token("Cl", -1))]
        )
        payload = prior.to_dict()
        self.assertIn("never a legality mask", payload["semantics"])
        self.assertEqual(payload["composition_count"], 1)
        rebuilt = CompositionPairPrior.from_dict(payload)
        self.assertEqual(rebuilt.node_counts, prior.node_counts)
        self.assertEqual(rebuilt.pair_counts, prior.pair_counts)


if __name__ == "__main__":
    unittest.main()
