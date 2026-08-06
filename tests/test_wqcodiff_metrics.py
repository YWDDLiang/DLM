from __future__ import annotations

import collections
import importlib.util
import time
import unittest
from unittest import mock


PYMATGEN_AVAILABLE = importlib.util.find_spec("pymatgen") is not None


class _FakeComposition:
    reduced_formula = "Li2O"


class _FakeStructure:
    composition = _FakeComposition()


class _SlowMatcher:
    def fit(self, first, second):
        del first, second
        time.sleep(1.0)
        return False


class BoundedMatcherPolicyTests(unittest.TestCase):
    @mock.patch("crystal_dlm.wqcodiff.metrics.MATCHER_FIT_TIMEOUT_SECONDS", 0.02)
    @mock.patch("crystal_dlm.wqcodiff.metrics.structure_matcher", return_value=_SlowMatcher())
    def test_duplicate_timeout_is_conservatively_nonunique(self, matcher) -> None:
        del matcher
        from crystal_dlm.wqcodiff.metrics import duplicate_components

        diagnostics = [collections.Counter(), collections.Counter()]
        clusters, unique = duplicate_components(
            ("a", "b"),
            (_FakeStructure(), _FakeStructure()),
            sensitivity="standard",
            diagnostics=diagnostics,
        )
        self.assertEqual(clusters[0], clusters[1])
        self.assertEqual(unique, (False, False))
        self.assertEqual(diagnostics[0]["duplicate_standard_timeout"], 1)
        self.assertEqual(diagnostics[1]["duplicate_standard_timeout"], 1)

    @mock.patch("crystal_dlm.wqcodiff.metrics.MATCHER_FIT_TIMEOUT_SECONDS", 0.02)
    @mock.patch("crystal_dlm.wqcodiff.metrics.structure_matcher", return_value=_SlowMatcher())
    def test_novelty_timeout_is_conservatively_non_novel(self, matcher) -> None:
        del matcher
        from crystal_dlm.wqcodiff.metrics import full_structure_novelty

        diagnostics = [collections.Counter()]
        result = full_structure_novelty(
            (_FakeStructure(),),
            (_FakeStructure(),),
            sensitivity="standard",
            diagnostics=diagnostics,
        )
        self.assertEqual(result, (False,))
        self.assertEqual(diagnostics[0]["full_novelty_standard_timeout"], 1)


@unittest.skipUnless(PYMATGEN_AVAILABLE, "pymatgen matcher path is tested locally/server")
class RelationalMetricTests(unittest.TestCase):
    def _structure(self, species):
        from pymatgen.core import Lattice, Structure

        return Structure(
            Lattice.cubic(5.0),
            species,
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        )

    def test_duplicate_component_uniqueness_marks_every_duplicate_nonunique(self) -> None:
        from crystal_dlm.wqcodiff.metrics import duplicate_components

        first = self._structure(["Li", "Li", "O"])
        second = first.copy()
        third = self._structure(["Na", "Na", "O"])
        clusters, unique = duplicate_components(
            ["a", "b", "c"], [first, second, third], sensitivity="standard"
        )
        self.assertEqual(clusters[0], clusters[1])
        self.assertEqual(unique, (False, False, True))

    def test_full_novelty_is_species_aware(self) -> None:
        from crystal_dlm.wqcodiff.metrics import full_structure_novelty

        train = self._structure(["Li", "Li", "O"])
        same = train.copy()
        substitution = self._structure(["Na", "Na", "O"])
        self.assertEqual(
            full_structure_novelty(
                [same, substitution], [train], sensitivity="standard"
            ),
            (False, True),
        )


if __name__ == "__main__":
    unittest.main()
