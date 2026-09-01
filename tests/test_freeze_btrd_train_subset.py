import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_btrd_train_subset", ROOT / "scripts/freeze_btrd_train_subset.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import BTRD freezer")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(source_idx, elements, counts):
    return {
        "source_row_idx": source_idx,
        "answer": "body",
        "answer_sha256": f"sha{source_idx}",
        "plan_state": {
            "N": sum(counts),
            "elements": elements,
            "counts": counts,
        },
    }


class BTRDSubsetTest(unittest.TestCase):
    def test_hash_selection_excludes_evaluation_identity_and_freezes_modes(self) -> None:
        train = [
            row(0, ["Na", "Cl"], [1, 1]),
            row(1, ["Li", "F"], [1, 1]),
            row(2, ["K", "Br"], [1, 1]),
            row(3, ["Mg", "O"], [1, 1]),
        ]
        evaluation = [row(10, ["Na", "Cl"], [1, 1])]
        selected, audit = MODULE.freeze_rows(
            train,
            evaluation,
            count=3,
            teacher_count=2,
            selection_seed=7,
        )
        self.assertEqual(len(selected), 3)
        self.assertNotIn(0, {item["btrd_source_row_idx"] for item in selected})
        self.assertEqual(
            [item["btrd_target_mode"] for item in selected].count("model494_tau200"),
            2,
        )
        self.assertEqual(
            [item["btrd_target_mode"] for item in selected].count("mp20_anchor"),
            1,
        )
        self.assertEqual(audit["excluded_exact_overlap"], 1)
        self.assertTrue(all(item["btrd_selection_outcomes_read"] is False for item in selected))

    def test_selection_is_deterministic(self) -> None:
        train = [row(index, ["Li", "F"], [1, index + 1]) for index in range(6)]
        first, _ = MODULE.freeze_rows(
            train, [], count=4, teacher_count=3, selection_seed=11
        )
        second, _ = MODULE.freeze_rows(
            train, [], count=4, teacher_count=3, selection_seed=11
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
