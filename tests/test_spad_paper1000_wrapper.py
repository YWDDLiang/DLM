from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "218_spad_basin_paper1000_generation_refine.sbatch"


class Paper1000WrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = WRAPPER.read_text(encoding="utf-8")

    def test_parser_only_selection_precedes_refinement(self) -> None:
        select = self.text.index("select_first_parseable_spad_body.py")
        refine = self.text.index("--proposal-graphs")
        self.assertLess(select, refine)
        self.assertIn('"${RUN}/body_valid1000/proposal_graphs.pt"', self.text)

    def test_fixed_paper_protocol(self) -> None:
        self.assertIn("SPAD_PAPER_SOURCE_REQUESTED:-1200", self.text)
        self.assertIn("SPAD_PAPER_VALID_TARGET:-1000", self.text)
        self.assertIn("SPAD_PLANNER_SEED:-25", self.text)
        self.assertIn("SPAD_EVAL_STREAM:-20", self.text)
        self.assertIn("SPAD_DLM_SEED:-94117", self.text)
        self.assertIn("SPAD_REFINER_SEED:-104117", self.text)
        self.assertIn('[[ "${SELECTED_TAU}" -eq 800 ]]', self.text)
        self.assertIn("SPAD_PAPER1000_AUTHORIZATION_MARKER", self.text)
        self.assertIn("preregistered_near_miss_counts", self.text)

    def test_no_outcome_selection_surface(self) -> None:
        self.assertIn("parser_only_before_outcomes", self.text)
        self.assertIn("outcome_rerank_replacement\\tfalse", self.text)
        self.assertNotIn("best-of", self.text.lower())
        self.assertNotIn("official", self.text.lower())

    def test_resource_contract_is_dynamic_up_to_four(self) -> None:
        self.assertIn('[[ "${WORLD_SIZE}" =~ ^[1-4]$ ]]', self.text)
        self.assertIn('"--nproc_per_node=${WORLD_SIZE}"', self.text)


if __name__ == "__main__":
    unittest.main()
