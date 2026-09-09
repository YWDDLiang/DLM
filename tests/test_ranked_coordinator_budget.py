import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'operations/r03_c3fd_main_20260907'))
spec=importlib.util.spec_from_file_location('ranked_coordinator_budget',ROOT/'operations/r03_c3fd_main_20260907/coordinate_ranked_rsi.py')
coordinator=importlib.util.module_from_spec(spec)
spec.loader.exec_module(coordinator)


class RankedCoordinatorBudgetTests(unittest.TestCase):
    def test_short_stage_can_dispatch_with_twenty_minutes_remaining(self):
        minutes=coordinator.dispatch_minutes(90,1200)
        self.assertGreater(minutes,0)
        self.assertLessEqual(minutes*60+90,1200)

    def test_requested_allocation_is_preserved_when_budget_allows(self):
        self.assertEqual(coordinator.dispatch_minutes(35,5400),35)

    def test_near_deadline_does_not_launch_new_work(self):
        with self.assertRaises(RuntimeError):coordinator.dispatch_minutes(90,179)

    def test_final_editor_does_not_reserve_an_already_completed_body(self):
        reserve=coordinator.training_reserve_seconds(3,'E')
        self.assertEqual(reserve,35*60)
        self.assertLess(reserve,coordinator.training_reserve_seconds(3,'G'))


if __name__=='__main__':unittest.main()
