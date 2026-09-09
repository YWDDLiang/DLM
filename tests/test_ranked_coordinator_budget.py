import importlib.util
from pathlib import Path
import sys
import unittest
from concurrent.futures import Future

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'operations/r03_c3fd_main_20260907'))
spec=importlib.util.spec_from_file_location('ranked_coordinator_budget',ROOT/'operations/r03_c3fd_main_20260907/coordinate_ranked_rsi.py')
coordinator=importlib.util.module_from_spec(spec)
spec.loader.exec_module(coordinator)
from dispatch_rsi import allocation_cpus_per_gpu


class RankedCoordinatorBudgetTests(unittest.TestCase):
    def test_collection_labels_keep_shared_allocation_while_teacher_runs(self):
        self.assertEqual(coordinator.collection_label_allocation(Future(),2,4),2)

    def test_collection_labels_use_single_allocation_after_teacher_finishes(self):
        teacher=Future();teacher.set_result(None)
        self.assertEqual(coordinator.collection_label_allocation(teacher,2,4),4)

    def test_teacher_failure_stops_new_collection_label_work(self):
        teacher=Future();teacher.set_exception(RuntimeError('teacher failed'))
        with self.assertRaisesRegex(RuntimeError,'teacher failed'):
            coordinator.collection_label_allocation(teacher,2,4)

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

    def test_thousand_request_reservation_scales_work_but_not_startup(self):
        reserve=coordinator.training_reserve_seconds(1,'G',1000,6)
        self.assertAlmostEqual(reserve,(3*(20+45*(1000/256)*(4/6))+15)*60)
        self.assertLess(coordinator.training_reserve_seconds(3,'E',1000,6),reserve)

    def test_three_card_allocation_preserves_large_effective_batch(self):
        values=coordinator.runtime_allocations(dict(single_GPUs=3,training_GPUs=3,
            parallel_main_GPUs=2,parallel_other_GPUs=1,training_batch_size=43))
        self.assertEqual(values['training_GPUs']*values['training_batch_size'],129)
        self.assertEqual(values['parallel_main_GPUs']+values['parallel_other_GPUs'],3)

    def test_parallel_resource_limit_is_enforced(self):
        with self.assertRaises(ValueError):coordinator.runtime_allocations(dict(parallel_other_GPUs=3))

    def test_zero_gpu_stage_is_rejected(self):
        with self.assertRaises(ValueError):coordinator.runtime_allocations(dict(single_GPUs=0))

    def test_three_generation_workers_fit_four_reserved_cpus(self):
        self.assertEqual(allocation_cpus_per_gpu('generate',3,cap=4),4)
        self.assertEqual(allocation_cpus_per_gpu('generate',3),6)

    def test_label_cpu_cap_does_not_require_eight_cores_per_gpu(self):
        self.assertEqual(allocation_cpus_per_gpu('label',1,label_workers=4,cap=4),4)

    def test_zero_cpu_cap_is_rejected(self):
        with self.assertRaises(ValueError):allocation_cpus_per_gpu('generate',3,cap=0)


if __name__=='__main__':unittest.main()
