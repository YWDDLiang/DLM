import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from concurrent.futures import Future
from unittest.mock import Mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'operations/r03_c3fd_main_20260907'))
spec=importlib.util.spec_from_file_location('ranked_coordinator_budget',ROOT/'operations/r03_c3fd_main_20260907/coordinate_ranked_rsi.py')
coordinator=importlib.util.module_from_spec(spec)
spec.loader.exec_module(coordinator)
from dispatch_rsi import allocation_cpus_per_gpu, action_source
from run_component import verify_deployed_source

fixture_spec=importlib.util.spec_from_file_location('ranked_reference_fixture',ROOT/'tests/test_r03_hull_union.py')
reference_fixture=importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(reference_fixture)


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


class PinnedTrialPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.physics=self.root/'old_physics';self.physics.mkdir()
        self.editor=self.root/'new_editor'
        (self.physics/'worker.py').write_text('original physics\n')
        (self.physics/'_CODE_READY').write_text('a'*40)
        (self.physics/'_SOURCE_FILES.json').write_text(json.dumps({'worker.py':hashlib.sha256((self.physics/'worker.py').read_bytes()).hexdigest()}))
        self.pin=self.root/'PHYSICS_SOURCE_PIN.json'
        self.pin.write_text(json.dumps({'source':str(self.physics),'identity':verify_deployed_source(self.physics)}))

    def test_only_trial_labeling_uses_verified_old_source(self):
        self.assertEqual(action_source(self.root,self.editor,'label',editor_trial=True),(self.physics,self.pin))
        for action in ('edit','train','score','generate','refine'):
            self.assertEqual(action_source(self.root,self.editor,action,editor_trial=True),(self.editor,None))

    def test_non_trial_and_mutated_physics_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'limited to'):
            action_source(self.root,self.editor,'label')
        (self.physics/'worker.py').write_text('changed physics\n')
        with self.assertRaisesRegex(ValueError,'immutable deployed source changed'):
            action_source(self.root,self.editor,'label',editor_trial=True)

    def test_wrong_pinned_identity_is_rejected(self):
        pin=json.loads(self.pin.read_text());pin['identity']['commit']='b'*40
        self.pin.write_text(json.dumps(pin))
        with self.assertRaisesRegex(ValueError,'pinned physics source identity changed'):
            action_source(self.root,self.editor,'label',editor_trial=True)


class ReferenceCoverageBeforeDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.cache=self.root/'references'
        plans=[dict(reference_fixture.planner_row(i,elements),source_split='train')
               for i,elements in enumerate([['Fe','O'],['F','Li']])]
        reference_fixture.write_rows(self.root/'fit/cohort/plans.jsonl',plans)
        reference_fixture.write_json(self.root/'fit/RUN_SPEC.json',
            {'requests':2,'assets':{'official_cache':str(self.cache)}})

    def test_old_complete_cache_cannot_launch_expanded_panel(self):
        reference_fixture.cache_fixture(self.cache,['Fe-O'])
        driver=coordinator.Coordinator.__new__(coordinator.Coordinator)
        driver.root=self.root;driver.job=Mock()
        with self.assertRaisesRegex(ValueError,'coverage incomplete for 1'):
            driver.run()
        driver.job.assert_not_called()
        self.assertFalse((self.root/'REFERENCE_COVERAGE.json').exists())

    def test_completed_reference_union_passes(self):
        reference_fixture.cache_fixture(self.cache,['Fe-O','F-Li'])
        report=coordinator.require_complete_reference_coverage(self.root)
        self.assertTrue(report['coverage_accounted'])
        self.assertEqual((report['requests'],report['chemical_systems'],report['resolved_systems']),(2,2,2))

    def test_verified_official_absence_remains_accounted_unknown(self):
        error={'type':'ContractError','http_status':None,'message':"missing unary references: ['Li']"}
        reference_fixture.cache_fixture(self.cache,['Fe-O'],{'F-Li':error})
        report=coordinator.require_complete_reference_coverage(self.root)
        self.assertEqual(report['officially_unresolved_systems'],['F-Li'])
        self.assertEqual(report['resolved_systems'],1)

    def test_transport_failure_does_not_satisfy_coverage(self):
        error={'type':'ReadTimeout','http_status':None,'message':'transport timed out'}
        reference_fixture.cache_fixture(self.cache,['Fe-O'],{'F-Li':error})
        with self.assertRaisesRegex(ValueError,'coverage incomplete'):
            coordinator.require_complete_reference_coverage(self.root)

    def write_extension(self):
        target=reference_fixture.cache_fixture(self.root/'extended_references',['Fe-O','F-Li'])
        value={'schema':'ranked_reference_coverage_completion_v1','directory':str(target),
            'cache_manifest_sha256':coordinator.file_hash(target/'completion_manifest.json'),
            'plans_sha256':coordinator.file_hash(self.root/'fit/cohort/plans.jsonl'),
            'database_version':'2026.04.13'}
        reference_fixture.write_json(self.root/'REFERENCE_CACHE_OVERRIDE.json',value)
        return target,value

    def test_extension_keeps_original_cache_and_routes_score_only(self):
        reference_fixture.cache_fixture(self.cache,['Fe-O'])
        old_hash=coordinator.file_hash(self.cache/'completion_manifest.json')
        target,_=self.write_extension()
        report=coordinator.require_complete_reference_coverage(self.root)
        self.assertEqual(report['resolved_systems'],2)
        self.assertEqual(coordinator.score_reference_arguments(self.root,()),('--cache',str(target)))
        self.assertEqual(coordinator.file_hash(self.cache/'completion_manifest.json'),old_hash)
        with self.assertRaisesRegex(ValueError,'cannot override'):
            coordinator.score_reference_arguments(self.root,('--cache',str(self.cache)))

    def test_extension_must_match_its_manifest_and_database_version(self):
        target,value=self.write_extension()
        value['cache_manifest_sha256']='0'*64
        reference_fixture.write_json(self.root/'REFERENCE_CACHE_OVERRIDE.json',value)
        with self.assertRaisesRegex(ValueError,'identity changed'):
            coordinator.require_complete_reference_coverage(self.root)
        value['cache_manifest_sha256']=coordinator.file_hash(target/'completion_manifest.json')
        value['database_version']='different-version'
        reference_fixture.write_json(self.root/'REFERENCE_CACHE_OVERRIDE.json',value)
        with self.assertRaisesRegex(ValueError,'database version changed'):
            coordinator.require_complete_reference_coverage(self.root)


class S0ResourceAllocationTests(unittest.TestCase):
    def test_temporary_comparator_allocation_preserves_training_and_expires(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            control={'execution_policy':{'single_GPUs':6,'parallel_main_GPUs':3,
                'parallel_other_GPUs':3,'training_GPUs':6,'training_batch_size':32}}
            (root/'RUN_SPEC.json').write_text(json.dumps(control))
            (root/'S0_RESOURCE_OVERRIDE.json').write_text(json.dumps(
                {'schema':'ranked_S0_resource_allocation_v1','parallel_other_GPUs':2}))
            driver=coordinator.Coordinator.__new__(coordinator.Coordinator)
            driver.root=root;driver.budget={'max_GPUs':6}
            values=driver.allocations()
            self.assertEqual(values['parallel_other_GPUs'],2)
            self.assertEqual((values['single_GPUs'],values['training_GPUs'],values['training_batch_size']),(6,6,32))
            self.assertEqual(json.loads((root/'RUN_SPEC.json').read_text()),control)
            (root/'S0_COMPLETE.json').write_text('{}')
            self.assertEqual(driver.allocations()['parallel_other_GPUs'],3)

    def test_temporary_allocation_still_enforces_shared_gpu_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'RUN_SPEC.json').write_text(json.dumps({'execution_policy':{'parallel_main_GPUs':3}}))
            (root/'S0_RESOURCE_OVERRIDE.json').write_text(json.dumps(
                {'schema':'ranked_S0_resource_allocation_v1','parallel_other_GPUs':4}))
            driver=coordinator.Coordinator.__new__(coordinator.Coordinator)
            driver.root=root;driver.budget={'max_GPUs':6}
            with self.assertRaisesRegex(ValueError,'exceed'):
                driver.allocations()


if __name__=='__main__':unittest.main()
