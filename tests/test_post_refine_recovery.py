import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from crystal_dlm import r03_geometry_bridge as bridge
from crystal_dlm.post_refine_contract import recovery_canvas

binding = importlib.util.spec_from_file_location('_recovery_constructor_test', ROOT / 'src/scripts/run_r03_integrated_body.py')
native = importlib.util.module_from_spec(binding)
sys.modules[binding.name] = native
binding.loader.exec_module(native)


class Tokenizer:
    def __call__(self, *_args, **_kwargs):
        return {'input_ids': torch.tensor([[1, 1]])}


class RecoveryWorkflow(unittest.TestCase):
    def setUp(self):
        self.task = {'plan_state': {'N': 2}, 'body_noise_seed': 99, 'body_prompt': 'p'}
        self.partial = [2, 40, 40, 40, 90, 90, 90, 14, 1, 2, 3, 14, 7, 8, 126336]
        self.complete = torch.tensor([[2, 40, 40, 40, 90, 90, 90, 14, 1, 2, 3, 14, 20, 30, 40]])

    def error(self):
        return bridge.GeometryNoLegalSupport({'reason': 'no_legal_Z_completion'},
                                              torch.tensor([[1, 1, *self.partial]]), 2)

    def invoke(self, check):
        return native.construct_with_recovery(None, Tokenizer(), self.task, SimpleNamespace(),
            constraints={}, geometry_api=bridge, complete_geometry=check,
            recovery_transform=recovery_canvas, recovery_seed=123, max_recoveries=1)

    def test_incomplete_site_recovery_preserves_complete_prefix_and_changes_noise(self):
        with patch.object(native, 'construct_batch', side_effect=[self.error(), (self.complete, {})]) as sample:
            result, report = self.invoke(lambda _: {'supported': True})
        self.assertTrue(torch.equal(result, self.complete))
        self.assertEqual(sample.call_count, 2)
        second = sample.call_args_list[1].kwargs
        self.assertEqual(second['initial_body'][:11], self.partial[:11])
        self.assertEqual(second['initial_body'][12:], [126336] * 3)
        self.assertEqual(second['noise_seed_override'], 123)
        self.assertEqual(report['construction_recovery']['recoveries_used'], 1)

    def test_self_image_failure_reopens_cell_and_all_coordinates(self):
        checks = iter([{'supported': False, 'reason': 'periodic_self_image_below_0.5A'}, {'supported': True}])
        with patch.object(native, 'construct_batch', side_effect=[(self.complete, {}), (self.complete, {})]) as sample:
            self.invoke(lambda _: next(checks))
        recovered = sample.call_args_list[1].kwargs['initial_body']
        self.assertEqual([recovered[i] for i in [0, 7, 11]], [2, 14, 14])
        self.assertTrue(all(recovered[i] == 126336 for i in [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14]))

    def test_repeated_failure_does_not_start_a_third_candidate(self):
        with patch.object(native, 'construct_batch', side_effect=[self.error(), self.error()]) as sample:
            with self.assertRaises(bridge.GeometryNoLegalSupport) as observed:
                self.invoke(lambda _: {'supported': True})
        self.assertEqual(sample.call_count, 2)
        self.assertEqual(observed.exception.details['construction_recoveries_used'], 1)
        self.assertEqual(len(observed.exception.details['recovery_episodes']), 1)


if __name__ == '__main__':
    unittest.main()
