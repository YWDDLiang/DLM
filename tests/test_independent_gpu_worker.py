import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('gpu_worker', Path(__file__).resolve().parents[1] / 'src/scripts/run_independent_gpu_worker.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DeviceBindingTests(unittest.TestCase):
    def test_all_loaders_see_only_the_assigned_gpu(self):
        visible = []
        for rank in range(12):
            env = {'CUDA_VISIBLE_DEVICES': '2,3,5,7', 'LOCAL_RANK': str(rank), 'RANK': str(rank)}
            visible.append(module.bind_device(env))
            self.assertEqual(env['RANK'], str(rank))
            self.assertNotIn(',', env['CUDA_VISIBLE_DEVICES'])
        self.assertEqual(visible, ['2', '3', '5', '7'] * 3)

    def test_refuses_to_guess_devices_outside_allocation(self):
        with self.assertRaises(ValueError):
            module.bind_device({'LOCAL_RANK': '0'})
