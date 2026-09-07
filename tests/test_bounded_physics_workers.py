"""A real process timeout must free a stuck slot without changing other labels."""
import importlib.util
from pathlib import Path
import time
from types import SimpleNamespace
import unittest


def fake_worker(connection, gpu_index, options):
    connection.send({'ready': True})
    while True:
        record = connection.recv()
        if record is None:
            return
        if record.get('hang'):
            time.sleep(30)
        time.sleep(.03)
        connection.send({'result': {'status': 'verified', 'verified': True, 'value': record['value']}})


class BoundedWorkerTests(unittest.TestCase):
    def test_timeout_is_unknown_and_later_queued_records_keep_their_own_budget(self):
        path = Path(__file__).resolve().parents[1]/'scripts/label_programmed_paths.py'
        spec = importlib.util.spec_from_file_location('physics_timeout_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = SimpleNamespace(gpu_count=1, workers_per_gpu=1, fmax=.1, stress_tolerance=.5,
                               max_steps=500, record_timeout=.15, worker_startup_timeout=10.)
        records = {'hung': [{'hang': True, 'value': -1}],
                   'good1': [{'value': 10}], 'good2': [{'value': 11}]}
        started = time.monotonic()
        result = {key: row for key, _, row in module.bounded_labels(records, args, worker_target=fake_worker)}
        self.assertLess(time.monotonic()-started, 12)
        self.assertEqual(set(result), set(records))
        self.assertEqual(result['hung']['status'], 'worker_error')
        self.assertIsNone(result['hung']['terminal_energy'])
        self.assertEqual(result['good1']['value'], 10)
        self.assertEqual(result['good2']['value'], 11)


if __name__ == '__main__':
    unittest.main()
