"""Resource, provenance and recovery contracts for manifest-driven execution."""
from contextlib import redirect_stdout
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


OPS = Path(__file__).resolve().parents[1] / 'operations/r03_c3fd_main_20260907'
sys.path.insert(0, str(OPS))
import run_component as runner
import submit_stage as submitter


class PipelineContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.root, self.source = base / 'run', base / 'run/code' / ('a' * 40)
        self.source.mkdir(parents=True)
        self.script = self.source / 'src/task.py'
        self.script.parent.mkdir()
        self.script.write_text('print("fixture")\n')
        (self.source / '_CODE_READY').write_text('a' * 40)
        manifest = {'src/task.py': hashlib.sha256(self.script.read_bytes()).hexdigest()}
        (self.source / '_SOURCE_FILES.json').write_text(json.dumps(manifest))
        now = dt.datetime.now(dt.timezone.utc)
        self.spec = {
            'schema': 'crystal_pipeline_run_v1', 'run_root': str(self.root),
            'source_root': str(self.source), 'source_identity': runner.verify_deployed_source(self.source),
            'purpose': 'train', 'python': sys.executable,
            'resources': {'extra_gpu_until_utc': (now + dt.timedelta(hours=4)).isoformat(),
                          'deadline_utc': (now + dt.timedelta(hours=10)).isoformat(),
                          'gpus_before_extra_window': 7, 'gpus_after_extra_window': 6},
            'environment': {'OMP_NUM_THREADS': '1'},
            'components': [{'id': 'shard0', 'output_dir': 'shard0', 'gpus': 1,
                            'stages': [{'name': 'task', 'script': 'src/task.py', 'args': [],
                                        'outputs': ['{output}/result.json']}]}],
            'jobs': {'collect': {'component_indices': [0], 'gpus_per_task': 1,
                                 'cpus_per_task': 6, 'parallel_tasks': 1, 'wall_minutes': 10}},
        }
        self.config = self.root / 'config.json'
        self.save()

    def save(self):
        self.config.write_text(json.dumps(self.spec))

    def dispatch(self, mock_run=None):
        result = subprocess.CompletedProcess([], 0, '12345\n', '')
        with patch.dict(os.environ, {'USER': 'fixture'}), \
                patch.object(submitter.sp, 'check_output', return_value='') as queue, \
                patch.object(submitter.sp, 'run', return_value=result) as call, redirect_stdout(io.StringIO()):
            submitter.configured_dispatch(['--config', str(self.config), '--job', 'collect'])
            return queue, call

    def test_dispatch_freezes_manifest_and_enforces_absolute_cutoff(self):
        self.spec['components'] = [dict(self.spec['components'][0], id=f'shard{i}',
                                       output_dir=f'shard{i}') for i in range(8)]
        self.spec['jobs']['collect'].update(component_indices=list(range(8)), parallel_tasks=7)
        self.save()
        _, call = self.dispatch()
        self.assertEqual(call.call_count, 1)
        receipt = json.loads((self.root / 'submissions/collect.json').read_text())
        self.assertEqual(receipt['absolute_stop_utc'], self.spec['resources']['extra_gpu_until_utc'])
        snapshot = Path(receipt['manifest'])
        self.assertNotEqual(snapshot, self.config)
        self.spec['purpose'] = 'evaluation'
        self.save()
        self.assertEqual(json.loads(snapshot.read_text())['purpose'], 'train')
        script = (self.root / 'submissions/collect.sbatch').read_text()
        self.assertIn('timeout --signal=TERM --kill-after=30s', script)
        self.assertIn('--config-sha256 ' + receipt['manifest_sha256'], script)

    def test_successful_retry_returns_before_queue_or_new_capacity_check(self):
        self.dispatch()
        queue, call = self.dispatch()
        queue.assert_not_called()
        call.assert_not_called()
        self.spec['purpose'] = 'evaluation'
        self.save()
        with self.assertRaisesRegex(ValueError, 'different frozen configuration'):
            self.dispatch()

    def test_global_dispatch_and_uncertain_reservations_block_new_jobs(self):
        guard = self.root / '.dispatch_guard.lock'
        guard.touch()
        with self.assertRaises(FileExistsError):
            self.dispatch()
        guard.unlink()
        records = self.root / 'submissions'
        records.mkdir()
        (records / 'uncertain.lock').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'uncertain dispatch'):
            self.dispatch()

    def test_invalid_or_overlapping_shards_fail_without_submission(self):
        self.spec['jobs']['collect']['component_indices'] = [-1]
        self.save()
        with self.assertRaisesRegex(ValueError, 'distinct component indices'):
            self.dispatch()
        self.spec['components'].append(dict(self.spec['components'][0], id='other', output_dir='shard0/nested'))
        self.spec['jobs']['collect']['component_indices'] = [0, 1]
        self.save()
        with self.assertRaisesRegex(ValueError, 'must not overlap'):
            self.dispatch()

    def run_component(self, *, execute=True):
        self.spec['job_runtime'] = {
            'hard_stop_utc': self.spec['resources']['deadline_utc'], 'cpus_per_task': 6}
        self.save()
        digest = hashlib.sha256(self.config.read_bytes()).hexdigest()
        def child(*args, **kwargs):
            (self.root / 'shard0/result.json').write_text('{"complete":true}')
            return subprocess.CompletedProcess([], 0)
        with patch.object(runner, 'SOURCE', self.source), \
                patch.dict(os.environ, {'SLURM_JOB_ID': '12345', 'CUDA_VISIBLE_DEVICES': '0',
                                        'SLURM_CPUS_PER_TASK': '6'}), \
                patch.object(runner.sp, 'run', side_effect=child) as call, redirect_stdout(io.StringIO()):
            runner.configured_component(['--config', str(self.config), '--config-sha256', digest])
            return call

    def test_stage_reuse_binds_source_purpose_and_actual_inputs(self):
        source_input = self.root / 'input.json'
        source_input.write_text('{"value":1}')
        self.spec['components'][0]['stages'][0]['inputs'] = [str(source_input)]
        self.assertEqual(self.run_component().call_count, 1)
        self.run_component().assert_not_called()
        source_input.write_text('{"value":2}')
        with self.assertRaisesRegex(ValueError, 'new component directory'):
            self.run_component()
        self.spec['purpose'] = 'evaluation'
        with self.assertRaisesRegex(ValueError, 'different configuration'):
            self.run_component()

    def test_source_mutation_and_allocation_overrides_fail(self):
        self.script.write_text('print("changed")\n')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            self.run_component()
        self.script.write_text('print("fixture")\n')
        self.spec['environment']['CUDA_VISIBLE_DEVICES'] = '0,1'
        with self.assertRaisesRegex(ValueError, 'must not override allocation'):
            self.run_component()

    def test_delayed_start_cannot_enter_expired_execution_window(self):
        self.spec['resources']['deadline_utc'] = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)).isoformat()
        with self.assertRaisesRegex(ValueError, 'absolute resource/deadline'):
            self.run_component()

    def test_tampered_submitted_manifest_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'manifest content changed'):
            runner.configured_component(['--config', str(self.config), '--config-sha256', '0' * 64])


if __name__ == '__main__':
    unittest.main()
