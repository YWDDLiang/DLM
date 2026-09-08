import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('archive_experiment', Path(__file__).resolve().parents[1]/'operations/archive_experiment.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class ExperimentArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='experiment-archive-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)/'run'
        self.output = Path(self.temporary.name)/'archive'
        for name in ['failed','retained']:
            directory = self.root/name/'train/checkpoint'
            directory.mkdir(parents=True)
            for filename in archive.WEIGHTS:
                (directory/filename).write_bytes((name+filename).encode())
            for filename in ['EXPERT_EDITOR.json','adapter_config.json','tokenizer.json']:
                (directory/filename).write_text('{}')
            (directory/'roundtrip_probe.pt').write_bytes(b'reload verification data')
        (self.root/'generated').mkdir()
        (self.root/'generated/outputs.pt').write_bytes(b'generated structures and trajectories')
        self.keep = ['retained/train/checkpoint']

    def make_archive(self):
        return archive.archive(self.root, self.output, self.keep, check_idle=False)

    def test_prunes_only_failed_weights_after_verified_data_archive(self):
        manifest, receipt = self.make_archive()
        self.assertTrue(receipt['complete'])
        self.assertIn('generated/outputs.pt', {x['path'] for x in manifest['data']})
        result = archive.prune(self.output, check_idle=False)
        self.assertEqual(result['cleaned_checkpoints'], 1)
        self.assertEqual(result['removed_weight_files'], 4)
        self.assertFalse((self.root/'failed/train/checkpoint/adapter_model.safetensors').exists())
        self.assertTrue((self.root/'retained/train/checkpoint/adapter_model.safetensors').exists())
        self.assertTrue((self.root/'failed/train/checkpoint/roundtrip_probe.pt').exists())
        self.assertEqual((self.root/'generated/outputs.pt').read_bytes(), b'generated structures and trajectories')

    def test_changed_data_blocks_deletion(self):
        self.make_archive()
        (self.root/'generated/outputs.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'retained data changed'):
            archive.prune(self.output, check_idle=False)
        self.assertTrue((self.root/'failed/train/checkpoint/adapter_model.safetensors').exists())

    def test_unarchived_new_data_blocks_deletion(self):
        self.make_archive()
        (self.root/'generated/new.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'inventory changed'):
            archive.prune(self.output, check_idle=False)

    def test_archive_tampering_blocks_deletion(self):
        self.make_archive()
        with (self.output/'experiment_data.tar.gz').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'archive binding changed'):
            archive.prune(self.output, check_idle=False)

    def test_unreadable_subtree_cannot_silently_pass_inventory(self):
        def unreadable(*args, **kwargs):
            kwargs['onerror'](PermissionError('unreadable source subtree'))
            return iter(())
        with mock.patch.object(archive.os, 'walk', side_effect=unreadable):
            with self.assertRaisesRegex(PermissionError, 'unreadable source'):
                self.make_archive()
            with self.assertRaisesRegex(PermissionError, 'unreadable source'):
                archive.check_inventory(self.root, [])
        self.assertFalse((self.output/'ARCHIVE_VERIFIED.json').exists())
        self.assertTrue((self.root/'failed/train/checkpoint/adapter_model.safetensors').exists())

    def test_paths_and_active_jobs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unsafe relative'):
            archive.inside(self.root.resolve(), '../outside')
        with self.assertRaisesRegex(ValueError, 'separate'):
            archive.archive(self.root, self.root/'archive', self.keep, check_idle=False)
        with mock.patch.object(archive.subprocess, 'check_output', side_effect=['123\n',f'JobId=123 WorkDir={self.root} Command=/source/run.py\n']), mock.patch.dict(archive.os.environ, {'SLURM_JOB_ID':'456'}):
            with self.assertRaisesRegex(ValueError, 'active or pending'):
                archive.require_idle(self.root)

    def test_jobs_bound_only_through_output_paths_block_cleanup(self):
        for field in ['StdOut','StdErr']:
            details = f'JobId=123 WorkDir=/source Command=/source/launch.sh {field}={self.root}/logs/run.out\n'
            with self.subTest(field=field), mock.patch.object(archive.subprocess, 'check_output', side_effect=['123\n',details]), mock.patch.dict(archive.os.environ, {'SLURM_JOB_ID':'456'}):
                with self.assertRaisesRegex(ValueError, 'active or pending'):
                    archive.require_idle(self.root)


if __name__ == '__main__':
    unittest.main()
