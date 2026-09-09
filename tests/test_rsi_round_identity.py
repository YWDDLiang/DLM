import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from scripts.run_post_refine_cycle import file_hash,write_json,write_rows,validate_rsi_checkpoint
from scripts.prepare_rsi_round import prepare_round,prepare_training


class RoundIdentity(unittest.TestCase):
    def training_fixture(self,root):
        self.fixture(root)
        for branch in ('G','E'):
            checkpoint=root/'training/round2'/branch/'result/checkpoint'
            checkpoint.mkdir(parents=True)
            (checkpoint/'weights.bin').write_bytes(b'second updated weights')
            write_json(checkpoint/'RSI_TRAINING_DONE.json',{'optimizer_steps':8,
                'parameter_delta_squared':.02,'contract':{'branch':branch},
                'checkpoint_files':{'weights.bin':file_hash(checkpoint/'weights.bin')}})
            write_json(root/'training_configs'/f'TRAIN_{branch}1.json',
                       {'branch':branch,'seed':7,'source_round':0,'update_index':1})
            for index in (1,2):
                pairs=root/'rounds'/f'round{index}'/'fit/pairs'
                data=pairs/f'{branch}.jsonl'
                write_rows(data,[{'source_split':'train'}])
                write_json(pairs/f'PAIRS_{branch}_FINAL.json',
                           {'source_split':'train','files_sha256':{data.name:file_hash(data)}})

    def fixture(self,root):
        for branch in ('G','E'):
            checkpoint=root/'training/round1'/branch/'result/checkpoint'
            checkpoint.mkdir(parents=True)
            (checkpoint/'weights.bin').write_bytes(b'updated weights')
            write_json(checkpoint/'RSI_TRAINING_DONE.json',{'optimizer_steps':8,
                'parameter_delta_squared':.01,'contract':{'branch':branch},
                'checkpoint_files':{'weights.bin':file_hash(checkpoint/'weights.bin')}})
        for cohort,split in [(root,'evaluation'),(root/'fit','train')]:
            write_rows(cohort/'cohort/plans.jsonl',[{'source_split':split,'sample_idx':7,
                'body_noise_seed':19,'refiner_noise_seed':23,'body_prompt':'fixed prompt'}])
            write_json(cohort/'cohort/MANIFEST.json',{'requests':1})
            write_json(cohort/'RUN_SPEC.json',{'run_root':str(cohort),'run_id':split,'requests':1,
                'policy':{'max_edited_sites':4,'max_numeric_bin_delta':1},'assets':{}})

    def test_same_plan_bytes_and_updated_checkpoint_are_both_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            result=prepare_round(root,1)
            for cohort,original in [('MAIN',root),('FIT',root/'fit')]:
                self.assertEqual(Path(result['cohorts'][cohort]['plans']).read_bytes(),
                                 (original/'cohort/plans.jsonl').read_bytes())
            self.assertEqual(prepare_round(root,1),result)
            checkpoint=root/'training/round1/G/result/checkpoint'
            (checkpoint/'weights.bin').write_bytes(b'unregistered replacement')
            with self.assertRaisesRegex(ValueError,'checkpoint changed'): prepare_round(root,1)

    def test_zero_update_or_wrong_branch_cannot_count_as_a_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            checkpoint=root/'training/round1/G/result/checkpoint'
            with self.assertRaisesRegex(ValueError,'actual update'): validate_rsi_checkpoint(checkpoint,'E')
            path=checkpoint/'RSI_TRAINING_DONE.json';receipt=json.loads(path.read_text())
            receipt['parameter_delta_squared']=0.;write_json(path,receipt)
            with self.assertRaisesRegex(ValueError,'actual update'): prepare_round(root,1)

    def test_generation_can_start_before_editor_without_mutating_its_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            receipt=root/'training/round1/E/result/checkpoint/RSI_TRAINING_DONE.json'
            saved=receipt.read_bytes();receipt.unlink()
            early=prepare_round(root,1,generation_only=True)
            config=Path(early['cohorts']['MAIN']['config']);before=config.read_bytes()
            self.assertFalse(early['editing_ready'])
            self.assertFalse((root/'rounds/round1/ROUND_READY.json').exists())
            with self.assertRaises(FileNotFoundError): prepare_round(root,1)
            receipt.write_bytes(saved)
            final=prepare_round(root,1)
            self.assertTrue(final['editing_ready']);self.assertEqual(config.read_bytes(),before)
            edit=json.loads(Path(final['cohorts']['MAIN']['edit_config']).read_text())
            self.assertEqual(edit['updated_checkpoint_receipts']['E']['receipt_sha256'],file_hash(receipt))

    def test_later_training_round_metadata_matches_parent_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.training_fixture(root)
            for index in (2,3):
                for branch in ('G','E'):
                    with self.subTest(index=index,branch=branch):
                        result=prepare_training(root,index,branch)
                        config=json.loads(Path(result['config']).read_text())
                        self.assertEqual(config['source_round'],index-1)
                        self.assertEqual(config['update_index'],index)
                        self.assertEqual(Path(config['checkpoint']),
                            root/'training'/f'round{index-1}'/branch/'result/checkpoint')
                        self.assertNotIn('round_metadata_correction',result)

    def test_legacy_metadata_is_reported_without_rewriting_registered_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.training_fixture(root)
            result=prepare_training(root,2,'G');path=Path(result['config'])
            legacy=json.loads(path.read_text());legacy.update(source_round=0,update_index=1)
            write_json(path,legacy);before=path.read_bytes()
            result=prepare_training(root,2,'G')
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(result['round_metadata_correction']['effective'],
                             {'source_round':1,'update_index':2})
            self.assertEqual(result['sha256'],file_hash(path))
            self.assertEqual(prepare_training(root,2,'G'),result)

    def test_round_metadata_compatibility_does_not_accept_other_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.training_fixture(root)
            result=prepare_training(root,2,'E');path=Path(result['config'])
            tampered=json.loads(path.read_text());tampered['source_round']=99
            write_json(path,tampered)
            with self.assertRaisesRegex(ValueError,'existing training configuration changed'):
                prepare_training(root,2,'E')


if __name__=='__main__': unittest.main()
