import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from scripts.run_post_refine_cycle import file_hash,write_json,write_rows,validate_rsi_checkpoint
from scripts.prepare_rsi_round import prepare_round


class RoundIdentity(unittest.TestCase):
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


if __name__=='__main__': unittest.main()
