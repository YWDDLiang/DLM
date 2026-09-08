import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from scripts.run_post_refine_cycle import write_json,write_rows,file_hash
from scripts.run_rsi_stages import register_refine_resume,validate_refine_resume


class RefinementResume(unittest.TestCase):
    def fixture(self,root):
        spec={'run_root':str(root),'_config_sha256':'frozen'}
        plans=[{'original_ordinal':i,'refiner_noise_seed':10+i} for i in range(2)]
        write_rows(root/'cohort/plans.jsonl',plans)
        decision={'run_diffusion':True,'input_fingerprint':'raw0'}
        write_json(root/'construction/GATE.json',{'decisions':[decision,dict(decision,input_fingerprint='raw1')]})
        value={'config_sha256':'frozen','gate':decision,'record':{'original_ordinal':0},
               'raw_refiner_output':{'seed':10,'diffusion_steps':800},'tokenization':{'fallback_to_raw':False}}
        for stage in ('refined','tokenized'):
            write_json(root/stage/'records/0000.json',value)
        return spec

    def test_preserves_complete_pairs_and_detects_endpoint_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);spec=self.fixture(root);manifest=root/'RESUME.json'
            original=file_hash(root/'refined/records/0000.json')
            result=register_refine_resume(spec,manifest)
            self.assertEqual((result['retained_requests'],result['missing_requests']),(1,1))
            self.assertEqual(validate_refine_resume(spec,manifest),result)
            self.assertEqual(file_hash(root/'refined/records/0000.json'),original)
            path=root/'tokenized/records/0000.json'
            value=json.loads(path.read_text());value['record']['body']='changed';write_json(path,value)
            with self.assertRaisesRegex(ValueError,'endpoint changed'):
                validate_refine_resume(spec,manifest)

    def test_rejects_changed_seed_gate_or_config_before_registration(self):
        for field in ('seed','gate','config'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);spec=self.fixture(root)
                path=root/'refined/records/0000.json';value=json.loads(path.read_text())
                if field=='seed': value['raw_refiner_output']['seed']=999
                elif field=='gate': value['gate']['run_diffusion']=False
                else: value['config_sha256']='different'
                write_json(path,value)
                with self.assertRaises(ValueError): register_refine_resume(spec,root/'RESUME.json')

    def test_rejects_partial_pair_and_changed_plan_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);spec=self.fixture(root);manifest=root/'RESUME.json'
            register_refine_resume(spec,manifest)
            with (root/'cohort/plans.jsonl').open('a') as stream: stream.write('\n')
            with self.assertRaisesRegex(ValueError,'inputs changed'): validate_refine_resume(spec,manifest)
            (root/'tokenized/records/0000.json').unlink()
            with self.assertRaisesRegex(ValueError,'partial'): register_refine_resume(spec,root/'OTHER.json')


if __name__=='__main__': unittest.main()
