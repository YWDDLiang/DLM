import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from scripts.run_post_refine_cycle import write_rows,write_json,file_hash
from scripts.run_rsi_stages import select_current


class TokenBoundary(unittest.TestCase):
    def test_raw_preference_does_not_replace_editor_token_F_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            write_rows(root/'cohort/plans.jsonl',[{'original_ordinal':0,'source_split':'evaluation'}])
            for stage,body,sun,hull in [('construction',[1,2],True,0.),('tokenized',[1,3],False,.05)]:
                record={'trajectory_id':stage+':0','sample_idx':0,'original_ordinal':0,'evaluation_ordinal':0,
                        'body_token_ids':body,'success':True,'source_split':'evaluation'}
                inputs=root/stage/'inputs.jsonl';write_rows(inputs,[record])
                directory=root/stage/'scoring/result'
                write_rows(directory/'attempt_results.jsonl',[{'sample_idx':0,'strict_sun':sun,
                    'strict_stable':sun,'meta_sun':True,'meta_stable':True,'e_above_hull_eV_atom':hull}])
                write_json(directory/'EVALUATION_FINAL.json',{'status':'complete','input_sha256':file_hash(inputs)})
            select_current({'run_root':str(root),'run_id':'fixture'})
            current=json.loads((root/'current/records/0000.json').read_text())
            self.assertEqual(current['record']['body_token_ids'],[1,3])
            self.assertEqual(current['preference_for_training_only']['chosen'],'before')
            self.assertFalse(current['physical_reranking_performed'])


if __name__=='__main__': unittest.main()
