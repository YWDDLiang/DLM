import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from crystal_dlm.post_refine_contract import registered_requests
from scripts.run_post_refine_cycle import write_rows, read_rows
from scripts.run_rsi_stages import prepare_fit


class FixedTrainPanel(unittest.TestCase):
    def test_expansion_preserves_complete_prefix_and_train_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'parent'
            plan={'N':1,'elements':['Na'],'counts':[1],
                  'rich_field_valid':True,'plan_end_marker_present':True}
            rows=[{'ancestor_id':str(i),'source_split':'train','plan_state':plan,
                   'source_row_idx':i,'prompt':'Plan '+str(i)} for i in range(1020)]
            rows += [dict(rows[0]),dict(rows[1],ancestor_id='heldout',
                plan_state=dict(plan,elements=['Mg'])),dict(rows[2],ancestor_id='eval',source_split='evaluation')]
            write_rows(parent/'pairs_pending.jsonl',rows)
            heldout=root/'heldout.jsonl'
            write_rows(heldout,[{'plan_state':dict(plan,elements=['Mg']),'body_eligible':True}])
            for n in (256,1000):
                spec={'requests':n,'run_root':str(root/str(n)),'run_id':str(n),'policy':{},
                      'assets':{'training_preparation':str(parent),'cohort':str(heldout)}}
                prepare_fit(spec)
            small=read_rows(root/'256/fit/cohort/plans.jsonl')
            large=read_rows(root/'1000/fit/cohort/plans.jsonl')
            self.assertEqual(small,large[:256]);self.assertEqual(len(large),1000)
            self.assertEqual(len({p['ancestor_id'] for p in large}),1000)
            self.assertFalse({'heldout','eval'} & {p['ancestor_id'] for p in large})
            report=json.loads((root/'1000/fit/cohort/MANIFEST.json').read_text())
            self.assertEqual(report['eligible_unique_sources'],1020)
            self.assertEqual(report['requests'],1000)
            with self.assertRaisesRegex(ValueError,'not enough'):
                prepare_fit(dict(spec,requests=1021,run_root=str(root/'too_many')))

    def test_bad_request_counts_fail_before_selection(self):
        for value in (None,True,0,-1,1000.0,'1000'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                registered_requests({'requests':value})


if __name__=='__main__': unittest.main()
