"""Physics recovery reuses only fully bound known endpoint results."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tests.test_label_programmed_paths import MODULE
reusable_labels=MODULE.reusable_labels
COMMON_RELAXATION_PROTOCOL=MODULE.COMMON_RELAXATION_PROTOCOL
TERMINAL_VERIFICATION_PROTOCOL=MODULE.TERMINAL_VERIFICATION_PROTOCOL
LABEL_GEOMETRY_PROTOCOL=MODULE.LABEL_GEOMETRY_PROTOCOL


class LabelReuseTests(unittest.TestCase):
    def fixture(self, root):
        by={f'key{i}':[{'trajectory_id':f'id{i}','group_id':f'g{i}','source_row_idx':i,
                       'source_split':'evaluation','endpoint':'native'}] for i in range(2)}
        rows=[{**by[f'key{i}'][0],'endpoint_cache_key':f'key{i}',
               'status':'verified' if i==0 else 'worker_error','versions':{'source':'same'},
               'terminal_energy':-1. if i==0 else None} for i in range(2)]
        (root/'labels.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        report={'input_sha256':'input-sha','purpose':'evaluation','protocol':COMMON_RELAXATION_PROTOCOL,
                'verification_protocol':TERMINAL_VERIFICATION_PROTOCOL,'geometry_validation_protocol':LABEL_GEOMETRY_PROTOCOL,
                'requested':2,'completed':2,'statuses':{'verified':1,'worker_error':1},'runtime_identities':[{'source':'same'}]}
        (root/'LABEL_FINAL.json').write_text(json.dumps(report))
        return by

    def load(self, root, by, **kwargs):
        options=dict(input_sha256='input-sha',purpose='evaluation',protocol=COMMON_RELAXATION_PROTOCOL,runtime={'source':'same'})
        return reusable_labels(root,by,**(options|kwargs))

    def test_completed_known_records_reused_and_engineering_unknown_retried(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);by=self.fixture(root)
            cached,report=self.load(root,by)
            self.assertEqual(set(cached),{'key0'})
            self.assertEqual(cached['key0']['terminal_energy'],-1.)
            self.assertFalse(report['worker_errors_reused'])
            self.assertEqual(report['reused_endpoints'],1)

    def test_input_endpoint_protocol_or_runtime_change_cannot_reuse(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);by=self.fixture(root)
            for kw in ({'input_sha256':'changed'},{'runtime':{'source':'changed'}},
                       {'protocol':COMMON_RELAXATION_PROTOCOL|{'fire_dt':.2}}):
                with self.subTest(kw=kw),self.assertRaises(ValueError):self.load(root,by,**kw)
            changed=copy.deepcopy(by);changed['different']=changed.pop('key0')
            with self.assertRaises(ValueError):self.load(root,changed)

    def test_inconsistent_old_report_is_not_repaired_into_scientific_success(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);by=self.fixture(root)
            report=json.loads((root/'LABEL_FINAL.json').read_text());report['statuses']={'verified':2}
            (root/'LABEL_FINAL.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'contradicts'):self.load(root,by)
            by=self.fixture(root);by['key0'][0]['success']=False
            with self.assertRaisesRegex(ValueError,'failed generation'):self.load(root,by)


if __name__=='__main__':unittest.main()
