"""Cache contracts with synthetic label fixtures; no physical worker runs."""
import copy
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from pymatgen.core import Lattice,Structure
from test_sun_only_evaluation import fixture,write_rows,EVAL

ROOT=Path(__file__).resolve().parents[1]
B=importlib.util.spec_from_file_location('rsi_cached_label_test',ROOT/'scripts/label_rsi_cached_endpoints.py')
M=importlib.util.module_from_spec(B);B.loader.exec_module(M)


class CachedPhysics(unittest.TestCase):
    def prepare(self,root):
        paths,old,_,_,records,labels,report=fixture(root,size=4)
        # The fixture's two identical Na endpoints must agree physically.
        labels[0].update(raw_energy=0.,terminal_energy=0.)
        write_rows(old/'labels.jsonl',labels)
        report['input_file']=str(paths)
        (old/'LABEL_FINAL.json').write_text(json.dumps(report))
        current=copy.deepcopy(records)
        for row in current: row['trajectory_id']='new:'+row['trajectory_id']
        target=root/'target.jsonl';write_rows(target,current)
        return target,old,current,report

    def invoke(self,target,old,output,report):
        argv=['cached','--input-jsonl',str(target),'--output-dir',str(output),'--purpose','evaluation',
              '--gpu-count','1','--reuse-endpoints',str(old)]
        with patch.object(sys,'argv',argv),patch.dict(os.environ,{'SLURM_JOB_ID':'fixture','CUDA_VISIBLE_DEVICES':'0'}),\
             patch.object(M.physics,'runtime_identity',return_value=report['runtime_identities'][0]),\
             contextlib.redirect_stdout(io.StringIO()): M.main()

    def test_identity_only_changes_reuse_all_physics_and_keep_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target,old,records,report=self.prepare(root);output=root/'result'
            with patch.object(M.physics,'bounded_labels',side_effect=AssertionError('no physical worker expected')):
                self.invoke(target,old,output,report)
            final=json.loads((output/'LABEL_FINAL.json').read_text())
            self.assertEqual(final['requested'],4);self.assertEqual(final['completed'],4)
            self.assertEqual(final['new_endpoint_evaluations'],0)
            self.assertEqual(final['generation_failures_without_model_calls'],1)
            labels,_=EVAL.load_bound_evaluation_labels(records,[output/'labels.jsonl'],paths_file=target,endpoint='native')
            self.assertEqual(set(labels),{r['trajectory_id'] for r in records})

    def test_changed_cell_requires_a_fresh_physical_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target,old,records,report=self.prepare(root)
            records[2]['structure']=Structure(Lattice.cubic(5),['Li'],[[0,0,0]]).as_dict();write_rows(target,records)
            with patch.object(M.physics,'bounded_labels',side_effect=RuntimeError('fresh required')) as fresh:
                with self.assertRaisesRegex(RuntimeError,'fresh required'): self.invoke(target,old,root/'result',report)
            pending=fresh.call_args.args[0]
            self.assertEqual(len(pending),1)
            self.assertEqual(next(iter(pending.values()))[0]['trajectory_id'],records[2]['trajectory_id'])

    def test_mutated_cache_source_ledger_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target,old,_,report=self.prepare(root)
            with (root/'paths.jsonl').open('a') as handle: handle.write('\n')
            with self.assertRaisesRegex(ValueError,'cache input ledger changed'):
                self.invoke(target,old,root/'result',report)


if __name__=='__main__': unittest.main()
