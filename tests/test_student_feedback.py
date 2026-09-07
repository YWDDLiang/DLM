import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from crystal_dlm.expert_edit_data import (compile_student_feedback, feedback_decision, quantize_arrays,
    certify_geometry, full_action, physics_input, endpoint_fingerprint, COMMON_RELAXATION_PROTOCOL,
    TERMINAL_VERIFICATION_PROTOCOL)
from crystal_dlm.fixed_slot import build_special_tokens


class Tokenizer:
    def get_vocab(self):
        return {token:i for i,token in enumerate(build_special_tokens())}


def write_rows(path,rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


class StudentFeedbackTests(unittest.TestCase):
    def test_missing_physics_does_not_become_a_stability_negative(self):
        result = feedback_decision('S',[1],[2],{'valid':True},{'valid':True},None,None)
        self.assertIsNone(result['accept_label'])
        self.assertIsNone(result['gain_eV_atom'])
        self.assertIsNone(result['old_reliable'])
        invalid = feedback_decision('S',[1],[2],{'valid':True},{'valid':False},None,None)
        self.assertFalse(invalid['accept_label'])
        self.assertIsNone(invalid['gain_eV_atom'])

    def test_compile_rejected_states_and_complete_student_to_teacher_corrections(self):
        tokenizer = Tokenizer()
        vocab = tokenizer.get_vocab()
        inverse = {v:k for k,v in vocab.items()}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepared,reference,samples,labels = [root/name for name in ('prepared','reference','samples','labels')]
            for path in (prepared,reference,samples,labels): path.mkdir()
            arrays = {'lengths':[4.]*3,'angles':[90.]*3,'species':['Na','Cl'],'frac_coords':[[0.,0.,0.],[.5,.5,.5]]}
            old,old_arrays,_ = quantize_arrays(arrays,vocab)
            arrays['lengths']=[4.2]*3
            target,target_arrays,_ = quantize_arrays(arrays,vocab)
            arrays['lengths']=[4.1]*3
            final,final_arrays,_ = quantize_arrays(arrays,vocab)
            bad=old.copy();bad[8:11]=old[12:15]
            pair={'schema':'expert_crystal_edit_v1','ancestor_id':'source','source_split':'train','source_row_idx':7,
                  'composition_key':'Cl:1|Na:1','plan_state':{'N':2,'elements':['Na','Cl'],'counts':[1,1]},
                  'prompt':'actual rich prompt','num_atoms':2,'old_body':old,'target_body':target,
                  'teacher_available':True,'old_physics_id':'old','target_physics_id':'teacher'}
            write_rows(prepared/'pairs_pending.jsonl',[pair])
            (prepared/'PREPARATION_FINAL.json').write_text(json.dumps({'heldout_cohort_sha256':'b'*64,'files_sha256':{
                'pairs_pending.jsonl':hashlib.sha256((prepared/'pairs_pending.jsonl').read_bytes()).hexdigest()}}))
            refs=[physics_input(name,'source',7,'train',endpoint,values,''.join(inverse[x] for x in body))
                  for name,endpoint,values,body in (('old','native',old_arrays,old),('teacher','expert_quantized',target_arrays,target))]
            current=physics_input('final','source',7,'train','expert_quantized',final_arrays,''.join(inverse[x] for x in final))
            write_rows(prepared/'all_inputs.jsonl',refs)
            write_rows(samples/'physics.jsonl',[current])
            versions={'model':'fixture'}
            def label_bundle(destination,input_path,inputs,energies):
                rows=[{**{key:r[key] for key in ('trajectory_id','group_id','source_row_idx','source_split','endpoint')},
                       'status':'verified','verified':True,'terminal_energy':energy,'final_structure':r['structure'],
                       'versions':versions,'endpoint_cache_key':endpoint_fingerprint(r)} for r,energy in zip(inputs,energies)]
                write_rows(destination/'labels.jsonl',rows)
                report={'purpose':'expert_edit','protocol':COMMON_RELAXATION_PROTOCOL,'verification_protocol':TERMINAL_VERIFICATION_PROTOCOL,
                        'requested':len(rows),'completed':len(rows),'statuses':{'verified':len(rows)},'runtime_identities':[versions],
                        'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest()}
                (destination/'LABEL_FINAL.json').write_text(json.dumps(report));(destination/'_SUCCESS').touch()
            label_bundle(reference,prepared/'all_inputs.jsonl',refs,[-1.,-1.3])
            label_bundle(labels,samples/'physics.jsonl',[current],[-1.1])
            trace=[{'task':'G','old_body':old,'proposal_body':bad,'accepted':False,**full_action(old,bad,2)},
                   {'task':'S','old_body':old,'proposal_body':final,'accepted':True,**full_action(old,final,2)}]
            row={key:pair[key] for key in ('ancestor_id','source_split','source_row_idx','prompt','num_atoms','old_body')}
            row['output']={'canonical_body':final,'trace':trace,'scope_policy':'learned',
                           'accept_all':False,'S_admission_policy':'learned'}
            write_rows(samples/'samples.jsonl',[row])
            (samples/'SAMPLE_FINAL.json').write_text(json.dumps({'split':'train','source_kind':'all_old_states','requested':1}))
            (samples/'_SUCCESS').touch()
            report=compile_student_feedback(samples,labels,[prepared],[reference],tokenizer,root/'compiled')
            rows=[json.loads(line) for line in (root/'compiled/train.jsonl').read_text().splitlines()]
            rejected=[r for r in rows if ':proposal_0:' in r['record_id']]
            self.assertEqual(len(rejected),1)
            self.assertFalse(rejected[0]['accept_label'])
            self.assertIsNone(rejected[0]['target_reliable'])
            repairs=[r for r in rows if r.get('original_teacher_reference') and r['old_body']==bad and r['task']=='G']
            self.assertEqual(len(repairs),1)
            self.assertTrue(repairs[0]['content_supervision'])
            self.assertEqual(repairs[0]['target_body'],target)
            unknown_s=[r for r in rows if r.get('original_teacher_reference') and r['old_body']==bad and r['task']=='S']
            self.assertIsNone(unknown_s[0]['accept_label'])
            self.assertTrue(all('target_body' not in r for r in rows if r.get('state_only')))
            self.assertFalse(report['development_records_used_for_training'])
            (samples/'SAMPLE_FINAL.json').write_text(json.dumps({'split':'dev','source_kind':'all_old_states','requested':1}))
            with self.assertRaisesRegex(ValueError,'training-only'):
                compile_student_feedback(samples,labels,[prepared],[reference],tokenizer,root/'wrong')


if __name__=='__main__': unittest.main()
