import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from scripts.run_post_refine_cycle import write_json,write_rows,read_rows,file_hash
from scripts.run_ranked_rsi import compile_dataset, require_train, historical_stable_targets


class RankedDataset(unittest.TestCase):
    def fixture(self,root,count=256):
        plans=[{'source_split':'train','ancestor_id':str(i),'source_row_idx':i,'original_ordinal':i,
            'body_prompt':'fixed Plan '+str(i),'plan_state':{'N':2}} for i in range(count)]
        write_rows(root/'cohort/plans.jsonl',plans)
        return {'ranked_training':True,'training_parent_root':str(root/'cohort'),
            'run_root':str(root),'_config_sha256':'fixture','round_index':1,'requests':count,'policy':{}}

    def stage(self,root,stage,body,hull,action=None):
        count=len(read_rows(root/'cohort/plans.jsonl'))
        records=[{'sample_idx':i,'trajectory_id':f'{stage}:{i}','body_token_ids':body} for i in range(count)]
        measured=[dict(r,e_above_hull_eV_atom=hull,strict_sun=False,terminal_verified=True,
                       terminal_status='verified') for r in records]
        write_rows(root/stage/'inputs.jsonl',records)
        write_rows(root/stage/'scoring/result/attempt_results.jsonl',measured)
        write_json(root/stage/'scoring/result/SCORE_FINAL.json',{'status':'complete',
            'input_sha256':file_hash(root/stage/'inputs.jsonl')})
        if action:
            for i,row in enumerate(records): write_json(root/stage/'records'/f'{i:04d}.json',
                {'record':row,'editor_trace':{'action':action}})

    def test_generator_targets_actual_RAW_with_own_F_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'new';other=Path(tmp)/'old'
            spec=self.fixture(root);self.fixture(other)
            old=[100,1,2,3,4,5,6,20,1,2,3,21,4,5,6]
            new=old.copy();new[1]=9
            refined=new.copy();refined[1]=11
            self.stage(root,'construction',new,.8);self.stage(root,'tokenized',refined,-.05)
            self.stage(other,'construction',old,-.01);self.stage(other,'tokenized',old,.2)
            compile_dataset(spec,'G',[other])
            rows=read_rows(root/'pairs/G.jsonl')
            self.assertEqual(len(rows),256)
            self.assertEqual(rows[0]['chosen_tokens'],new)
            self.assertEqual(rows[0]['rejected_tokens'],old)
            self.assertEqual(rows[0]['objective_level'],'strict_stable')

    def test_editor_excludes_outside_action_pair_but_keeps_valid_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=self.fixture(root)
            before=[100,1,2,3,4,5,6,20,1,2,3,21,4,5,6]
            bad=before.copy();bad[1]=9
            good=before.copy();good[12]=8
            action={'mode':1,'sites':[1],'positions':[12,13,14]}
            self.stage(root,'current',before,.3)
            self.stage(root,'proposal',bad,-.01,action)
            self.stage(root,'teacher',good,.05,action)
            compile_dataset(spec,'E')
            rows=read_rows(root/'pairs/E.jsonl');audit=read_rows(root/'pairs/pair_audit_E.jsonl')
            self.assertEqual(len(rows),256)
            self.assertEqual(sum('excluded' in row for row in audit),256)
            self.assertEqual(rows[0]['action_positions'],[12,13,14])
            self.assertEqual(rows[0]['mode_target'],1)
            self.assertEqual(rows[0]['chosen_tokens'],good)

    def test_thousand_requests_compile_without_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'new';other=Path(tmp)/'old'
            spec=self.fixture(root,1000);self.fixture(other,1000)
            old=[100,1,2,3,4,5,6,20,1,2,3,21,4,5,6]
            new=old.copy();new[1]=9
            self.stage(root,'construction',new,.8);self.stage(root,'tokenized',new,-.05)
            self.stage(other,'construction',old,.8);self.stage(other,'tokenized',old,.2)
            compile_dataset(spec,'G',[other])
            self.assertEqual(len(read_rows(root/'pairs/G.jsonl')),1000)
            with self.assertRaisesRegex(ValueError,'registered TRAIN'):
                require_train(dict(spec,requests=999))

    def test_candidate_acceptance_labels_survive_one_mode_per_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=self.fixture(root,2)
            before=[100,1,2,3,4,5,6,20,1,2,3,21,4,5,6]
            bad=before.copy();bad[12]=7
            good=before.copy();good[12]=8
            action={'mode':1,'sites':[1],'positions':[12,13,14]}
            self.stage(root,'current',before,.3)
            self.stage(root,'proposal',bad,.4,action)
            self.stage(root,'teacher',good,-.05,action)
            compile_dataset(spec,'E')
            rows=read_rows(root/'pairs/E.jsonl')
            self.assertEqual(len(rows),4)
            for source_id in ('0','1'):
                group=[r for r in rows if r['source_id']==source_id]
                self.assertEqual([r['mode_target'] for r in group],[None,1])
                self.assertEqual([r['accept_target'] for r in group],[0,1])
                self.assertEqual(group[0]['chosen_tokens'],before)
                self.assertEqual(group[1]['chosen_tokens'],good)

    def test_historical_teacher_requires_same_plan_and_physics_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            from crystal_dlm.post_refine_contract import SCHEMA
            root=Path(tmp)/'new';prior=Path(tmp)/'prior'
            spec=self.fixture(root,2);old_spec=self.fixture(prior,2)
            write_json(prior/'RUN_SPEC.json',dict(old_spec,schema=SCHEMA))
            before=[100,1,2,3,4,5,6,20,1,2,3,21,4,5,6]
            good=before.copy();good[12]=8
            self.stage(root,'current',before,.2);self.stage(prior,'current',good,-.02)
            for at in (root,prior):
                path=at/'current/labeling/result';path.mkdir(parents=True)
                write_json(path/'LABEL_FINAL.json',{'input_sha256':file_hash(at/'current/inputs.jsonl'),
                    'protocol':{'max_steps':1000},'verification_protocol':'same',
                    'geometry_validation_protocol':'same','runtime_identities':['same']})
                write_rows(path/'labels.jsonl',[]);(path/'_SUCCESS').touch()
            plans=read_rows(root/'cohort/plans.jsonl');current=read_rows(root/'current/inputs.jsonl')
            targets,pins=historical_stable_targets(spec,plans,current,[prior])
            self.assertEqual(len(targets[0]),1)
            self.assertEqual(targets[0][0]['tokens'],good)
            self.assertFalse(targets[0][0]['historical_U_label_reused'])
            self.assertTrue(pins)
            old_plans=read_rows(prior/'cohort/plans.jsonl');old_plans[0]['body_prompt']='different Plan'
            (prior/'cohort/plans.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in old_plans))
            with self.assertRaisesRegex(ValueError,'Plan or ancestor'):
                historical_stable_targets(spec,plans,current,[prior])


if __name__=='__main__':unittest.main()
