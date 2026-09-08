"""Register one RSI stage under the common six-GPU and absolute-deadline cap."""
import argparse
import json
from pathlib import Path
from run_component import verify_deployed_source
from submit_stage import configured_dispatch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--job',required=True)
    p.add_argument('--action',choices=['generate','label','score','refine','edit','train'],required=True)
    p.add_argument('--stage',default='construction')
    p.add_argument('--gpus',type=int,default=2)
    p.add_argument('--minutes',type=int,default=90)
    p.add_argument('--training-config',type=Path)
    p.add_argument('--cache',type=Path)
    p.add_argument('--score-attempt',default='scoring')
    p.add_argument('--reuse-endpoints',type=Path,nargs='*',default=[])
    a=p.parse_args();root=a.root.resolve();cfg=json.loads(a.config.read_text())
    cohort=Path(cfg['run_root']);relative=cohort.relative_to(root)
    source=Path(__file__).resolve().parents[2]
    pipeline=json.loads((root/'RAW0_PIPELINE.json').read_text())
    training='training_parent_root' in cfg
    pipeline.update(source_root=str(source),source_identity=verify_deployed_source(source),
                    purpose='training_feedback' if training else 'evaluation')
    gpu=0 if a.action=='score' else a.gpus
    script='src/scripts/run_rsi_stages.py'
    args=['--config',str(a.config)];inputs=[str(a.config)];distributed=False
    if a.action=='generate':
        directory=relative/'construction';script='src/scripts/run_post_refine_cycle.py'
        args+=['--stage','construct'];distributed=True
        inputs+=[str(cohort/'cohort/MANIFEST.json')]
        outputs=[f'{{output}}/worker_{i}_DONE.json' for i in range(gpu)]
    elif a.action=='label':
        directory=relative/a.stage/'labeling';script='scripts/label_programmed_paths.py'
        input_file=cohort/a.stage/'inputs.jsonl'
        args=['--input-jsonl',str(input_file),'--output-dir','{output}/result','--purpose',
              'training_feedback' if training else 'evaluation','--gpu-count',str(gpu),
              '--workers-per-gpu','2','--record-timeout','300','--deterministic']
        inputs+=[str(input_file)]
        if training:
            feedback=cohort/a.stage/'FEEDBACK_MANIFEST.json'
            args+=['--feedback-manifest',str(feedback)];inputs+=[str(feedback)]
        if a.reuse_endpoints:
            script='scripts/label_rsi_cached_endpoints.py'
            args+=['--reuse-endpoints',*[str(v) for v in a.reuse_endpoints]]
            inputs+=[str(v/'LABEL_FINAL.json') for v in a.reuse_endpoints]
        outputs=['{output}/result/LABEL_FINAL.json','{output}/result/labels.jsonl']
    elif a.action=='score':
        if not a.score_attempt.replace('_','').isalnum(): p.error('invalid score attempt name')
        directory=relative/a.stage/a.score_attempt;script='scripts/evaluate_programmed_paths.py'
        pointer=cohort/a.stage/'SCORING_DIRECTORY.json'
        if pointer.exists():
            prior=json.loads(pointer.read_text())['directory']
            if prior!=a.score_attempt and (cohort/a.stage/prior/'result/_SUCCESS').exists():
                raise ValueError('cannot replace a completed score ledger')
        pointer.write_text(json.dumps({'directory':a.score_attempt,'job':a.job})+'\n')
        input_file=cohort/a.stage/'inputs.jsonl';labels=cohort/a.stage/'labeling/result/labels.jsonl'
        args=['--paths-jsonl',str(input_file),'--labels-jsonl',str(labels),
            '--frozen-config',cfg['assets']['frozen_config'],'--official-cache',str(a.cache or cfg['assets']['official_cache']),
            '--output-dir','{output}/result','--expected-requests',str(cfg['requests']),
            '--endpoint','native','--cohort-role','training_feedback' if training else 'fixed_development',
            '--policy-stage','round0_diagnostic','--sun-only','--nu-workers','7',
            '--nu-cache',str(root/'nu_cache')]
        inputs+=[str(input_file),str(labels),str(labels.parent/'LABEL_FINAL.json')]
        if training:
            feedback=cohort/a.stage/'FEEDBACK_MANIFEST.json'
            args+=['--feedback-manifest',str(feedback)];inputs+=[str(feedback)]
        outputs=['{output}/result/_SUCCESS','{output}/result/attempt_results.jsonl']
    elif a.action in ('refine','edit'):
        directory=relative/('refined' if a.action=='refine' else 'proposal')
        args+=['--action',a.action];distributed=True
        inputs+=[str(cohort/'construction/GATE.json' if a.action=='refine' else cohort/'current/inputs.jsonl')]
        outputs=[f'{{output}}/worker_{i}_DONE.json' for i in range(gpu)]
    else:
        if not a.training_config: p.error('train requires --training-config')
        script='src/scripts/train_rsi_preferences.py'
        train=json.loads(a.training_config.read_text());training_output=Path(train['output_dir'])
        directory=training_output.parent.relative_to(root)
        if training_output.name!='result': p.error('training output must be component/result')
        args=['--config',str(a.training_config)];inputs+=[str(a.training_config),train['data']]
        outputs=['{output}/result/TRAINING_FINAL.json','{output}/result/checkpoint/RSI_TRAINING_DONE.json'];distributed=True
    stage={'name':a.action,'script':script,'args':args,'inputs':inputs,'outputs':outputs}
    if distributed: stage['distributed_processes']=gpu
    stages=[stage]
    if a.action=='score':
        stages.append({'name':'validity','script':'src/scripts/run_rsi_stages.py',
            'args':['--config',str(a.config),'--action','validity','--stage',a.stage],
            'inputs':[str(a.config),'{output}/result/attempt_results.jsonl'],
            'outputs':['{output}/result/BASIC_METRICS.json','{output}/result/four_metrics.jsonl']})
    pipeline['components']=[{'id':a.job,'output_dir':str(directory),'gpus':gpu,'stages':stages}]
    pipeline['jobs']={a.job:{'component_indices':[0],'gpus_per_task':gpu,'cpus_per_task':4*gpu if gpu else 8,
        'parallel_tasks':1,'wall_minutes':a.minutes,'memory':f'{(32 if a.action in ("label","refine") else 96)*gpu}G' if gpu else '64G',
        'partition':'gpu' if gpu else 'normal'}}
    manifest=root/(a.job+'_PIPELINE.json')
    if manifest.exists():
        if json.loads(manifest.read_text())!=pipeline: raise ValueError('registered job differs')
    else: manifest.write_text(json.dumps(pipeline,sort_keys=True,indent=2)+'\n')
    configured_dispatch(['--config',str(manifest),'--job',a.job])


if __name__=='__main__': main()
