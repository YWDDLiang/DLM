"""Count actual deterministic content-view supervision on an existing training run."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import sys
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path)
parser.add_argument('source', type=Path)
parser.add_argument('--runs', nargs='+', default=['mechanism_gauge64_control', 'mechanism_data64_next1600', 'mechanism_data256_next1600'])
parser.add_argument('--output', default='CONTENT_COVERAGE.json')
options = parser.parse_args()
root, source = options.root, options.source
sys.path.insert(0,str(source/'src'))
from crystal_dlm.expert_edit import ExpertEditDataset
class PrefixOnlyTokenizer:
    def __call__(self, text, **kwargs): return {'input_ids':[0]}
output = {'schema':'expert_content_coverage_v1','prefix_stub_only':True,
          'sampling_RNG_source_task_cut_unchanged':True,'runs':{}}
for name in options.runs:
    path = root/name/'train/TRAIN_CONFIG.json'
    config = json.loads(path.read_text())
    args = config['args']
    data = ExpertEditDataset(args['data_dirs'],PrefixOnlyTokenizer(),seed=args['seed'],split='train',
        smoke_sources=args['smoke_sources'],geometry_aux_fraction=args['geometry_aux_fraction'],
        content_fraction=args['content_fraction'],inspect_fraction=args['inspect_fraction'],
        student_feedback_fraction=args['student_feedback_fraction'],healthy_state_fraction=args['healthy_state_fraction'],
        content_target_mode=args['content_target_mode'],m2t_probability=args['m2t_probability'],
        target_representative='none')
    total = args['updates']*config['effective_batch']
    counters = {'G':Counter(),'S':Counter()}
    views = Counter()
    for index in range(total):
        row = data[index]
        views[row['kind']]+=1
        if row['kind']=='content':
            task = 'G' if row['task']==0 else 'S'
            for position, token in enumerate(row['targets']):
                if token!=-100: counters[task][(row['record_id'],position)]+=1
    result = {'updates':args['updates'],'batch':config['effective_batch'],'total_views':total,'view_counts':dict(views),
              'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'tasks':{}}
    for task in ['G','S']:
        keys = [(row['record_id'],p) for row in data.content[task] for p in row['action']['positions']]
        exposure = [counters[task][k] for k in keys]
        assert len(keys)==len(set(keys))
        result['tasks'][task] = {'positive_records':len(data.content[task]),
            'positive_sources':len({row['ancestor_id'] for row in data.content[task]}),
            'possible_record_positions':len(keys),'supervised_tokens':sum(exposure),
            'unseen_record_positions':sum(x==0 for x in exposure),
            'mean_supervisions_per_record_position':sum(exposure)/len(exposure),
            'histogram':dict(sorted(Counter(exposure).items()))}
    output['runs'][name] = result
out = root/options.output
assert not out.exists(), 'coverage audits must not overwrite prior evidence'
out.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output))
