"""Collect complete gauge repeats with input identity and learned-action evidence."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
read = lambda path: [json.loads(line) for line in path.open()]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
def labels(path):
    return {'sha256':sha(path), 'rows':[{k:x[k] for k in
        ['trajectory_id','group_id','status','verified','terminal_energy','actual_steps','error','versions']}
        for x in read(path)]}
def inputs(path):
    return {x['group_id']:{k:x[k] for k in ['body','structure','source_row_idx','source_split']} for x in read(path)}
seeds = [2026090913,2026091013,2026091113]
data = {'schema':'expert_gauge64_repeats_capsule_v1','labels':{},'reports':{},
        'autonomous_samples':{},'input_identity':{}}
reference = inputs(root/'mechanism_gauge64_control/how_dev/old_physics.jsonl')
assert len(reference) == 64
for arm in ['control','aligned']:
    p = root/('mechanism_gauge64_repeat_'+arm)
    for seed in seeds:
        name = str(seed)
        path = p/('sample_'+name)/'old_physics.jsonl'
        same = inputs(path) == reference
        assert same
        data['input_identity'][arm+':'+name] = {'equal_to_first_seed':same,'sha256':sha(path)}
        data['labels'][arm+':'+name] = labels(p/('labels_'+name)/'labels.jsonl')
        for stage, report in [('sample_'+name,'SAMPLE_FINAL.json'),('labels_'+name,'LABEL_FINAL.json')]:
            data['reports'][arm+':'+stage] = json.loads((p/stage/report).read_text())
            assert (p/stage/'_SUCCESS').is_file()
    for stage, report in [('autonomous','SAMPLE_FINAL.json'),('labels_autonomous','LABEL_FINAL.json')]:
        data['reports'][arm+':'+stage] = json.loads((p/stage/report).read_text())
        assert (p/stage/'_SUCCESS').is_file()
    data['labels'][arm+':autonomous'] = labels(p/'labels_autonomous/labels.jsonl')
    data['autonomous_samples'][arm] = [{k:x[k] for k in
        ['ancestor_id','old_body','old_geometry','output','proposal_geometry']} for x in read(p/'autonomous/samples.jsonl')]
paths = [root/('mechanism_gauge64_repeat_'+arm)/'autonomous/old_physics.jsonl' for arm in ['control','aligned']]
assert inputs(paths[0]) == inputs(paths[1])
data['input_identity']['autonomous'] = {'normalized_equal':True,'byte_equal':paths[0].read_bytes()==paths[1].read_bytes(),'sha256':[sha(p) for p in paths]}
p = root/'mechanism_gauge64_repeat_control/labels_autonomous_old'
assert (p/'_SUCCESS').is_file()
data['reports']['autonomous_old'] = json.loads((p/'LABEL_FINAL.json').read_text())
data['labels']['autonomous_old'] = labels(p/'labels.jsonl')
out = root/'GAUGE64_REPEATS_COMPACT.json'
assert not out.exists()
out.write_text(json.dumps(data,separators=(',',':'))+'\n')
print(json.dumps({'output':str(out),'sha256':sha(out),'bytes':out.stat().st_size}))
