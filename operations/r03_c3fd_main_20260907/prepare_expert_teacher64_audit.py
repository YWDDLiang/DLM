"""Audit exact teacher inputs for the existing gauge64 panels, without relabelling training data."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
read = lambda path: [json.loads(line) for line in path.open()]
prepared = root / 'expanded_training_data/prepared'
pairs = {x['target_physics_id']: x for x in read(prepared/'pairs_pending.jsonl') if x.get('target_physics_id')}
targets = {x['trajectory_id']: x for x in read(prepared/'target_inputs.jsonl')}
out = root/'mechanism_teacher64_inputs'
out.mkdir(exist_ok=False)
manifest = {'schema':'expert_teacher64_exact_inputs_v1', 'selection':'same actual train/dev source-task rows as gauge64, no outcome selection', 'splits':{}}
for split in ['train','dev']:
    panel = root/f'mechanism_gauge64_control/how_{split}/samples.jsonl'
    samples = read(panel)
    assert len(samples) == 64
    rows, selected = [], []
    for sample in samples:
        pid = sample['target_physics_id']
        pair, target = pairs[pid], targets[pid]
        assert pair['old_body'] == sample['old_body']
        assert pair['target_body'] == sample['target_body']
        assert target['group_id'] == sample['ancestor_id']
        assert target['source_split'] == split
        rows.append(target)
        selected.append({k:sample[k] for k in ['record_id','ancestor_id','task','old_physics_id','target_physics_id','old_body','target_body']})
    assert len({x['group_id'] for x in rows}) == 64
    path = out/f'{split}.jsonl'
    path.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))
    manifest['splits'][split] = {'panel_sha256':hashlib.sha256(panel.read_bytes()).hexdigest(),'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selected':selected}
(out/'INPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({s:{k:v for k,v in x.items() if k!='selected'} for s,x in manifest['splits'].items()}))
