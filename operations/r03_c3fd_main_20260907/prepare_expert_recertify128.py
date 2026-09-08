"""Create an auditable exact subset for fresh deterministic physics compilation."""
import hashlib
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
parent = root/'expanded_training_data/prepared'
out = root/'mechanism_recertify128/prepared'
out.mkdir(parents=True,exist_ok=False)
sha = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda path:[json.loads(line) for line in path.open()]
report = json.loads((parent/'PREPARATION_FINAL.json').read_text())
for name in ['pairs_pending.jsonl','old_inputs.jsonl','target_inputs.jsonl']:
    assert sha(parent/name)==report['files_sha256'][name]
selection_path = root/'mechanism_teacher64_inputs/INPUT_MANIFEST.json'
selection = json.loads(selection_path.read_text())
chosen = [x for s in ['train','dev'] for x in selection['splits'][s]['selected']]
assert len(chosen)==len({x['ancestor_id'] for x in chosen})==128
ancestors = {x['ancestor_id'] for x in chosen}
pairs = [x for x in read(parent/'pairs_pending.jsonl') if x['ancestor_id'] in ancestors]
selected = {x['ancestor_id']:x for x in chosen}
assert len(pairs)==128
for pair in pairs:
    item = selected[pair['ancestor_id']]
    assert pair['old_body']==item['old_body'] and pair['target_body']==item['target_body']
    assert pair['old_physics_id']==item['old_physics_id'] and pair['target_physics_id']==item['target_physics_id']
comps = {s:{p['composition_key'] for p in pairs if p['source_split']==s} for s in ['train','dev']}
assert not comps['train'] & comps['dev']
old = [x for x in read(parent/'old_inputs.jsonl') if x['group_id'] in ancestors]
target = [x for x in read(parent/'target_inputs.jsonl') if x['group_id'] in ancestors]
assert len(old)==len(target)==128
files = {'pairs_pending.jsonl':pairs,'old_inputs.jsonl':old,'target_inputs.jsonl':target,
         'all_inputs.jsonl':old+target,'rejected.jsonl':[]}
for name,rows in files.items():
    (out/name).write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))
result = {'schema':report['schema'],'attempted_requests':128,'admitted_old':128,
    'quantized_teacher_targets':128,'teacher_unavailable':0,
    'geometry_protocol':report['geometry_protocol'],
    'selection':'exact same previously fixed gauge64 train/dev positive panels; no new generated sources',
    'selection_manifest_sha256':sha(selection_path),'parent_preparation_sha256':sha(parent/'PREPARATION_FINAL.json'),
    'parent_files_sha256':{name:sha(parent/name) for name in ['pairs_pending.jsonl','old_inputs.jsonl','target_inputs.jsonl']},
    'source_provenance':report['source_provenance'],'files_sha256':{name:sha(out/name) for name in files}}
(out/'PREPARATION_FINAL.json').write_text(json.dumps(result,indent=2)+'\n')
(out/'_SUCCESS').touch()
print(json.dumps({'path':str(out),'sha256':sha(out/'PREPARATION_FINAL.json'),'input_sha256':sha(out/'all_inputs.jsonl')}))
