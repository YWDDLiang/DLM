"""Evaluate saved rejected S proposals without changing or resampling the policy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples-dir', type=Path, required=True)
    p.add_argument('--b0-checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer
    from crystal_dlm.expert_edit_data import decode_body, physics_input, read_rows, write_rows
    receipt = json.loads((a.samples_dir/'ROUND_FINAL.json').read_text())
    if receipt['source_split'] != 'dev' or not (a.samples_dir/'_SUCCESS').is_file():
        raise ValueError('expected complete development-only round probe')
    if receipt['outputs_sha256']['samples.jsonl'] != sha(a.samples_dir/'samples.jsonl'):
        raise ValueError('saved round trajectories changed')
    if sha(a.b0_checkpoint/'tokenizer.json') != '3a21588abca8e56155cc7b6cabb81df51992ccd2e89704aec770912f24e75509':
        raise ValueError('original tokenizer identity changed')
    tokenizer = AutoTokenizer.from_pretrained(str(a.b0_checkpoint), trust_remote_code=True, local_files_only=True)
    inverse = {int(v): k for k, v in tokenizer.get_vocab().items()}
    inputs, decisions = [], []
    for sample in read_rows(a.samples_dir/'samples.jsonl'):
        for history in sample['rounds']:
            for action in history['output']['trace']:
                if action['task'] != 'S' or action['mode'] == 'none':
                    continue
                index = len(decisions)
                decisions.append({'decision_idx': index, 'case_idx': sample['probe_case'],
                    'ancestor_id': sample['ancestor_id'], 'round': history['round'],
                    'mode': action['mode'], 'accepted': action['accepted'], 'quality': action['quality'],
                    'reason': action['reason']})
                for variant, field in [('old', 'old_body'), ('proposal', 'proposal_body')]:
                    body = action[field]
                    record = physics_input(f'round-S-audit:{index}:{variant}', sample['ancestor_id'],
                        sample['source_row_idx'], 'dev', 'expert_quantized', decode_body(body, inverse),
                        ''.join(inverse[t] for t in body))
                    record.update(decision_idx=index, probe_case=sample['probe_case'], edit_round=history['round'],
                                  audit_variant=variant, training_use_allowed=False)
                    inputs.append(record)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    write_rows(a.output_dir/'physics.jsonl', inputs)
    write_rows(a.output_dir/'decisions.jsonl', decisions)
    report = {'purpose': 'saved_proposal_gate_audit', 'source_split': 'dev', 'decisions': len(decisions),
              'source_groups': len({x['ancestor_id'] for x in decisions}),
              'all_attempted_S_proposals_included': True, 'new_sampling': False, 'training_use_allowed': False,
              'samples_sha256': sha(a.samples_dir/'samples.jsonl'), 'receipt_sha256': sha(a.samples_dir/'ROUND_FINAL.json'),
              'source_sha256': sha(__file__), 'outputs_sha256': {n: sha(a.output_dir/n) for n in ['physics.jsonl', 'decisions.jsonl']}}
    (a.output_dir/'PREPARATION_FINAL.json').write_text(json.dumps(report, indent=2)+'\n')
    (a.output_dir/'_SUCCESS').touch()
    print(json.dumps(report))


if __name__ == '__main__':
    main()
