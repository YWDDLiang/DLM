"""Fixed-source sequential G/S editing probe; every prefix is saved exactly once."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def round_seed(seed, ancestor, round_index):
    return int(hashlib.sha256(f'{seed}:{ancestor}:round{round_index}'.encode()).hexdigest()[:15], 16)


def run_rounds(model, tokenizer, rows, *, seed, rounds, max_calls, batch_size, sampler):
    """No physical evaluator is available while these trajectories are generated."""
    current = [list(row['old_body']) for row in rows]
    totals = [0] * len(rows)
    seen = [{tuple(body)} for body in current]
    histories = [[] for _ in rows]
    for round_index in range(1, rounds + 1):
        requests = [{'prompt': row['prompt'], 'body': body.copy(), 'num_sites': row['num_atoms'],
                     'tasks': ('G', 'S'), 'seed': round_seed(seed, row['ancestor_id'], round_index)}
                    for row, body in zip(rows, current)]
        sampled = sampler(model, tokenizer, requests, allowed_modes=model.training_modes,
                          block_size=1, max_calls=max_calls, batch_size=batch_size)
        if len(sampled['results']) != len(rows):
            raise ValueError('a round lost a source')
        for index, (row, output) in enumerate(zip(rows, sampled['results'])):
            body = list(output['body'])
            if not 0 <= output['forward_calls'] <= max_calls or len(body) != len(current[index]):
                raise ValueError('invalid round output or budget')
            # Species and atom-count marker are immutable even across round boundaries.
            fixed = [0] + list(range(7, len(body), 4))
            if any(body[i] != row['old_body'][i] for i in fixed):
                raise ValueError('multi-round editing changed composition')
            totals[index] += output['forward_calls']
            histories[index].append({'round': round_index, 'input_body': current[index].copy(),
                'seed': requests[index]['seed'], 'output': output,
                'cumulative_forward_calls': totals[index],
                'changed_from_previous': sum(a != b for a, b in zip(current[index], body)),
                'changed_from_original': sum(a != b for a, b in zip(row['old_body'], body)),
                'revisited_saved_body': tuple(body) in seen[index]})
            current[index] = body
            seen[index].add(tuple(body))
        print(json.dumps({'round_completed': round_index, 'sources': len(rows),
                          'forward_calls_sum': sum(totals)}), flush=True)
    return histories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-path', 'b0-checkpoint', 'checkpoint', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--data-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--sources', type=int, default=64)
    parser.add_argument('--seed', type=int, default=2026090823)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--max-calls-per-round', type=int, default=160)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.sources <= 256 or args.rounds != 4 or args.max_calls_per_round != 160:
        parser.error('registered probe requires 1..256 sources, four rounds and 160 calls per round')
    import torch
    import torch.distributed as dist
    from crystal_dlm.expert_edit import ExpertEditDataset, edit_structures, load_editor_model
    from crystal_dlm.expert_edit_data import decode_body, physics_input
    from scripts.train_r03_physics_transfer import verify_b0, verify_saved_tables, verify_saved_lora, write_json, write_jsonl, read_jsonl, file_sha256
    rank, world, local_rank = (int(os.environ.get(k, d)) for k, d in
                              [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    if not torch.cuda.is_available() or not 1 <= world <= 6:
        raise ValueError('probe requires assigned CUDA devices')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl')
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        verify_b0(args.b0_checkpoint)
    if world > 1:
        dist.barrier()
    model, tokenizer = load_editor_model(args.model_path, args.checkpoint, device, trainable=False)
    tables = verify_saved_tables(model, args.b0_checkpoint)
    lora = verify_saved_lora(model, args.checkpoint)
    model.eval()
    data = ExpertEditDataset(args.data_dirs, tokenizer, seed=args.seed, split='dev')
    states = [row for row in data.states if row.get('source_kind') == 'current_B0_full_rich']
    if len({row['ancestor_id'] for row in states}) != len(states):
        raise ValueError('original B0 development states contain duplicate source groups')
    selected = sorted(states, key=lambda row: hashlib.sha256(
        f'{args.seed}:{row["ancestor_id"]}'.encode()).hexdigest())[:args.sources]
    if len(selected) != args.sources:
        raise ValueError('not enough registered old states')
    local = [(index, row) for index, row in enumerate(selected) if index % world == rank]
    histories = run_rounds(model, tokenizer, [row for _, row in local], seed=args.seed,
        rounds=args.rounds, max_calls=args.max_calls_per_round, batch_size=args.batch_size, sampler=edit_structures)
    inverse = {int(v): k for k, v in tokenizer.get_vocab().items()}
    samples, physics = [], []
    # Decode only after all editing rounds finish. Geometry never controls stopping or acceptance.
    for (index, row), history in zip(local, histories):
        samples.append({'probe_case': index, 'ancestor_id': row['ancestor_id'], 'source_split': 'dev',
                        'source_row_idx': row['source_row_idx'], 'num_atoms': row['num_atoms'],
                        'prompt': row['prompt'], 'old_body': row['old_body'], 'rounds': history})
        for endpoint in (0, 1, 2, 4):
            body = row['old_body'] if endpoint == 0 else history[endpoint-1]['output']['body']
            record = physics_input(f'round-probe:{index}:r{endpoint}', row['ancestor_id'],
                row['source_row_idx'], 'dev', 'native' if endpoint == 0 else 'expert_quantized',
                decode_body(body, inverse), ''.join(inverse[token] for token in body))
            record.update(probe_case=index, edit_rounds=endpoint, training_use_allowed=False)
            physics.append(record)
    write_jsonl(args.output_dir/f'samples.rank{rank}.jsonl', samples)
    write_jsonl(args.output_dir/f'physics.rank{rank}.jsonl', physics)
    if world > 1:
        dist.barrier()
    if rank == 0:
        all_samples = sorted([row for worker in range(world) for row in read_jsonl(args.output_dir/f'samples.rank{worker}.jsonl')], key=lambda row: row['probe_case'])
        all_physics = [row for worker in range(world) for row in read_jsonl(args.output_dir/f'physics.rank{worker}.jsonl')]
        if [row['probe_case'] for row in all_samples] != list(range(args.sources)):
            raise ValueError('probe sources lost or duplicated')
        write_jsonl(args.output_dir/'samples.jsonl', all_samples)
        for endpoint in (0, 1, 2, 4):
            records = sorted([row for row in all_physics if row['edit_rounds'] == endpoint], key=lambda row: row['probe_case'])
            if len(records) != args.sources:
                raise ValueError('probe endpoint lost sources')
            write_jsonl(args.output_dir/f'physics_round{endpoint}.jsonl', records)
        write_json(args.output_dir/'ROUND_FINAL.json', {'schema': 'expert_round_probe_v1',
            'source_split': 'dev', 'sources': args.sources, 'rounds': args.rounds,
            'seed': args.seed, 'max_calls_per_round': args.max_calls_per_round,
            'prefixes_generated_once': True, 'external_inference_geometry_gate': False,
            'inference_mlip': False, 'training_use_allowed': False,
            'selection': 'hash ordered original B0 compiled dev states; no teacher outcome filtering',
            'source_files': data.provenance, 'source_sha256': file_sha256(__file__),
            'checkpoint': str(args.checkpoint), 'checkpoint_receipt_sha256': file_sha256(args.checkpoint/'CHECKPOINT_FINAL.json'),
            'tables': tables, 'lora': lora,
            'outputs_sha256': {p.name: file_sha256(p) for p in args.output_dir.glob('*.jsonl')}})
        (args.output_dir/'_SUCCESS').touch()
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
