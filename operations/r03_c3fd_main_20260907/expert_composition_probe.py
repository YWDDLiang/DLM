"""Paired old/quantized-teacher/student F800 mechanism observations, never training."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
VARIANTS = ('old', 'teacher', 'student')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def schedule(cases, repeats=2):
    jobs = []
    for repeat in range(repeats):
        for case in cases:
            shift = case['case_idx'] % len(VARIANTS)
            order = VARIANTS[shift:] + VARIANTS[:shift]
            if repeat % 2:
                order = tuple(reversed(order))
            for variant in order:
                jobs.append({'sample_idx': len(jobs), 'case_idx': case['case_idx'],
                             'variant': variant, 'technical_repeat': repeat,
                             'refiner_seed': case['refiner_seed']})
    return jobs


def prepare(args):
    import torch
    from transformers import AutoTokenizer
    from crystal_dlm.expert_edit_data import read_rows, bound_labels, physics_input, decode_body
    from scripts.run_r03_integrated_body import load_frozen_runtime, frozen_imports, materialize_record
    source = args.run_root
    sample_dir = source / 'stage2_diagnostics/autonomous_selected'
    samples = read_rows(sample_dir / 'samples.jsonl')
    sample_report = json.loads((sample_dir / 'SAMPLE_FINAL.json').read_text())
    if sample_report['split'] != 'dev' or not (sample_dir / '_SUCCESS').is_file():
        raise ValueError('source development sample is incomplete')
    for pin in sample_report['source_files']:
        if digest(pin['path']) != pin['sha256'] or digest(Path(pin['path']).parent / 'DATA_FINAL.json') != pin['report_sha256']:
            raise ValueError('original compiled development data changed')
    old_labels, _ = bound_labels(sample_dir / 'old_physics.jsonl', source / 'stage2_diagnostics/labels_original')
    old_labels = {row['group_id']: row for row in old_labels.values()}
    pairs, positive, inputs = {}, {}, [sample_dir / 'samples.jsonl', sample_dir / 'SAMPLE_FINAL.json',
                                        source / 'stage2_diagnostics/labels_original/labels.jsonl']
    for directory in ('headroom256', 'data_remaining'):
        path = source / directory / 'prepared/pairs_pending.jsonl'
        report = json.loads((path.parent / 'PREPARATION_FINAL.json').read_text())
        if digest(path) != report['files_sha256']['pairs_pending.jsonl']:
            raise ValueError('original teacher pairs changed')
        pairs.update({row['ancestor_id']: row for row in read_rows(path)})
        inputs.append(path)
    for directory in ('headroom256/compiled', 'editor_smoke32/compiled'):
        path = source / directory / 'dev.jsonl'
        for row in read_rows(path):
            if row.get('content_supervision'):
                positive.setdefault(row['ancestor_id'], set()).add(tuple(row['target_body']))
        inputs.extend([path, path.parent / 'DATA_FINAL.json'])
    pools = {'invalid_geometry': [], 'valid_unreliable': [], 'reliable': []}
    for sample in samples:
        ancestor = sample['ancestor_id']
        pair = pairs.get(ancestor)
        if (not pair or not pair.get('teacher_available')
                or tuple(pair['target_body']) not in positive.get(ancestor, set())):
            continue
        if sample['source_split'] != 'dev' or sample['old_body'] != pair['old_body']:
            raise ValueError('mechanism observations require the original development state')
        kind = ('invalid_geometry' if sample['old_geometry']['valid'] is not True else
                'reliable' if old_labels[ancestor]['verified'] else 'valid_unreliable')
        pools[kind].append((sample, pair))
    cases, compositions = [], set()
    for kind, quota in (('invalid_geometry', 6), ('valid_unreliable', 5), ('reliable', 5)):
        ordered = sorted(pools[kind], key=lambda x: hashlib.sha256(f'{args.seed}:{x[0]["ancestor_id"]}'.encode()).hexdigest())
        chosen = []
        for sample, pair in ordered:
            if pair['composition_key'] in compositions:
                continue
            compositions.add(pair['composition_key'])
            chosen.append((sample, pair))
            if len(chosen) == quota:
                break
        if len(chosen) != quota:
            raise ValueError(f'insufficient predetermined {kind} cases: {len(chosen)}/{quota}')
        for sample, pair in chosen:
            seed = int.from_bytes(hashlib.sha256(f'{args.seed}:{sample["ancestor_id"]}:F800'.encode()).digest()[:8], 'big') % 2**63
            cases.append({'case_idx': len(cases), 'ancestor_id': sample['ancestor_id'],
                'source_row_idx': sample['source_row_idx'], 'source_split': 'dev', 'stratum': kind,
                'composition_key': pair['composition_key'], 'plan_state': pair['plan_state'], 'prompt': pair['prompt'],
                'refiner_seed': seed, 'student_trace': sample['output']['trace'],
                'bodies': {'old': pair['old_body'], 'teacher': pair['target_body'],
                           'student': sample['output']['canonical_body']}})
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.b0_checkpoint, local_files_only=True, trust_remote_code=True)
    inverse = {int(value): key for key, value in tokenizer.get_vocab().items()}
    runtime = load_frozen_runtime(args.frozen_runtime_root)
    with frozen_imports(runtime):
        process_one = runtime.module.import_process_one(args.crysllmgen_dir)
    graphs, raw, failures = {}, [], []
    for case in cases:
        for variant in VARIANTS:
            body = case['bodies'][variant]
            index = case['case_idx'] * 3 + VARIANTS.index(variant)
            task = {'sample_idx': index, 'ordinal': index, 'attempt_id': f'composition-probe:{index}',
                'body_noise_seed': 0, 'eligible': True, 'source_row': {'plan_state': case['plan_state']},
                'plan_state': case['plan_state'], 'body_prompt': case['prompt'], 'schedule_sha256': None}
            with frozen_imports(runtime):
                record, graph = materialize_record(task, body, runtime=runtime, tokenizer=tokenizer, process_one=process_one)
            key = f'{case["case_idx"]}:{variant}'
            graphs[key] = graph
            if graph is None:
                failures.append({'key': key, 'reason': record['reason'], 'message': record.get('message')})
            value = physics_input('composition-raw:' + key, case['ancestor_id'], case['source_row_idx'],
                                  'dev', 'native', decode_body(body, inverse), ''.join(inverse[x] for x in body))
            value.update(probe_case=case['case_idx'], probe_variant=variant)
            raw.append(value)
    from crystal_dlm.expert_edit_data import write_rows
    write_rows(output / 'raw_inputs.jsonl', raw)
    torch.save(graphs, output / 'graphs.pt')
    write_json(output / 'STUDY.json', {'schema': 'expert_composition_probe_v1', 'cases': cases,
        'schedule': schedule(cases), 'graph_failures': failures, 'training_use_allowed': False,
        'teacher_definition': 'actual saved quantized training target; not the floating teacher',
        'source_files_sha256': {str(path): digest(path) for path in inputs},
        'graphs_sha256': digest(output / 'graphs.pt'), 'raw_inputs_sha256': digest(output / 'raw_inputs.jsonl'),
        'selection': 'fixed original-state strata and hash order, unique compositions; no new outcomes used',
        'R_repeat_cases': [0, 3, 6, 8, 11, 14], 'source_sha256': digest(__file__)})
    (output / '_SUCCESS').touch()
    print(json.dumps({'cases': len(cases), 'planned_refinements': len(schedule(cases)), 'graph_failures': failures}), flush=True)


def refine(args):
    import numpy as np
    import torch
    from crystal_dlm.expert_edit_data import physics_input, write_rows
    from scripts.refine_dlm_with_crysllmgen import setup_crysllmgen_imports, ProposalDataset, lattices_to_params_shape, init_distributed
    from crystal_dlm.fixed_slot import Z_TO_SYMBOL
    prepared = args.prepared_dir
    study = json.loads((prepared / 'STUDY.json').read_text())
    if study['graphs_sha256'] != digest(prepared / 'graphs.pt') or study['training_use_allowed'] is not False:
        raise ValueError('paired probe input identity changed')
    if digest(args.checkpoint) != '573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e':
        raise ValueError('model494 identity changed')
    info = init_distributed()
    rank, world, device = info['rank'], info['world_size'], info['device']
    if not torch.cuda.is_available() or not 1 <= world <= 6:
        raise ValueError('paired cases require one through six allocated GPUs')
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    if world > 1:
        torch.distributed.barrier()
    output = args.output_dir / f'rank{rank}'
    output.mkdir()
    _, Model, Data, DataLoader = setup_crysllmgen_imports(args.crysllmgen_dir)
    model = Model(1000, 'train').to(device)
    model.device = device
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved['model'] if 'model' in saved else saved)
    del saved
    model.eval()
    graphs = torch.load(prepared / 'graphs.pt', map_location='cpu', weights_only=False)
    results, tensors = [], []
    local_jobs = [job for job in study['schedule'] if job['case_idx'] % world == rank]
    with torch.no_grad():
        for job in local_jobs:
            case = study['cases'][job['case_idx']]
            key = f'{job["case_idx"]}:{job["variant"]}'
            graph = graphs[key]
            arrays = None
            if graph is not None:
                seed = job['refiner_seed']
                random.seed(seed); np.random.seed(seed % 2**32)
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                batch = next(iter(DataLoader(ProposalDataset([graph], Data), batch_size=1, shuffle=False))).to(device)
                value, _ = model.sample(batch, diff_steps=800)
                value = {k: v.detach().cpu() for k, v in value.items() if isinstance(v, torch.Tensor)}
                lengths, angles = lattices_to_params_shape(value['lattices'])
                arrays = {'lengths': lengths.reshape(-1, 3)[0].tolist(), 'angles': angles.reshape(-1, 3)[0].tolist(),
                          'species': [Z_TO_SYMBOL[int(z)] for z in value['atom_types'].reshape(-1)],
                          'frac_coords': value['frac_coords'].reshape(-1, 3).tolist()}
                tensors.append({'job': job, 'output': value})
            pid = f'composition-refined:{job["technical_repeat"]}:{key}'
            row = physics_input(pid, case['ancestor_id'], case['source_row_idx'], 'dev', 'tau800', arrays)
            row.update(probe_case=job['case_idx'], probe_variant=job['variant'], technical_repeat=job['technical_repeat'],
                       probe_job_index=job['sample_idx'], probe_gpu_rank=rank)
            results.append(row)
            write_json(output / 'PROGRESS.json', {'completed': len(results), 'planned': len(local_jobs)})
            print(json.dumps({'rank': rank, 'completed': len(results), 'planned': len(local_jobs)}), flush=True)
    write_rows(output / 'refined_inputs.jsonl', results)
    torch.save(tensors, output / 'outputs.pt')
    if world > 1:
        torch.distributed.barrier()
    if rank != 0:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        return
    from crystal_dlm.expert_edit_data import read_rows
    results = sorted([row for worker in range(world)
                      for row in read_rows(args.output_dir / f'rank{worker}/refined_inputs.jsonl')],
                     key=lambda row: row['probe_job_index'])
    if [row['probe_job_index'] for row in results] != list(range(len(study['schedule']))):
        raise ValueError('paired probe lost or duplicated a scheduled observation')
    write_rows(args.output_dir / 'refined_inputs.jsonl', results)
    repeats = [x for x in results if x['technical_repeat'] == 0 and x['probe_case'] in study['R_repeat_cases']]
    for repeat in (1, 2):
        write_rows(args.output_dir / f'R_repeat_{repeat}.jsonl', [dict(x, trajectory_id=x['trajectory_id'] + f':R{repeat}') for x in repeats])
    write_json(args.output_dir / 'REFINE_FINAL.json', {'planned': len(results), 'input_study_sha256': digest(prepared / 'STUDY.json'),
        'same_seed_within_source_and_technical_repeats': True, 'balanced_variant_order': True,
        'world_size': world, 'source_group_assigned_to_one_gpu': True,
        'forward_noise_added': False, 'diff_steps': 800, 'timesteps': 1000, 'source_sha256': digest(__file__),
        'model_source': inspect.getfile(Model), 'model_source_sha256': digest(inspect.getfile(Model)),
        'outputs_sha256': {p.name: digest(p) for p in args.output_dir.glob('*') if p.is_file()}})
    (args.output_dir / '_SUCCESS').touch()
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'refine'), required=True)
    for name in ('run-root', 'prepared-dir', 'b0-checkpoint', 'frozen-runtime-root', 'crysllmgen-dir', 'checkpoint'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026090817)
    args = parser.parse_args()
    prepare(args) if args.mode == 'prepare' else refine(args)
