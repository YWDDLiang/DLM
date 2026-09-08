#!/usr/bin/env python3
"""Immutable-cohort stages for token revision with a measured draft SUN gate."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from crystal_dlm.post_refine_contract import SCHEMA, derived_seed, fingerprint, recovery_canvas


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True, allow_nan=False) + '\n')


def load_config(path):
    spec = json.loads(Path(path).read_text(encoding='utf-8'))
    if spec.get('schema') != SCHEMA:
        raise ValueError('unknown token revision configuration')
    spec['_config_sha256'] = file_hash(path)
    spec['_config_path'] = str(Path(path).resolve())
    return spec


def prepare(spec):
    root = Path(spec['run_root']) / 'cohort'
    if (root / '_SUCCESS').is_file():
        if json.loads((root / 'MANIFEST.json').read_text())['config_sha256'] != spec['_config_sha256']:
            raise ValueError('cohort belongs to another registration')
        return
    rows = read_rows(spec['assets']['cohort'])
    ledger = read_rows(spec['assets']['seed_ledger'])
    if len(rows) != 1200 or len(ledger) != 1200:
        raise ValueError('saved Plan cohort or seed ledger cardinality changed')
    for name in ('cohort', 'seed_ledger', 'control_body', 'candidate_body', 'control_graphs', 'candidate_graphs'):
        if file_hash(spec['assets'][name]) != spec['source_sha256'][name]:
            raise ValueError('saved source changed: ' + name)
    valid, excluded = [], []
    for original, row in enumerate(rows):
        if row.get('cohort_ordinal') != original:
            raise ValueError('source Plan ordinals changed')
        if row.get('body_eligible') is not True:
            excluded.append({'original_ordinal': original, 'reason': row.get('ineligible_reason')})
            continue
        plan = row['plan_state']
        if (not 1 <= plan['N'] <= 20 or sum(plan['counts']) != plan['N']
                or plan.get('rich_field_valid') is not True or plan.get('plan_end_marker_present') is not True):
            raise ValueError('an additional Plan hard error needs explicit accounting')
        seed_row = ledger[original]
        if seed_row['ordinal'] != original:
            raise ValueError('original seed ledger order differs')
        valid.append(dict(row, sample_idx=original, original_ordinal=original,
                          body_noise_seed=seed_row['body_noise_seed'],
                          refiner_noise_seed=seed_row['refiner_noise_seed']))
    if [row['original_ordinal'] for row in excluded] != spec['excluded_original_ordinals']:
        raise ValueError('Plan exclusion set changed')
    requested = int(spec['requests'])
    if not 1 <= requested <= len(valid):
        raise ValueError('requested cohort exceeds the fixed valid sources')
    order = sorted(valid, key=lambda r: fingerprint({'selection': spec['selection_tag'],
                                                     'original_ordinal': r['original_ordinal']}))
    selected = sorted(order[:requested], key=lambda r: r['original_ordinal'])
    for index, row in enumerate(selected):
        row['evaluation_ordinal'] = index
    root.mkdir(parents=True, exist_ok=True)
    write_rows(root / 'plans.jsonl', selected)
    write_rows(root / 'excluded_plans.jsonl', excluded)
    write_json(root / 'MANIFEST.json', {'schema': SCHEMA, 'config_sha256': spec['_config_sha256'],
              'source_count': len(rows), 'valid_count': len(valid), 'selected_count': len(selected),
              'original_ordinals': [r['original_ordinal'] for r in selected],
              'plans_sha256': file_hash(root / 'plans.jsonl'), 'source_sha256': spec['source_sha256']})
    (root / '_SUCCESS').touch()


def load_refiner(spec, device):
    import torch
    api = refiner_api()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = Path(spec['assets']['model494'])
    if file_hash(checkpoint) != '573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e':
        raise ValueError('model494 checkpoint identity changed')
    _, CSPDiffusion, Data, DataLoader = api.setup_crysllmgen_imports(Path(spec['assets']['crysllmgen']))
    model = CSPDiffusion(1000, 'train').to(device)
    model.device = device
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved['model'] if 'model' in saved else saved)
    model.eval()
    return model, Data, DataLoader


def refiner_api():
    name = '_post_refine_bound_refiner'
    if name not in sys.modules:
        binding = importlib.util.spec_from_file_location(name, ROOT / 'src/scripts/refine_dlm_with_crysllmgen.py')
        module = importlib.util.module_from_spec(binding)
        sys.modules[name] = module
        binding.loader.exec_module(module)
    return sys.modules[name]


def refine_one(graph, *, model, Data, DataLoader, seed, steps=800):
    import torch
    api = refiner_api()
    value = dict(graph, refiner_noise_seed=int(seed))
    batch = next(api.frozen_seeded_batches([value], Data, DataLoader, 'refiner_noise_seed')).to(model.device)
    with torch.no_grad():
        result, _ = model.sample(batch, diff_steps=int(steps))
        lengths, angles = api.lattices_to_params_shape(result['lattices'])
    if not all(bool(torch.isfinite(t).all()) for t in (result['frac_coords'], result['lattices'], lengths, angles)):
        raise FloatingPointError('nonfinite_refiner_geometry')
    return {'frac_coords': result['frac_coords'].detach().cpu().tolist(),
            'atom_types': result['atom_types'].detach().cpu().reshape(-1).tolist(),
            'lattice_matrix': result['lattices'].detach().cpu().reshape(3, 3).tolist(),
            'lengths': lengths.detach().cpu().reshape(3).tolist(),
            'angles': angles.detach().cpu().reshape(3).tolist(),
            'seed': int(seed), 'diffusion_steps': int(steps), 'geometry_decoder_calls': 2 * int(steps)}


def structure_from_refined(value):
    from pymatgen.core import Lattice, Structure
    # Match the existing F800 endpoint readout for every arm. Keep the raw
    # matrix separately so canonicalization cannot masquerade as new sampling.
    lattice = Lattice.from_parameters(*value['lengths'], *value['angles'])
    return Structure(lattice, value['atom_types'], value['frac_coords']).as_dict()


def physics_record(plan, *, state_id, structure=None, body=None, reason=None):
    training = plan.get('source_split') == 'train'
    return {'trajectory_id': state_id, 'sample_idx': plan['original_ordinal'],
            'original_ordinal': plan['original_ordinal'], 'evaluation_ordinal': plan['evaluation_ordinal'],
            'success': structure is not None or bool(body), 'structure': structure, 'body': body,
            'endpoint': 'native', 'source_split': 'train' if training else 'evaluation',
            'purpose': 'training_feedback' if training else 'evaluation',
            'group_id': plan.get('ancestor_id'), 'source_row_idx': plan.get('source_row_idx'),
            'declared_composition': dict(zip(plan['plan_state']['elements'], plan['plan_state']['counts'])),
            'reason': reason}


def constructor_api():
    name = '_post_refine_bound_constructor'
    if name not in sys.modules:
        binding = importlib.util.spec_from_file_location(name, ROOT / 'src/scripts/run_r03_integrated_body.py')
        module = importlib.util.module_from_spec(binding)
        sys.modules[name] = module
        binding.loader.exec_module(module)
    return sys.modules[name]


def construct(spec, shard, shards):
    import torch
    from crystal_dlm import r03_geometry_bridge as bridge
    from crystal_dlm.r03_physics_transfer import build_repair_constraints, geometry_support_report
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available():
        raise RuntimeError('DLM construction requires its Slurm GPU allocation')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root, native = Path(spec['run_root']), constructor_api()
    plans = read_rows(root / 'cohort/plans.jsonl')
    runtime = native.load_frozen_runtime(Path(spec['assets']['frozen_runtime']))
    checkpoint = spec['assets'].get('generator_checkpoint', spec['assets']['b0_checkpoint'])
    if checkpoint == spec['assets']['b0_checkpoint']:
        checkpoint_identity = native.validate_b0_checkpoint(Path(checkpoint))
    else:
        marker = Path(checkpoint) / 'RSI_TRAINING_DONE.json'
        checkpoint_identity = json.loads(marker.read_text())
        if checkpoint_identity.get('optimizer_steps', 0) < 1:
            raise ValueError('updated generator has no actual optimizer steps')
        for name, digest in checkpoint_identity['checkpoint_files'].items():
            if file_hash(Path(checkpoint) / name) != digest:
                raise ValueError('updated generator checkpoint changed: ' + name)
    with native.frozen_imports(runtime):
        tasks = native.prepare_tasks(plans, runtime, seed=17029)
        api = runtime.module
        model, tokenizer = api.load_model_and_tokenizer(spec['assets']['base_model'],
            checkpoint, torch.device('cuda', local_rank))
        tokenizer_identity = api.assert_body_tokenizer_identity(tokenizer, expected_vocab_sha256=native.B0_VOCAB_SHA256)
        constraints = api.build_dynamic_lightweight_constraints(tokenizer, duplicate_coordinate_mask=True,
            lattice_volume_mask=True, min_lattice_rad=1e-4)
        process_one = api.import_process_one(Path(spec['assets']['crysllmgen']))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    repair_constraints = build_repair_constraints(tokenizer)
    complete_geometry = lambda body: geometry_support_report(body, constraints=repair_constraints)
    source_signature = fingerprint({'constructor': file_hash(ROOT / 'src/scripts/run_r03_integrated_body.py'),
        'geometry_bridge': file_hash(ROOT / 'src/crystal_dlm/r03_geometry_bridge.py'),
        'support': file_hash(ROOT / 'src/crystal_dlm/r03_physics_transfer.py'),
        'config': spec['_config_sha256'], 'tokenizer': tokenizer_identity, 'checkpoint': checkpoint_identity})
    directory = root / 'construction'
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / f'model_{shard}.json', {'source_signature': source_signature,
        'checkpoint': checkpoint_identity, 'tokenizer': tokenizer_identity,
        'runtime': runtime.provenance, 'current_source': str(ROOT), 'shard': shard, 'shards': shards})
    forwards = {'count': 0}
    def count_forward(_model, _inputs):
        forwards['count'] += 1
    hook = model.register_forward_pre_hook(count_forward)
    started, completed = time.monotonic(), 0
    for index in range(shard, len(plans), shards):
        plan, task = plans[index], tasks[index]
        original = plan['original_ordinal']
        path = directory / 'records' / f'{original:04d}.json'
        if path.is_file():
            existing = json.loads(path.read_text())
            if existing['source_signature'] != source_signature:
                raise ValueError('construction resume has different source or configuration')
            continue
        before_calls = forwards['count']
        graph = None
        try:
            with torch.no_grad(), native.frozen_imports(runtime):
                suffix, metadata = native.construct_with_recovery(model, tokenizer, task, runtime,
                    constraints=constraints, geometry_api=bridge, complete_geometry=complete_geometry,
                    recovery_transform=recovery_canvas,
                    recovery_seed=derived_seed(str(task['body_noise_seed']), 'construction_recovery', 1),
                    max_recoveries=spec['policy']['construction_recoveries'])
                ids = suffix[0].tolist()
                generated, graph = native.materialize_record(task, ids, runtime=runtime,
                                                            tokenizer=tokenizer, process_one=process_one)
            body = generated['text']
            record = physics_record(plan, state_id=f"{spec['run_id']}:raw0:{original}", body=body)
            record['body_token_ids'] = ids
            record['body_prompt'] = task['body_prompt']
            trace = {'constructor': metadata, 'materialization_status': generated['status'],
                     'materialization_reason': generated.get('reason')}
        except bridge.GeometryNoLegalSupport as error:
            record = physics_record(plan, state_id=f"{spec['run_id']}:raw0:{original}",
                                    reason='construction_geometry_exhausted')
            trace = {'construction_failure': error.to_dict()}
        trace['DLM_forward_calls'] = forwards['count'] - before_calls
        if graph is not None:
            graph = dict(graph, sample_idx=original, refiner_noise_seed=plan['refiner_noise_seed'])
            graph_path = directory / 'graphs' / f'{original:04d}.pt'
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = graph_path.with_suffix('.tmp')
            torch.save(graph, temporary)
            temporary.replace(graph_path)
        write_json(path, {'source_signature': source_signature, 'config_sha256': spec['_config_sha256'],
                          'record': record, 'trace': trace, 'graph_available': graph is not None})
        completed += 1
        if completed % 5 == 0:
            progress = {'shard': shard, 'new_records': completed, 'forward_calls': forwards['count'],
                        'elapsed_seconds': time.monotonic() - started, 'latest_original_ordinal': original}
            write_json(directory / f'progress_{shard}.json', progress)
            print(json.dumps(progress), flush=True)
    hook.remove()
    write_json(directory / f'worker_{shard}_DONE.json', {'source_signature': source_signature,
        'new_records': completed, 'forward_calls': forwards['count'], 'elapsed_seconds': time.monotonic() - started})


def baselines(spec, shard, shards):
    import torch
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available():
        raise RuntimeError('refinement requires its Slurm GPU allocation')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    root = Path(spec['run_root'])
    plans = read_rows(root / 'cohort/plans.jsonl')
    model, Data, DataLoader = load_refiner(spec, torch.device('cuda', local_rank))
    started, count = time.monotonic(), 0
    for arm, asset in (('H1A2', 'control_graphs'), ('R03', 'candidate_graphs')):
        entries = torch.load(spec['assets'][asset], map_location='cpu', weights_only=False)
        by_id = {int(row['ordinal']): row['graph'] for row in entries}
        output = root / 'baselines' / arm
        for plan in plans[shard::shards]:
            original = plan['original_ordinal']
            path = output / 'records' / f'{original:04d}.json'
            if path.is_file():
                saved = json.loads(path.read_text())
                if saved['config_sha256'] != spec['_config_sha256']:
                    raise ValueError('baseline resume belongs to another configuration')
                continue
            state_id = f"{spec['run_id']}:{arm}:F:{original}"
            raw = None
            if original not in by_id:
                record = physics_record(plan, state_id=state_id, reason='saved_draft_graph_missing')
            else:
                graph = dict(by_id[original], sample_idx=original)
                try:
                    raw = refine_one(graph, model=model, Data=Data, DataLoader=DataLoader,
                                     seed=plan['refiner_noise_seed'], steps=800)
                    record = physics_record(plan, state_id=state_id, structure=structure_from_refined(raw))
                except (ValueError, FloatingPointError) as error:
                    record = physics_record(plan, state_id=state_id, reason='refiner_output:' + str(error))
            write_json(path, {'config_sha256': spec['_config_sha256'], 'record': record, 'raw_refiner_output': raw})
            count += 1
            if count % 10 == 0:
                progress = {'arm': arm, 'shard': shard, 'new_records': count,
                            'seconds': time.monotonic() - started, 'latest_original_ordinal': original}
                write_json(root / 'baselines' / f'progress_{shard}.json', progress)
                print(json.dumps(progress), flush=True)
    write_json(root / 'baselines' / f'worker_{shard}_DONE.json',
               {'config_sha256': spec['_config_sha256'], 'new_records': count,
                'elapsed_seconds': time.monotonic() - started})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare', 'baselines', 'construct'), required=True)
    parser.add_argument('--shard', type=int, default=None)
    parser.add_argument('--shards', type=int, default=None)
    args = parser.parse_args(argv)
    args.shard = int(os.environ.get('RANK', '0')) if args.shard is None else args.shard
    args.shards = int(os.environ.get('WORLD_SIZE', '1')) if args.shards is None else args.shards
    if not 0 <= args.shard < args.shards:
        parser.error('invalid disjoint worker shard')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    spec = load_config(args.config)
    if args.stage == 'prepare':
        prepare(spec)
    elif args.stage == 'baselines':
        baselines(spec, args.shard, args.shards)
    else:
        construct(spec, args.shard, args.shards)


if __name__ == '__main__':
    main()
