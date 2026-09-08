"""Registered, training-only teacher action headroom before SUN policy fitting."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crystal_dlm.sun_feedback_contract import SCHEMA, composition_counts, reduced_key, rows, sha256, validate_training_feedback

VARIANTS = ('keep', 'local1', 'local4', 'all_xyz', 'full_cell')
MODEL_SHA = '573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e'
MODEL_CODE_SHA = '88c38d3fc237001e163c01adeb6296c795497f15a54067a487a9527b3859d208'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n', encoding='utf-8')


def write_rows(path, values):
    with Path(path).open('w', encoding='utf-8') as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True)+'\n')


def keyed_seed(seed, *parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, (seed, *parts))).encode()).digest()[:8], 'big') % (2**63)


def feedback_manifest(path, spec, endpoint, count):
    value = {'schema': SCHEMA, 'purpose': 'training_feedback', 'endpoint': endpoint, 'expected_requests': count,
             'paths': {'path': str(path.resolve()), 'sha256': sha256(path)},
             **{key: spec[key] for key in ['parent_preparation', 'parent_pairs', 'heldout_cohort']}}
    target = path.parent/'FEEDBACK_INPUTS.json'
    write_json(target, value)
    validate_training_feedback(rows(path), path, target, endpoint=endpoint)
    return str(target.resolve())


def materializer(args, tokenizer):
    from scripts.run_r03_integrated_body import load_frozen_runtime, frozen_imports, materialize_record
    runtime = load_frozen_runtime(args.frozen_runtime_root)
    with frozen_imports(runtime):
        process_one = runtime.module.import_process_one(args.crysllmgen_dir)
    def convert(case, body):
        task = {'sample_idx': case['case_idx'], 'ordinal': case['case_idx'],
                'attempt_id': 'sun-headroom:'+case['ancestor_id'], 'body_noise_seed': 0, 'eligible': True,
                'source_row': {'plan_state': case['plan_state']}, 'plan_state': case['plan_state'],
                'body_prompt': case['prompt'], 'schedule_sha256': None}
        with frozen_imports(runtime):
            record, graph = materialize_record(task, body, runtime=runtime, tokenizer=tokenizer, process_one=process_one)
        return graph, {'reason': record.get('reason'), 'message': record.get('message')}
    return convert


def native_teacher_graph(arrays, sample_idx):
    """Teacher-only native frame; final F800 still uses the frozen materializer."""
    import numpy as np
    from crystal_dlm.expert_edit_data import lattice_from_parameters
    from crystal_dlm.fixed_slot import SYMBOL_TO_Z
    lattice = lattice_from_parameters(arrays['lengths'], arrays['angles'])
    if not np.isfinite(lattice).all() or abs(float(np.linalg.det(lattice))) <= 1e-10:
        raise ValueError('native teacher input has a nonfinite or degenerate lattice')
    n = len(arrays['species'])
    coordinates = np.asarray(arrays['frac_coords'], dtype=np.float32)
    if coordinates.shape != (n,3) or not np.isfinite(coordinates).all():
        raise ValueError('native teacher input coordinates are incomplete')
    return {'n_atom':n, 'a_type':np.array([SYMBOL_TO_Z[x] for x in arrays['species']],dtype=np.int64),
        'x_coord':coordinates, 'length':np.asarray(arrays['lengths'],dtype=np.float32),
        'angle':np.asarray(arrays['angles'],dtype=np.float32), 'sample_idx':sample_idx,
        'edge_indices':np.array([(i,j) for i in range(n) for j in range(n)],dtype=np.int64),
        'to_jimages':np.zeros((n*n,3),dtype=np.int64)}


def prepare(args):
    import torch
    from transformers import AutoTokenizer
    report_path = args.parent_prepared/'PREPARATION_FINAL.json'
    parent_path = args.parent_prepared/'pairs_pending.jsonl'
    report = json.loads(report_path.read_text())
    if (not (args.parent_prepared/'_SUCCESS').is_file() or sha256(parent_path) != report['files_sha256'][parent_path.name]
            or sha256(args.heldout_cohort) != report['heldout_cohort_sha256']):
        raise ValueError('registered parent training preparation changed')
    if not (args.official_cache/'completion_SUCCESS').is_file():
        raise ValueError('public reference cache is incomplete')
    covered = {row['chemsys'] for row in rows(args.official_cache/'official_slim_cache.jsonl')}
    unresolved = {row['chemsys'] for row in rows(args.official_cache/'unresolved_chemsys.jsonl')}
    covered -= unresolved
    forbidden = {reduced_key(composition_counts(row['plan_state'])) for row in rows(args.heldout_cohort)
                 if row.get('body_eligible') and row.get('plan_state')}
    pools, counts = {False: [], True: []}, Counter()
    for row in rows(parent_path):
        if row['source_split'] != 'train':
            continue
        counts['all_training_parents'] += 1
        if reduced_key(composition_counts(row['plan_state'])) in forbidden:
            raise ValueError('parent training preparation overlaps heldout chemistry')
        system = '-'.join(sorted(row['plan_state']['elements']))
        if system not in covered:
            counts['no_resolved_cached_reference'] += 1
            continue
        pools[bool(row['old_geometry']['valid'])].append(row)
    if args.sources < 2 or args.sources % 2:
        raise ValueError('headroom panel has equal original geometry-valid/invalid strata')
    selected, compositions = [], set()
    for valid in [False, True]:
        ordered = sorted(pools[valid], key=lambda x: keyed_seed(args.seed, x['ancestor_id'], 'select'))
        chosen = []
        for row in ordered:
            if row['composition_key'] in compositions:
                continue
            compositions.add(row['composition_key'])
            chosen.append(row)
            if len(chosen) == args.sources//2:
                break
        if len(chosen) != args.sources//2:
            raise ValueError('insufficient unique training compositions in registered stratum')
        selected.extend(chosen)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.b0_checkpoint, local_files_only=True, trust_remote_code=True)
    from crystal_dlm.expert_edit_data import decode_body
    inverse = {int(value):key for key,value in tokenizer.get_vocab().items()}
    cases, graphs = [], {}
    for index, row in enumerate(selected):
        case = {k: row[k] for k in ['ancestor_id', 'source_row_idx', 'source_split', 'plan_state', 'prompt',
                                     'composition_key', 'num_atoms', 'old_body', 'old_geometry']}
        case.update(case_idx=index, teacher_seed=keyed_seed(args.seed, row['ancestor_id'], 'teacher200'),
                    refiner_seeds=[keyed_seed(args.seed, row['ancestor_id'], 'F800', repeat) for repeat in range(2)])
        try:
            graph = native_teacher_graph(decode_body(row['old_body'], inverse), index)
            status = {'reason': 'native_body_frame'}
        except ValueError as error:
            graph, status = None, {'reason':'native_frame_unavailable', 'message':str(error)}
        graphs[str(index)] = graph
        case['initial_graph_status'] = status
        cases.append(case)
    torch.save(graphs, args.output_dir/'graphs.pt')
    spec = {'schema': 'sun_teacher_headroom_preparation_v1', 'cases': cases, 'seed': args.seed,
            'sources': args.sources, 'variants': list(VARIANTS), 'teacher_steps': 200,
            'selection': 'training-only, public-reference-covered, original geometry strata, hash order, unique composition; no new outcome selection',
            'previous_teacher_success_required': False,
            'teacher_input_frame': 'native_body_order_and_fractional_basis_float32_no_niggli',
            'teacher_protocol_is_not_standard_reduced_F200': True,
            'coverage_counts': dict(counts), 'training_use_allowed': True, 'independent_evaluation': False,
            'graphs_sha256': sha256(args.output_dir/'graphs.pt'),
            'public_reference_sha256': sha256(args.official_cache/'official_slim_cache.jsonl')}
    for name, path in [('parent_preparation', report_path), ('parent_pairs', parent_path), ('heldout_cohort', args.heldout_cohort)]:
        spec[name] = {'path': str(path.resolve()), 'sha256': sha256(path)}
    write_json(args.output_dir/'STUDY.json', spec)
    (args.output_dir/'_SUCCESS').touch()
    print(json.dumps({'prepared_sources': len(cases), 'coverage': dict(counts), 'graph_failures': sum(g is None for g in graphs.values())}))


def candidate_bodies(old, targets, coordinates):
    """Whole valid scopes; teacher movement ranks only choose offline supervision."""
    import numpy as np
    n = (len(old)-7)//4
    result = {'keep': {'body': list(old), 'positions': [], 'sites': [], 'mode': 'none'}}
    old_xyz = np.asarray(coordinates['old'], dtype=float)
    for variant in VARIANTS[1:]:
        target_name = 'first' if variant == 'local1' else 'middle' if variant == 'local4' else 'last'
        target = targets.get(target_name)
        if target is None:
            result[variant] = {'body': None, 'positions': [], 'sites': [], 'mode': None, 'failure': 'teacher_state_not_encodable'}
            continue
        if len(target) != len(old) or any(target[p] != old[p] for p in [0]+[7+4*i for i in range(n)]):
            raise ValueError('continuous teacher changed immutable atom slots')
        if variant.startswith('local'):
            delta = (np.asarray(coordinates[target_name])-old_xyz+.5) % 1. - .5
            rank = sorted(range(n), key=lambda i: (-float((delta[i]**2).sum()), i))
            sites = sorted(rank[:min(n, 1 if variant == 'local1' else 4)])
            positions = [8+4*i+a for i in sites for a in range(3)]
            mode = 'local_xyz' if len(sites) < n else 'all_xyz'
        else:
            sites = list(range(n))
            positions = ([*range(1,7)] if variant == 'full_cell' else []) + [8+4*i+a for i in sites for a in range(3)]
            mode = variant
        body = list(old)
        for position in positions:
            body[position] = target[position]
        result[variant] = {'body': body, 'positions': positions, 'sites': sites, 'mode': mode,
                           'teacher_target': target_name, 'changed': body != old}
    return result


def generate(args):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from crystal_dlm.expert_edit_data import decode_body, quantize_arrays, physics_input
    from crystal_dlm.fixed_slot import Z_TO_SYMBOL
    from scripts.refine_dlm_with_crysllmgen import setup_crysllmgen_imports, ProposalDataset, lattices_to_params_shape, init_distributed
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    spec = json.loads((args.prepared_dir/'STUDY.json').read_text())
    if (spec['schema'] != 'sun_teacher_headroom_preparation_v1' or sha256(args.prepared_dir/'graphs.pt') != spec['graphs_sha256']
            or sha256(args.checkpoint) != MODEL_SHA):
        raise ValueError('teacher or prepared graph identity changed')
    info = init_distributed()
    rank, world, device = info['rank'], info['world_size'], info['device']
    if not torch.cuda.is_available() or not 1 <= world <= 6:
        raise ValueError('registered allocated GPUs are required')
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    if world > 1:
        torch.distributed.barrier()
    worker = args.output_dir/f'rank{rank}'
    worker.mkdir()
    tokenizer = AutoTokenizer.from_pretrained(args.b0_checkpoint, local_files_only=True, trust_remote_code=True)
    vocabulary = tokenizer.get_vocab()
    inverse = {int(v): k for k,v in vocabulary.items()}
    _, Model, Data, Loader = setup_crysllmgen_imports(args.crysllmgen_dir)
    if sha256(inspect.getfile(Model)) != MODEL_CODE_SHA:
        raise ValueError('teacher imported a different CrysLLMGen model implementation')
    model = Model(1000, 'train').to(device)
    model.device = device
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved['model'] if 'model' in saved else saved)
    del saved
    model.eval()
    if model.decoder.edge_style != 'fc' or model.decoder.pred_type is not False:
        raise ValueError('native-frame teacher requires internally rebuilt FC edges and fixed atom types')
    graphs = torch.load(args.prepared_dir/'graphs.pt', map_location='cpu', weights_only=False)
    output_cases, inputs = [], []
    for case in spec['cases']:
        if case['case_idx'] % world != rank:
            continue
        old_arrays = decode_body(case['old_body'], inverse)
        targets, coordinates, diagnostics = {}, {'old': old_arrays['frac_coords']}, {}
        graph = graphs[str(case['case_idx'])]
        if graph is not None:
            seed = case['teacher_seed']
            random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            batch = next(iter(Loader(ProposalDataset([graph], Data), batch_size=1, shuffle=False))).to(device)
            if (not torch.equal(batch.frac_coords.cpu(), torch.tensor(old_arrays['frac_coords'],dtype=torch.float32))
                    or not torch.equal(batch.lengths.cpu().reshape(-1), torch.tensor(old_arrays['lengths'],dtype=torch.float32))
                    or not torch.equal(batch.angles.cpu().reshape(-1), torch.tensor(old_arrays['angles'],dtype=torch.float32))):
                raise ValueError('teacher input changed OLD coordinate frame or numeric values')
            initial_lattice = sys.modules[Model.__module__].lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            if not bool(torch.isfinite(initial_lattice).all()) or not bool((torch.linalg.det(initial_lattice).abs() > 1e-10).all()):
                raise ValueError('actual float32 teacher lattice is nonfinite or degenerate')
            with torch.no_grad():
                final, trajectory = model.sample(batch, diff_steps=200)
            if (not torch.equal(trajectory['all_frac_coords'][0], batch.frac_coords.remainder(1.))
                    or not torch.equal(trajectory['all_lattices'][0], initial_lattice)):
                raise ValueError('teacher trajectory does not start at OLD')
            sparse = {key: value.detach().cpu() for key, value in trajectory.items() if key not in ['all_frac_coords', 'all_lattices']}
            sparse['time_values'] = [200,199,150,0]
            sparse['all_frac_coords'] = trajectory['all_frac_coords'][[0,1,50,200]].detach().cpu()
            sparse['all_lattices'] = trajectory['all_lattices'][[0,1,50,200]].detach().cpu()
            torch.save(sparse, worker/f'teacher-{case["case_idx"]:04d}.pt')
            species = [Z_TO_SYMBOL[int(z)] for z in final['atom_types'].reshape(-1)]
            if species != old_arrays['species']:
                raise ValueError('teacher changed OLD atom order')
            for name, offset in [('first',1), ('middle',50), ('last',200)]:
                lengths, angles = lattices_to_params_shape(trajectory['all_lattices'][offset].detach().cpu())
                arrays = {'lengths': lengths.reshape(-1,3)[0].tolist(), 'angles': angles.reshape(-1,3)[0].tolist(),
                    'frac_coords': trajectory['all_frac_coords'][offset].detach().cpu().reshape(-1,3).tolist(), 'species': species}
                try:
                    targets[name], decoded, diagnostics[name] = quantize_arrays(arrays, vocabulary)
                    coordinates[name] = decoded['frac_coords']
                except ValueError as error:
                    diagnostics[name] = {'error': str(error)}
                    targets[name] = None
            del trajectory, final
        else:
            diagnostics['initial'] = {'error': 'original_graph_unavailable'}
        candidates = candidate_bodies(case['old_body'], targets, coordinates)
        for variant, candidate in candidates.items():
            body = candidate['body']
            arrays = decode_body(body, inverse) if body is not None else None
            text = ''.join(inverse[token] for token in body) if body is not None else None
            value = physics_input(f'sun-teacher-native:{variant}:{case["case_idx"]}', case['ancestor_id'], case['source_row_idx'], 'train', 'native', arrays, text)
            value.update(purpose='training_feedback', sample_idx=case['case_idx'], evaluation_ordinal=case['case_idx'],
                         probe_case=case['case_idx'], probe_variant=variant, declared_composition=composition_counts(case['plan_state']))
            inputs.append(value)
        trace_path = worker/f'teacher-{case["case_idx"]:04d}.pt'
        output_cases.append({**case, 'candidates': candidates, 'teacher_quantization': diagnostics,
            'teacher_trace': {'path':str(trace_path.resolve()),'sha256':sha256(trace_path)} if trace_path.is_file() else None})
        write_json(worker/'PROGRESS.json', {'completed_sources': len(output_cases), 'rank': rank})
        print(json.dumps({'rank':rank, 'completed_sources':len(output_cases)}), flush=True)
    write_json(worker/'CASES.json', output_cases)
    write_rows(worker/'inputs.jsonl', inputs)
    if world > 1:
        torch.distributed.barrier()
    if rank == 0:
        cases = sorted([c for i in range(world) for c in json.loads((args.output_dir/f'rank{i}/CASES.json').read_text())], key=lambda c:c['case_idx'])
        if [c['case_idx'] for c in cases] != list(range(spec['sources'])):
            raise ValueError('teacher proposal collection lost original sources')
        all_inputs = []
        for i in range(world):
            all_inputs.extend(rows(args.output_dir/f'rank{i}/inputs.jsonl'))
        manifests = []
        for variant in VARIANTS:
            arm = args.output_dir/'native'/variant
            arm.mkdir(parents=True)
            values = sorted([x for x in all_inputs if x['probe_variant'] == variant], key=lambda x:x['sample_idx'])
            path = arm/'inputs.jsonl'
            write_rows(path, values)
            manifests.append(feedback_manifest(path, spec, 'native', spec['sources']))
        jobs = []
        for repeat in range(2):
            for case in cases:
                shift = case['case_idx'] % len(VARIANTS)
                order = VARIANTS[shift:]+VARIANTS[:shift]
                if repeat:
                    order = order[::-1]
                for variant in order:
                    jobs.append({'sample_idx':len(jobs), 'case_idx':case['case_idx'], 'variant':variant,
                                 'technical_repeat':repeat, 'refiner_seed':case['refiner_seeds'][repeat]})
        output = {**spec, 'schema':'sun_teacher_candidates_v1', 'cases':cases, 'schedule':jobs,
                  'R_repeat_cases':[], 'candidate_graphs_ready':False,
                  'native_feedback_manifests':manifests, 'repeat_semantics':'independent_refiner_noise_seeds',
                  'teacher_checkpoint_sha256':MODEL_SHA, 'teacher_model_source_sha256':MODEL_CODE_SHA,
                  'source_sha256':sha256(__file__)}
        write_json(args.output_dir/'STUDY.json', output)
        (args.output_dir/'_SUCCESS').touch()
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def materialize_candidates(args):
    """Keep the frozen CIF/Niggli graph importer separate from teacher imports."""
    import torch
    from transformers import AutoTokenizer
    spec_path = args.prepared_dir/'STUDY.json'
    spec = json.loads(spec_path.read_text())
    if spec['schema'] != 'sun_teacher_candidates_v1' or not (args.prepared_dir/'_SUCCESS').is_file():
        raise ValueError('teacher candidate generation is incomplete')
    input_rows = {}
    for manifest_path in spec['native_feedback_manifests']:
        manifest = json.loads(Path(manifest_path).read_text())
        path = Path(manifest['paths']['path'])
        values = rows(path)
        validate_training_feedback(values, path, manifest_path, endpoint='native')
        input_rows[path.parent.name] = {row['group_id']: row for row in values}
    tokenizer = AutoTokenizer.from_pretrained(args.b0_checkpoint, local_files_only=True, trust_remote_code=True)
    inverse = {int(value):token for token,value in tokenizer.get_vocab().items()}
    convert = materializer(args, tokenizer)
    graphs = {}
    for case in spec['cases']:
        for variant, candidate in case['candidates'].items():
            text = ''.join(inverse[token] for token in candidate['body']) if candidate['body'] is not None else None
            if text != input_rows[variant][case['ancestor_id']]['body']:
                raise ValueError('candidate graph and native physical input refer to different token bodies')
            graph, status = convert(case, candidate['body']) if candidate['body'] is not None else (None, {'reason':'teacher_state_not_encodable'})
            graphs[f'{case["case_idx"]}:{variant}'] = graph
            candidate['graph_status'] = status
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(graphs, args.output_dir/'graphs.pt')
    spec.update(schema='sun_teacher_headroom_v1', candidate_graphs_ready=True,
                graphs_sha256=sha256(args.output_dir/'graphs.pt'), candidate_study_sha256=sha256(spec_path),
                candidate_study_path=str(spec_path.resolve()), endpoint_graph_importer_root=str(args.crysllmgen_dir.resolve()))
    write_json(args.output_dir/'STUDY.json', spec)
    (args.output_dir/'_SUCCESS').touch()


def validate_refinement_receipt(spec, study_path, refined_dir, values):
    report = json.loads((refined_dir/'REFINE_FINAL.json').read_text())
    if (report.get('input_study_sha256') != sha256(study_path) or report.get('training_use_allowed') is not True
            or report.get('repeat_semantics') != 'independent_refiner_noise_seeds'
            or report.get('deterministic_algorithms_enabled') is not True
            or report.get('deterministic_algorithms_requested') is not True
            or report.get('cublas_workspace_config') != ':4096:8'
            or report.get('diff_steps') != 800 or report.get('timesteps') != 1000
            or report.get('forward_noise_added') is not False
            or report.get('outputs_sha256',{}).get('refined_inputs.jsonl') != sha256(refined_dir/'refined_inputs.jsonl')):
        raise ValueError('refinement receipt does not bind the declared deterministic training study')
    if len(values) != len(spec['schedule']) or report.get('planned') != len(values):
        raise ValueError('refinement denominator changed')
    observed = {x['probe_job_index']:x for x in values}
    if len(observed) != len(values):
        raise ValueError('duplicate refinement job observation')
    for job in spec['schedule']:
        row = observed[job['sample_idx']]
        case = spec['cases'][job['case_idx']]
        expected = {'probe_case':job['case_idx'], 'probe_variant':job['variant'], 'technical_repeat':job['technical_repeat'],
                    'refiner_noise_seed_index':job['technical_repeat'], 'refiner_seed':job['refiner_seed'],
                    'group_id':case['ancestor_id'], 'source_row_idx':case['source_row_idx'],
                    'source_split':'train', 'purpose':'training_feedback', 'endpoint':'tau800'}
        if any(row.get(key) != value for key,value in expected.items()):
            raise ValueError('refinement source, alternative or seed differs from its registered schedule')


def export_refined(args):
    spec = json.loads((args.prepared_dir/'STUDY.json').read_text())
    if spec['schema'] != 'sun_teacher_headroom_v1' or not (args.refined_dir/'_SUCCESS').is_file():
        raise ValueError('training feedback refinement is incomplete')
    values = rows(args.refined_dir/'refined_inputs.jsonl')
    validate_refinement_receipt(spec, args.prepared_dir/'STUDY.json', args.refined_dir, values)
    if len(values) != spec['sources']*len(VARIANTS)*2:
        raise ValueError('refinement lost declared alternatives')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = []
    for repeat in range(2):
        for variant in VARIANTS:
            arm = args.output_dir/f'seed{repeat}'/variant
            arm.mkdir(parents=True)
            selected = sorted([x for x in values if x['probe_variant'] == variant and x['technical_repeat'] == repeat], key=lambda x:x['probe_case'])
            selected = [dict(x, sample_idx=x['probe_case'], evaluation_ordinal=x['probe_case']) for x in selected]
            path = arm/'inputs.jsonl'
            write_rows(path, selected)
            outputs.append(feedback_manifest(path, spec, 'tau800', spec['sources']))
    write_json(args.output_dir/'EXPORT_FINAL.json', {'manifests':outputs, 'source_sha256':sha256(args.refined_dir/'refined_inputs.jsonl')})
    (args.output_dir/'_SUCCESS').touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare','generate','materialize','export-refined'], required=True)
    for name in ['parent-prepared','heldout-cohort','official-cache','prepared-dir','refined-dir','b0-checkpoint','frozen-runtime-root','crysllmgen-dir','checkpoint']:
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sources', type=int, default=64)
    parser.add_argument('--seed', type=int, default=2026090841)
    args = parser.parse_args()
    {'prepare':prepare, 'generate':generate, 'materialize':materialize_candidates, 'export-refined':export_refined}[args.mode](args)


if __name__ == '__main__':
    main()
