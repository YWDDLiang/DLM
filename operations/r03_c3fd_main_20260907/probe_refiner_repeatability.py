"""Compare frozen/current refiner inputs and repeated sampling on the same GPU.

This diagnostic does not select structures or alter an editor or refiner policy.
Each rank repeats the same registered input, with the original request seed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal-graphs', type=Path, required=True)
    parser.add_argument('--input-receipt', type=Path, required=True)
    parser.add_argument('--frozen-refiner-module', type=Path, required=True)
    parser.add_argument('--frozen-refiner-sha256', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--crysllmgen-dir', type=Path, required=True)
    parser.add_argument('--sample-index', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from scripts.refine_dlm_with_crysllmgen import ProposalDataset, setup_crysllmgen_imports

    receipt = json.loads(args.input_receipt.read_text())
    if digest(args.proposal_graphs) != receipt['output_hashes']['proposal_graphs.pt']:
        raise ValueError('replay graphs changed after preparation')
    if digest(args.frozen_refiner_module) != args.frozen_refiner_sha256:
        raise ValueError('the original frozen refiner module changed')
    if digest(args.checkpoint) != '573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e':
        raise ValueError('the model494 checkpoint changed')
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('this replay requires an allocated Slurm GPU')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    spec = importlib.util.spec_from_file_location('frozen_refiner_for_replay', args.frozen_refiner_module)
    frozen = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = frozen
    spec.loader.exec_module(frozen)
    _, CSPDiffusion, Data, DataLoader = setup_crysllmgen_imports(args.crysllmgen_dir)
    graphs = torch.load(args.proposal_graphs, map_location='cpu', weights_only=False)
    matches = [row for row in graphs if row['sample_idx'] == args.sample_index]
    if len(matches) != 1:
        raise ValueError('replay needs exactly one registered sample index')
    graph = matches[0]
    seed = graph['refiner_noise_seed']
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('replay lacks its original request seed')
    model = CSPDiffusion(1000, 'train').to(device)
    model.device = device
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    del checkpoint
    model.eval()
    output = args.output_dir / f'rank{rank}'
    output.mkdir(parents=True, exist_ok=False)
    results, inputs, rngs = {}, {}, {}
    fields = ('num_atoms', 'lengths', 'angles', 'frac_coords', 'atom_types',
              'edge_index', 'to_jimages', 'batch', 'ptr')

    def rng_hash(value):
        return hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()

    with torch.no_grad():
        for name in ('frozen_first', 'current', 'frozen_repeat'):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            dataset = (ProposalDataset([graph], Data, seed_from_graph_field='refiner_noise_seed')
                       if name == 'current' else frozen.ProposalDataset([graph], Data))
            batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False))).to(device)
            inputs[name] = {key: getattr(batch, key).detach().cpu().clone() for key in fields}
            rngs[name] = {'cpu_before_sample': rng_hash(torch.get_rng_state()),
                          'cuda_before_sample': rng_hash(torch.cuda.get_rng_state(device))}
            sampled, _ = model.sample(batch, diff_steps=800)
            torch.cuda.synchronize(device)
            results[name] = {key: value.detach().cpu().clone() for key, value in sampled.items()
                             if isinstance(value, torch.Tensor)}
            rngs[name].update(cpu_after_sample=rng_hash(torch.get_rng_state()),
                              cuda_after_sample=rng_hash(torch.cuda.get_rng_state(device)))
            torch.save(results[name], output / (name + '.pt'))

    def compare(left, right):
        result = {}
        for key in sorted(set(left) & set(right)):
            a, b = left[key], right[key]
            item = {'shape_equal': a.shape == b.shape, 'exact': torch.equal(a, b)}
            if a.shape == b.shape and a.numel():
                delta = (a.to(torch.float64) - b.to(torch.float64)).abs()
                item.update(max_abs=float(delta.max()), mean_abs=float(delta.mean()))
                if key == 'frac_coords':
                    periodic = torch.remainder(delta + .5, 1.) - .5
                    item['max_periodic_abs'] = float(periodic.abs().max())
            result[key] = item
        return result

    source_files = {}
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file and str(file).endswith('.py'):
            path = Path(file).resolve()
            if args.crysllmgen_dir.resolve() in path.parents:
                source_files[str(path)] = digest(path)
    report = {'sample_idx': args.sample_index, 'seed': seed, 'rank': rank,
              'device': str(device), 'gpu_name': torch.cuda.get_device_name(device),
              'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
              'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
              'tf32_cudnn': torch.backends.cudnn.allow_tf32,
              'cudnn_benchmark': torch.backends.cudnn.benchmark,
              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'input_comparison': compare(inputs['frozen_first'], inputs['current']),
              'frozen_vs_current': compare(results['frozen_first'], results['current']),
              'frozen_self_repeat': compare(results['frozen_first'], results['frozen_repeat']),
              'rngs': rngs, 'crysllmgen_source_files_sha256': source_files,
              'source_sha256': digest(__file__), 'frozen_module_sha256': args.frozen_refiner_sha256,
              'output_sha256': {p.name: digest(p) for p in output.glob('*.pt')}}
    (output / 'PROBE_FINAL.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    (output / '_SUCCESS').touch()
    print(json.dumps({'rank': rank, 'input_equal': all(x['exact'] for x in report['input_comparison'].values()),
                      'frozen_vs_current': report['frozen_vs_current'],
                      'frozen_self_repeat': report['frozen_self_repeat']}), flush=True)


if __name__ == '__main__':
    main()
