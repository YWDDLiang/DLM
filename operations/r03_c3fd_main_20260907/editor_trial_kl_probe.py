"""Measure whole-population reference KL and teacher CE without weight updates."""
from collections import defaultdict
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE/'src'))
from scripts.run_post_refine_cycle import read_rows, write_json, file_hash


def moments(values):
    import numpy as np
    x = np.asarray(values, dtype=float)
    if not len(x): return dict(count=0)
    return dict(count=len(x), mean=float(x.mean()), minimum=float(x.min()), maximum=float(x.max()),
                median=float(np.median(x)), p90=float(np.quantile(x, .9)), p95=float(np.quantile(x, .95)),
                p99=float(np.quantile(x, .99)))


def row_values(live, reference, size):
    import torch
    sums = {k: torch.zeros(size) for k in ('KL', 'CE', 'reference_CE', 'tokens')}
    for (r, p, target), (qr, q, qt) in zip(live, reference, strict=True):
        r, p, target = r.cpu(), p.cpu(), target.cpu()
        if not torch.equal(r, qr) or not torch.equal(target, qt): raise ValueError('KL probe views changed')
        terms = dict(KL=(q.exp()*(q-p)).sum(-1), CE=-p.gather(1, target[:, None]).squeeze(1),
                     reference_CE=-q.gather(1, target[:, None]).squeeze(1), tokens=torch.ones(len(r)))
        for k, value in terms.items(): sums[k].scatter_add_(0, r, value)
    if bool((sums['tokens'] == 0).any()): raise ValueError('KL probe missed target tokens')
    return [{**{k: float(sums[k][i]/sums['tokens'][i]) for k in ('KL', 'CE', 'reference_CE')},
             'tokens': int(sums['tokens'][i])} for i in range(size)]


def main(root):
    import torch
    import torch.distributed as dist
    from crystal_dlm.expert_edit import load_editor_model, ExpertEditObjective
    from crystal_dlm.editor_t2t import dense_vectors
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available(): raise RuntimeError('KL probe requires Slurm GPUs')
    rank = int(os.environ.get('RANK', '0')); world = int(os.environ.get('WORLD_SIZE', '1'))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))
    torch.cuda.set_device(device); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    if world > 1: dist.init_process_group('nccl')
    root = Path(root); spec = json.loads((root/'KL_PROBE_PLAN.json').read_text())
    path = root/'data/E.jsonl'
    if file_hash(path) != spec['data_sha256']: raise ValueError('KL diagnostic data changed')
    all_rows = [r for r in read_rows(path) if r.get('content_target_tokens')]
    if any(r.get('trial_split') != 'train' for r in all_rows): raise ValueError('KL diagnostic must not inspect held-out targets')
    rows = all_rows[rank::world]; batches = [rows[i:i+8] for i in range(0, len(rows), 8)]
    started = time.monotonic(); ref_batches = {}
    model, tok = load_editor_model(spec['base_model'], spec['reference_checkpoint'], device)
    objective = ExpertEditObjective(tok, device, temperature=.7)
    model.eval()
    with torch.no_grad():
        for masked in (False, True):
            for i, batch in enumerate(batches):
                vectors = dense_vectors(model, tok, batch, objective, masked=masked)
                ref_batches[(masked, i)] = [(r.cpu(), p.cpu(), t.cpu()) for r, p, t in vectors]
                del vectors
    del model; gc.collect(); torch.cuda.empty_cache()
    result = []
    for arm, checkpoint in spec['candidate_checkpoints'].items():
        model, tok = load_editor_model(spec['base_model'], checkpoint, device); model.eval()
        with torch.no_grad():
            for masked in (False, True):
                for i, batch in enumerate(batches):
                    vectors = dense_vectors(model, tok, batch, objective, masked=masked)
                    values = row_values(vectors, ref_batches[(masked, i)], len(batch))
                    for row, value in zip(batch, values, strict=True):
                        value.update(pair_id=row['pair_id'], ordinal=row['source_ordinal'], arm=arm,
                            view='M2T' if masked else 'T2T', content_kind=row['content_kind'],
                            CE_improvement=value['reference_CE']-value['CE'])
                        result.append(value)
                    del vectors
        del model; gc.collect(); torch.cuda.empty_cache()
        if rank == 0: print(json.dumps(dict(arm=arm, seconds=time.monotonic()-started)), flush=True)
    out = root/'KL_audit'; out.mkdir(exist_ok=True)
    write_json(out/f'PROBE_rank{rank}.json', dict(rank=rank, world=world, rows=result, seconds=time.monotonic()-started))
    if world > 1: dist.barrier()
    if rank == 0:
        merged = [row for i in range(world) for row in json.loads((out/f'PROBE_rank{i}.json').read_text())['rows']]
        grouped = defaultdict(list)
        for row in merged:
            grouped[(row['arm'], row['view'], 'all')].append(row)
            grouped[(row['arm'], row['view'], row['content_kind'])].append(row)
        summaries = {}
        for key, group in grouped.items():
            summary = {k: moments([r[k] for r in group]) for k in ('KL', 'CE', 'reference_CE', 'CE_improvement')}
            summary.update(fraction_rows_KL_over_002=sum(r['KL']>.02 for r in group)/len(group),
                token_weighted_KL=sum(r['KL']*r['tokens'] for r in group)/sum(r['tokens'] for r in group),
                fraction_targets_CE_improved=sum(r['CE_improvement']>0 for r in group)/len(group))
            summaries['|'.join(key)] = summary
        expected = len(all_rows)*2*len(spec['candidate_checkpoints'])
        if len(merged) != expected: raise ValueError('full-population KL probe coverage differs')
        write_json(out/'GLOBAL_KL_AUDIT.json', dict(schema='full_population_editor_KL_CE_audit_v1',
            plan_sha256=file_hash(root/'KL_PROBE_PLAN.json'), data_sha256=file_hash(path), summaries=summaries,
            rows=merged, unique_content_targets=len(all_rows), optimizer_updates=0,
            diagnostic_source_split='train_only', reference_direction='KL(previous_E3 || trained_content)',
            density='per_content_example_mean_over_all_action_numeric_tokens; report_token_weighting_separately'))
        print(json.dumps(dict(unique_content_targets=len(all_rows), summaries=summaries)), flush=True)
    if world > 1: dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
