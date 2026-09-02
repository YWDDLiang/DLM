#!/usr/bin/env python3
"""Combine main1000/remainder raw graphs for paired model494 endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import torch


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--baseline-run',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(); run=a.baseline_run.resolve(); out=a.output_dir.resolve()
    if out.exists(): raise FileExistsError(out)
    cells=[*(f'main{i}' for i in range(5)),'remainder']
    accounting=[]; proposals=[]; accepted=0; requested=0
    for cell in cells:
        body=run/cell/'body'; rows=read_jsonl(body/'raw_generations.jsonl'); graphs=torch.load(body/'proposal_graphs.pt',map_location='cpu')
        parsed=[r for r in rows if r.get('parsed') is True]
        if len(parsed)!=len(graphs): raise ValueError(f'parsed/graph mismatch {cell}')
        graph_iter=iter(graphs)
        split='main1000' if cell.startswith('main') else 'remainder159'
        split_offset=int(cell[4:])*200 if cell.startswith('main') else 0
        for local,row in enumerate(rows):
            source_idx=int(row.get('source_sample_idx',requested)); ok=row.get('parsed') is True
            item={'requested_index':requested,'split':split,'split_requested_index':split_offset+local,'source_sample_idx':source_idx,'plan_state':row.get('plan_state'),'parsed':ok,'reason':None if ok else row.get('reason')}
            if ok:
                g=dict(next(graph_iter)); g['sample_idx']=accepted; g['source_sample_idx']=source_idx; g['accepted_index']=accepted; g['refiner_seed']=101117+source_idx
                item['accepted_index']=accepted; proposals.append(g); accepted+=1
            accounting.append(item); requested+=1
    if requested!=1159 or accepted!=1139: raise ValueError((requested,accepted))
    out.mkdir(parents=True); graph=out/'proposal_graphs.pt'; torch.save(proposals,graph)
    acc=out/'all_requested_accounting.jsonl'; acc.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in accounting))
    manifest={'schema':'known1200_combined_proposals_v1','requested':requested,'accepted_cif':accepted,'invalid_cif_excluded_default_denominator':20,'main_requested':1000,'main_accepted':983,'remainder_requested':159,'remainder_accepted':156,'proposal_sha256':sha(graph),'accounting_sha256':sha(acc)}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n'); (out/'_SUCCESS').touch(); print(json.dumps(manifest,sort_keys=True))


if __name__=='__main__': main()
