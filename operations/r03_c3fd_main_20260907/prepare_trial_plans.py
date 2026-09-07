"""Sample and export the shared I/G/P Plans once on one allocated GPU."""
import argparse
import json
import os
from pathlib import Path
from run_component import export_programs, sample_plans, write_json

parser = argparse.ArgumentParser()
parser.add_argument('--run-root', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--requests', type=int, required=True)
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()
if not os.environ.get('SLURM_JOB_ID') or len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) != 1:
    raise ValueError('shared candidate preparation requires one allocated GPU')
assets = args.run_root / 'pointer_40395/assets'
plans = sample_plans(args.output_dir, assets, count=args.requests, seed=args.seed, offset=0, constrained=True)
programs = export_programs(args.output_dir, assets, args.run_root / 'pointer_40395/train/r03_control_pointer.pt', plans)
write_json(args.output_dir / 'SHARED_PLANS_FINAL.json', {'requests': args.requests, 'seed': args.seed,
           'plans': str(plans), 'programs': str(programs), 'shared_by': ['I', 'G', 'P']})
(args.output_dir / '_SUCCESS').touch()
