"""Finish the registered fixed-G/F Stable/SUN comparison after KL analysis."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import time
import traceback

from coordinate_ranked_rsi import Coordinator
from editor_trial_analysis import (clone_collection, policy_panel, freeze_selection,
                                   final_report, file_hash, write_json)


def run(root):
    root = Path(root).resolve(); c = Coordinator(root)
    while not (root/'TRAIN_ARMS_DONE.json').exists():
        if (root/'TRAIN_ARMS_FAILED.json').exists(): raise RuntimeError('registered training failed; inspect preserved failure receipt')
        if c.remaining() < 3600: raise RuntimeError('trial deadline leaves no room for held-out panels')
        time.sleep(20)
    decision_path = root/'KL_AUDIT_DECISION.json'
    if not decision_path.exists(): raise RuntimeError('write the planned KL audit decision before evaluation')
    decision = json.loads(decision_path.read_text())
    arm = decision['primary_arm']
    reg = json.loads((root/'PREREGISTRATION.json').read_text())
    old = Path(reg['fixed_current'])

    def collect(name, checkpoint, stream='E', gpus=3):
        cfg = clone_collection(root, name, checkpoint, stream)
        panel = cfg.parent; tag = panel.name
        c.job(tag+'_collect', cfg, 'edit', gpus=gpus)
        c.rsi(tag+'_proposal_inputs', cfg, 'materialize', 'proposal')
        c.evaluate(tag+'_proposal', cfg, 'proposal', gpus,
                   [panel/'current/labeling/result', old/'proposal/labeling/result'])
        return panel

    def evaluate_policy(collection, respect_keep, threshold):
        cfg = policy_panel(root, collection, threshold=threshold, respect_keep=respect_keep)
        c.job(cfg.parent.name+'_score', cfg, 'score', 'edited', 0, 35)
        return cfg.parent

    with ThreadPoolExecutor(max_workers=2) as pool:
        t = pool.submit(collect, arm, root/'training'/arm/'result/checkpoint')
        h = pool.submit(collect, 'heads', root/'training/heads/result/checkpoint')
        t2t, heads = t.result(), h.result()
    policies = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(evaluate_policy, t2t, keep, threshold)
                   for keep in reg['calibration']['respect_learned_KEEP']
                   for threshold in reg['calibration']['acceptance_thresholds']]
        policies = [f.result() for f in futures]
    frozen = freeze_selection(root, arm, policies)
    selected = frozen['selected']['policy']
    print(json.dumps(dict(event='selection_frozen', arm=arm,
        threshold=selected['acceptance_threshold'], respect_KEEP=selected['respect_learned_KEEP'],
        dev=frozen['selected']['dev']['counts'], coverage=frozen['coverage'])), flush=True)
    heads_policy = evaluate_policy(heads, selected['respect_learned_KEEP'], selected['acceptance_threshold'])
    primary = final_report(root, heads_policy)

    write_json(root/'EDITOR_TRIAL_DONE.json', dict(status='complete', work=primary['work'],
        primary_result_sha256=file_hash(root/'PRIMARY_TRIAL_RESULT.json'),
        scope_amendment_sha256=file_hash(root/'SCOPE_AMENDMENT_STABLE_SUN.json'),
        remaining_seconds=c.remaining()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    try: run(args.root)
    except Exception:
        write_json(args.root/'EDITOR_TRIAL_FAILED.json', dict(error=traceback.format_exc()))
        raise
