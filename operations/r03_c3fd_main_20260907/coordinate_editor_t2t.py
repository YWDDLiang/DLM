"""Finish the registered fixed-G/F Stable/SUN comparison after KL analysis."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import time
import traceback

from coordinate_ranked_rsi import Coordinator
from editor_trial_analysis import (clone_collection, policy_panel, freeze_selection,
                                   final_report, file_hash, write_json, read_rows)


def run(root):
    root = Path(root).resolve(); c = Coordinator(root)
    while not (root/'TRAIN_ARMS_DONE.json').exists():
        if (root/'TRAIN_ARMS_FAILED.json').exists(): raise RuntimeError('registered training failed; inspect preserved failure receipt')
        if c.remaining() < 3600: raise RuntimeError('trial deadline leaves no room for held-out panels')
        time.sleep(20)
    decision_path = root/'KL_AUDIT_DECISION.json'
    if not decision_path.exists(): raise RuntimeError('write the planned KL audit decision before evaluation')
    decision = json.loads(decision_path.read_text())
    arms = decision['candidate_arms']
    amendment = json.loads((root/'KL_ABLATION_AMENDMENT.json').read_text())
    if arms != amendment['candidate_arms'] or len(set(arms)) != 4:
        raise ValueError('candidate arms differ from the final bounded amendment')
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
        futures = {arm: pool.submit(collect, arm, root/'training'/arm/'result/checkpoint') for arm in arms}
        repeat = pool.submit(collect, 'old_E3_matched', json.loads((old/'RUN_SPEC.json').read_text())['assets']['editor_checkpoint'])
        collections = {arm: task.result() for arm, task in futures.items()}
        old_repeat = repeat.result()
    old_repeat_policy = evaluate_policy(old_repeat, False, .5)
    previous_outputs = read_rows(old/'edited/inputs.jsonl')
    repeated_outputs = read_rows(old_repeat_policy/'edited/inputs.jsonl')
    write_json(root/'OLD_E3_MATCHED_EXECUTION_AUDIT.json', dict(requests=len(previous_outputs),
        equal_final_token_rows=sum(a.get('body_token_ids') == b.get('body_token_ids') for a, b in zip(previous_outputs, repeated_outputs, strict=True)),
        fixed_model=True, fixed_current=True, fixed_sampling_seeds=True, matched_trial_GPU_sharding=True,
        repeated_policy=str(old_repeat_policy)))
    policies = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(evaluate_policy, collection, keep, threshold)
                   for collection in collections.values() for keep in reg['calibration']['respect_learned_KEEP']
                   for threshold in reg['calibration']['acceptance_thresholds']]
        policies = [f.result() for f in futures]
    frozen = freeze_selection(root, 'four_fixed_arms', policies)
    selected = frozen['selected']['policy']
    print(json.dumps(dict(event='selection_frozen', arm=frozen['arm'],
        threshold=selected['acceptance_threshold'], respect_KEEP=selected['respect_learned_KEEP'],
        dev=frozen['selected']['dev']['counts'], coverage=frozen['coverage'])), flush=True)
    comparison = {arm: evaluate_policy(panel, selected['respect_learned_KEEP'], selected['acceptance_threshold'])
                  for arm, panel in collections.items()}
    comparison['old_E3_matched'] = old_repeat_policy
    write_json(root/'COMMON_POLICY_COMPARISON_PANELS.json', {k: str(v) for k, v in comparison.items()})
    primary = final_report(root, comparison)

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
