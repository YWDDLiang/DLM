import importlib.util
from pathlib import Path
from types import SimpleNamespace


def test_round_probe_preserves_prefix_and_replays_only_accepted_state():
    path = Path(__file__).resolve().parents[1] / 'operations/r03_c3fd_main_20260907/expert_round_probe.py'
    spec = importlib.util.spec_from_file_location('round_probe', path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    original = [100, 1, 2, 3, 4, 5, 6, 200, 8, 9, 10]
    row = {'ancestor_id': 'a', 'old_body': original, 'num_atoms': 1, 'prompt': 'p'}
    calls = []

    def sample(model, tokenizer, requests, **options):
        body = requests[0]['body'].copy()
        calls.append(body.copy())
        # A rejected second-round proposal leaves the committed structure unchanged.
        if len(calls) != 2:
            body[8] += 1
        return {'results': [{'body': body, 'forward_calls': 7, 'trace': []}]}

    history = probe.run_rounds(SimpleNamespace(training_modes={'G': [1]}), None, [row],
                              seed=3, rounds=4, max_calls=160, batch_size=1, sampler=sample)[0]
    assert calls[1] == history[0]['output']['body']
    assert calls[2] == history[1]['output']['body'] == calls[1]
    assert [h['cumulative_forward_calls'] for h in history] == [7, 14, 21, 28]
    assert history[1]['revisited_saved_body'] is True
    assert history[1]['changed_from_previous'] == 0
    assert len({h['seed'] for h in history}) == 4
    assert original == [100, 1, 2, 3, 4, 5, 6, 200, 8, 9, 10]
