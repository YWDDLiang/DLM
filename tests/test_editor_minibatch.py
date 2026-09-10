from pathlib import Path
import random

import pytest
import torch

from crystal_dlm.editor_minibatch import permute_atoms, train_editor_minibatches


def test_atom_reordering_preserves_aligned_pairs_and_site_targets():
    row = dict(num_sites=3, current_tokens=list(range(19)), proposal_tokens=list(range(30, 49)),
               content_target_tokens=list(range(60, 79)), site_targets=[0, 1, 0],
               action_positions=[1, 12, 13, 14], content_positions=[12, 13, 14])
    changed = permute_atoms(row, [2, 0, 1])
    assert changed['current_tokens'] == list(range(7)) + list(range(15, 19)) + list(range(7, 15))
    assert changed['proposal_tokens'][15:19] == row['proposal_tokens'][11:15]
    assert changed['content_target_tokens'][15:19] == row['content_target_tokens'][11:15]
    assert changed['site_targets'] == [0, 0, 1]
    assert changed['action_positions'] == [1, 16, 17, 18]
    assert changed['content_positions'] == [16, 17, 18]
    assert permute_atoms(changed, [1, 2, 0]) == row
    with pytest.raises(ValueError, match='bijection'):
        permute_atoms(row, [0, 0, 2])


def test_every_minibatch_updates_content_despite_large_reference_kl(monkeypatch, tmp_path):
    import crystal_dlm.editor_minibatch as module

    class TinyEditor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.content = torch.nn.Parameter(torch.tensor(0.))
            self.mode_head = torch.nn.Linear(1, 1)

    model = TinyEditor()
    selected = list(model.named_parameters())
    reference = {name: value.detach().clone() for name, value in selected}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.05, weight_decay=0.)
    monkeypatch.setattr(module, 'ExpertEditObjective', lambda *a, **k: None)
    monkeypatch.setattr(module, 'dense_vectors', lambda model, *a, **k: model.content.square() + model.content)
    monkeypatch.setattr(module, 'dense_loss', lambda p, q, rows: ((p - 2).square() * len(rows),
                       (p - q).square() * len(rows) + 10., len(rows)))
    monkeypatch.setattr(module, 'editor_head_loss', lambda model, tokenizer, rows, device, **k:
                       (model.mode_head(torch.ones(len(rows), 1)).square().mean(), len(rows)))
    rows = [dict(pair_id=str(i), num_sites=2, current_tokens=list(range(15)),
                 content_target_tokens=list(range(15)), content_positions=[8, 9, 10],
                 mode_target=1, site_targets=[1, 0]) for i in range(6)]
    spec = dict(branch='E', seed=12, batch_size=2, epochs=2, permute_atoms=True,
                reference_kl_weight=.01, max_reference_kl=.0001, max_training_seconds=100.)
    reports = {}
    result = train_editor_minibatches(model, None, rows, spec, selected, reference, optimizer,
                                      None, tmp_path, lambda p, v: reports.setdefault(p.name, v))
    report = reports['EXPOSURE_rank0.json']
    assert result[0] == 6
    assert report['content_optimizer_steps'] == 6
    assert report['head_optimizer_steps'] == 6
    assert report['content_updated_pair_visits'] == {str(i): 2 for i in range(6)}
    assert report['KL_stop'] is False
    assert float(model.content) != float(reference['content'])
    assert set(report['local_site_target_visits']) == {0, 1}
    assert all(event['reference_KL'] >= 5. for event in result[3])
