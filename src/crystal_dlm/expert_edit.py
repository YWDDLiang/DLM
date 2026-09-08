"""A B0-compatible crystal editor with explicit current/old state and decisions.

Only learned modules change content logits. Typed-token support preserves the
representation; no online force, energy or external geometry veto is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import hashlib
import random
import re
from typing import NamedTuple

import torch
from torch import nn
from torch.utils.data import Dataset

from crystal_dlm.periodic_repair_model import FP32Module
from crystal_dlm.periodic_state_conditioning import PeriodicStateConditioner, PeriodicStateConfig
from crystal_dlm.state_conditioned_model import CrystalStateContext, StateConditionedDLM, set_state_lora_trainable
from crystal_dlm.periodic_geometry_objective import build_geometry_token_support
from crystal_dlm.fixed_slot import MASK_TOKEN_ID


EDITOR_SCHEMA = 'expert_crystal_editor_v1'
MODE_NAMES = ('none', 'local_xyz', 'all_xyz', 'full_cell')
LOCAL_COUNTS = (1, 2, 4, 8)


@dataclass(frozen=True)
class ExpertEditConfig:
    hidden_size: int
    width: int = 128
    max_sites: int = 20
    schema: str = EDITOR_SCHEMA

    def __post_init__(self):
        if self.hidden_size < 1 or self.width < 1 or not 1 <= self.max_sites <= 20 or self.schema != EDITOR_SCHEMA:
            raise ValueError('invalid expert editor configuration')


@dataclass(frozen=True)
class EditContext:
    old_token_ids: torch.Tensor
    prompt_lengths: torch.Tensor
    num_sites: torch.Tensor
    active_token_mask: torch.Tensor
    task_ids: torch.Tensor
    remaining_steps: torch.Tensor
    reveal_fraction: torch.Tensor


class EditOutput(NamedTuple):
    logits: torch.Tensor
    mode_logits: torch.Tensor
    site_logits: torch.Tensor
    count_logits: torch.Tensor
    quality_logits: torch.Tensor


class FloatMLP(FP32Module):
    def __init__(self, inputs, width, outputs, *, zero_output=False):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, width, dtype=torch.float32), nn.SiLU(),
                                    nn.Linear(width, outputs, dtype=torch.float32))
        if zero_output:
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(self, value):
        with torch.autocast(device_type=value.device.type, enabled=False):
            return self.layers(value.float())


class ExpertEditDLM(StateConditionedDLM):
    def __init__(self, base_model, tokenizer, config: ExpertEditConfig):
        state = PeriodicStateConfig(hidden_size=config.hidden_size, width=config.width, max_sites=config.max_sites)
        super().__init__(base_model, tokenizer, state)
        self.editor_config = config
        self.current_state_conditioner = PeriodicStateConditioner(state)
        self.raw_cell_adapter = FloatMLP(27, config.width, config.hidden_size, zero_output=True)
        self.raw_site_adapter = FloatMLP(21, config.width, config.hidden_size, zero_output=True)
        self.task_adapter = FloatMLP(4, config.width, config.hidden_size, zero_output=True)
        self.mode_head = FloatMLP(2 * config.hidden_size, config.width, 4)
        self.site_head = FloatMLP(config.hidden_size, config.width, 1)
        self.count_head = FloatMLP(2 * config.hidden_size, config.width, 4)
        self.quality_head = FloatMLP(2 * config.hidden_size, config.width, 4)
        self.forward_calls = 0

    def extra_modules(self):
        return {name: getattr(self, name) for name in (
            'current_state_conditioner', 'raw_cell_adapter', 'raw_site_adapter', 'task_adapter',
            'mode_head', 'site_head', 'count_head', 'quality_head')}

    def content_modules(self):
        return {'state_conditioner': self.state_conditioner, **{name: module for name, module in self.extra_modules().items()
                 if name not in ('mode_head', 'site_head', 'count_head', 'quality_head')}}

    def _context(self, value: EditContext, ids):
        rank = torch.arange(self.editor_config.max_sites, device=ids.device)[None].expand(ids.shape[0], -1)
        return CrystalStateContext(ids, value.prompt_lengths, value.num_sites, rank,
                                   value.active_token_mask, value.task_ids)

    def _raw_values(self, ids, context):
        batch, length = ids.shape
        rows = torch.arange(batch, device=ids.device)[:, None]
        cell_positions = context.prompt_lengths[:, None] + torch.arange(1, 7, device=ids.device)[None]
        cell = self.geometry_values[torch.arange(6, device=ids.device)[None], ids[rows, cell_positions]]
        sites = torch.arange(self.editor_config.max_sites, device=ids.device)[None]
        valid = sites < context.num_sites[:, None]
        coordinates = []
        for axis in range(3):
            positions = (context.prompt_lengths[:, None] + 8 + 4 * sites + axis).clamp_max(length - 1)
            coordinates.append(self.geometry_values[6 + axis, ids[rows, positions]])
        coords = torch.stack(coordinates, -1)
        cell_known = torch.isfinite(cell)
        coord_known = torch.isfinite(coords) & valid[..., None]
        cell = torch.where(cell_known, cell, 0.)
        scale = cell.new_tensor([50., 50., 50., 180., 180., 180.])
        cell = cell / scale
        coords = torch.where(coord_known, coords, 0.).remainder(1.)
        phase = 2 * torch.pi * coords
        periodic = torch.cat((phase.sin(), phase.cos()), -1)
        periodic = periodic * torch.cat((coord_known, coord_known), -1)
        return cell, cell_known.float(), periodic, coord_known.float(), valid

    def _validate(self, input_ids, context):
        if context.old_token_ids.shape != input_ids.shape or context.active_token_mask.shape != input_ids.shape:
            raise ValueError('old/current/active canvases must align')
        if any(value.shape != (input_ids.shape[0],) for value in (
                context.prompt_lengths, context.num_sites, context.task_ids, context.remaining_steps, context.reveal_fraction)):
            raise ValueError('one task, budget, length and site count is required per row')
        if bool(((context.task_ids < 0) | (context.task_ids > 1)).any()):
            raise ValueError('task must be G=0 or S=1')
        if (not bool(torch.isfinite(context.remaining_steps).all() & torch.isfinite(context.reveal_fraction).all())
                or bool((context.remaining_steps < 0).any())
                or bool(((context.reveal_fraction < 0) | (context.reveal_fraction > 1)).any())):
            raise ValueError('invalid edit budget or reveal fraction')

    def forward(self, input_ids, attention_mask=None, *, edit_context: EditContext, **kwargs):
        self._validate(input_ids, edit_context)
        self.forward_calls += 1
        old_context = self._context(edit_context, edit_context.old_token_ids)
        current_context = self._context(edit_context, input_ids)
        old_geometry = self.geometry_inputs(old_context)
        current_geometry = self.geometry_inputs(current_context)
        embeddings = self.state_embeddings(input_ids, old_context)
        current = self.current_state_conditioner(**current_geometry)
        old_cell, old_ck, old_xyz, old_xk, valid = self._raw_values(edit_context.old_token_ids, edit_context)
        new_cell, new_ck, new_xyz, new_xk, _ = self._raw_values(input_ids, edit_context)
        batch, length, hidden = embeddings.shape
        row = torch.arange(batch, device=input_ids.device)[:, None]
        cell_positions = edit_context.prompt_lengths[:, None] + torch.arange(1, 7, device=input_ids.device)[None]
        slots = torch.arange(self.editor_config.max_sites, device=input_ids.device)[None]
        cell_active = edit_context.active_token_mask[row, cell_positions].any(-1).float()
        cell_features = torch.cat((old_cell, new_cell, old_ck, new_ck,
                                   old_geometry['lattice_known'][:, None].float(),
                                   current_geometry['lattice_known'][:, None].float(), cell_active[:, None]), -1)
        site_active = old_geometry['active_sites'].float()
        site_features = torch.cat((old_xyz, new_xyz, old_xk, new_xk, site_active[..., None],
                                   old_geometry['lattice_known'][:, None, None].expand(-1, slots.shape[1], 1).float(),
                                   current_geometry['lattice_known'][:, None, None].expand(-1, slots.shape[1], 1).float()), -1)
        task_features = torch.cat((nn.functional.one_hot(edit_context.task_ids.long(), 2).float(),
                                   (edit_context.remaining_steps.float() / 32.)[:, None],
                                   edit_context.reveal_fraction.float()[:, None]), -1)
        cell_delta = current['cell_embedding'] + self.raw_cell_adapter(cell_features)
        site_delta = (current['site_embeddings'] + self.raw_site_adapter(site_features)) * valid[..., None]
        residual = embeddings.new_zeros(batch, length, hidden)
        residual[row, cell_positions] = cell_delta[:, None].to(embeddings.dtype).expand(-1, cell_positions.shape[1], -1).contiguous()
        for offset in range(4):
            positions = (edit_context.prompt_lengths[:, None] + 7 + 4 * slots + offset).clamp_max(length - 1)
            residual = residual.scatter_add(1, positions[..., None].expand(-1, -1, hidden),
                                            site_delta.to(embeddings.dtype))
        relative = torch.arange(length, device=input_ids.device)[None] - edit_context.prompt_lengths[:, None]
        body_mask = (relative >= 0) & (relative < 7 + 4 * edit_context.num_sites[:, None])
        residual += self.task_adapter(task_features)[:, None].to(embeddings.dtype) * body_mask[..., None]
        kwargs.pop('output_hidden_states', None)
        output = self.base_model(input_ids=None, inputs_embeds=embeddings + residual,
                                 attention_mask=attention_mask, output_hidden_states=True, **kwargs)
        if not output.hidden_states:
            raise RuntimeError('B0 must expose its final normalized hidden state')
        final = output.hidden_states[-1]
        cell_hidden = final[row, cell_positions].float().mean(1)
        positions = (edit_context.prompt_lengths[:, None] + 7 + 4 * slots).clamp_max(length - 1)
        site_hidden = final[row, positions].float() * valid[..., None]
        pooled = site_hidden.sum(1) / edit_context.num_sites[:, None].clamp_min(1)
        global_hidden = torch.cat((cell_hidden, pooled), -1)
        sites = self.site_head(site_hidden).squeeze(-1).masked_fill(~valid, -1e4)
        return EditOutput(output.logits, self.mode_head(global_hidden), sites,
                          self.count_head(global_hidden), self.quality_head(global_hidden))

    def save_pretrained(self, output_dir, **kwargs):
        super().save_pretrained(output_dir, **kwargs)
        root = Path(output_dir)
        (root / 'EXPERT_EDITOR.json').write_text(json.dumps({'schema': EDITOR_SCHEMA,
                         'allowed_modes': getattr(self, 'training_modes', None)}) + '\n')
        (root / 'expert_edit_config.json').write_text(json.dumps(asdict(self.editor_config), indent=2) + '\n')
        torch.save({name: module.state_dict() for name, module in self.extra_modules().items()},
                   root / 'expert_edit_modules.pt')

    def load_editor(self, checkpoint):
        root = Path(checkpoint)
        required = ('EXPERT_EDITOR.json', 'expert_edit_config.json', 'expert_edit_modules.pt', 'periodic_state_config.json', 'periodic_state.pt')
        if any(not (root / name).is_file() for name in required):
            raise ValueError('editor checkpoint is incomplete; fallback to plain B0 is forbidden')
        if json.loads((root / 'expert_edit_config.json').read_text()) != asdict(self.editor_config):
            raise ValueError('editor configuration differs')
        self.load_state_conditioner(root)
        saved = torch.load(root / 'expert_edit_modules.pt', map_location='cpu', weights_only=True)
        if set(saved) != set(self.extra_modules()):
            raise ValueError('editor checkpoint module partition differs')
        for name, module in self.extra_modules().items():
            module.load_state_dict(saved[name], strict=True)


def set_editor_trainable(model):
    counts = set_state_lora_trainable(model)
    for name, module in model.extra_modules().items():
        module.requires_grad_(True)
        counts[name] = sum(parameter.numel() for parameter in module.parameters())
    if model.get_input_embeddings().weight.requires_grad or model.get_output_embeddings().weight.requires_grad:
        raise ValueError('original B0 embedding/head tables must remain frozen')
    return counts


def load_editor_model(model_path, checkpoint, device, *, trainable=False):
    from scripts.sample_llada_dynamic_crystals import load_model_and_tokenizer
    root = Path(checkpoint)
    # The entire trained wte/head tables are restored and verified afterwards;
    # computing a covariance initialization for soon-overwritten rows is wasted.
    base, tokenizer = load_model_and_tokenizer(str(model_path), str(root), device, mean_resizing=False)
    marker = root / 'expert_edit_config.json'
    config = (ExpertEditConfig(**json.loads(marker.read_text())) if marker.exists()
              else ExpertEditConfig(hidden_size=base.get_input_embeddings().weight.shape[1]))
    model = ExpertEditDLM(base, tokenizer, config).to(device)
    if any((root / name).exists() for name in ('EXPERT_EDITOR.json', 'expert_edit_config.json', 'expert_edit_modules.pt')):
        model.load_editor(root)
    elif (root / 'periodic_state_config.json').exists():
        raise ValueError('an unrelated state-conditioned checkpoint cannot initialize this B0 editor')
    if trainable:
        set_editor_trainable(model)
    else:
        model.requires_grad_(False)
    model.eval()
    if marker.exists():
        capability = json.loads((root/'EXPERT_EDITOR.json').read_text()).get('allowed_modes')
        if capability is None:
            for parent in (root.parent, root.parent.parent):
                config_path = parent/'TRAIN_CONFIG.json'
                if config_path.is_file():
                    capability = json.loads(config_path.read_text()).get('allowed_modes')
                    break
        if (not isinstance(capability, dict) or set(capability) != {'G','S'} or
                any(type(mode) is not int or not 0 <= mode < len(MODE_NAMES) for modes in capability.values() for mode in modes)):
            raise ValueError('trained editor action capabilities are missing or invalid')
        model.training_modes = capability
        probe_path = root/'roundtrip_probe.pt'
        if not probe_path.is_file():
            raise ValueError('editor checkpoint lacks its required reload verification probe')
        probe = torch.load(probe_path, map_location='cpu', weights_only=False)
        batch = materialize_edit_batch(probe['examples'], tokenizer, device)
        with torch.no_grad():
            actual = model(batch['input_ids'], attention_mask=batch['attention_mask'], edit_context=batch['edit_context'])
        if any(not torch.equal(getattr(actual,name).cpu(), probe[name]) for name in
               ('logits','mode_logits','site_logits','count_logits','quality_logits')):
            raise ValueError('loaded editor differs from its recorded content/decision probe')
        model.reload_verified = True
    return model, tokenizer


def reverse_geometry_example(record):
    """A witnessed G recovery supplies its reverse rejection without new physics."""
    row = dict(record)
    row.update(record_id='reverse:' + record['record_id'], old_body=record['target_body'],
               target_body=record['old_body'], old_geometry=record['target_geometry'],
               target_geometry=record['old_geometry'], old_reliable=record.get('target_reliable'),
               target_reliable=record.get('old_reliable'), old_physics_id=record.get('target_physics_id'),
               target_physics_id=record.get('old_physics_id'), content_supervision=False,
               accept_label=False, label_reason='reverse_witnessed_G_recovery', gain_eV_atom=None)
    row['action'] = {key: list(value) if isinstance(value,list) else value for key,value in record['action'].items()}
    return row


class ExpertEditDataset(Dataset):
    """Explicit sampling of content, scope/state and complete-proposal decisions."""
    def __init__(self, data_dirs, tokenizer, *, seed=20260908, size=10000, split='train', smoke_sources=0,
                 geometry_aux_fraction=.2, content_fraction=.65, inspect_fraction=.17,
                 student_feedback_fraction=0., healthy_state_fraction=0.,
                 content_target_mode='all_pending', m2t_probability=.7, target_representative='none'):
        from crystal_dlm.expert_edit_data import SCHEMA, read_rows, sha256
        self.seed, self.size, self.epoch, self.tokenizer = int(seed), int(size), 0, tokenizer
        if not 0 <= geometry_aux_fraction <= 1:
            raise ValueError('invalid auxiliary data fraction')
        self.geometry_aux_fraction, self._source_pools = geometry_aux_fraction, {}
        if (not 0 < content_fraction < 1 or not 0 <= inspect_fraction < 1
                or content_fraction+inspect_fraction >= 1 or not 0 <= student_feedback_fraction <= 1
                or not 0 <= healthy_state_fraction <= 1):
            raise ValueError('invalid editor curriculum view fractions')
        self.content_fraction, self.inspect_fraction = content_fraction, inspect_fraction
        self.student_feedback_fraction = student_feedback_fraction
        self.healthy_state_fraction = healthy_state_fraction
        if content_target_mode not in ('all_pending', 'next_token') or not 0 <= m2t_probability <= 1:
            raise ValueError('invalid content supervision or masked-view probability')
        self.content_target_mode, self.m2t_probability = content_target_mode, m2t_probability
        if target_representative not in ('none', 'old_aligned_grid'):
            raise ValueError('unknown target representative')
        self.target_representative = target_representative
        records, provenance = [], []
        for directory in map(Path, data_dirs):
            if not (directory / '_SUCCESS').is_file():
                raise ValueError('editor data is incomplete')
            report = json.loads((directory / 'DATA_FINAL.json').read_text())
            if report['schema'] != SCHEMA:
                raise ValueError('unknown editor data schema')
            path = directory / (split + '.jsonl')
            if report.get('output_sha256') and report['output_sha256'].get(path.name) != sha256(path):
                raise ValueError('compiled editor examples changed after certification')
            rows = read_rows(path)
            if any(row['source_split'] != split for row in rows):
                raise ValueError('editor dataset mixes source splits')
            records.extend(rows)
            provenance.append({'path': str(path.resolve()), 'sha256': sha256(path),
                               'report_sha256': sha256(directory / 'DATA_FINAL.json')})
        if len({row['record_id'] for row in records}) != len(records):
            raise ValueError('duplicate editor records would double-count source supervision')
        if any('training_target_body' in row or 'training_target_certificate' in row for row in records):
            raise ValueError('compiled physics records must retain original targets; derive representatives after loading')
        if smoke_sources:
            pools = {task: sorted({row['ancestor_id'] for row in records
                                  if row['task'] == task and row.get('content_supervision')}) for task in ('G', 'S')}
            selected = set()
            for task in ('G', 'S'):
                selected.update(pools[task][:max(1, smoke_sources // 2)])
            remaining = [x for x in pools['G'] + pools['S'] if x not in selected]
            selected.update(remaining[:max(0, smoke_sources - len(selected))])
            records = [row for row in records if row['ancestor_id'] in selected]
        if target_representative == 'old_aligned_grid':
            from crystal_dlm.expert_target_representative import training_representative
            records = [training_representative(row, tokenizer) for row in records]
        self.records = records
        # Equivalent zero edits provide no nonzero action supervision. Keep their
        # original physical records and state observations, without inventing STOP.
        actions = [row for row in records if not row.get('training_representative_zero_edit')]
        self.content = {task: [row for row in actions if row['task'] == task and row.get('content_supervision')]
                        for task in ('G', 'S')}
        self.positive = [row for row in actions if row.get('content_supervision')]
        self.acceptance_positive = [row for row in actions if row.get('accept_label') is True]
        self.states = [row for row in records if row.get('state_only')]
        self.healthy_states = [row for row in self.states if row['old_geometry'].get('valid') is True
                               and row.get('old_reliable') is True]
        self.student_states = [row for row in self.states if row.get('source_kind') == 'student_proposal_feedback']
        self.student_judgements = [row for row in records if row.get('source_kind') == 'student_proposal_feedback'
            and re.search(r':proposal_\d+:[GS]$',row['record_id']) and type(row.get('accept_label')) is bool
            and not row.get('state_only') and row.get('original_teacher_reference') is False]
        self.negative = [row for row in records if row.get('accept_label') is False]
        self.negative += [reverse_geometry_example(row) for row in self.content['G']
                          if row.get('accept_label') is True and row['old_body'] != row['target_body']]
        if not self.positive:
            raise ValueError('editor has no positively supervised executable edits')
        self.provenance = provenance
        self.prefixes = {}
        for row in records:
            if row['prompt'] not in self.prefixes:
                self.prefixes[row['prompt']] = tokenizer(row['prompt'], add_special_tokens=False)['input_ids']
        self.allowed_modes = {task: sorted({MODE_NAMES.index(row['action']['mode'])
                                           for row in self.content[task] if row['action']['mode'] in MODE_NAMES})
                              for task in ('G', 'S')}
        self.allowed_modes['G'] = sorted(set(self.allowed_modes['G']) | {0})

    def __len__(self):
        return self.size

    def _choose_content(self, rng):
        task = 'S' if self.content['S'] and rng.random() < .3 else 'G'
        return self._draw(self.content[task] or self.positive, rng)

    def _draw(self, rows, rng):
        key = id(rows)
        if key not in self._source_pools:
            auxiliary = [row for row in rows if row.get('source_kind') == 'mp20_geometry_auxiliary']
            real = [row for row in rows if row.get('source_kind') != 'mp20_geometry_auxiliary']
            def groups(values):
                result = {}
                for row in values:
                    result.setdefault(row['ancestor_id'],[]).append(row)
                return list(result.values())
            self._source_pools[key] = groups(real), groups(auxiliary)
        real, auxiliary = self._source_pools[key]
        pool = auxiliary if real and auxiliary and rng.random() < self.geometry_aux_fraction else real
        group = rng.choice(pool or auxiliary)
        return rng.choice(group)

    def __getitem__(self, index):
        value = hashlib.sha256(f'{self.seed}:{self.epoch}:{index}'.encode()).digest()[:8]
        rng = random.Random(int.from_bytes(value, 'big'))
        choice = rng.random()
        if choice < self.content_fraction:
            row, view = self._choose_content(rng), 'content'
        elif choice < self.content_fraction+self.inspect_fraction:
            healthy = bool(self.healthy_state_fraction and self.healthy_states and rng.random() < self.healthy_state_fraction)
            if healthy:
                row = dict(self._draw(self.healthy_states,rng),task='G')
            elif self.student_feedback_fraction and self.student_states and rng.random() < self.student_feedback_fraction:
                row = self._draw(self.student_states,rng)
            else:
                row = self._draw(self.states, rng) if self.states and rng.random() < .35 else self._draw(self.positive, rng)
            view = 'inspect'
            if row.get('state_only') and not healthy and rng.random() < .5:
                # Old-only bad/unknown states train S admission too. S receives
                # quality labels here, never a fabricated NONE/stop target.
                row = dict(row, task='S')
        else:
            if self.student_feedback_fraction and self.student_judgements and rng.random() < self.student_feedback_fraction:
                row = self._draw(self.student_judgements,rng)
            else:
                pool = self.negative if self.negative and rng.random() < .5 else self.acceptance_positive
                row = self._draw(pool or self.negative, rng)
            view = 'judge'
        return make_edit_view(row, view, rng, self.prefixes[row['prompt']],
                              content_target_mode=self.content_target_mode, m2t_probability=self.m2t_probability)


def make_edit_view(record, kind, rng, prefix, *, content_target_mode='all_pending', m2t_probability=.7):
    from crystal_dlm.expert_edit_data import numeric_positions
    if content_target_mode not in ('all_pending', 'next_token') or not 0 <= m2t_probability <= 1:
        raise ValueError('invalid content supervision or masked-view probability')
    old, n = list(record['old_body']), int(record['num_atoms'])
    task = 0 if record['task'] == 'G' else 1
    if record.get('state_only'):
        kind = 'inspect'
    active = list(record.get('action', {}).get('positions', []))
    current, targets = old.copy(), [-100] * len(old)
    mode, count = -100, -100
    sites = [-1.] * n
    quality = [None, None, None, None]
    reveal = 0.
    remaining = len(active) + 2
    if kind == 'content':
        target = record.get('training_target_body', record['target_body'])
        if not active:
            active = numeric_positions(n)
        cut = rng.randrange(len(active))
        prefix_positions = active[:cut]
        pending = active[cut:]
        m2t = rng.random() < m2t_probability
        for position in prefix_positions:
            current[position] = target[position]
        for position in pending:
            if m2t:
                current[position] = MASK_TOKEN_ID
            if content_target_mode == 'all_pending' or position == pending[0]:
                targets[position] = int(target[position])
        reveal = len(prefix_positions) / max(1, len(active))
        remaining = len(pending) + 1
    elif kind == 'inspect':
        active = []
        # The unknown teacher scope cannot be encoded in the inspection budget.
        # Deployment uses this same public budget for every structure of size N.
        remaining = 8 + 3*n
        quality[:2] = [record['old_geometry'].get('valid'), record.get('old_reliable')]
        if record.get('state_only'):
            if task == 0 and quality[0] is True and quality[1] is True:
                mode = 0
        elif record.get('content_supervision'):
            action_mode = record['action']['mode']
            mode = MODE_NAMES.index(action_mode) if action_mode in MODE_NAMES else -100
            if mode == 1:
                selected = record['action']['sites']
                sites = [float(i in selected) for i in range(n)]
                if len(selected) not in LOCAL_COUNTS:
                    raise ValueError('local training action is not a registered count')
                count = LOCAL_COUNTS.index(len(selected))
    elif kind == 'judge':
        current = list(record['target_body'])
        quality = [record['target_geometry'].get('valid'), record.get('target_reliable'),
                   None, record.get('accept_label')]
        if task == 1 and record.get('gain_eV_atom') is not None:
            quality[2] = record['gain_eV_atom'] >= .01
        reveal = 1.
    else:
        raise ValueError('unknown editor view')
    return {'record_id': record['record_id'], 'ancestor_id': record['ancestor_id'], 'kind': kind,
            'source_kind': record.get('source_kind'),
            'prefix': list(prefix), 'old_body': old, 'input_body': current, 'targets': targets,
            'active': active, 'num_sites': n, 'task': task, 'remaining': remaining, 'reveal': reveal,
            'mode_target': mode, 'site_targets': sites, 'count_target': count, 'quality_targets': quality}


def materialize_edit_batch(examples, tokenizer, device, *, max_length=1024):
    lengths = [len(row['prefix']) + len(row['old_body']) for row in examples]
    width, batch = max(lengths), len(examples)
    if width > max_length:
        raise ValueError('editor view exceeds max length; truncation is forbidden')
    pad = int(tokenizer.pad_token_id)
    ids = torch.full((batch, width), pad, dtype=torch.long, device=device)
    old = ids.clone()
    targets = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    active = torch.zeros_like(ids, dtype=torch.bool)
    site_targets = torch.full((batch, 20), -1., device=device)
    quality_targets = torch.zeros(batch, 4, device=device)
    quality_mask = torch.zeros(batch, 4, dtype=torch.bool, device=device)
    for i, row in enumerate(examples):
        p, n = len(row['prefix']), row['num_sites']
        if len(row['old_body']) != 7 + 4 * n or len(row['input_body']) != len(row['old_body']):
            raise ValueError('editor view changed its native body layout')
        ids[i, :lengths[i]] = torch.tensor(row['prefix'] + row['input_body'], device=device)
        old[i, :lengths[i]] = torch.tensor(row['prefix'] + row['old_body'], device=device)
        targets[i, p:lengths[i]] = torch.tensor(row['targets'], device=device)
        attention[i, :lengths[i]] = 1
        active[i, [p + j for j in row['active']]] = True
        site_targets[i, :n] = torch.tensor(row['site_targets'], device=device)
        for j, value in enumerate(row['quality_targets']):
            if value is not None:
                quality_targets[i, j], quality_mask[i, j] = float(value), True
    vector = lambda key, dtype=torch.long: torch.tensor([row[key] for row in examples], dtype=dtype, device=device)
    context = EditContext(old, torch.tensor([len(row['prefix']) for row in examples], device=device),
                          vector('num_sites'), active, vector('task'), vector('remaining', torch.float32),
                          vector('reveal', torch.float32))
    return {'input_ids': ids, 'attention_mask': attention, 'edit_context': context, 'targets': targets,
            'mode_targets': vector('mode_target'), 'count_targets': vector('count_target'),
            'site_targets': site_targets, 'quality_targets': quality_targets, 'quality_mask': quality_mask,
            'examples': examples}


class ExpertEditObjective:
    def __init__(self, tokenizer, device, *, temperature=.7):
        self.temperature, self.tables = temperature, {}
        for family, axes in build_geometry_token_support(tokenizer).items():
            for axis, table in axes.items():
                self.tables[(family, axis)] = (torch.tensor(table['ids'], device=device),
                                               torch.tensor(table['values'], device=device))

    def typed_vector(self, logits, rows, positions, family, axis):
        ids, values = self.tables[(family, axis)]
        vector = logits[rows[:, None], positions[:, None], ids[None]]
        if family == 'coord':
            zero, alias = int((values == 0).nonzero()[0]), int((values == 1).nonzero()[0])
            merged = torch.logaddexp(vector[:, zero], vector[:, alias])
            vector = vector.clone()
            vector[:, zero] = merged
            keep = torch.arange(len(ids), device=ids.device) != alias
            vector, ids = vector[:, keep], ids[keep]
        return vector.float() / self.temperature, ids

    def __call__(self, output, batch):
        targets, context = batch['targets'], batch['edit_context']
        sums = output.logits.new_zeros(len(targets), dtype=torch.float32)
        counts = sums.clone()
        rows, positions = torch.nonzero(targets != -100, as_tuple=True)
        relative = positions - context.prompt_lengths[rows]
        field_sums = {name: output.logits.new_zeros((), dtype=torch.float32)
                      for name in ('length', 'angle', 'coord', 'first_lattice')}
        field_counts = dict.fromkeys(field_sums, 0)
        for kind in range(9):
            if kind < 6:
                select = relative == kind + 1
                family, axis = ('length', 'ABC'[kind]) if kind < 3 else ('angle', 'ABG'[kind - 3])
            else:
                select = (relative >= 8) & ((relative - 8).remainder(4) == kind - 6)
                family, axis = 'coord', 'XYZ'[kind - 6]
            rr, pp = rows[select], positions[select]
            if not len(rr):
                continue
            vector, ids = self.typed_vector(output.logits, rr, pp, family, axis)
            matches = targets[rr, pp, None] == ids[None]
            if not bool(matches.any(-1).all()):
                raise ValueError('corrected target is outside its typed B0 vocabulary')
            ce = nn.functional.cross_entropy(vector, matches.long().argmax(-1), reduction='none')
            field_sums[family] += ce.detach().sum()
            field_counts[family] += len(rr)
            if kind == 0:
                field_sums['first_lattice'] += ce.detach().sum()
                field_counts['first_lattice'] += len(rr)
            sums = sums.scatter_add(0, rr, ce)
            counts = counts.scatter_add(0, rr, torch.ones_like(ce))
        content = (sums / counts.clamp_min(1)).mean()
        def classification(logits, expected):
            selected = expected != -100
            if not bool(selected.any()):
                return logits.sum() * 0
            return nn.functional.cross_entropy(logits[selected], expected[selected], reduction='sum') / len(expected)
        mode = classification(output.mode_logits, batch['mode_targets'])
        count = classification(output.count_logits, batch['count_targets'])
        valid_sites = batch['site_targets'] >= 0
        site_loss = nn.functional.binary_cross_entropy_with_logits(
            output.site_logits, batch['site_targets'].clamp_min(0), reduction='none')
        site = ((site_loss * valid_sites).sum(-1) / valid_sites.sum(-1).clamp_min(1)).mean()
        quality_loss = nn.functional.binary_cross_entropy_with_logits(
            output.quality_logits, batch['quality_targets'], reduction='none')
        quality = ((quality_loss * batch['quality_mask']).sum(-1)
                   / batch['quality_mask'].sum(-1).clamp_min(1)).mean()
        # All heads participate even when a particular rank has no labels for
        # one of them. Finite padding sentinels avoid inf*0 during this sum.
        connected_zero = sum(value.float().sum() * 0 for value in (
            output.logits[..., :1], output.mode_logits, output.site_logits,
            output.count_logits, output.quality_logits))
        loss = content + .2 * mode + .1 * count + .2 * site + .3 * quality + connected_zero
        stats = {'content_ce': float(content.detach()), 'mode_ce': float(mode.detach()),
                      'site_bce': float(site.detach()), 'quality_bce': float(quality.detach()),
                      'supervised_tokens': int(counts.sum()), 'loss': float(loss.detach())}
        for field in field_sums:
            stats[field+'_content_ce_sum'] = float(field_sums[field])
            stats[field+'_content_tokens'] = field_counts[field]
        for task_id, name in ((0, 'G'), (1, 'S')):
            selected = (context.task_ids == task_id) & (counts > 0)
            stats[name + '_content_views'] = int(selected.sum())
            stats[name + '_content_ce_sum'] = float((sums[selected] / counts[selected]).sum().detach())
            auxiliary = torch.tensor([row.get('source_kind') == 'mp20_geometry_auxiliary'
                                      for row in batch.get('examples', [{}]*len(targets))],
                                     device=counts.device, dtype=torch.bool)
            for origin, choose in (('real', ~auxiliary), ('aux', auxiliary)):
                subset = selected & choose
                stats[name+'_'+origin+'_content_views'] = int(subset.sum())
                stats[name+'_'+origin+'_content_ce_sum'] = float((sums[subset]/counts[subset]).sum().detach())
        return loss, stats


def inference_view(prefix, old, current, n, task, active=(), *, remaining=160, reveal=0.):
    """An inference input constructed entirely from the current structure."""
    return {'prefix': list(prefix), 'old_body': list(old), 'input_body': list(current),
            'targets': [-100] * len(old), 'active': list(active), 'num_sites': n,
            'task': task, 'remaining': remaining, 'reveal': reveal, 'mode_target': -100,
            'site_targets': [-1.] * n, 'count_target': -100, 'quality_targets': [None] * 4,
            'kind': 'inference', 'record_id': '', 'ancestor_id': ''}


@torch.no_grad()
def edit_structure(model, tokenizer, *, prompt, body, num_sites, allowed_modes,
                   tasks=('G', 'S'), seed=0, block_size=1, max_calls=160,
                   temperature=.7, accept_threshold=.5, force_mode=None, accept_all=False,
                   proposal_input='masked'):
    """One-request form of the same batched conditional sampler."""
    request = {'prompt': prompt, 'body': body, 'num_sites': num_sites, 'tasks': tasks, 'seed': seed}
    return edit_structures(model, tokenizer, [request], allowed_modes=allowed_modes, block_size=block_size,
                           max_calls=max_calls, temperature=temperature, accept_threshold=accept_threshold,
                           force_mode=force_mode, accept_all=accept_all, proposal_input=proposal_input,
                           batch_size=1)['results'][0]


@torch.no_grad()
def edit_structures(model, tokenizer, requests, *, allowed_modes, block_size=1, max_calls=160,
                    temperature=.7, accept_threshold=.5, force_mode=None, accept_all=False,
                    batch_size=8, progress=None, proposal_input='masked'):
    """Batch independent structures while retaining conditional scalar reveals.

    Every request has its own random generator, old/current state, task cursor,
    and forward budget. Finished requests release their slot immediately.
    """
    from crystal_dlm.expert_edit_data import canonical_body, numeric_positions
    if (block_size not in (1,4,8) or max_calls < 1 or not 1 <= batch_size <= 64
            or not 0 <= accept_threshold <= 1 or proposal_input not in ('masked', 'old_values')):
        raise ValueError('invalid editor sampling limits')
    device = next(model.parameters()).device
    key = (id(tokenizer), str(device), float(temperature))
    cached = getattr(model, '_sampling_support_cache', {})
    if key not in cached:
        vocabulary = tokenizer.get_vocab()
        cached[key] = (vocabulary, {int(value): name for name,value in vocabulary.items()},
                       ExpertEditObjective(tokenizer, device, temperature=temperature))
        model._sampling_support_cache = cached
    vocabulary, inverse, support = cached[key]
    states = []
    for request in requests:
        original = list(map(int, request['body']))
        n = int(request['num_sites'])
        current = canonical_body(original, vocabulary, inverse)
        tasks = tuple(request.get('tasks', ('G','S')))
        if len(current) != 7+4*n or not 1 <= n <= 20 or any(task not in ('G','S') for task in tasks):
            raise ValueError('invalid fixed-composition editor request')
        states.append({'original': original, 'initial': current.copy(), 'current': current,
                       'n': n, 'tasks': tasks, 'task_cursor': 0, 'stage': 'inspect', 'used': 0,
                       'prefix': tokenizer(request['prompt'], add_special_tokens=False)['input_ids'],
                       'generator': torch.Generator(device=device).manual_seed(int(request.get('seed',0))),
                       'trace': []})
    results, active, next_index, completed = [None]*len(states), [], 0, 0
    batches, forward_rows, started_calls = 0, 0, model.forward_calls
    def finish(index):
        nonlocal completed
        state = states[index]
        original, initial, current = state['original'], state['initial'], state['current']
        final = [original[i] if token == initial[i] else token for i,token in enumerate(current)]
        fixed = set(range(len(final)))-set(numeric_positions(state['n']))
        if any(final[i] != original[i] for i in fixed) or state['used'] > max_calls:
            raise RuntimeError('editor changed composition or exceeded its actual per-request budget')
        results[index] = {'body': final, 'canonical_body': current, 'trace': state['trace'],
                          'forward_calls': state['used'], 'changed_numeric_tokens': sum(a != b for a,b in zip(initial,current)),
                          'block_size': block_size, 'scope_policy': force_mode or 'learned', 'accept_all': accept_all,
                          'proposal_input': proposal_input,
                          'S_admission_policy': 'bypassed_for_single_task_forced_scope_diagnostic'
                              if force_mode is not None and len(state['tasks']) == 1 else 'learned'}
        completed += 1
        if progress is not None:
            progress(completed, len(states), batches)
    def next_task(state):
        state['task_cursor'] += 1
        state['stage'] = 'inspect'
    while next_index < len(states) or active:
        while next_index < len(states) and len(active) < batch_size:
            active.append(next_index)
            next_index += 1
        indices, views = [], []
        for index in active:
            state = states[index]
            if state['stage'] == 'inspect':
                while state['task_cursor'] < len(state['tasks']):
                    task = state['tasks'][state['task_cursor']]
                    if allowed_modes.get(task) and state['used']+3 <= max_calls:
                        break
                    state['task_cursor'] += 1
                if state['task_cursor'] >= len(state['tasks']):
                    finish(index)
                    continue
                state['old'] = state['current'].copy()
                current, positions, reveal = state['old'], [], 0.
                remaining = min(max_calls-state['used'], 8+3*state['n'])
            else:
                current, positions = state['candidate'], state['positions']
                reveal = state['offset']/len(positions) if state['stage'] == 'fill' else 1.
                remaining = len(positions)-state['offset']+1 if state['stage'] == 'fill' else len(positions)+2
            task = state['tasks'][state['task_cursor']]
            indices.append(index)
            views.append(inference_view(state['prefix'], state['old'], current, state['n'], int(task=='S'),
                                        positions, remaining=remaining, reveal=reveal))
        active = indices
        if not active:
            continue
        batch = materialize_edit_batch(views, tokenizer, device)
        output = model(batch['input_ids'], attention_mask=batch['attention_mask'], edit_context=batch['edit_context'])
        batches += 1
        forward_rows += len(active)
        for row_index, index in enumerate(active):
            state = states[index]
            state['used'] += 1
            task = state['tasks'][state['task_cursor']]
            if state['stage'] == 'inspect':
                modes = sorted(set(allowed_modes[task]))
                admission = output.quality_logits[row_index].sigmoid().float().tolist()
                diagnostic = force_mode is not None and len(state['tasks']) == 1
                reason = 'learned_S_admission_reject' if task=='S' and not diagnostic and (
                          admission[0] < .5 or admission[1] < .5) else None
                mode = (modes[int(output.mode_logits[row_index,modes].argmax())] if force_mode is None
                        else MODE_NAMES.index(force_mode))
                if mode not in modes:
                    raise ValueError('cannot test an action family absent from training')
                if reason or mode == 0:
                    state['trace'].append({'task': task, 'mode': 'none', 'old_body': state['old'],
                        'proposal_body': state['old'], 'accepted': False, 'quality': admission,
                        'reason': reason or 'learned_none', 'calls': 1})
                    next_task(state)
                    continue
                sites = list(range(state['n']))
                if mode == 1:
                    counts = [i for i,value in enumerate(LOCAL_COUNTS) if value <= state['n']]
                    count = LOCAL_COUNTS[counts[int(output.count_logits[row_index,counts].argmax())]]
                    sites = sorted(output.site_logits[row_index,:state['n']].topk(count).indices.tolist())
                positions = ([1,2,3,4,5,6] if mode == 3 else [])+[8+4*site+axis for site in sites for axis in range(3)]
                needed = (len(positions)+block_size-1)//block_size+1
                if state['used']+needed > max_calls:
                    state['trace'].append({'task': task, 'mode': MODE_NAMES[mode], 'old_body': state['old'],
                        'proposal_body': state['old'], 'accepted': False, 'calls': 1,
                        'reason': 'insufficient_complete_proposal_budget'})
                    next_task(state)
                    continue
                candidate = state['old'].copy()
                # The visible OLD canvas matches the existing T2T training view.
                # Earlier sampled positions still replace OLD before later draws.
                if proposal_input == 'masked':
                    for position in positions:
                        candidate[position] = MASK_TOKEN_ID
                state.update(stage='fill', mode=mode, sites=sites, positions=positions, candidate=candidate,
                             offset=0, proposal_calls=needed+1)
            elif state['stage'] == 'fill':
                for position in state['positions'][state['offset']:state['offset']+block_size]:
                    if position < 4:
                        family, axis = 'length', 'ABC'[position-1]
                    elif position < 7:
                        family, axis = 'angle', 'ABG'[position-4]
                    else:
                        family, axis = 'coord', 'XYZ'[(position-8)%4]
                    logits, ids = support.typed_vector(output.logits, torch.tensor([row_index],device=device),
                         torch.tensor([len(state['prefix'])+position],device=device), family, axis)
                    selected = torch.multinomial(logits[0].softmax(-1), 1, generator=state['generator'])
                    state['candidate'][position] = int(ids[selected])
                state['offset'] = min(len(state['positions']),state['offset']+block_size)
                if state['offset'] == len(state['positions']):
                    state['stage'] = 'judge'
            else:
                quality = output.quality_logits[row_index].sigmoid().float().tolist()
                accepted = quality[3] >= accept_threshold and state['candidate'] != state['old']
                state['trace'].append({'task': task, 'mode': MODE_NAMES[state['mode']], 'sites': state['sites'],
                    'positions': state['positions'], 'old_body': state['old'], 'proposal_body': state['candidate'],
                    'quality': quality, 'accepted': accepted, 'applied': bool(accepted or accept_all),
                    'calls': state['proposal_calls'], 'reason': 'learned_accept' if accepted else 'learned_reject'})
                if accepted or accept_all:
                    state['current'] = state['candidate']
                next_task(state)
        del output
    if model.forward_calls-started_calls != batches or sum(row['forward_calls'] for row in results) != forward_rows:
        raise RuntimeError('actual model batches or per-request forward evaluations were miscounted')
    return {'results': results, 'forward_batches': batches, 'forward_rows': forward_rows}
