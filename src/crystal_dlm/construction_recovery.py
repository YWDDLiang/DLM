"""Three-stage recovery at DLM geometry support, with a final Z-only escape."""
from itertools import product
import torch

from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.llada_generation import _lattice_matrix_from_token_ids
from crystal_dlm.post_refine_contract import derived_seed


def restrict_training_content(example, branch, supported):
    """Retain physical/head supervision without inventing strict-support logp."""
    fields = ['chosen_tokens','rejected_tokens','healthy_anchor_tokens','content_target_tokens']
    bad = [key for key in fields if example.get(key) is not None and not supported(example[key])]
    if not bad: return True, []
    chosen = example.get('chosen_tokens')
    anchor = example.get('healthy_anchor_tokens') if branch == 'G' else example.get('content_target_tokens')
    target = chosen if chosen is not None and supported(chosen) else anchor if anchor is not None and supported(anchor) else None
    example['chosen_tokens'] = example['rejected_tokens'] = None
    example.pop('healthy_anchor_tokens',None);example.pop('content_target_tokens',None)
    if target is not None: example['healthy_anchor_tokens' if branch == 'G' else 'content_target_tokens'] = target
    example['strict_content_support_exclusions'] = bad
    return branch == 'E' or target is not None, bad


def failed_sites(body, n, failure):
    sites = {(int(p)-8)//4 for p in failure.get('failure_positions', []) if 8 <= int(p) < len(body)}
    if not sites:
        sites = {i for i in range(n) if MASK_TOKEN_ID in body[8+4*i:11+4*i]}
    return sorted(sites or set(range(n)))


def neighbor_sites(body, n, centers, constraints, radius=2.0):
    """All nearby sites plus four nearest; unknown Z is projected out."""
    cell = _lattice_matrix_from_token_ids(torch.tensor(body), prompt_length=0, constraints=constraints)
    if cell is None or not bool(torch.isfinite(cell).all()):
        return list(range(n))
    cell = cell.double(); selected = set(centers)
    maps = constraints['coord_token_to_bin']
    coords = [[maps[a].get(int(body[8+4*i+j])) for j, a in enumerate('XYZ')] for i in range(n)]
    shifts = torch.tensor(list(product(range(-2, 3), repeat=3)), dtype=torch.double)
    for center in centers:
        distances = []
        for other in range(n):
            if other == center: continue
            missing = [j for j in range(3) if coords[center][j] is None or coords[other][j] is None]
            delta = torch.tensor([(coords[center][j]-coords[other][j])/100. if j not in missing else 0.
                                  for j in range(3)], dtype=torch.double)
            vectors = (shifts+delta) @ cell
            if missing:
                basis = cell[missing]
                vectors = vectors - (vectors @ torch.linalg.pinv(basis)) @ basis
            distances.append((float(torch.linalg.vector_norm(vectors, dim=1).min()), other))
        distances.sort()
        selected.update(i for distance, i in distances if distance <= radius)
        selected.update(i for _, i in distances[:4])
    return sorted(selected)


def reopen(body, n, sites, *, cell=False, z_only=False):
    result = list(body)
    positions = (list(range(1, 7)) if cell else []) + [8+4*i+j for i in sites for j in ([2] if z_only else range(3))]
    for position in positions: result[position] = MASK_TOKEN_ID
    return result, positions


def construct_cascade(model, tokenizer, task, runtime, *, construct, constraints, repair_constraints,
                      geometry_api, complete_geometry):
    n = int(task['plan_state']['N']); initial = None; episodes = []; stages = ['draft', 'failed_XYZ', 'neighbor_XYZ', 'all_numeric', 'final_Z_relaxed']
    centers = []; opened = []
    for stage_index, stage in enumerate(stages):
        relaxed = stage == 'final_Z_relaxed'
        seed = None if stage_index == 0 else derived_seed(str(task['body_noise_seed']), 'geometry_recovery_'+stage, 1)
        try:
            body, metadata = construct(model, tokenizer, [task], runtime, constraints=constraints,
                geometry_api=geometry_api, initial_body=initial, noise_seed_override=seed, relax_final_z=relaxed)
            support = complete_geometry(body[0].tolist())
            if support['supported'] or (relaxed and support.get('reason') == 'native_pair_below_0.5A'):
                episodes.append({'stage':stage,'seed':seed,'opened_positions':opened,'completed':True,
                                 'complete_geometry':support,'distance_hard_limit_relaxed':relaxed})
                metadata.update(complete_geometry=support, construction_recovery={
                    'schema':'DLM_geometry_three_stage_v1','episodes':episodes,'recoveries_used':stage_index,
                    'final_Z_relaxed':relaxed,'Plan_replacement_or_resampling':False,
                    'relaxed_generation_is_not_physical_validity':True,'neighbor_radius_A':2.0})
                return body, metadata
            error = geometry_api.GeometryNoLegalSupport({'reason':support['reason'],'complete_geometry':support}, body, 0)
        except geometry_api.GeometryNoLegalSupport as caught:
            error = caught
        partial = error.partial_canvas[0, error.prompt_length:].tolist()
        episodes.append({'stage':stage,'seed':seed,'opened_positions':opened,'completed':False,'failure':error.to_dict()})
        if stage_index == 4:
            error.details['recovery_episodes'] = episodes
            raise error
        centers = sorted(set(centers) | set(failed_sites(partial, n, error.details)))
        if stage_index == 0:
            initial, opened = reopen(partial, n, centers)
        elif stage_index == 1:
            sites = neighbor_sites(partial, n, centers, repair_constraints)
            initial, opened = reopen(partial, n, sites)
        elif stage_index == 2:
            initial, opened = reopen(partial, n, range(n), cell=True)
        else:
            # Only the last coordinate group loses distance support. Lattice,
            # element identities and completed X/Y remain exactly as generated.
            initial, opened = reopen(partial, n, range(n), z_only=True)
