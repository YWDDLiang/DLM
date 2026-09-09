"""Versioned TRAIN-only quality and action contracts for ranked RSI.

All ranks describe measured endpoints, never a claim about an unscored target.
Ordinary improvement is a pairwise relation, not a permanent structure class.
"""
from collections import Counter
import math

SCHEMA = 'ranked_physics_feedback_v2'
LEVELS = ('unstable', 'ordinary_improvement', 'meta_stable', 'strict_stable', 'SUN')
INVALID = frozenset(('invalid_raw', 'invalid_terminal', 'relaxation_energy_increased'))
MODES = ('none', 'local_xyz', 'all_xyz', 'full_cell')
COUNTS = (1, 2, 4, 8)


def endpoint_quality(score):
    hull = score.get('e_above_hull_eV_atom')
    finite = isinstance(hull, (float, int)) and not isinstance(hull, bool) and math.isfinite(hull)
    status = score.get('terminal_status', score.get('status'))
    invalid = status in INVALID
    verified = score.get('terminal_verified', score.get('verified')) is True
    reliable = bool(verified and finite and not invalid)
    rank = None
    if reliable:
        sun = score.get('strict_sun')
        rank = 4 if sun is True else 3 if hull <= 0 else 2 if hull <= .1 else 0
    return dict(reliable=reliable, known_failure=invalid, status=status,
                rank=rank, level=LEVELS[rank] if rank is not None else None,
                hull=float(hull) if finite else None, SUN=score.get('strict_sun'))


def ranked_preference(before, after, *, margin=.01):
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError('positive fixed comparison margin required')
    a, b = endpoint_quality(before), endpoint_quality(after)
    result = dict(schema=SCHEMA, before=a, after=b, chosen=None,
                  objective_level=None, priority=0., reason='unknown_quality',
                  margin_eV_atom=margin)
    if a['known_failure'] and b['reliable'] and b['rank'] >= 2:
        result.update(chosen='after', objective_level=b['level'], priority=float(2**b['rank']),
                      reason='reliable_target_over_invalid_endpoint')
    elif b['known_failure'] and a['reliable'] and a['rank'] >= 2:
        result.update(chosen='before', objective_level=a['level'], priority=float(2**a['rank']),
                      reason='preserve_reliable_target_from_invalid_endpoint')
    elif a['reliable'] and b['reliable']:
        if a['rank'] != b['rank']:
            best = b if b['rank'] > a['rank'] else a
            result.update(chosen='after' if b['rank'] > a['rank'] else 'before',
                          objective_level=best['level'], priority=float(2**best['rank']),
                          reason='quality_level_transition')
        elif a['rank'] == 4:
            result.update(objective_level='SUN', reason='preserve_equal_SUN')
        elif abs(a['hull']-b['hull']) > margin:
            level = a['level'] if a['rank'] >= 2 else 'ordinary_improvement'
            result.update(chosen='after' if b['hull'] < a['hull'] else 'before',
                          objective_level=level, priority=1., reason='reliable_same_level_energy')
        else:
            result.update(objective_level=a['level'], reason='quality_tie')
    result['fallback_reason'] = (None if result['objective_level'] == 'SUN' else
                                'NU_unavailable' if a['SUN'] is None or b['SUN'] is None else 'no_SUN_transition')
    return result


def align_fixed_slots(target, reference):
    """Reorder entire atom blocks without changing their physical information."""
    if len(target) != len(reference) or len(target) < 11 or (len(target)-7) % 4:
        raise ValueError('fixed-count body shape mismatch')
    if target[0] != reference[0]:
        raise ValueError('teacher changed atom count')
    remaining = [tuple(target[i:i+4]) for i in range(7, len(target), 4)]
    original = Counter(remaining)
    aligned = list(target[:7])
    permutation = []
    indices = list(range(len(remaining)))
    for element in reference[7::4]:
        index = next((j for j, block in enumerate(remaining) if block[0] == element), None)
        if index is None:
            raise ValueError('teacher changed elemental composition')
        aligned.extend(remaining.pop(index)); permutation.append(indices.pop(index))
    if remaining or Counter(tuple(aligned[i:i+4]) for i in range(7,len(aligned),4)) != original:
        raise ValueError('atom permutation lost physical information')
    return aligned, permutation


def fixed_slots_match(target, reference):
    return (len(target) == len(reference) and target[0] == reference[0]
            and target[7::4] == reference[7::4])


def action_positions(n, mode, sites=()):
    if mode not in range(4) or not 1 <= n <= 20:
        raise ValueError('invalid edit action')
    if mode == 0:
        return []
    selected = sorted(set(sites)) if mode == 1 else list(range(n))
    if not selected or any(i < 0 or i >= n for i in selected):
        raise ValueError('invalid site selection')
    return (list(range(1,7)) if mode == 3 else []) + [8+4*i+k for i in selected for k in range(3)]


def target_action(current, target):
    if not fixed_slots_match(target, current):
        raise ValueError('content target differs from fixed inference slots')
    n=(len(current)-7)//4
    sites=[i for i in range(n) if current[8+4*i:11+4*i] != target[8+4*i:11+4*i]]
    if current[1:7] != target[1:7]:
        mode=3
    elif not sites:
        mode=0
    elif len(sites) <= min(8,n):
        supported=[v for v in COUNTS if len(sites) <= v <= n]
        if supported:
            count=supported[0]
            sites=sorted(sites+[i for i in range(n) if i not in sites][:count-len(sites)])
            mode=1
        else: mode=2
    else: mode=2
    return dict(mode=mode, name=MODES[mode], sites=sites if mode==1 else list(range(n)) if mode else [],
                positions=action_positions(n,mode,sites))


def validate_action_target(current, target, positions):
    if not fixed_slots_match(target,current):
        raise ValueError('off-support fixed element/count target')
    allowed=set(positions)
    if any(a!=b and i not in allowed for i,(a,b) in enumerate(zip(current,target))):
        raise ValueError('target edits positions outside its conditioned action')


def joint_stop_status(forces, stresses_GPa, *, fmax=.1, stress_tolerance=.5):
    force=max((math.sqrt(sum(float(v)**2 for v in row)) for row in forces), default=float('inf'))
    stress=max((abs(float(v)) for row in stresses_GPa for v in row), default=float('inf'))
    if not math.isfinite(force) or not math.isfinite(stress):
        raise ValueError('nonfinite physical convergence observation')
    return dict(force_max_eV_A=force, stress_max_GPa=stress,
                physical_converged=force<=fmax and stress<=stress_tolerance)


def ranked_relaxation_protocol(common):
    return {**common, 'max_steps':1000, 'optimizer_stop':'joint_atomic_force_and_stress_v1'}
