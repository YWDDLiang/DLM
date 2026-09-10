"""SUN-specific gain targets, observable state features and fixed candidate choice."""
from __future__ import annotations
import math


EXTRA_FEATURES=('current_stable','current_meta','current_novel_known','current_novel',
    'current_verified','current_known_failure','current_not_converged','current_hull_known',
    'current_hull_clipped','current_hull_signed_log','raw_force_scaled','raw_stress_scaled',
    'sites_scaled','changed_site_fraction','changed_token_fraction','cell_changed',
    'fractional_displacement_rms','fractional_displacement_max','cartesian_displacement_rms',
    'cartesian_displacement_max','geometry_delta_available')


def nested_probabilities(logits):
    """The NS event is a subset of NMS; both outputs describe candidates."""
    probability=logits.sigmoid()
    return __import__('torch').stack((probability[...,0]*probability[...,1],probability[...,1]),dim=-1)


def select_scored_state(candidates):
    """Compare the learned KEEP view with edits, without acceptance floors."""
    valid=[c for c in candidates if c['valid']]
    for candidate in valid:
        if not all(math.isfinite(candidate[key]) for key in ('sun_gain','ms_gain')):
            raise ValueError('nonfinite state prediction')
    return max(valid,key=lambda c:(c['sun_gain'],c['ms_gain'],c['stream']=='keep'))['stream'] if valid else 'keep'


def endpoint_targets(score):
    from crystal_dlm.ranked_feedback import endpoint_quality
    q=endpoint_quality(score)
    if q['known_failure']:return (0.,0.),'known_failure'
    if score.get('terminal_status')=='not_converged' and score.get('e_above_hull_eV_atom') is not None:
        return (0.,0.),'not_converged_operational_zero'
    if not q['reliable']:return None,'unknown_physics'
    stable=score.get('strict_stable') is True;meta=score.get('meta_stable') is True
    if not stable and not meta:return (0.,0.),'verified_nonstable'
    if score.get('novel') is None:return None,'unknown_novelty'
    novel=score['novel'] is True
    return (float(stable and novel),float(meta and novel)),'verified'


def extra_features(score,current,candidate,trace,n):
    import numpy as np
    from crystal_dlm.ranked_feedback import endpoint_quality
    from crystal_dlm.continuous_keep_edit import structure_of
    q=endpoint_quality(score);hull=score.get('e_above_hull_eV_atom');known=hull is not None and math.isfinite(hull)
    hull=float(hull) if known else 0.
    raw=score.get('raw') or {}
    def raw_value(key):
        value=raw.get(key)
        return math.tanh(float(value)/10) if isinstance(value,(int,float)) and math.isfinite(value) else 0.
    positions=trace.get('action',{}).get('positions',[])
    sites={int((p-8)//4) for p in positions if p>=8}
    values=[float(q['reliable'] and score.get('strict_stable') is True),
        float(q['reliable'] and score.get('meta_stable') is True),float(score.get('novel') is not None),
        float(score.get('novel') is True),float(q['reliable']),float(q['known_failure']),
        float(score.get('terminal_status')=='not_converged'),float(known),max(-1.,min(1.,hull)),
        math.copysign(math.log1p(abs(hull)*100),hull)/5,raw_value('force_max_eV_A'),raw_value('stress_max_GPa'),
        n/20,len(sites)/n,len(positions)/(6+3*n),float(any(p<7 for p in positions))]
    try:
        a,b=structure_of(current),structure_of(candidate)
        if [str(s.specie) for s in a]!=[str(s.specie) for s in b]:raise ValueError('site order changed')
        delta=np.asarray(b.frac_coords)-np.asarray(a.frac_coords);delta-=np.round(delta)
        frac=np.linalg.norm(delta,axis=1);cart=np.linalg.norm(delta@np.asarray(a.lattice.matrix),axis=1)
        values += [float(np.sqrt(np.mean(frac**2))),float(frac.max()),
            float(np.sqrt(np.mean(cart**2)))/10,float(cart.max())/10,1.]
    except (ValueError,KeyError,TypeError):values += [0.]*5
    if len(values)!=len(EXTRA_FEATURES) or not all(math.isfinite(v) for v in values):
        raise ValueError('invalid observable ranker features')
    return values


def select_candidate(candidates,*,sun_threshold,ms_floor,tie_width=.01,known_sun=False):
    """Candidates carry only model outputs and geometry validity, never outcomes."""
    if known_sun:return None
    eligible=[]
    for order,c in enumerate(candidates):
        if not c['valid']:continue
        sun,ms=c['sun_gain'],c['ms_gain']
        if not all(math.isfinite(x) for x in (sun,ms)):raise ValueError('nonfinite SUN rank prediction')
        if sun>=sun_threshold and ms>=ms_floor:
            eligible.append((math.floor(sun/tie_width+1e-10),ms,-order,c['stream']))
    return max(eligible)[3] if eligible else None
