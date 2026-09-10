"""Lossless continuous KEEP and fixed-slot token patches on the same crystal."""
from __future__ import annotations
import copy
import numpy as np


def structure_of(record):
    from pymatgen.core import Structure
    from crystal_dlm.dynamic_crystal import parse_dynamic_answer, arrays_to_structure
    if record.get('structure') is not None:
        return Structure.from_dict(record['structure'])
    if record.get('body'):
        return arrays_to_structure(parse_dynamic_answer(record['body'], strict=True))
    raise ValueError('no_structure_or_token_body')


def geometry(record):
    from crystal_dlm.expert_edit_data import arrays_from_structure, certify_geometry
    try:
        return certify_geometry(arrays_from_structure(structure_of(record).as_dict()))
    except (ValueError, TypeError, KeyError, FloatingPointError) as error:
        return dict(valid=False, certified=True, reason=str(error))


def native_current(refined_wrapper, token_record):
    """Choose representation without consulting endpoint stability or novelty."""
    native = refined_wrapper['record']
    trace = refined_wrapper['tokenization']
    if refined_wrapper.get('raw_refiner_output') is None:
        return copy.deepcopy(token_record), dict(source='existing_token_bypass_or_F_failure',
            editable=bool(token_record.get('body_token_ids')), reason=trace.get('reason'))
    check = geometry(native)
    if check['valid'] is not True:
        return copy.deepcopy(token_record), dict(source='existing_token_geometry_fallback',
            editable=bool(token_record.get('body_token_ids')), reason=check['reason'], geometry=check)
    editable = not trace.get('fallback_to_raw', False) and bool(token_record.get('body_token_ids'))
    return copy.deepcopy(native), dict(source='continuous_F', editable=editable,
        reason=None if editable else 'continuous_valid_without_matching_token_view', geometry=check)


def commit_patch(current_record, old_tokens, new_tokens, inverse, *, editable=True):
    """Unchanged numeric fields retain their exact stored continuous values."""
    from pymatgen.core import Lattice
    from crystal_dlm.expert_edit_data import decode_body, fixed_composition
    if not editable:
        return copy.deepcopy(current_record), dict(applied=False, reason='no_matching_token_view', changed=[])
    if list(old_tokens) == list(new_tokens):
        return copy.deepcopy(current_record), dict(applied=False, reason='KEEP', changed=[])
    try:
        old = decode_body(old_tokens, inverse); new = decode_body(new_tokens, inverse)
        n = len(old['species'])
        if not fixed_composition(old_tokens, new_tokens, n):
            raise ValueError('fixed_atom_count_or_species_changed')
        native = structure_of(current_record)
        if [str(s.specie) for s in native] != old['species']:
            raise ValueError('continuous_and_token_site_order_differ')
        changed = []
        for i,(a,b) in enumerate(zip(old_tokens,new_tokens,strict=True)):
            if a==b:continue
            if i>=8:
                site,axis=divmod(i-8,4)
                if axis<=2 and float(old['frac_coords'][site][axis])%1. == float(new['frac_coords'][site][axis])%1.:
                    continue
            changed.append(i)
        if not changed:
            return copy.deepcopy(current_record),dict(applied=False,reason='periodic_equivalent_KEEP',changed=[])
        structure = copy.deepcopy(current_record.get('structure') or native.as_dict())
        lattice_changed = any(1 <= i <= 6 for i in changed)
        if lattice_changed:
            lengths, angles = list(native.lattice.abc), list(native.lattice.angles)
            for i in changed:
                if 1 <= i <= 3: lengths[i - 1] = new['lengths'][i - 1]
                if 4 <= i <= 6: angles[i - 4] = new['angles'][i - 4]
            lattice = Lattice.from_parameters(*lengths, *angles)
            structure['lattice'] = dict(matrix=lattice.matrix.tolist(), pbc=list(lattice.pbc),
                a=lattice.a, b=lattice.b, c=lattice.c, alpha=lattice.alpha,
                beta=lattice.beta, gamma=lattice.gamma, volume=lattice.volume)
        matrix = np.asarray(structure['lattice']['matrix'], dtype=float)
        changed_sites = set()
        for position in changed:
            if position >= 8:
                site, axis = divmod(position - 8, 4)
                if axis > 2: raise ValueError('non_numeric_edit')
                structure['sites'][site]['abc'][axis] = float(new['frac_coords'][site][axis]) % 1.
                changed_sites.add(site)
        for site in range(n):
            if lattice_changed or site in changed_sites:
                structure['sites'][site]['xyz'] = (np.asarray(structure['sites'][site]['abc']) @ matrix).tolist()
        result = copy.deepcopy(current_record)
        result.update(structure=structure, body=None, success=True, reason=None)
        result.pop('body_token_ids', None)
        check = geometry(result)
        if check['valid'] is not True:
            return copy.deepcopy(current_record), dict(applied=False, reason='hybrid_geometry_revert',
                changed=changed, geometry=check)
        return result, dict(applied=True, reason='changed_numeric_fields_only', changed=changed,
            lattice_changed=lattice_changed, changed_sites=sorted(changed_sites), geometry=check)
    except (ValueError, KeyError, TypeError, IndexError, FloatingPointError) as error:
        return copy.deepcopy(current_record), dict(applied=False, reason='invalid_patch_revert:'+str(error), changed=[])


def quantization_error(native, token_record):
    a, b = structure_of(native), structure_of(token_record)
    if [str(s.specie) for s in a] != [str(s.specie) for s in b]:
        raise ValueError('precision_comparison_changed_species_order')
    delta = a.frac_coords - b.frac_coords
    delta -= np.round(delta)
    return dict(max_length_error_A=float(np.max(np.abs(np.asarray(a.lattice.abc)-b.lattice.abc))),
        max_angle_error_deg=float(np.max(np.abs(np.asarray(a.lattice.angles)-b.lattice.angles))),
        max_fractional_error=float(np.max(np.abs(delta))),
        max_coordinate_error_A=float(np.max(np.linalg.norm(delta @ a.lattice.matrix, axis=-1))))
