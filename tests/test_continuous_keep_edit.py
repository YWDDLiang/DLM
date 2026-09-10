import copy
import importlib.util
from pathlib import Path
import unittest
from pymatgen.core import Lattice, Structure
from crystal_dlm.continuous_keep_edit import commit_patch, native_current, quantization_error
from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens
from crystal_dlm.expert_edit_data import arrays_from_structure
from crystal_dlm.fixed_slot import FixedSlotConfig
from crystal_dlm.utility_acceptance import materialize_continuous_patch,materialized_continuous_decision


class ContinuousKeepEditTests(unittest.TestCase):
    def setUp(self):
        s = Structure(Lattice.from_parameters(5.031234, 5.071234, 5.021234, 90.11, 90.22, 90.33),
                      ['Na','Cl'], [[.123456,.234567,.345678],[.623456,.734567,.845678]])
        a = arrays_from_structure(s.as_dict())
        tokens, _ = arrays_to_dynamic_tokens(a['lengths'], a['angles'], a['species'], a['frac_coords'], config=FixedSlotConfig())
        self.inverse = dict(enumerate(tokens)); self.tokens = list(self.inverse)
        self.current = dict(structure=s.as_dict(), body=None, success=True)

    def replace(self, position, value):
        new = list(self.tokens); ident = max(self.inverse)+1
        self.inverse[ident] = value; new[position] = ident
        return new

    def test_keep_exact_values_and_independent_copy(self):
        result, trace = commit_patch(self.current, self.tokens, self.tokens, self.inverse)
        self.assertEqual(result, self.current); self.assertFalse(trace['applied'])
        result['structure']['sites'][0]['abc'][0] = 0
        self.assertNotEqual(result, self.current)

    def test_local_edit_retains_lattice_other_sites_and_axes(self):
        result, trace = commit_patch(self.current, self.tokens, self.replace(8,'<X_020>'),self.inverse)
        self.assertTrue(trace['applied'])
        self.assertEqual(result['structure']['lattice'], self.current['structure']['lattice'])
        self.assertEqual(result['structure']['sites'][1], self.current['structure']['sites'][1])
        self.assertEqual(result['structure']['sites'][0]['abc'][1:], self.current['structure']['sites'][0]['abc'][1:])
        self.assertEqual(result['structure']['sites'][0]['abc'][0], .2)

    def test_lattice_edit_retains_fractional_values(self):
        token = self.inverse[1]; prefix, number = token.rsplit('_',1)
        new = self.replace(1, prefix+'_'+str(int(number[:-1])+1).zfill(len(number)-1)+'>')
        result, trace = commit_patch(self.current, self.tokens, new, self.inverse)
        self.assertTrue(trace['applied']); self.assertTrue(trace['lattice_changed'])
        self.assertEqual([s['abc'] for s in result['structure']['sites']], [s['abc'] for s in self.current['structure']['sites']])
        self.assertAlmostEqual(result['structure']['lattice']['b'], self.current['structure']['lattice']['b'])

    def test_geometry_failure_reverts(self):
        changed = list(self.tokens)
        for j, axis in enumerate('XYZ'):
            token = '<'+axis+'_'+['062','073','085'][j]+'>'
            ident = max(self.inverse)+1; self.inverse[ident]=token; changed[8+j]=ident
        result, trace = commit_patch(self.current,self.tokens,changed,self.inverse)
        self.assertEqual(result,self.current); self.assertEqual(trace['reason'],'hybrid_geometry_revert')

    def test_no_token_view_and_species_change_are_kept(self):
        result, trace = commit_patch(self.current,[],[],self.inverse,editable=False)
        self.assertEqual(result,self.current); self.assertEqual(trace['reason'],'no_matching_token_view')
        result, trace = commit_patch(self.current,self.tokens,self.replace(7,'<E_K>'),self.inverse)
        self.assertEqual(result,self.current); self.assertFalse(trace['applied'])

    def test_valid_continuous_without_quantized_view_is_preserved(self):
        wrapper = dict(record=self.current, raw_refiner_output={'existing':True},
                       tokenization={'fallback_to_raw':True})
        result, trace = native_current(wrapper,dict(success=True,body_token_ids=self.tokens))
        self.assertEqual(result,self.current); self.assertFalse(trace['editable'])

    def test_periodic_coordinate_edit_wraps_and_preserves_other_values(self):
        result, trace = commit_patch(self.current,self.tokens,self.replace(8,'<X_100>'),self.inverse)
        self.assertTrue(trace['applied']); self.assertEqual(result['structure']['sites'][0]['abc'][0],0.)
        self.assertEqual(result['structure']['sites'][0]['abc'][1:],self.current['structure']['sites'][0]['abc'][1:])

    def test_periodic_token_alias_does_not_erase_continuous_residual(self):
        structure=Structure.from_dict(self.current['structure'])
        structure.translate_sites([0],[.003-structure[0].frac_coords[0],0.,0.],frac_coords=True,to_unit_cell=False)
        self.current['structure']=structure.as_dict();self.inverse[8]='<X_000>'
        result,trace=commit_patch(self.current,self.tokens,self.replace(8,'<X_100>'),self.inverse)
        self.assertEqual(result,self.current);self.assertFalse(trace['applied'])
        self.assertEqual(trace['reason'],'periodic_equivalent_KEEP')

    def test_materialized_selection_keeps_exact_committed_bytes(self):
        native=dict(record=self.current,continuous_trace=dict(editable=True))
        proposal=self.replace(8,'<X_020>');trace=dict(proposal_generated=True,proposal_tokens=proposal)
        bound=materialize_continuous_patch(native,self.tokens,proposal,self.inverse)
        selected,decision=materialized_continuous_decision(native,self.tokens,trace,bound,.1,.05)
        self.assertEqual(selected,bound['record']);self.assertTrue(decision['actual_edit'])
        selected,decision=materialized_continuous_decision(native,self.tokens,trace,bound,.01,.05)
        self.assertEqual(selected,self.current);self.assertFalse(decision['actual_edit'])

    def test_materialized_proposal_rejects_changed_source_or_candidate(self):
        native=dict(record=self.current,continuous_trace=dict(editable=True))
        proposal=self.replace(8,'<X_020>');trace=dict(proposal_generated=True,proposal_tokens=proposal)
        bound=materialize_continuous_patch(native,self.tokens,proposal,self.inverse)
        changed=copy.deepcopy(native);changed['record']['structure']['sites'][0]['abc'][1]+=.001
        with self.assertRaises(ValueError):materialized_continuous_decision(changed,self.tokens,trace,bound,.1,.05)
        bound['record']['structure']['sites'][0]['xyz'][0]+=.000001
        with self.assertRaises(ValueError):materialized_continuous_decision(native,self.tokens,trace,bound,.1,.05)

    def test_partition_co_locates_scaled_compositions(self):
        path=Path(__file__).resolve().parents[1]/'operations/r03_c3fd_main_20260907/prepare_keep_edit_focus.py'
        spec=importlib.util.spec_from_file_location('focus_partition_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        formulas=['Mg2Pb2','Mg4Pb4','Cu2O2','Cu4O4','NaCl','LiBr','KF']
        plans=[dict(ancestor_id=str(i),original_ordinal=i,plan_state=dict(reduced_formula=f)) for i,f in enumerate(formulas)]
        rows=module.partition(plans,[plans[0]])
        self.assertEqual(rows[1]['split'],'train');self.assertTrue(rows[1]['old_E_seen_formula'])
        self.assertEqual(rows[0]['canonical_reduced_formula'],rows[1]['canonical_reduced_formula'])
        self.assertEqual(rows[2]['split'],rows[3]['split'])


if __name__ == '__main__': unittest.main()
