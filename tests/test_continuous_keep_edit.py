import copy
import unittest
from pymatgen.core import Lattice, Structure
from crystal_dlm.continuous_keep_edit import commit_patch, native_current, quantization_error
from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens
from crystal_dlm.expert_edit_data import arrays_from_structure
from crystal_dlm.fixed_slot import FixedSlotConfig


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


if __name__ == '__main__': unittest.main()
