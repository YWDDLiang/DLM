"""Explicit old official-cache adapter, without network or generated energies."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tests.test_r03_hull_union import H, cache_fixture, input_manifest, planner_row, write_json, write_rows, ROOT


def legacy_cache(root):
    cache_fixture(root,resolved=['Cl-Na','Cu-S-Se-V'],errors={'O-Yb':{
        'type':'ContractError','http_status':None,'message':"missing unary references: ['Yb']"}})
    original=H.read_json(root/'completion_manifest.json')
    audits=H.read_jsonl(root/'query_audit.jsonl')
    for row in audits:
        row['query_method']='MPRester.get_entries_in_chemsys'
        if row['query_status']=='resolved':
            row['entries_sha256']=row.pop('slim_entries_sha256')
    write_rows(root/'query_audit.jsonl',audits)
    report={'schema':'h1_r03_best_fresh_official_mp_query_v1','database_version':'2026.04.13',
            'query_method':'MPRester.get_entries_in_chemsys','compatible_only':True,'thermo_type':'GGA_GGA+U',
            'additional_criteria':{'thermo_types':['GGA_GGA+U']},'fresh_empty_cache':True,
            'wanted_chemsys':3,'resolved_chemsys':2,'unresolved_chemsys':1,
            'cache_identities':{key:H.identity(root/name) for key,name in (
                ('resolved','official_slim_cache.jsonl'),('unresolved','unresolved_chemsys.jsonl'),('query_audit','query_audit.jsonl'))}}
    write_json(root/'completion_manifest.json',report)
    return root


class LegacySupplementTests(unittest.TestCase):
    def test_only_missing_resolved_entries_are_imported_with_original_audit(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);legacy=legacy_cache(root/'legacy')
            result=H.load_legacy_resolved_supplement(legacy,{'Cu-S-Se-V','O-Yb'})
            self.assertEqual(set(result['resolved']),{'Cu-S-Se-V'})
            self.assertEqual(result['official_unresolved'],{})
            self.assertFalse(result['audits']['Cu-S-Se-V']['new_query'])
            self.assertEqual(result['identity']['schema'],'h1_r03_best_fresh_official_mp_query_v1')
            self.assertEqual(result['audits']['Cu-S-Se-V']['legacy_original_query_audit']['query_index'],1)

    def test_union_roundtrip_preserves_primary_and_fills_one_gap(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);primary=cache_fixture(root/'primary',resolved=['Cl-Na'])
            legacy=legacy_cache(root/'legacy')
            source=input_manifest(root,[('R','planner',[planner_row(0,['Na','Cl']),planner_row(1,['Cu','S','Se','V'])])])
            config=root/'config.json'
            write_json(config,{'thermo':{**H.THERMO,'fresh_empty_cache':True,'reuse_any_historical_or_august_cache':False},
                               'runtime':{'official_mp_python':'fixture'}})
            report=H.prepare(inputs_manifest=source,known_caches=[primary],legacy_resolved_caches=[legacy],
                             query_config=config,query_source=ROOT/'eval_runtime',run_root=root/'union')
            self.assertEqual(report['missing_chemsys'],[])
            H.finalize(run_root=root/'union')
            result=H.load_cache(root/'union/official_mp_cache')
            self.assertEqual(set(result['resolved']),{'Cl-Na','Cu-S-Se-V'})
            provenance=H.read_jsonl(root/'union/official_mp_cache/cache_provenance.jsonl')
            self.assertEqual(provenance[0]['cache']['directory'],str(primary.resolve()))
            self.assertEqual(provenance[1]['cache']['directory'],str(legacy.resolve()))

    def test_changed_entries_fail_even_if_outer_file_hash_was_rewritten(self):
        with TemporaryDirectory() as directory:
            root=legacy_cache(Path(directory)/'legacy')
            rows=H.read_jsonl(root/'official_slim_cache.jsonl');rows[1]['entries'][0]['energy']=-99.
            write_rows(root/'official_slim_cache.jsonl',rows)
            report=H.read_json(root/'completion_manifest.json')
            report['cache_identities']['resolved']=H.identity(root/'official_slim_cache.jsonl')
            write_json(root/'completion_manifest.json',report)
            with self.assertRaisesRegex(H.HullUnionError,'recorded official query'):
                H.load_legacy_resolved_supplement(root,{'Cu-S-Se-V'})


if __name__=='__main__':unittest.main()
