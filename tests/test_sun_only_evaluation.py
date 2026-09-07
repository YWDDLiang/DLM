"""All-request SUN accounting and exact input/label/protocol binding."""
import copy
import hashlib
import importlib.util
import json
import os
import contextlib
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pymatgen.core import Lattice, Structure
from crystal_dlm.terminal_energy_consistency import COMMON_RELAXATION_PROTOCOL, LABEL_GEOMETRY_PROTOCOL, TERMINAL_VERIFICATION_PROTOCOL


SPEC = importlib.util.spec_from_file_location('sun_only_eval_test', Path(__file__).resolve().parents[1]/'scripts/evaluate_programmed_paths.py')
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def fixture(root, size=1200):
    paths, labels_dir, cache = root/'paths.jsonl', root/'labels', root/'official'
    labels_dir.mkdir()
    cache.mkdir()
    structures = [Structure(Lattice.cubic(4), [symbol], [[0.,0.,0.]]).as_dict() for symbol in ('Na','Na','Li')]
    rows = [{'trajectory_id': f'eval:{i}', 'sample_idx': i, 'evaluation_ordinal': i,
             'group_id': f'group:{i}', 'source_row_idx': i, 'source_split': 'evaluation', 'endpoint': 'native',
             'success': i < 3, 'parseable': i < 3, 'structure': structures[i] if i < 3 else None,
             'body': None} for i in range(size)]
    write_rows(paths, rows)
    model = root/'model.pth'
    model.write_bytes(b'pinned-physical-model')
    runtime = {'model': 'CHGNet-0.3.0', 'model_checkpoint_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
               'chgnet_package': 'fixture', 'ase_package': 'fixture', 'torch_package': 'fixture',
               'pymatgen_package': 'fixture', 'labeler_sha256': 'a'*64}
    labels = [{**{key: row[key] for key in ('trajectory_id','group_id','source_row_idx','source_split','endpoint')},
               'status': 'verified' if row['success'] else 'generation_failure', 'verified': row['success'],
               'raw_energy': [1.,0.,.1][i] if i < 3 else None,
               'terminal_energy': [1.,0.,.1][i] if i < 3 else None,
               'versions': runtime, 'endpoint_cache_key': EVAL.endpoint_cache_key(row)} for i,row in enumerate(rows)]
    write_rows(labels_dir/'labels.jsonl', labels)
    report = {'purpose': 'evaluation', 'requested': size, 'completed': size,
              'statuses': {'verified': 3, 'generation_failure': size-3},
              'protocol': COMMON_RELAXATION_PROTOCOL, 'verification_protocol': TERMINAL_VERIFICATION_PROTOCOL,
              'geometry_validation_protocol': LABEL_GEOMETRY_PROTOCOL,
              'runtime_identities': [runtime], 'input_sha256': EVAL.sha256_file(paths)}
    (labels_dir/'LABEL_FINAL.json').write_text(json.dumps(report))
    (labels_dir/'_SUCCESS').touch()
    write_rows(cache/'official_slim_cache.jsonl', [{'chemsys': symbol, 'entries':
               [{'composition': {symbol:1}, 'energy': 0., 'entry_id': symbol}]} for symbol in ('Na','Li')])
    (cache/'unresolved_chemsys.jsonl').write_text('')
    (cache/'completion_SUCCESS').touch()
    reference = root/'frozen_eval.py'
    reference.write_text('from pymatgen.analysis.structure_matcher import StructureMatcher\n'
                         'def load_training_index(path): return [], {}\n')
    training = root/'train.csv'
    training.write_text('frozen-training-reference')
    config = {'assets': {'eval_sun_py': str(reference), 'train_csv': str(training),
                          'chgnet_runtime_checkpoint': str(model)},
              'frozen_code': {'eval_sun_sha256': EVAL.sha256_file(reference)}}
    config_path = root/'CONFIG.json'
    config_path.write_text(json.dumps(config))
    return paths, labels_dir, cache, config_path, rows, labels, report


def run_cli(paths, labels, cache, config, output):
    argv = ['evaluate', '--paths-jsonl', str(paths), '--labels-jsonl', str(labels/'labels.jsonl'),
            '--official-cache', str(cache), '--frozen-config', str(config), '--output-dir', str(output),
            '--expected-requests', '1200', '--endpoint', 'native', '--cohort-role', 'independent_main',
            '--sun-only', '--nu-workers', '1']
    with patch.object(sys, 'argv', argv), patch.dict(os.environ, {'SLURM_JOB_ID':'test', 'SLURM_CPUS_PER_TASK':'2'}):
        with contextlib.redirect_stdout(io.StringIO()):
            EVAL.main()


class SunOnlyEvaluationTests(unittest.TestCase):
    def test_unverified_retained_energy_still_counts_in_main_sun(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths, labels_dir, cache, config, _, labels, report = fixture(root)
            labels[2].update(status='not_converged', verified=False, terminal_energy=0.)
            write_rows(labels_dir/'labels.jsonl', labels)
            report['statuses'] = {'verified': 2, 'not_converged': 1, 'generation_failure': 1197}
            (labels_dir/'LABEL_FINAL.json').write_text(json.dumps(report))
            run_cli(paths, labels_dir, cache, config, root/'score')
            observed = json.loads((root/'score/EVALUATION_FINAL.json').read_text())
            self.assertEqual(observed['counts']['strict_sun'], 1)
            self.assertEqual(observed['counts']['meta_sun'], 1)
            self.assertEqual(observed['counts']['verified_strict_sun'], 0)
            self.assertEqual(observed['counts']['verified_meta_sun'], 0)
            self.assertTrue((root/'score/_SUCCESS').is_file())

    def test_accounted_official_unresolved_is_preserved_and_needs_no_nu(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths, labels, cache, config, *_ = fixture(root)
            write_rows(cache/'official_slim_cache.jsonl', [{'chemsys': 'Na', 'entries':
                       [{'composition': {'Na': 1}, 'energy': 0., 'entry_id': 'Na'}]}])
            write_rows(cache/'unresolved_chemsys.jsonl', [{'chemsys': 'Li', 'reason': 'official_reference_unresolved'}])
            with patch.object(EVAL, 'evaluate_sun_predicates', wraps=EVAL.evaluate_sun_predicates) as nu:
                run_cli(paths, labels, cache, config, root/'score')
                self.assertEqual(nu.call_args.args[3], [1])
            observed = json.loads((root/'score/EVALUATION_FINAL.json').read_text())
            rows = EVAL.read_jsonl(root/'score/attempt_results.jsonl')
            self.assertEqual(observed['counts']['meta_sun'], 0)
            self.assertEqual(rows[2]['official_hull_status'], 'official_cache_unresolved')
            self.assertIsNone(rows[2]['e_above_hull_eV_atom'])
            self.assertIsNone(rows[2]['novel'])
            self.assertFalse(rows[2]['meta_sun'])
            self.assertTrue((root/'score/_SUCCESS').is_file())

    def test_full_1200_denominator_and_unstable_earlier_duplicate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths, labels, cache, config, *_ = fixture(root)
            run_cli(paths, labels, cache, config, root/'score')
            report = json.loads((root/'score/EVALUATION_FINAL.json').read_text())
            rows = EVAL.read_jsonl(root/'score/attempt_results.jsonl')
            self.assertEqual(report['counts']['requests'], 1200)
            self.assertEqual(report['counts']['strict_sun'], 0)
            self.assertEqual(report['counts']['meta_sun'], 1)
            self.assertIsNone(report['counts']['novel'])
            self.assertIsNone(rows[0]['novel'])
            self.assertFalse(rows[1]['unique_representative'])
            self.assertTrue(rows[2]['meta_sun'])
            self.assertEqual(len(rows), 1200)
            self.assertTrue((root/'score/_SUCCESS').is_file())

    def test_required_nu_timeout_cannot_create_success_or_zero_sun(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths, labels, cache, config, *_ = fixture(root)
            result = {'novel': [None,None,True], 'unique': [None,True,True],
                      'complete_sun_predicates': False, 'unresolved_sun_indices': [1]}
            with patch.object(EVAL, 'evaluate_sun_predicates', return_value=result):
                with self.assertRaises(SystemExit) as error:
                    run_cli(paths, labels, cache, config, root/'score')
            self.assertEqual(error.exception.code, 2)
            report = json.loads((root/'score/EVALUATION_FINAL.json').read_text())
            self.assertIsNone(report['counts']['strict_sun'])
            self.assertFalse((root/'score/_SUCCESS').exists())
            self.assertTrue((root/'score/_INCOMPLETE_NU').exists())

    def test_missing_official_coverage_is_an_engineering_failure_before_nu(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths, labels, cache, config, *_ = fixture(root)
            write_rows(cache/'official_slim_cache.jsonl', [])
            with patch.object(EVAL, 'evaluate_sun_predicates') as nu:
                with self.assertRaisesRegex(ValueError, 'coverage is missing'):
                    run_cli(paths, labels, cache, config, root/'score')
                nu.assert_not_called()

    def test_endpoint_change_cannot_reuse_same_id_or_repinned_input_hash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths, labels_dir, _, _, rows, _, report = fixture(root)
            rows[0]['structure']['sites'][0]['abc'][0] += .00000001
            write_rows(paths, rows)
            report['input_sha256'] = EVAL.sha256_file(paths)
            (labels_dir/'LABEL_FINAL.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'different exact endpoint'):
                EVAL.load_bound_evaluation_labels(rows, [labels_dir/'labels.jsonl'], paths_file=paths, endpoint='native')

    def test_complete_relaxation_protocol_includes_fire_and_cell_mask(self):
        for key, value in (('fire_dt', .2), ('fire_maxstep', .3), ('cell_mask', 'diagonal')):
            with self.subTest(key=key), TemporaryDirectory() as directory:
                paths, labels_dir, _, _, rows, _, report = fixture(Path(directory))
                report = copy.deepcopy(report)
                report['protocol'][key] = value
                (labels_dir/'LABEL_FINAL.json').write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, 'full physical protocol'):
                    EVAL.load_bound_evaluation_labels(rows, [labels_dir/'labels.jsonl'], paths_file=paths, endpoint='native')


if __name__ == '__main__':
    unittest.main()
