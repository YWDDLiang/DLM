"""Directed/nontransitive N-U semantics and relevance of unresolved comparisons."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from crystal_dlm.exact_sun_nu import PairCache, conjunction, evaluate_sun_predicates
from crystal_dlm.isolated_workers import isolated_results


def controlled_cpu_worker(connection, gate=None):
    """Real killable CPU worker; no model, network, or scientific evaluator."""
    import time
    connection.send({'ready': True})
    while True:
        payload = connection.recv()
        if payload is None:
            return
        if payload.get('wait_for_peer'):
            if not gate.wait(timeout=10):
                raise RuntimeError('fixture peer did not start')
        if payload.get('signal_peer'):
            gate.set()
        time.sleep(payload.get('delay', 0.))
        connection.send({'result': {'status': 'complete', 'value': payload['name']}})


class Crystal:
    def __init__(self, name, formula='F', coordinate=0.):
        self.name, self.coordinate = name, coordinate
        self.composition = SimpleNamespace(reduced_formula=formula)
    def as_dict(self):
        return {'name': self.name, 'coordinate': self.coordinate}


class PredicateTests(unittest.TestCase):
    def evaluate(self, structures, required, *, training=(), matches=(), unknown=(), cache=None):
        self.calls = []
        def runner(tasks, admitted):
            for key, payload in tasks:
                if not admitted(key, payload):
                    continue
                pair = (payload['left']['name'], payload['right']['name'])
                self.calls.append(pair)
                result = ({'status': 'worker_error', 'error': 'simulated timeout'} if pair in unknown else
                          {'status': 'complete', 'matched': pair in matches})
                yield key, payload, result
        index = {}
        for i, structure in enumerate(training):
            index.setdefault(structure.composition.reduced_formula, []).append(i)
        return evaluate_sun_predicates(structures, training, index, required, cache_dir=cache,
                    frozen_nu_sha256='frozen-reference', training_identity='training-v1',
                    pair_runner=runner, contract={'fixture': 'directed-nontransitive'})

    def test_earlier_unstable_duplicate_still_defeats_later_uniqueness(self):
        with TemporaryDirectory() as directory:
            result = self.evaluate([Crystal('unstable'), Crystal('stable')], [1],
                                   matches={('stable', 'unstable')}, cache=directory)
            self.assertEqual(result['novel'], [None, True])
            self.assertEqual(result['unique'], [None, False])
            self.assertTrue(result['complete_sun_predicates'])
            self.assertFalse(result['standalone_NU_complete'])

    def test_nonrepresentative_intermediate_is_not_discarded(self):
        with TemporaryDirectory() as directory:
            result = self.evaluate([Crystal('A'),Crystal('B'),Crystal('C')], [0,1,2],
                                   matches={('B','A'),('C','B')}, cache=directory)
            self.assertEqual(result['unique'], [True,False,False])
            self.assertIn(('C','B'), self.calls)

    def test_novelty_and_uniqueness_preserve_asymmetric_argument_order(self):
        with TemporaryDirectory() as directory:
            result = self.evaluate([Crystal('A'),Crystal('B')], [1], training=[Crystal('T')],
                                   matches={('T','B'),('A','B')}, cache=directory)
            self.assertEqual(self.calls, [('B','T'),('B','A')])
            self.assertTrue(result['novel'][1])
            self.assertTrue(result['unique'][1])

    def test_known_non_novel_skips_an_irrelevant_hanging_unique_pair(self):
        with TemporaryDirectory() as directory:
            result = self.evaluate([Crystal('A'),Crystal('B')], [1], training=[Crystal('T')],
                                   matches={('B','T')}, unknown={('B','A')}, cache=directory)
            self.assertEqual(self.calls, [('B','T')])
            self.assertEqual(result['novel'][1], False)
            self.assertIsNone(result['unique'][1])
            self.assertTrue(result['complete_sun_predicates'])

    def test_known_nonunique_resolves_sun_despite_unknown_novelty(self):
        with TemporaryDirectory() as directory:
            result = self.evaluate([Crystal('A'),Crystal('B')], [1], training=[Crystal('T')],
                                   matches={('B','A')}, unknown={('B','T')}, cache=directory)
            self.assertIsNone(result['novel'][1])
            self.assertEqual(result['unique'][1], False)
            self.assertTrue(result['complete_sun_predicates'])
            self.assertEqual(conjunction(result['novel'][1], result['unique'][1]), False)

    def test_required_timeout_is_unresolved_and_is_not_cached_as_false(self):
        with TemporaryDirectory() as directory:
            first = self.evaluate([Crystal('B')], [0], training=[Crystal('T')],
                                  unknown={('B','T')}, cache=directory)
            self.assertFalse(first['complete_sun_predicates'])
            self.assertEqual(first['unresolved_sun_indices'], [0])
            self.assertIsNone(conjunction(first['novel'][0], first['unique'][0]))
            second = self.evaluate([Crystal('B')], [0], training=[Crystal('T')], cache=directory)
            self.assertEqual(self.calls, [('B','T')])
            self.assertTrue(second['complete_sun_predicates'])

    def test_exact_pair_cache_reuses_known_results(self):
        with TemporaryDirectory() as directory:
            self.evaluate([Crystal('B')], [0], training=[Crystal('T')], cache=directory)
            second = self.evaluate([Crystal('B')], [0], training=[Crystal('T')], cache=directory)
            self.assertEqual(self.calls, [])
            self.assertEqual(second['counters']['pair_cache_hits'], 1)

    def test_sub_six_decimal_geometry_and_direction_have_distinct_cache_keys(self):
        with TemporaryDirectory() as directory:
            cache = PairCache(directory, {'matcher': 'v1'})
            left = Crystal('A', coordinate=.123456701).as_dict()
            changed = Crystal('A', coordinate=.123456702).as_dict()
            right = Crystal('B').as_dict()
            a, _ = cache.descriptor(left, right)
            b, _ = cache.descriptor(changed, right)
            reverse, _ = cache.descriptor(right, left)
            self.assertEqual(len({a,b,reverse}), 3)

    def test_new_training_member_and_changed_input_order_recompute_aggregate(self):
        with TemporaryDirectory() as directory:
            first = self.evaluate([Crystal('A'),Crystal('B')], [1], cache=directory)
            second = self.evaluate([Crystal('B'),Crystal('A')], [0], training=[Crystal('T')],
                                   matches={('B','T')}, cache=directory)
            self.assertNotEqual(first['input_geometry_order_sha256'], second['input_geometry_order_sha256'])
            self.assertFalse(second['novel'][0])

    def test_no_stability_contributor_requires_no_nu_comparison(self):
        with TemporaryDirectory() as directory:
            result = self.evaluate([Crystal('A')], [], training=[Crystal('T')], cache=directory)
            self.assertEqual(self.calls, [])
            self.assertEqual(result['novel'], [None])
            self.assertTrue(result['complete_sun_predicates'])

    def test_irrelevant_nonfinite_geometry_cannot_block_an_unrelated_contributor(self):
        with TemporaryDirectory() as directory:
            result = self.evaluate([Crystal('bad', formula='BAD', coordinate=float('nan')),
                                    Crystal('good', formula='GOOD')], [1], cache=directory)
            self.assertEqual(self.calls, [])
            self.assertTrue(result['complete_sun_predicates'])
            self.assertTrue(conjunction(result['novel'][1], result['unique'][1]))
            self.assertIsNone(result['novel'][0])

    def test_cached_nonunique_witness_avoids_an_unneeded_novelty_timeout(self):
        with TemporaryDirectory() as directory:
            self.evaluate([Crystal('A'), Crystal('B')], [1], matches={('B', 'A')}, cache=directory)
            result = self.evaluate([Crystal('A'), Crystal('B')], [1], training=[Crystal('T')],
                                   unknown={('B', 'T')}, cache=directory)
            self.assertEqual(self.calls, [])
            self.assertTrue(result['complete_sun_predicates'])
            self.assertFalse(conjunction(result['novel'][1], result['unique'][1]))

    def test_lazy_sun_matches_actual_frozen_functions_on_directed_match_matrices(self):
        import ast
        from collections import defaultdict
        import random
        import numpy as np
        reference = Path(__file__).resolve().parents[1] / 'docs/h1a2_to_current_review_20260907/evidence/legacy_eval_sun.py.txt'
        tree = ast.parse(reference.read_text(encoding='utf-8'))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('index_by_formula', 'compute_novelty', 'compute_uniqueness')]
        self.assertEqual(len(functions), 3)
        namespace = {'np': np, 'defaultdict': defaultdict, 'print': lambda *a, **kw: None,
                     'tqdm': lambda iterable, **kw: iterable}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(reference), 'exec'), namespace)
        structures = [Crystal('A', 'F'), Crystal('B', 'G'), Crystal('C', 'F'), Crystal('D', 'F')]
        training = [Crystal('T', 'F'), Crystal('V', 'G')]
        train_index = {'F': [0], 'G': [1]}
        rng = random.Random(9173)
        names = [s.name for s in structures + training]
        with TemporaryDirectory() as directory:
            for trial in range(48):
                matches = {(left, right) for left in names for right in names if rng.random() < .28}
                matcher = SimpleNamespace(fit=lambda left, right: (left.name, right.name) in matches)
                full_novel = namespace['compute_novelty'](structures, training, train_index, matcher)
                classes, _ = namespace['compute_uniqueness'](structures, matcher)
                seen, representatives = set(), set()
                for index, cls in enumerate(classes):
                    if int(cls) not in seen:
                        seen.add(int(cls))
                        representatives.add(index)
                required = [i for i in range(len(structures)) if rng.random() < .75]
                result = self.evaluate(structures, required, training=training, matches=matches,
                                       cache=Path(directory) / str(trial))
                self.assertTrue(result['complete_sun_predicates'])
                for index in required:
                    self.assertEqual(conjunction(result['novel'][index], result['unique'][index]),
                                     bool(full_novel[index]) and index in representatives,
                                     (trial, index, matches))

    def test_real_matcher_runs_in_a_killable_worker(self):
        from pymatgen.core import Structure, Lattice
        crystal = Structure(Lattice.cubic(4), ['Na'], [[0.,0.,0.]])
        with TemporaryDirectory() as directory:
            result = evaluate_sun_predicates([crystal], [crystal], {'Na':[0]}, [0],
                         cache_dir=directory, frozen_nu_sha256='frozen', training_identity='fixture',
                         workers=1, pair_timeout=10.)
            self.assertEqual(result['novel'], [False])
            self.assertTrue(result['complete_sun_predicates'])


class IsolatedCPUWorkerTests(unittest.TestCase):
    def test_replacement_started_before_peer_startup_failure_is_still_cleaned_up(self):
        from unittest.mock import patch
        import crystal_dlm.isolated_workers as workers_module

        class Connection:
            def __init__(self, messages=()):
                self.messages = list(messages)
                self.closed = False
            def recv(self):
                message = self.messages.pop(0)
                if isinstance(message, Exception):
                    raise message
                return message
            def send(self, payload):
                pass
            def close(self):
                self.closed = True

        class Process:
            def __init__(self, **kwargs):
                self.alive = False
            def start(self):
                self.alive = True
            def is_alive(self):
                return self.alive
            def terminate(self):
                self.alive = False
            def kill(self):
                self.alive = False
            def join(self, timeout):
                pass

        processes, connections = [], []
        def make_pipe():
            messages = ([{'ready': True}, {'result': {'status': 'worker_error'}}] if not connections else
                        [EOFError()] if len(connections) == 1 else [])
            parent, child = Connection(messages), Connection()
            connections.append(parent)
            return parent, child
        def make_process(**kwargs):
            process = Process(**kwargs)
            processes.append(process)
            return process
        rounds = 0
        def ready_connections(current, timeout):
            nonlocal rounds
            rounds += 1
            return [connections[0]] if rounds == 1 else connections[:2]
        context = SimpleNamespace(Pipe=make_pipe, Process=make_process)
        with patch.object(workers_module.mp, 'get_context', return_value=context), \
                patch.object(workers_module, 'wait', side_effect=ready_connections):
            with self.assertRaisesRegex(RuntimeError, 'initialization'):
                list(isolated_results([('first', {}), ('queued', {})], worker_target=controlled_cpu_worker,
                                      worker_arguments=[(None,), (None,)], task_timeout=10., startup_timeout=10.))
        self.assertEqual(len(processes), 3, 'fixture must create the replacement before the second slot fails')
        self.assertTrue(all(not process.is_alive() for process in processes))
        self.assertTrue(all(connection.closed for connection in connections))

    def test_timed_out_active_task_is_killed_and_queued_task_gets_a_fresh_worker(self):
        import multiprocessing as mp
        before = {process.pid for process in mp.active_children()}
        results = list(isolated_results(
            [('slow', {'name': 'slow', 'delay': 10.}), ('next', {'name': 'next'})],
            worker_target=controlled_cpu_worker, worker_arguments=[(None,)],
            task_timeout=.15, startup_timeout=10.))
        self.assertEqual([key for key, _, _ in results], ['slow', 'next'])
        self.assertEqual(results[0][2]['status'], 'worker_error')
        self.assertIn('exceeded', results[0][2]['error'])
        self.assertEqual(results[1][2], {'status': 'complete', 'value': 'next'})
        self.assertFalse({process.pid for process in mp.active_children()} - before)

    def test_running_irrelevant_task_is_cancelled_without_waiting_for_its_deadline(self):
        import multiprocessing as mp
        import time
        before = {process.pid for process in mp.active_children()}
        gate = mp.get_context('spawn').Event()
        admitted_slow = True

        def admitted(key, payload):
            return key != 'slow' or admitted_slow

        began = time.monotonic()
        results = []
        for item in isolated_results(
            [('winner', {'name': 'winner', 'wait_for_peer': True}),
             ('slow', {'name': 'slow', 'signal_peer': True, 'delay': 20.})],
            worker_target=controlled_cpu_worker, worker_arguments=[(gate,), (gate,)],
            task_timeout=10., startup_timeout=10., admitted=admitted):
            results.append(item)
            if item[0] == 'winner':
                admitted_slow = False
        self.assertEqual({key: result['status'] for key, _, result in results},
                         {'winner': 'complete', 'slow': 'cancelled_irrelevant'})
        self.assertLess(time.monotonic() - began, 8.)
        self.assertFalse({process.pid for process in mp.active_children()} - before)


if __name__ == '__main__':
    unittest.main()
