#!/usr/bin/env python3
"""Offline original/terminal energy labels for every requested native path."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import partial
import gzip
import hashlib
import importlib.metadata
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from crystal_dlm.terminal_energy_consistency import COMMON_RELAXATION_PROTOCOL, LABEL_GEOMETRY_PROTOCOL, TERMINAL_VERIFICATION_PROTOCOL, check_terminal_energy

EV_A3_TO_GPA = 160.21766208
_MODEL = None
_OPTIMIZER = None
_OPT_STATUS: dict[str, Any] = {}
_VERSIONS: dict[str, str] = {}


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def finite_scalar(value):
    values = array(value).reshape(-1)
    if len(values) != 1 or not np.isfinite(values).all():
        raise ValueError("energy must be one finite scalar")
    return float(values[0])


def force_and_stress(forces, stress, *, stress_unit):
    forces, stress = array(forces), array(stress)
    if forces.ndim != 2 or forces.shape[-1] != 3 or not len(forces):
        raise ValueError("forces must have shape [N,3]")
    if stress.shape not in ((3, 3), (6,)):
        raise ValueError("stress must be a matrix or Voigt six-vector")
    if not np.isfinite(forces).all() or not np.isfinite(stress).all():
        raise ValueError("nonfinite force/stress")
    if stress_unit == "eV/A3":
        stress = stress * EV_A3_TO_GPA
    elif stress_unit != "GPa":
        raise ValueError("unknown stress unit")
    norms = np.linalg.norm(forces, axis=-1)
    return {"force_max_eV_A": float(norms.max()),
            "force_rms_eV_A": float(np.sqrt(np.mean(norms ** 2))),
            "stress_max_GPa": float(np.abs(stress).max()), "stress_GPa": stress.tolist()}


def structure_from_record(record):
    if record.get("structure") is not None:
        from pymatgen.core import Structure
        return Structure.from_dict(record["structure"])
    from crystal_dlm.dynamic_crystal import parse_dynamic_answer, arrays_to_structure
    return arrays_to_structure(parse_dynamic_answer(record["body"], strict=True))


class GeometryCertificationUnavailable(RuntimeError):
    pass


class InvalidPeriodicGeometry(ValueError):
    pass


def _validate_structure_geometry(structure):
    lattice, coords = array(structure.lattice.matrix), array(structure.frac_coords)
    if lattice.shape != (3, 3) or coords.shape != (int(structure.num_sites), 3):
        raise InvalidPeriodicGeometry("invalid periodic geometry dimensions")
    volume = float(abs(np.linalg.det(lattice)))
    if (not np.isfinite(lattice).all() or not np.isfinite(coords).all()
            or not math.isfinite(volume) or volume <= 1e-10):
        raise InvalidPeriodicGeometry("nonfinite or degenerate periodic structure")
    from pymatgen.core import Lattice
    try:
        reduced = np.asarray(Lattice(lattice).get_lll_reduced_lattice().matrix)
        transform = reduced@np.linalg.inv(lattice)
        rounded = np.rint(transform)
        # Any nonzero integer image is already a valid short-contact witness;
        # it does not require the full coordinate change to be well conditioned.
        if np.isfinite(rounded).all() and np.max(np.abs(rounded)) <= 1_000_000_000:
            images_from_original = rounded@lattice
            component_error = (np.abs(rounded)@np.abs(lattice))*np.finfo(float).eps*16
            upper_lengths = np.linalg.norm(images_from_original,axis=-1)*(1+4*np.finfo(float).eps)+np.linalg.norm(component_error,axis=-1)
            if bool(((upper_lengths < .5-1e-8)&np.any(rounded!=0,axis=-1)).any()):
                raise InvalidPeriodicGeometry('periodic geometry violates the common 0.5 Angstrom support')
        if not np.isfinite(transform).all() or np.max(np.abs(rounded))>1_000_000_000 or not np.allclose(transform,rounded,rtol=0,atol=1e-7):
            raise GeometryCertificationUnavailable('LLL basis change is not certified integral')
        a,b,c,d,e,f,g,h,i = map(int,rounded.reshape(-1))
        determinant = a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
        if abs(determinant) != 1:
            raise GeometryCertificationUnavailable('LLL basis change is not unimodular')
        # Use the original represented crystal and the certified integer map,
        # never the approximately returned LLL vectors as a replacement cell.
        reduced = rounded@lattice
        integer_inverse = np.asarray([[e*i-f*h,c*h-b*i,b*f-c*e],
                                      [f*g-d*i,a*i-c*g,c*d-a*f],
                                      [d*h-e*g,b*g-a*h,a*e-b*d]],dtype=float)*determinant
        numeric_bound = max(float(np.max(np.abs(rounded)@np.abs(lattice))),
                            float(np.max(np.abs(integer_inverse)))*float(np.linalg.norm(reduced,axis=-1).max()))
        if numeric_bound*np.finfo(float).eps*32 > 1e-9:
            raise GeometryCertificationUnavailable('integer basis transformation exceeds the numerical error budget')
        inverse = np.linalg.inv(reduced)
        reduced_coords = np.mod(coords,1.)@integer_inverse
        radii_float = np.ceil(.5+.5*np.linalg.norm(inverse,axis=0)+1e-12)
        if not np.isfinite(radii_float).all() or np.max(radii_float)>1_000_000:
            raise GeometryCertificationUnavailable('periodic contact bound is not numerically established')
    except (GeometryCertificationUnavailable,InvalidPeriodicGeometry):
        raise
    except Exception as error:
        raise GeometryCertificationUnavailable('periodic contact reduction could not be certified') from error
    minimum = float(np.linalg.norm(reduced,axis=-1).min())
    if minimum < .5-1e-8:
        raise InvalidPeriodicGeometry('periodic geometry violates the common 0.5 Angstrom support')
    radii = radii_float.astype(int)
    images = math.prod(2*int(radius)+1 for radius in radii)
    if len(coords)**2*images > LABEL_GEOMETRY_PROTOCOL['max_pair_images']:
        raise GeometryCertificationUnavailable('periodic contact certification exceeded its explicit work budget')
    delta = reduced_coords[:,None]-reduced_coords[None,:]
    delta -= np.round(delta)
    shifts = itertools.product(*(range(-int(radius),int(radius)+1) for radius in radii))
    while True:
        chunk = list(itertools.islice(shifts,256))
        if not chunk:
            break
        offsets = np.asarray(chunk,dtype=float)
        distances = np.linalg.norm((delta[:,:,None,:]+offsets)@reduced,axis=-1)
        if not np.isfinite(distances).all():
            raise GeometryCertificationUnavailable('periodic distance computation is nonfinite')
        zero = np.flatnonzero((offsets==0).all(-1))
        if len(zero):
            distances[np.arange(len(coords)),np.arange(len(coords)),int(zero[0])] = np.inf
        minimum = min(minimum,float(distances.min()))
        if minimum < .5-1e-8:
            raise InvalidPeriodicGeometry('periodic geometry violates the common 0.5 Angstrom support')
    return minimum


def validate_structure_geometry(structure):
    try:
        return _validate_structure_geometry(structure)
    except (InvalidPeriodicGeometry,GeometryCertificationUnavailable):
        raise
    except Exception as error:
        raise GeometryCertificationUnavailable('periodic certification could not complete') from error


def label_record(record, *, model, optimizer, structure_factory=structure_from_record,
                 fmax=.1, stress_tolerance=.5, max_steps=500, optimizer_status=None,
                 terminal_energy_checker=check_terminal_energy):
    result = {key: record.get(key) for key in ("trajectory_id", "group_id", "source_row_idx", "source_split", "endpoint")}
    result.update(raw_energy=None, terminal_energy=None, gap=None, verified=False,
                  status="unknown", error=None, raw=None, terminal=None, actual_steps=None,
                  optimizer_converged=None, final_structure=None)
    if not record.get("success", False):
        result["status"] = "generation_failure"
        result["error"] = (record.get("trace") or {}).get("failure")
        return result
    try:
        structure = structure_factory(record)
        count = int(structure.num_sites)
        if count < 1:
            raise ValueError("empty structure")
        result["raw_min_distance_A"] = validate_structure_geometry(structure)
    except GeometryCertificationUnavailable as error:
        result.update(status='worker_error', error=f'{type(error).__name__}: {error}')
        return result
    except Exception as error:
        result.update(status="invalid_raw", error=f"{type(error).__name__}: {error}")
        return result
    try:
        raw = model.predict_structure(structure, task="efs")
        if isinstance(raw, list):
            raw = raw[0]
        result["raw_energy"] = finite_scalar(raw["e"])  # CHGNet predict returns eV/atom.
        result["raw"] = force_and_stress(raw["f"], raw["s"], stress_unit="GPa")
        if optimizer_status is not None:
            optimizer_status.clear()
        relaxed = optimizer.relax(structure, fmax=fmax, steps=max_steps, relax_cell=True,
                                  ase_filter="FrechetCellFilter", verbose=False)
        final = relaxed["final_structure"]
        trajectory = relaxed["trajectory"]
        if int(final.num_sites) != count or final.composition != structure.composition:
            raise ValueError("relaxation changed the fixed composition")
        geometry_valid = True
        try:
            result["terminal_min_distance_A"] = validate_structure_geometry(final)
        except ValueError as error:
            geometry_valid = False
            result["terminal_geometry_error"] = str(error)
        energies = list(trajectory.energies)
        if not energies:
            raise ValueError("missing relaxation energy trajectory")
        result["terminal_energy"] = finite_scalar(energies[-1]) / count
        result["terminal"] = force_and_stress(trajectory.forces[-1], trajectory.stresses[-1], stress_unit="eV/A3")
        result["gap"] = result["raw_energy"] - result["terminal_energy"]
        first_energy = finite_scalar(energies[0]) / count
        result["raw_vs_trajectory_first_delta"] = result["raw_energy"] - first_energy
        status = optimizer_status or relaxed.get("optimizer_status") or {}
        result["actual_steps"] = status.get("steps")
        result["optimizer_converged"] = status.get("converged")
        result["trajectory_frames"] = len(energies)
        result["relaxation_trajectory"] = {
            "energies_eV_atom": [finite_scalar(e) / count for e in energies],
            "forces_eV_A": [array(f).tolist() for f in trajectory.forces],
            "stresses_GPa": [(array(s) * EV_A3_TO_GPA).tolist() for s in trajectory.stresses],
            "cells_A": [array(v).tolist() for v in getattr(trajectory, "cells", [])],
            "positions_A": [array(v).tolist() for v in getattr(trajectory, "atom_positions", [])],
        }
        result["final_structure"] = final.as_dict()
        physical = (result["terminal"]["force_max_eV_A"] <= fmax + 1e-8
                    and result["terminal"]["stress_max_GPa"] <= stress_tolerance + 1e-8)
        same_energy = abs(result["raw_vs_trajectory_first_delta"]) <= .001
        monotone = result["gap"] >= -.001
        # Missing optimizer status is explicit, never synthesized as a success.
        stop_verified = result["optimizer_converged"] is True
        preliminary_verified = bool(geometry_valid and physical and same_energy and monotone and stop_verified)
        if preliminary_verified:
            result["terminal_consistency"] = terminal_energy_checker(model, final, result["terminal_energy"])
        result["verified"] = bool(preliminary_verified and result["terminal_consistency"]["status"] == "consistent")
        if not same_energy:
            result["status"] = "energy_protocol_mismatch"
        elif not monotone:
            result["status"] = "relaxation_energy_increased"
        elif not geometry_valid:
            result["status"] = "invalid_terminal"
        elif not physical:
            result["status"] = "not_converged"
        elif not stop_verified:
            result["status"] = "optimizer_stop_unverified"
        elif result["terminal_consistency"]["status"] != "consistent":
            result["status"] = "terminal_consistency_unverified"
        else:
            result["status"] = "verified"
    except GeometryCertificationUnavailable as error:
        result.update(status='worker_error', verified=False, error=f'{type(error).__name__}: {error}')
    except Exception as error:
        result.update(status="evaluation_error", verified=False, error=f"{type(error).__name__}: {error}")
    return result


def configure_deterministic_execution(enabled):
    """Opt-in numerical execution control; physical relaxation parameters are unchanged."""
    if enabled:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.use_deterministic_algorithms(bool(enabled))
    if enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def worker_init(gpu_index):
    global _MODEL, _OPTIMIZER, _VERSIONS
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import torch
    configure_deterministic_execution(os.environ.get('R03_DETERMINISTIC_LABELING') == '1')
    from ase.optimize import FIRE
    from ase.filters import FrechetCellFilter
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import StructOptimizer
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu_index)
    class RecordedFIRE(FIRE):
        def run(self, *args, **kwargs):
            outcome = super().run(*args, **kwargs)
            _OPT_STATUS.update(steps=int(self.nsteps), converged=None if outcome is None else bool(outcome))
            return outcome
    class PinnedOptimizer(StructOptimizer):
        def relax(self, structure, **kwargs):
            if kwargs.pop("ase_filter") != "FrechetCellFilter":
                raise ValueError("teacher protocol requires FrechetCellFilter")
            return super().relax(structure, ase_filter=partial(
                FrechetCellFilter, mask=np.ones((3, 3)), hydrostatic_strain=False,
                constant_volume=False, scalar_pressure=0.), loginterval=1,
                assign_magmoms=False, dt=.1, maxstep=.2, **kwargs)
    device = f"cuda:{gpu_index}"
    _MODEL = CHGNet.load(model_name="0.3.0", use_device=device)
    _MODEL.eval()
    _OPTIMIZER = PinnedOptimizer(model=_MODEL, optimizer_class=RecordedFIRE, use_device=device,
                                stress_weight=1 / EV_A3_TO_GPA)
    _VERSIONS = runtime_identity()


def runtime_identity():
    import importlib
    import importlib.metadata
    import torch
    versions = {"model": "CHGNet-0.3.0", "chgnet_package": importlib.metadata.version("chgnet"),
                 "ase_package": importlib.metadata.version("ase"), "torch_package": torch.__version__,
                 "pymatgen_package": importlib.metadata.version("pymatgen")}
    package_root = Path(importlib.import_module('chgnet').__file__).resolve().parent
    checkpoint = package_root / 'pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar'
    versions['model_checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    versions['labeler_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    versions['deterministic_algorithms_enabled'] = torch.are_deterministic_algorithms_enabled()
    versions['cublas_workspace_config'] = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    return versions


def worker_label(record, fmax, stress_tolerance, max_steps):
    result = label_record(record, model=_MODEL, optimizer=_OPTIMIZER, fmax=fmax,
                          stress_tolerance=stress_tolerance, max_steps=max_steps,
                          optimizer_status=_OPT_STATUS)
    result["versions"] = _VERSIONS
    error = str(result.get('error') or '').lower()
    if any(token in error for token in ('out of memory', 'cuda error', 'brokenprocesspool', 'modulenotfounderror')):
        result['status'] = 'worker_error'
    return result


def validate_record_purpose(record, purpose):
    if purpose == 'train':
        if record.get('source_split') != 'train':
            raise ValueError('training labels cannot read heldout conditions')
        if record.get('endpoint') not in (None, 'native'):
            raise ValueError('native path teachers cannot use model494 endpoints')
    elif purpose == 'expert_edit':
        if (record.get('source_split') not in ('train', 'dev') or record.get('purpose') != 'expert_edit'
                or record.get('endpoint') not in ('native', 'expert_quantized')):
            raise ValueError('expert labels require explicit train/development editing provenance')
    elif purpose == 'training_feedback':
        if (record.get('source_split') != 'train' or record.get('purpose') != 'training_feedback'
                or record.get('endpoint') not in ('native', 'tau800')):
            raise ValueError('SUN feedback labels require explicit training-only endpoint provenance')


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _isolated_worker(connection, gpu_index, options):
    """Persistent single-record worker; a parent can kill a stuck C++ call."""
    worker_init(gpu_index)
    connection.send({'ready': True})
    while True:
        record = connection.recv()
        if record is None:
            return
        result = worker_label(record, *options)
        connection.send({'result': result})
        if result['status'] == 'worker_error':
            return  # Never reuse a potentially damaged CUDA context.


def bounded_labels(by_endpoint, args, *, worker_target=_isolated_worker):
    from crystal_dlm.isolated_workers import isolated_results
    options = (args.fmax, args.stress_tolerance, args.max_steps)
    arguments = [(gpu, options) for gpu in range(args.gpu_count) for _ in range(args.workers_per_gpu)]
    tasks = [(key, occurrences[0]) for key, occurrences in by_endpoint.items()]
    for key, _, result in isolated_results(tasks, worker_target=worker_target, worker_arguments=arguments,
                                           task_timeout=args.record_timeout,
                                           startup_timeout=args.worker_startup_timeout):
        if result.get('status') == 'worker_error':
            result = {'raw_energy': None, 'terminal_energy': None, 'gap': None, 'verified': False,
                      'raw': None, 'terminal': None, 'actual_steps': None, 'optimizer_converged': None,
                      'final_structure': None, **result}
        yield key, by_endpoint[key], result


def reusable_labels(directory, by_endpoint, *, input_sha256, purpose, protocol, runtime):
    directory = Path(directory)
    report_path, labels_path = directory/'LABEL_FINAL.json', directory/'labels.jsonl'
    report = json.loads(report_path.read_text())
    rows = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
    expected = {row['trajectory_id']:(key,row) for key,records in by_endpoint.items() for row in records}
    if (report.get('input_sha256') != input_sha256 or report.get('purpose') != purpose
            or report.get('protocol') != protocol or report.get('verification_protocol') != TERMINAL_VERIFICATION_PROTOCOL
            or report.get('geometry_validation_protocol') != LABEL_GEOMETRY_PROTOCOL
            or report.get('completed') != len(expected) or report.get('requested') != len(expected)
            or len(rows) != len(expected) or {row['trajectory_id'] for row in rows} != set(expected)):
        raise ValueError('reused physics requires the same complete input ledger and full protocols')
    actual_versions = {json.dumps(row['versions'],sort_keys=True) for row in rows if row.get('versions')}
    reported_versions = [json.dumps(value,sort_keys=True) for value in report.get('runtime_identities',[])]
    if (report.get('statuses') != dict(Counter(row['status'] for row in rows))
            or set(reported_versions) != actual_versions or len(reported_versions) != len(actual_versions)):
        raise ValueError('reused physics report contradicts its row statuses or runtime identities')
    cached = {}
    for row in rows:
        key,record = expected[row['trajectory_id']]
        if (row.get('endpoint_cache_key') != key or any(row.get(field) != record.get(field) for field in
                ('group_id','source_row_idx','source_split','endpoint'))):
            raise ValueError('reused physics is not bound to the current endpoint occurrence')
        if row.get('status') == 'worker_error':
            continue
        if record.get('success') is False and (row.get('status') != 'generation_failure'
                or row.get('verified') is not False or any(row.get(field) is not None for field in
                    ('raw_energy','terminal_energy','gap','raw','terminal','final_structure'))):
            raise ValueError('a failed generation acquired physical labels in the reused ledger')
        if row.get('versions') != runtime:
            raise ValueError('reused physics model/package/source identity changed')
        result = {name:value for name,value in row.items() if name not in
                  ('trajectory_id','group_id','source_row_idx','source_split','endpoint')}
        if key in cached and cached[key] != result:
            raise ValueError('duplicate old physics endpoints have inconsistent results')
        cached[key] = result
    return cached, {'directory':str(directory.resolve()),'labels_sha256':hashlib.sha256(labels_path.read_bytes()).hexdigest(),
                    'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
                    'reused_endpoints':len(cached),'worker_errors_reused':False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument('--reuse-labels', type=Path,
                   help='Verified complete prior ledger; retry only engineering-unknown endpoints into a new directory')
    p.add_argument("--gpu-count", type=int, default=2)
    p.add_argument("--workers-per-gpu", type=int, default=2)
    p.add_argument("--purpose", choices=("train", "evaluation", "expert_edit", "training_feedback"), default="train")
    p.add_argument('--feedback-manifest', type=Path)
    p.add_argument('--deterministic', action='store_true',
                   help='Explicit deterministic CUDA diagnostic; does not replace frozen stochastic-runtime labels')
    p.add_argument("--fmax", type=float, default=.1)
    p.add_argument("--stress-tolerance", type=float, default=.5)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument('--record-timeout', type=float, default=180.,
                   help='Hard wall limit per active structure; timeouts are engineering unknowns')
    p.add_argument('--worker-startup-timeout', type=float, default=120.)
    p.add_argument("--shard-rank", type=int, default=0)
    p.add_argument("--shard-ranks", type=int, nargs="+")
    p.add_argument("--shard-count", type=int, default=1)
    args = p.parse_args()
    os.environ['R03_DETERMINISTIC_LABELING'] = '1' if args.deterministic else '0'
    configure_deterministic_execution(args.deterministic)
    if args.shard_count < 1 or not 0 <= args.shard_rank < args.shard_count:
        raise ValueError("invalid label shard")
    shard_ranks = args.shard_ranks if args.shard_ranks is not None else [args.shard_rank]
    if not shard_ranks or len(set(shard_ranks)) != len(shard_ranks) or any(not 0 <= r < args.shard_count for r in shard_ranks):
        raise ValueError("invalid disjoint label shard ranks")
    if not 1 <= args.gpu_count <= 6 or not 1 <= args.workers_per_gpu <= 4:
        raise ValueError("respect six GPUs and four CPUs per GPU")
    if (not math.isfinite(args.record_timeout) or args.record_timeout <= 0
            or not math.isfinite(args.worker_startup_timeout) or args.worker_startup_timeout <= 0):
        raise ValueError('finite positive worker deadlines are required')
    if "SLURM_JOB_ID" not in os.environ or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("labeling requires its declared GPU allocation")
    if args.gpu_count > len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")):
        raise ValueError("requested workers exceed the GPUs actually allocated")
    feedback_scope = None
    if args.purpose == 'training_feedback':
        if args.feedback_manifest is None or args.shard_count != 1:
            raise ValueError('training feedback requires its bound manifest and a complete input ledger')
        from crystal_dlm.sun_feedback_contract import validate_training_feedback, rows as feedback_rows
        feedback_scope = validate_training_feedback(feedback_rows(args.input_jsonl), args.input_jsonl, args.feedback_manifest)
    elif args.feedback_manifest is not None:
        raise ValueError('feedback manifests cannot change another labeling purpose')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    with args.input_jsonl.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip() or index % args.shard_count not in shard_ranks:
                continue
            record = json.loads(line)
            validate_record_purpose(record, args.purpose)
            # Do not ship the complete token trace to each physics worker.
            records.append({key: record.get(key) for key in
                            ("trajectory_id", "group_id", "source_row_idx", "source_split", "success", "body", "structure", "endpoint")})
    identities = [r["trajectory_id"] for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate trajectory occurrence in label input")
    # Identical endpoints share physics only. Every path occurrence is retained.
    by_endpoint = {}
    for record in records:
        geometry = json.dumps(record["structure"], sort_keys=True) if record.get("structure") is not None else str(record["body"])
        key = hashlib.sha256(geometry.encode()).hexdigest() if record["success"] else record["trajectory_id"]
        by_endpoint.setdefault(key, []).append(record)
    protocol = {**COMMON_RELAXATION_PROTOCOL,'fmax':args.fmax,
                'stress_tolerance_GPa':args.stress_tolerance,'max_steps':args.max_steps}
    input_sha256 = hashlib.sha256(args.input_jsonl.read_bytes()).hexdigest()
    cached, reuse = ({},None) if args.reuse_labels is None else reusable_labels(args.reuse_labels,by_endpoint,
        input_sha256=input_sha256,purpose=args.purpose,protocol=protocol,runtime=runtime_identity())
    pending = {key:value for key,value in by_endpoint.items() if key not in cached}
    def results():
        for key,result in cached.items():
            yield key,by_endpoint[key],dict(result)
        if pending:
            yield from bounded_labels(pending,args)
    (args.output_dir / "trajectories").mkdir()
    started = time.monotonic()
    counts = {}
    completed = 0
    versions_seen = {}
    with (args.output_dir / "labels.jsonl").open("x", encoding="utf-8") as handle:
        for key, occurrences, result in results():
            trajectory = result.pop("relaxation_trajectory", None)
            if result.get('versions'):
                versions_seen[json.dumps(result['versions'], sort_keys=True)] = result['versions']
            if trajectory is not None:
                destination = args.output_dir / "trajectories" / f"{key}.json.gz"
                with gzip.open(destination, "wt", encoding="utf-8") as stream:
                    json.dump(trajectory, stream)
                result["trajectory_file"] = str(destination)
            for occurrence in occurrences:
                labelled = dict(result, **{name: occurrence.get(name) for name in
                                ("trajectory_id", "group_id", "source_row_idx", "source_split", "endpoint")})
                labelled["endpoint_cache_key"] = key
                handle.write(json.dumps(labelled, default=json_default) + "\n")
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                completed += 1
            handle.flush()
            if completed % 64 == 0:
                print(json.dumps({"completed": completed, "requested": len(records), "statuses": counts,
                                  "seconds": time.monotonic() - started}), flush=True)
    report = {"requested": len(records), "completed": completed, "statuses": counts,
              "verification_protocol": TERMINAL_VERIFICATION_PROTOCOL,
              "distinct_endpoint_evaluations": len(by_endpoint),
              "protocol": protocol, 'reused_label_source':reuse,'new_endpoint_evaluations':len(pending),
              "gpu_count": args.gpu_count, "workers_per_gpu": args.workers_per_gpu,
              "shard_count": args.shard_count, "shard_ranks": shard_ranks,
              "elapsed_seconds": time.monotonic() - started, "purpose": args.purpose}
    if feedback_scope is not None:
        report['training_feedback_scope'] = feedback_scope
    report['engineering_deadlines'] = {'record_timeout_seconds': args.record_timeout,
                                       'worker_startup_timeout_seconds': args.worker_startup_timeout,
                                       'timeout_is_physical_failure': False}
    report['runtime_identities'] = list(versions_seen.values())
    report['deterministic_algorithms_requested'] = args.deterministic
    report['input_sha256'] = input_sha256
    report['input_file'] = str(args.input_jsonl.resolve())
    report['geometry_validation_protocol'] = LABEL_GEOMETRY_PROTOCOL
    (args.output_dir / "LABEL_FINAL.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if counts.get('worker_error', 0):
        (args.output_dir / '_ENGINEERING_FAILED').touch()
        print(json.dumps(report), flush=True)
        raise SystemExit(2)
    (args.output_dir / "_SUCCESS").touch()  # Complete accounting, not universal convergence.
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
