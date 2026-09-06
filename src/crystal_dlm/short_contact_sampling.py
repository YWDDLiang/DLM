"""Frozen, optional Z-completion soft tilt after the original legal-set transform."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch

from crystal_dlm.llada_generation import _lattice_matrix_from_token_ids
from crystal_dlm.programmed_path_runtime import ProgrammedPathSampler
from crystal_dlm.spad_generation import _transaction_candidate_tokens


SCHEMA = "k4k8_short_contact_z_soft_v1"
RADII_SHA256 = "74bf9289545246bead1fa2b2a70bd5cbd73498ae3cef74f5d7ec24007c78bec7"


@dataclass(frozen=True)
class ContactSpec:
    radii: dict[str, float]
    spec_sha256: str
    radii_sha256: str

    @classmethod
    def load(cls, path):
        path = Path(path)
        raw = path.read_bytes()
        value = json.loads(raw)
        expected = {"schema": SCHEMA, "temperature": .7, "strength": 1., "image_radius": 2,
                    "distance_floor_A": .5, "radii_sum_factor": .5, "penalty_cap": 1.,
                    "radii_sha256": RADII_SHA256, "hard_support_changed": False}
        if any(value.get(k) != v for k,v in expected.items()):
            raise ValueError("short-contact spec differs from the single fixed candidate")
        radius_path = path.parent/value["radii_file"]
        radius_raw = radius_path.read_bytes()
        if hashlib.sha256(radius_raw).hexdigest() != RADII_SHA256:
            raise ValueError("Cordero radius table hash differs from the frozen table")
        radii = json.loads(radius_raw)
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in radii.values()):
            raise ValueError("invalid frozen covalent radius")
        return cls(radii, hashlib.sha256(raw).hexdigest(), RADII_SHA256)

    def metadata(self):
        return {"schema": SCHEMA, "spec_sha256": self.spec_sha256, "radii_sha256": self.radii_sha256,
                "strength": 1., "temperature": .7, "action_scope": "current_known_Z_completion",
                "old_geometry_imputation": False, "hard_support_changed": False}


def validate_legacy_contact_policy(checkpoint_path):
    root = Path(checkpoint_path).resolve()
    train = root.parent.parent
    record = json.loads((train/"TRAIN_FINAL.json").read_text())
    if (not (train/"_SUCCESS").is_file() or record.get("eligible_policy") is not True
            or record.get("phase") != "dual_objective_full_path_policy"
            or Path(record["policy_path"]).resolve() != root
            or not record.get("passes") or not all(p.get("complete_pass") for p in record["passes"])
            or record.get("raw_periodic_posttraining", False)
            or not (root/"POST_STATE.pt").is_file()
            or not (root/"periodic_state.pt").is_file()):
        raise ValueError("short-contact candidate requires the completed legacy K4/K8 policy")
    return record


class ShortContactProgrammedPathSampler(ProgrammedPathSampler):
    def __init__(self, *args, contact_spec: ContactSpec, probe_only=False, collect_vectors=False, **kwargs):
        super().__init__(*args, **kwargs)
        if self.temperature != .7:
            raise ValueError("the registered contact tilt retains temperature .7")
        required = {"pbc_min_distance_mask": True, "pbc_min_distance_A": .5, "pbc_image_radius": 2,
                    "canonicalize_periodic_alias": True, "lattice_volume_mask": True,
                    "duplicate_coordinate_mask": True}
        if any(self.constraints.get(k) != v for k,v in required.items()):
            raise ValueError("short-contact tilt requires the unchanged original hard supports")
        self.contact_spec = contact_spec
        self.probe_only, self.collect_vectors = bool(probe_only), bool(collect_vectors)
        self.radius_by_row = []
        for program in self.programs:
            values = [None]*program.num_atoms
            for entry in program.entries:
                if entry.symbol not in contact_spec.radii:
                    raise ValueError(f"frozen radius table does not cover {entry.symbol}")
                for site in entry.slot_indices:
                    values[site] = float(contact_spec.radii[entry.symbol])
            if any(v is None for v in values):
                raise ValueError("program did not assign every atom radius")
            self.radius_by_row.append(values)
        self.contact_diagnostics = []
        self._last = {}
        self._contact_salt = None

    def penalty(self, x, row, position, legal_ids):
        """Return b for original legal ids; current tokens only, never the old canvas."""
        penalty = torch.zeros(len(legal_ids), dtype=torch.float64, device=x.device)
        if position < 10 or (position-10) % 4:
            return penalty, "not_Z", 0
        site = (position-10)//4
        if site >= self.num_sites:
            raise ValueError("Z action is outside the crystal")
        lattice = _lattice_matrix_from_token_ids(x[row], prompt_length=self.prompt_length, constraints=self.constraints)
        if lattice is None:
            return penalty, "incomplete_current_cell", 0
        maps = self.constraints["coord_token_to_bin"]
        period = int(self.constraints["coord_period"])
        def coordinate(j, axis):
            token = int(x[row,self.prompt_length+8+4*j+axis])
            value = maps["XYZ"[axis]].get(token)
            return None if value is None else (int(value) % period)/period
        a, b = coordinate(site,0), coordinate(site,1)
        if a is None or b is None:
            return penalty, "incomplete_current_XY", 0
        neighbours, radii = [], []
        for other in range(self.num_sites):
            if other == site:
                continue
            value = [coordinate(other,axis) for axis in range(3)]
            if all(v is not None for v in value):
                neighbours.append(value)
                radii.append(max(.5,.5*(self.radius_by_row[row][site]+self.radius_by_row[row][other])))
        if not neighbours:
            return penalty, "no_complete_current_neighbours", 0
        tokens = legal_ids.detach().cpu().tolist()
        if any(token not in maps["Z"] or maps["Z"][token] >= period for token in tokens):
            raise ValueError("post-alias Z support contains a noncanonical candidate")
        candidates = torch.tensor([[a,b,maps["Z"][token]/period] for token in tokens], dtype=torch.float64, device=x.device)
        other = torch.tensor(neighbours,dtype=torch.float64,device=x.device)
        delta = candidates[:,None]-other[None]
        delta -= torch.round(delta)
        shell = torch.arange(-2,3,dtype=torch.float64,device=x.device)
        shifts = torch.cartesian_prod(shell,shell,shell)
        distance = torch.linalg.vector_norm((delta[:,:,None]+shifts)@lattice,dim=-1).min(-1).values
        cutoffs = torch.tensor(radii,dtype=torch.float64,device=x.device)
        penalty = torch.log(cutoffs[None]/distance).clamp_min(0).max(-1).values.clamp_max(1)
        if not bool(torch.isfinite(penalty).all()):
            raise FloatingPointError("short-contact penalty is nonfinite")
        return penalty, "evaluated", len(neighbours)

    def processed_logits(self, x, old, positions, transaction_positions, attention_mask, *, phase=None):
        hard, unavailable = super().processed_logits(x,old,positions,transaction_positions,attention_mask,phase=phase)
        minimum = torch.finfo(hard.dtype).min
        output = hard
        self._last = {}
        for row,position in positions.items():
            if row in unavailable or position < 10 or (position-10) % 4:
                continue
            vector = hard[row,self.prompt_length+position]
            legal = torch.isfinite(vector)&(vector>minimum)
            ids = legal.nonzero(as_tuple=True)[0]
            b,reason,neighbours = self.penalty(x,row,position,ids)
            baseline = vector[ids].double()
            constant = bool(len(b) and (b==b[0]).all())
            # An exactly constant barrier cancels from exp(-b)/Z. Preserve its
            # original floating-point policy instead of rounding a common shift.
            tilted = baseline if constant else baseline-self.temperature*b
            changed = bool((b>0).any()) and not constant
            if changed and not self.probe_only:
                if output is hard:
                    target_dtype = torch.float64 if hard.dtype==torch.float64 else torch.float32
                    output = hard.to(target_dtype).clone()
                # Keep original sentinels untouched; never infer legal ids after casting.
                output[row,self.prompt_length+position,ids] = tilted.to(output.dtype)
            actual_tilted = tilted if hard.dtype==torch.float64 else tilted.float().double()
            if not changed:
                actual_tilted = baseline
            p = torch.softmax(baseline/self.temperature,-1)
            q = torch.softmax(actual_tilted/self.temperature,-1)
            diagnostic = {"row":row,"position":position,"phase":phase,"salt":self._contact_salt,"reason":reason,
                          "known_neighbours":neighbours,"legal_count":len(ids),"positive_candidates":int((b>0).sum()),
                          "constant_penalty_removed":float(b[0]) if constant else 0.,
                          "maximum_penalty":float(b.max()) if len(b) else 0.,
                          "tv":float(.5*(p-q).abs().sum()),"expected_penalty_before":float((p*b).sum()),
                          "expected_penalty_after":float((q*b).sum()),"input_dtype":str(hard.dtype),
                          "baseline_probability_sum":float(p.sum()),"tilted_probability_sum":float(q.sum()),
                          "penalty_in_unit_interval":bool(((b>=0)&(b<=1)).all()),
                          "cast_before_mask_inference":False,"applied":changed and not self.probe_only}
            self._last[row] = {"position":position,"ids":ids,"penalty":b,"diagnostic":diagnostic,
                               "hard_vector":vector,"tilted_legal":actual_tilted,"legal_mask":legal}
            if self.collect_vectors:
                entry={**diagnostic,"legal_ids":ids.cpu().tolist(),
                    "baseline_logits":baseline.cpu().tolist(),"tilted_logits":actual_tilted.cpu().tolist(),
                    "penalties":b.cpu().tolist()}
                self._last[row]["vector_record"]=entry
                self.contact_diagnostics.append(entry)
        return output, unavailable

    def _draw(self,*args,**kwargs):
        self._last = {}
        self._contact_salt = kwargs.get("salt")
        unavailable = super()._draw(*args,**kwargs)
        for row,value in self._last.items():
            if row in unavailable or not self.traces[row].events or self.traces[row].events[-1]["op"]!="draw":
                continue
            event = self.traces[row].events[-1]
            token = event["token"]
            where = (value["ids"]==token).nonzero(as_tuple=True)[0]
            if len(where)!=1:
                raise RuntimeError("sampled Z is outside the original legal support")
            event["short_contact"] = {**value["diagnostic"],"chosen_penalty":float(value["penalty"][where[0]])}
            if self.collect_vectors:
                hard=value["hard_vector"]
                dtype=torch.float64 if hard.dtype==torch.float64 else torch.float32
                tilted=hard.to(dtype).clone()
                tilted[value["ids"]]=value["tilted_legal"].to(dtype)
                same_support=torch.equal(tilted[~value["legal_mask"]],hard[~value["legal_mask"]].to(dtype))
                choices=[]
                for vector in (hard,tilted):
                    chosen=_transaction_candidate_tokens(vector[None,None],active_absolute_positions={0:0},
                        temperature=self.temperature,remasking="low_confidence",sampling_seeds_by_batch=[self.seeds[row]],
                        salt=int(self._contact_salt))
                    choices.append(int(chosen[0,0]))
                old_index=(value["ids"]==choices[0]).nonzero(as_tuple=True)[0]
                new_index=(value["ids"]==choices[1]).nonzero(as_tuple=True)[0]
                if not same_support or len(old_index)!=1 or len(new_index)!=1:
                    raise RuntimeError("counterfactual tilt changed original hard action support")
                old_b=float(value["penalty"][old_index[0]])
                if self.probe_only and choices[0]!=token:
                    raise RuntimeError("counterfactual baseline did not reproduce original row-local Gumbel draw")
                if old_b==0 and choices[1]!=choices[0]:
                    raise RuntimeError("zero-penalty original Gumbel winner was not preserved")
                value["vector_record"].update(baseline_selected_token=choices[0],tilted_selected_token=choices[1],
                    baseline_winner_penalty=old_b,tilted_winner_penalty=float(value["penalty"][new_index[0]]),
                    zero_penalty_winner_preserved=choices[0]==choices[1] if old_b==0 else None,
                    hard_support_unchanged=same_support)
        return unavailable

    def run(self,*args,**kwargs):
        output,traces = super().run(*args,**kwargs)
        for trace in traces:
            trace["short_contact_policy"] = {**self.contact_spec.metadata(),"probe_only":self.probe_only}
        return output,traces
