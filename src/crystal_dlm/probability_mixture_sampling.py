"""One fixed equal probability mixture of same-state legacy K4/K8 policies."""
from __future__ import annotations

import math
import torch

from crystal_dlm.programmed_path_runtime import ProgrammedPathSampler

SCHEMA = "k4k8_equal_same_state_probability_mixture_v1"


def mix_post_hard_vectors(left, right, temperature=.7):
    """Mix normalized post-hard/alias probabilities; never logits or supports."""
    if (temperature != .7 or left.ndim != 1 or left.shape != right.shape or left.device != right.device
            or not left.is_floating_point() or not right.is_floating_point()):
        raise ValueError("mixture requires same-device scalar policy vectors at fixed temperature .7")
    if any(bool(torch.isnan(v).any() or torch.isposinf(v).any()) for v in (left,right)):
        raise ValueError("a mixture component is nonfinite")
    a=torch.isfinite(left)&(left>torch.finfo(left.dtype).min)
    b=torch.isfinite(right)&(right>torch.finfo(right.dtype).min)
    if not torch.equal(a,b) or not bool(a.any()):
        raise ValueError("component hard supports differ or are empty")
    logp4=torch.log_softmax(left[a].double()/temperature,-1)
    logp8=torch.log_softmax(right[a].double()/temperature,-1)
    logq=torch.logaddexp(logp4,logp8)-math.log(2)
    if not bool(torch.isfinite(logq).all()):
        raise ValueError("normalized mixture probability is nonfinite")
    output=torch.full_like(left,torch.finfo(torch.float64).min,dtype=torch.float64)
    output[a]=temperature*logq
    p4,p8,q=logp4.exp(),logp8.exp(),logq.exp()
    return output,{"support_equal":True,"legal_count":int(a.sum()),"probability_sum":float(q.sum()),
                   "component_tv":float(.5*(p4-p8).abs().sum()),
                   "kl_k4_to_mixture":float((p4*(logp4-logq)).sum()),
                   "kl_k8_to_mixture":float((p8*(logp8-logq)).sum()),
                   "temperature":temperature,"weights":[.5,.5]}


class ProbabilityMixtureProgrammedPathSampler(ProgrammedPathSampler):
    def __init__(self,*args,peer_model,**kwargs):
        super().__init__(*args,**kwargs)
        if self.temperature != .7 or peer_model is self.model:
            raise ValueError("fixed mixture requires two separate policies and temperature .7")
        self.peer=ProgrammedPathSampler(peer_model,prompt_length=self.prompt_length,gen_length=self.gen_length,
            mask_id=self.mask_id,programs=self.programs,allowed_token_ids=self.allowed_ids,
            atom_count_grammar=self.grammar,constraints=self.constraints,temperature=self.temperature,sampling_seeds=self.seeds)
        self.component_forward_calls=[0,0]
        self._mixture_last={}

    def _prepare(self,x):
        super()._prepare(x)
        if self.peer.model.get_output_embeddings().weight.shape[0] != self.model.get_output_embeddings().weight.shape[0]:
            raise ValueError("mixture vocabulary sizes differ")
        self.peer._allowed=self._allowed
        self.peer._prepared_grammar=self._prepared_grammar

    def processed_logits(self,x,old,positions,transaction_positions,attention_mask,*,phase=None):
        self.component_forward_calls[0]+=1
        first,bad4=super().processed_logits(x,old,positions,transaction_positions,attention_mask,phase=phase)
        self.component_forward_calls[1]+=1
        second,bad8=self.peer.processed_logits(x,old,positions,transaction_positions,attention_mask,phase=phase)
        if first.shape!=second.shape or first.device!=second.device:
            raise ValueError("same-state mixture component canvases differ")
        unavailable=set(bad4)|set(bad8)
        # The scalar runtime reads only one active position per row. Back its
        # canvas view with B*V values, not a resident B*T*V float64 allocation.
        active_vectors=torch.full((first.shape[0],first.shape[-1]),torch.finfo(torch.float64).min,
                                  dtype=torch.float64,device=first.device)
        sequence_length=first.shape[1]
        self._mixture_last={}
        for row,position in positions.items():
            if row in unavailable:
                self._mixture_last[row]={"component_unavailable":[row in bad4,row in bad8],"single_component_fallback":False}
                continue
            try:
                vector,diagnostic=mix_post_hard_vectors(first[row,self.prompt_length+position],
                                                       second[row,self.prompt_length+position],self.temperature)
                active_vectors[row]=vector
                self._mixture_last[row]=diagnostic
            except ValueError as error:
                unavailable.add(row)
                self._mixture_last[row]={"support_or_numeric_error":str(error),"single_component_fallback":False}
        del first,second
        return active_vectors[:,None,:].expand(-1,sequence_length,-1),unavailable

    def _draw(self,*args,**kwargs):
        self._mixture_last={}
        unavailable=super()._draw(*args,**kwargs)
        for row,diagnostic in self._mixture_last.items():
            if self.traces[row].events and self.traces[row].events[-1]["op"] in {"draw","no_support"}:
                self.traces[row].events[-1]["probability_mixture"]=diagnostic
        return unavailable

    def run(self,*args,**kwargs):
        output,traces=super().run(*args,**kwargs)
        for trace in traces:
            trace["probability_mixture_policy"]={"schema":SCHEMA,"weights":[.5,.5],"temperature":.7,
                "components":["k4","k8"],"same_state":True,"contact":False,"single_component_fallback":False}
        return output,traces
