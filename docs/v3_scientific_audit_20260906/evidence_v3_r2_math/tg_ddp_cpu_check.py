"""Tiny CPU DDP accumulation check; no LLaDA/checkpoint/data/MLIP is loaded."""
from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import sys
import uuid

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint


class TwoModeToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(2,3,dtype=torch.float64)
        self.token = nn.Linear(3,1,dtype=torch.float64)
        self.geometry = nn.Linear(3,2,dtype=torch.float64)

    def forward(self,x,mode):
        hidden = checkpoint(lambda value: torch.tanh(self.shared(value)),x,use_reentrant=False)
        return self.token(hidden) if mode == "T" else self.geometry(hidden)


def example(window,micro,rank,world):
    index = window*world*4+micro*world+rank
    x = torch.tensor([[.1+.03*index,-.4+.07*(index%5)]],dtype=torch.float64)
    mode = "T" if micro%2 == 0 else "G"
    target = torch.tensor([[.2-.01*index]] if mode == "T" else [[.03*index,-.08]],dtype=torch.float64)
    # In window two, rank one's final T/G pair is a zero-weight padded pair.
    weight = 0. if window == 1 and rank == 1 and micro >= 2 else 1.
    return x,target,mode,weight


def worker(rank,world,store_uri,result_path):
    torch.set_num_threads(1)
    torch.manual_seed(731)
    dist.init_process_group("gloo",rank=rank,world_size=world,init_method=store_uri,timeout=timedelta(seconds=45))
    model = TwoModeToy()
    ddp = DistributedDataParallel(model,find_unused_parameters=True)
    reference = deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0.,foreach=False)
    reference_optimizer = torch.optim.AdamW(reference.parameters(),lr=.01,weight_decay=0.,foreach=False)
    results = []
    for window in range(2):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        for micro in range(4):
            x,y,mode,weight = example(window,micro,rank,world)
            loss = .5*(ddp(x,mode)-y).square().mean()*weight/4
            loss.backward()  # Synchronize every microbatch; no no_sync context.
        for micro in range(4):
            for other_rank in range(world):
                x,y,mode,weight = example(window,micro,other_rank,world)
                (.5*(reference(x,mode)-y).square().mean()*weight/(4*world)).backward()
        gradient_error = max(float((a.grad-b.grad).abs().max()) for a,b in zip(model.parameters(),reference.parameters(),strict=True))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(reference.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        reference_optimizer.step()
        parameter_error = max(float((a-b).detach().abs().max()) for a,b in zip(model.parameters(),reference.parameters(),strict=True))
        combined = torch.tensor([gradient_error,parameter_error],dtype=torch.float64)
        dist.all_reduce(combined,op=dist.ReduceOp.MAX)
        assert float(combined.max()) < 1e-12
        results.append({"window": window,"max_gradient_difference":float(combined[0]),"max_parameter_difference":float(combined[1]),
                        "zero_weight_padding":window == 1})
    if rank == 0:
        report = {"scope":"21-parameter CPU toy with unused task heads; no real model, training data, GPU or MLIP",
                  "torch_version":str(torch.__version__),"backend":"gloo","world":world,"microbatch":1,"accumulation":4,
                  "modes":["T","G","T","G"],"find_unused_parameters":True,"sync_every_microbatch":True,
                  "checkpoint_use_reentrant":False,"clip_once_after_window":True,"windows":results,
                  "deployment_limit":"This CPU Torch version is not a substitute for the registered Torch 2.4/CUDA LLaDA multi-GPU acceptance."}
        Path(result_path).write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        print(json.dumps(report,indent=2),flush=True)
    dist.destroy_process_group()


def algebra_only():
    """Verify explicit per-micro rank averaging, not the unavailable local Gloo runtime."""
    reports = []
    for world,accumulation in ((4,6),(6,4)):
        torch.manual_seed(731)
        original = TwoModeToy()
        copies = [deepcopy(original) for _ in range(world)]
        reference = deepcopy(original)
        for micro in range(accumulation):
            for rank,model in enumerate(copies):
                index = micro*world+rank
                mode = "T" if micro%2 == 0 else "G"
                x = torch.tensor([[.2+.01*index,-.3]],dtype=torch.float64)
                y = torch.zeros((1,1 if mode == "T" else 2),dtype=torch.float64)
                # One padded pair on the final rank, keeping T/G padding balanced.
                weight = 0. if rank == world-1 and micro >= accumulation-2 else 1.
                (.5*(model(x,mode)-y).square().mean()*weight/accumulation).backward()
                (.5*(reference(x,mode)-y).square().mean()*weight/(world*accumulation)).backward()
            for parameters in zip(*(m.parameters() for m in copies),strict=True):
                present = [p.grad for p in parameters if p.grad is not None]
                if present:
                    mean = sum(present)/world
                    for parameter in parameters:
                        parameter.grad = mean.clone()
        maximum = max(float((a.grad-b.grad).abs().max()) for model in copies for a,b in
                      zip(model.parameters(),reference.parameters(),strict=True))
        assert maximum < 1e-12
        reports.append({"world":world,"accumulation":accumulation,"max_gradient_difference":maximum})
    torch.manual_seed(731)
    toy = TwoModeToy()
    x = torch.tensor([[.2,-.3]],dtype=torch.float64)
    (.5*toy(x,"T").square().mean()).backward()
    token_before = toy.token.weight.grad.clone()
    (.5*toy(x,"G").square().mean()).backward()
    preserved = {"token_head_grad_norm_after_T":float(token_before.norm()),
                 "token_head_grad_norm_after_G":float(toy.token.weight.grad.norm()),
                 "token_head_grad_increment_from_G":float((toy.token.weight.grad-token_before).abs().max())}
    assert preserved["token_head_grad_norm_after_G"] > 0
    assert preserved["token_head_grad_increment_from_G"] == 0
    output = Path(__file__).resolve().with_name("tg_window_algebra_check.json")
    output.write_text(json.dumps({"scope":"single-process explicit rank-averaging algebra; not a DDP runtime acceptance",
                                  "torch_version":str(torch.__version__),"cases":reports,
                                  "existing_grad_is_not_mode_leakage":preserved},indent=2)+"\n",encoding="utf-8")
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    if "--algebra-only" in sys.argv:
        algebra_only()
        raise SystemExit(0)
    output = Path(__file__).resolve().with_suffix(".json")
    store = output.parent/f"rendezvous_{uuid.uuid4().hex}.store"
    mp.spawn(worker,args=(2,store.as_uri(),str(output)),nprocs=2,join=True)
