"""Offline Llama proposal/edit engine with exact tokenizer-level support masks."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

from ..bridge import ChartCatalog
from ..state import GeometryEvidence, StratifiedState
from .chemistry_constraint import ChargeAwareStopConstraint
from .constrained import (
    ProposalTokenConstraint,
    TopologyEditTokenConstraint,
    grammar_token_fragments,
)
from .formula_plan import (
    FORMULA_BODY_SYSTEM_PROMPT,
    FORMULA_PLAN_SYSTEM_PROMPT,
    FORMULA_PLAN_USER_PROMPT,
    FormulaPlan,
    FormulaPlanTokenConstraint,
    PlannedProposalTokenConstraint,
    formula_body_user_prompt,
    formula_plan_matches_state,
    parse_formula_plan,
)
from .sft_data import (
    UNCONDITIONAL_USER_PROMPT,
    WQ_EDIT_SYSTEM_PROMPT,
    WQ_SYSTEM_PROMPT,
    serialize_geometry_evidence,
)
from .wq_text import (
    TopologyEdit,
    parse_topology_edit,
    parse_wq_proposal,
    serialize_wq_proposal,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class WQLlamaEngine:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        base_root: Path,
        adapter_root: Path,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.base_root = base_root
        self.adapter_root = adapter_root
        self.token_fragments = grammar_token_fragments(tokenizer)
        adapter_model = adapter_root / "adapter_model.safetensors"
        adapter_config = adapter_root / "adapter_config.json"
        self.identity = {
            "base_root": str(base_root),
            "adapter_root": str(adapter_root),
            "adapter_model_sha256": _sha256(adapter_model),
            "adapter_config_sha256": _sha256(adapter_config),
            "tokenizer_size": len(tokenizer),
            "grammar_token_fragments": len(self.token_fragments),
            "parameter_count_with_adapter": sum(
                value.numel() for value in model.parameters()
            ),
        }

    @classmethod
    def load(
        cls,
        *,
        base_root: str | Path,
        adapter_root: str | Path,
    ) -> "WQLlamaEngine":
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = Path(base_root).resolve()
        adapter = Path(adapter_root).resolve()
        for path in (
            adapter / "adapter_model.safetensors",
            adapter / "adapter_config.json",
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        tokenizer = AutoTokenizer.from_pretrained(
            base,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
            model_max_length=512,
            padding_side="right",
        )
        if tokenizer.eos_token_id is None or not tokenizer.chat_template:
            raise RuntimeError("registered tokenizer lacks EOS or chat template")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            device_map={"": 0},
        )
        model = PeftModel.from_pretrained(
            model,
            adapter,
            local_files_only=True,
            is_trainable=False,
        )
        model.eval()
        model.config.use_cache = True
        return cls(model=model, tokenizer=tokenizer, base_root=base, adapter_root=adapter)

    def _prompt(self, system: str, user: str) -> Any:
        tokens = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return tokens.to(next(self.model.parameters()).device)

    def _generate(
        self,
        prompt: Any,
        *,
        constraint: Any,
        seed: int,
        max_new_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        import torch

        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        started = time.monotonic()
        prompt_width = int(prompt.shape[1])
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                renormalize_logits=True,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=constraint,
            )
        generated = output[0, prompt_width:]
        emitted_eos = bool(
            generated.numel()
            and int(generated[-1]) == int(self.tokenizer.eos_token_id)
        )
        if emitted_eos:
            generated = generated[:-1]
        text = self.tokenizer.decode(
            generated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return text, {
            "prompt_tokens": prompt_width,
            "generated_tokens": int(output.shape[1] - prompt_width),
            "emitted_eos": emitted_eos,
            "llama_invocations": 1,
            "walltime_s": time.monotonic() - started,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }

    def propose(
        self,
        *,
        catalog: ChartCatalog,
        seed: int,
        attempt_id: str,
    ) -> tuple[StratifiedState, str, dict[str, Any]]:
        prompt = self._prompt(WQ_SYSTEM_PROMPT, UNCONDITIONAL_USER_PROMPT)
        constraint = ProposalTokenConstraint(
            self.tokenizer,
            catalog,
            prompt_width=int(prompt.shape[1]),
            token_fragments=self.token_fragments,
        )
        remaining = 512 - int(prompt.shape[1])
        if remaining <= 0:
            raise ValueError("registered proposal prompt consumes the complete context")
        text, usage = self._generate(
            prompt,
            constraint=constraint,
            seed=seed,
            max_new_tokens=remaining,
        )
        state = parse_wq_proposal(text, catalog, attempt_id=attempt_id, timestep=1.0)
        return state, text, usage

    def propose_charge_aware_stop(
        self,
        *,
        catalog: ChartCatalog,
        seed: int,
        attempt_id: str,
    ) -> tuple[StratifiedState, str, dict[str, Any]]:
        """Generate once while deferring only charge-invalid STOP decisions."""

        prompt = self._prompt(WQ_SYSTEM_PROMPT, UNCONDITIONAL_USER_PROMPT)
        constraint = ChargeAwareStopConstraint(
            self.tokenizer,
            catalog,
            prompt_width=int(prompt.shape[1]),
            token_fragments=self.token_fragments,
        )
        remaining = 512 - int(prompt.shape[1])
        if remaining <= 0:
            raise ValueError("registered proposal prompt consumes the complete context")
        text, usage = self._generate(
            prompt,
            constraint=constraint,
            seed=seed,
            max_new_tokens=remaining,
        )
        state = parse_wq_proposal(text, catalog, attempt_id=attempt_id, timestep=1.0)
        usage["chemistry_constraint"] = constraint.diagnostics()
        return state, text, usage

    def generate_formula_plan(
        self,
        *,
        plan_seed: int,
    ) -> tuple[FormulaPlan, str, dict[str, Any]]:
        """Generate and validate one formula plan before any body decoding."""

        plan_prompt = self._prompt(
            FORMULA_PLAN_SYSTEM_PROMPT,
            FORMULA_PLAN_USER_PROMPT,
        )
        plan_constraint = FormulaPlanTokenConstraint(
            self.tokenizer,
            prompt_width=int(plan_prompt.shape[1]),
            token_fragments=self.token_fragments,
        )
        plan_remaining = min(128, 512 - int(plan_prompt.shape[1]))
        if plan_remaining <= 0:
            raise ValueError("formula-plan prompt consumes the complete context")
        plan_text, plan_usage = self._generate(
            plan_prompt,
            constraint=plan_constraint,
            seed=plan_seed,
            max_new_tokens=plan_remaining,
        )
        return parse_formula_plan(plan_text), plan_text, plan_usage

    def generate_formula_body(
        self,
        *,
        catalog: ChartCatalog,
        plan: FormulaPlan,
        body_seed: int,
        attempt_id: str,
    ) -> tuple[StratifiedState, str, dict[str, Any]]:
        """Generate one WQ body on exact count-reachable support."""

        body_prompt = self._prompt(
            FORMULA_BODY_SYSTEM_PROMPT,
            formula_body_user_prompt(plan),
        )
        body_constraint = PlannedProposalTokenConstraint(
            self.tokenizer,
            catalog,
            plan,
            prompt_width=int(body_prompt.shape[1]),
            token_fragments=self.token_fragments,
        )
        body_remaining = 640 - int(body_prompt.shape[1])
        if body_remaining <= 0:
            raise ValueError("formula-conditioned body prompt consumes the context")
        body_text, body_usage = self._generate(
            body_prompt,
            constraint=body_constraint,
            seed=body_seed,
            max_new_tokens=body_remaining,
        )
        state = parse_wq_proposal(
            body_text,
            catalog,
            attempt_id=attempt_id,
            timestep=1.0,
        )
        if not formula_plan_matches_state(plan, state):
            raise RuntimeError("formula-conditioned body did not match its frozen plan")
        body_usage["formula_plan"] = plan.as_dict()
        body_usage["plan_body_exact_match"] = True
        return state, body_text, body_usage

    def propose_formula_plan(
        self,
        *,
        catalog: ChartCatalog,
        plan_seed: int,
        body_seed: int,
        attempt_id: str,
    ) -> tuple[StratifiedState, str, str, dict[str, Any]]:
        """Generate one chemistry plan and one exactly matching WQ body.

        This is deliberately two forward-generation calls through the same
        adapter.  The first call chooses primitive-cell stoichiometry; the
        second call can only consume those counts through legal Wyckoff
        primitive multiplicities.  There is no repair, retry, or reranking.
        """

        plan, plan_text, plan_usage = self.generate_formula_plan(
            plan_seed=plan_seed,
        )
        state, body_text, body_usage = self.generate_formula_body(
            catalog=catalog,
            plan=plan,
            body_seed=body_seed,
            attempt_id=attempt_id,
        )
        usage = {
            "llama_invocations": 2,
            "prompt_tokens": int(plan_usage["prompt_tokens"])
            + int(body_usage["prompt_tokens"]),
            "generated_tokens": int(plan_usage["generated_tokens"])
            + int(body_usage["generated_tokens"]),
            "walltime_s": float(plan_usage["walltime_s"])
            + float(body_usage["walltime_s"]),
            "plan": plan_usage,
            "body": body_usage,
            "formula_plan": plan.as_dict(),
            "plan_body_exact_match": True,
        }
        return state, plan_text, body_text, usage

    def edit(
        self,
        state: StratifiedState,
        evidence: Sequence[GeometryEvidence],
        *,
        catalog: ChartCatalog,
        seed: int,
    ) -> tuple[TopologyEdit, str, dict[str, Any]]:
        proposal = serialize_wq_proposal(state, catalog)
        geometry = serialize_geometry_evidence([value.as_tuple() for value in evidence])
        user = f"P={proposal};G={geometry}"
        prompt = self._prompt(WQ_EDIT_SYSTEM_PROMPT, user)
        if int(prompt.shape[1]) >= 512:
            raise ValueError("registered edit prompt exceeds the training context")
        constraint = TopologyEditTokenConstraint(
            self.tokenizer,
            state,
            catalog,
            prompt_width=int(prompt.shape[1]),
            token_fragments=self.token_fragments,
        )
        text, usage = self._generate(
            prompt,
            constraint=constraint,
            seed=seed,
            max_new_tokens=min(64, 512 - int(prompt.shape[1])),
        )
        return parse_topology_edit(text, state, catalog), text, usage
