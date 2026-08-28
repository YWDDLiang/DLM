"""Train-only soft co-occurrence prior for semantic composition actions.

This is intentionally not a tokenizer.  Nodes are typed element/oxidation
states and edges summarize their formula-level co-occurrence in the training
split.  The score may be added to a model's semantic action logits, but it can
never override the CCFD atom/charge/reachability mask.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

from crystal_dlm.ccfd import FormulaToken
from crystal_dlm.fixed_slot import Z_TO_SYMBOL


@dataclass(frozen=True, order=True)
class ValenceNode:
    atomic_number: int
    oxidation_state: int

    @classmethod
    def from_token(cls, token: FormulaToken) -> "ValenceNode":
        return cls(int(token.atomic_number), int(token.oxidation_state))

    @property
    def label(self) -> str:
        symbol = Z_TO_SYMBOL.get(int(self.atomic_number), f"Z{int(self.atomic_number)}")
        sign = "+" if int(self.oxidation_state) > 0 else ""
        return f"{symbol}{sign}{int(self.oxidation_state)}"


def _pair(left: ValenceNode, right: ValenceNode) -> tuple[ValenceNode, ValenceNode]:
    if left == right:
        raise ValueError("co-occurrence edges require two distinct nodes")
    return tuple(sorted((left, right)))  # type: ignore[return-value]


@dataclass(frozen=True)
class CompositionPairPrior:
    """Smoothed pairwise PMI estimated only from training compositions."""

    composition_count: int
    node_counts: dict[ValenceNode, int]
    pair_counts: dict[tuple[ValenceNode, ValenceNode], int]
    alpha: float = 1.0
    clip: float = 6.0

    @classmethod
    def fit(
        cls,
        compositions: Iterable[Sequence[FormulaToken | ValenceNode]],
        *,
        alpha: float = 1.0,
        clip: float = 6.0,
    ) -> "CompositionPairPrior":
        if float(alpha) <= 0:
            raise ValueError("alpha must be positive")
        node_counts: Counter[ValenceNode] = Counter()
        pair_counts: Counter[tuple[ValenceNode, ValenceNode]] = Counter()
        composition_count = 0
        for composition in compositions:
            nodes = sorted(
                {
                    value
                    if isinstance(value, ValenceNode)
                    else ValenceNode.from_token(value)
                    for value in composition
                }
            )
            if not nodes:
                continue
            composition_count += 1
            node_counts.update(nodes)
            for left_index, left in enumerate(nodes):
                for right in nodes[left_index + 1 :]:
                    pair_counts[_pair(left, right)] += 1
        if composition_count <= 0:
            raise ValueError("cannot fit a pair prior without compositions")
        return cls(
            composition_count=int(composition_count),
            node_counts=dict(node_counts),
            pair_counts=dict(pair_counts),
            alpha=float(alpha),
            clip=float(clip),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CompositionPairPrior":
        node_counts = {
            ValenceNode(int(row["atomic_number"]), int(row["oxidation_state"])): int(
                row["count"]
            )
            for row in payload.get("node_records") or ()
        }
        pair_counts = {
            _pair(
                ValenceNode(int(row["left_atomic_number"]), int(row["left_oxidation_state"])),
                ValenceNode(int(row["right_atomic_number"]), int(row["right_oxidation_state"])),
            ): int(row["count"])
            for row in payload.get("edge_records") or ()
        }
        return cls(
            composition_count=int(payload["composition_count"]),
            node_counts=node_counts,
            pair_counts=pair_counts,
            alpha=float(payload.get("alpha", 1.0)),
            clip=float(payload.get("clip", 6.0)),
        )

    @property
    def nodes(self) -> tuple[ValenceNode, ...]:
        return tuple(sorted(self.node_counts))

    def pair_pmi(self, left: ValenceNode, right: ValenceNode) -> float:
        """Return symmetric add-alpha PMI, clipped for robust logit use."""

        if left == right:
            return 0.0
        pair_count = float(self.pair_counts.get(_pair(left, right), 0))
        left_count = float(self.node_counts.get(left, 0))
        right_count = float(self.node_counts.get(right, 0))
        n = float(self.composition_count)
        alpha = float(self.alpha)
        # The same additive correction is applied to the joint and marginals.
        # This is a ranking prior rather than a calibrated probability model.
        value = math.log(
            ((pair_count + alpha) * (n + alpha))
            / ((left_count + alpha) * (right_count + alpha))
        )
        return max(-float(self.clip), min(float(self.clip), float(value)))

    def context_score(
        self,
        candidate: ValenceNode | FormulaToken,
        context: Sequence[ValenceNode | FormulaToken],
    ) -> float:
        """Mean pair compatibility with already emitted semantic nodes."""

        candidate_node = (
            candidate
            if isinstance(candidate, ValenceNode)
            else ValenceNode.from_token(candidate)
        )
        context_nodes = sorted(
            {
                value if isinstance(value, ValenceNode) else ValenceNode.from_token(value)
                for value in context
                if (
                    value if isinstance(value, ValenceNode) else ValenceNode.from_token(value)
                )
                != candidate_node
            }
        )
        if not context_nodes:
            return 0.0
        return sum(self.pair_pmi(candidate_node, node) for node in context_nodes) / len(
            context_nodes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "h1a2_semantic_composition_pair_prior_v1",
            "semantics": "soft train-only element-valence co-occurrence; never a legality mask",
            "composition_count": int(self.composition_count),
            "node_count": len(self.node_counts),
            "edge_count": len(self.pair_counts),
            "alpha": float(self.alpha),
            "clip": float(self.clip),
            "nodes": {
                node.label: int(count)
                for node, count in sorted(self.node_counts.items())
            },
            "edges": {
                f"{left.label}|{right.label}": int(count)
                for (left, right), count in sorted(self.pair_counts.items())
            },
            "node_records": [
                {
                    "atomic_number": int(node.atomic_number),
                    "oxidation_state": int(node.oxidation_state),
                    "count": int(count),
                }
                for node, count in sorted(self.node_counts.items())
            ],
            "edge_records": [
                {
                    "left_atomic_number": int(left.atomic_number),
                    "left_oxidation_state": int(left.oxidation_state),
                    "right_atomic_number": int(right.atomic_number),
                    "right_oxidation_state": int(right.oxidation_state),
                    "count": int(count),
                }
                for (left, right), count in sorted(self.pair_counts.items())
            ],
        }


__all__ = ["CompositionPairPrior", "ValenceNode"]
