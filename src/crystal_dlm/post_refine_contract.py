"""SUN-gated, token-boundary revision contracts without model dependencies.

Scoring is explicitly cohort-dependent. A gate may only consume the complete
bound draft score ledger; preferences can point in either direction.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA = "sun_gated_token_revision_v1"
PAIR_SCHEMA = "sun_ranked_revision_pair_v1"
QUALITY_ORDER = ("other", "meta_stable", "meta_sun", "stable", "sun")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def derived_seed(source_id: str, stage: str, round_index: int = 0) -> int:
    if not source_id or not stage or type(round_index) is not int or round_index < 0:
        raise ValueError("a source, stage and nonnegative round are required")
    value = f"{SCHEMA}|{source_id}|{stage}|{round_index}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % (2**63)


def strict_flag(value: Any, name: str) -> bool | None:
    if value is not None and type(value) is not bool:
        raise ValueError(f"{name} must be a boolean or explicit unknown")
    return value


@dataclass(frozen=True)
class Quality:
    sun: bool | None
    stable: bool | None
    meta_sun: bool | None
    meta_stable: bool | None
    hull: float | None
    consistent: bool = True

    @classmethod
    def from_score(cls, row: Mapping[str, Any]) -> "Quality":
        flags = [strict_flag(row.get(key), key) for key in
                 ("strict_sun", "strict_stable", "meta_sun", "meta_stable")]
        hull = row.get("e_above_hull_eV_atom")
        if hull is not None:
            if isinstance(hull, bool) or not isinstance(hull, (int, float)) or not math.isfinite(hull):
                raise ValueError("energy above hull must be finite or unknown")
            hull = float(hull)
        if flags[0] is True and (flags[1] is not True or flags[2] is not True):
            raise ValueError("Strict SUN contradicts stability or MSUN")
        if flags[1] is True and flags[3] is not True:
            raise ValueError("strict stability contradicts meta stability")
        if flags[2] is True and flags[3] is not True:
            raise ValueError("MSUN contradicts meta stability")
        # Verification is NOT part of formal SUN. Only explicit contradictory
        # physical evidence prevents a preference label.
        consistent = row.get("preference_physics_consistent", True)
        if type(consistent) is not bool:
            raise ValueError("preference consistency must be explicit")
        return cls(*flags, hull, consistent)

    @property
    def tier(self) -> str | None:
        if not self.consistent or self.hull is None:
            return None
        for name, flag in (("sun", self.sun), ("stable", self.stable),
                           ("meta_sun", self.meta_sun), ("meta_stable", self.meta_stable)):
            if flag is True:
                return name
            if flag is None:
                return None
        return "other"


def preference(before: Quality, after: Quality, *, energy_margin: float = 0.01) -> dict[str, Any]:
    """Return the better endpoint, never assume the refiner is the winner."""
    if not math.isfinite(energy_margin) or energy_margin < 0:
        raise ValueError("energy margin must be finite and nonnegative")
    a, b = before.tier, after.tier
    result = {"before_tier": a, "after_tier": b, "chosen": None,
              "reason": "unknown_or_inconsistent", "energy_margin_eV_atom": energy_margin}
    if a is None or b is None:
        return result
    if a != b:
        result.update(chosen="after" if QUALITY_ORDER.index(b) > QUALITY_ORDER.index(a) else "before",
                      reason="SUN_then_Stable_then_meta")
        return result
    gain = before.hull - after.hull
    result["hull_gain_eV_atom"] = gain
    if abs(gain) > energy_margin:
        result.update(chosen="after" if gain > 0 else "before", reason="same_tier_hull_margin")
    else:
        result["reason"] = "quality_tie"
    return result


def make_gate(records: Sequence[Mapping[str, Any]], scores: Sequence[Mapping[str, Any]], *,
              input_sha256: str, scored_input_sha256: str, score_identity: Mapping[str, Any],
              diffusion_steps: int = 800) -> dict[str, Any]:
    if type(diffusion_steps) is not int or not 1 <= diffusion_steps <= 1000:
        raise ValueError("diffusion steps must be in 1..1000")
    if not records or input_sha256 != scored_input_sha256:
        raise ValueError("draft score ledger belongs to a different input")
    ids = [row["trajectory_id"] for row in records]
    if len(set(ids)) != len(ids) or len(scores) != len(ids):
        raise ValueError("a gate needs one score per complete cohort occurrence")
    by_id = {row["trajectory_id"]: row for row in scores}
    if len(by_id) != len(scores) or set(by_id) != set(ids):
        raise ValueError("draft score identities differ from the complete cohort")
    decisions = []
    for ordinal, record in enumerate(records):
        if record.get("evaluation_ordinal", ordinal) != ordinal:
            raise ValueError("cohort order must be fixed before computing uniqueness")
        score = by_id[record["trajectory_id"]]
        if score.get("sample_idx") != record["sample_idx"]:
            raise ValueError("draft score original sample identity differs")
        quality = Quality.from_score(score)
        success = record.get("success") is True
        skip = success and quality.sun is True
        if not success:
            reason = "upstream_generation_failure"
        elif skip:
            reason = "draft_strict_SUN"
        elif quality.meta_sun is True:
            reason = "MSUN_still_requires_diffusion"
        else:
            reason = "not_confirmed_strict_SUN"
        decisions.append({"trajectory_id": record["trajectory_id"],
                          "original_ordinal": record.get("original_ordinal", record["sample_idx"]),
                          "evaluation_ordinal": ordinal, "input_fingerprint": fingerprint(record),
                          "skip_diffusion": skip, "run_diffusion": success and not skip,
                          "diffusion_steps": diffusion_steps if success and not skip else 0,
                          "reason": reason, "draft_tier": quality.tier})
    return {"schema": SCHEMA, "draft_input_sha256": input_sha256,
            "score_identity": dict(score_identity), "cohort_denominator": len(records),
            "decisions": decisions, "draft_SUN_skip_count": sum(d["skip_diffusion"] for d in decisions),
            "diffusion_count": sum(d["run_diffusion"] for d in decisions),
            "relaxed_label_geometry_is_not_written_back": True,
            "N_U_recomputed_after_round": True}


def make_pair(*, source_id: str, conditioning: Mapping[str, Any], before_tokens: Sequence[int],
              after_tokens: Sequence[int], before_score: Mapping[str, Any], after_score: Mapping[str, Any],
              before_identity: Mapping[str, Any], after_identity: Mapping[str, Any],
              round_index: int, purpose: str, energy_margin: float = 0.01) -> dict[str, Any]:
    if purpose not in ("train", "cal", "dev", "evaluation"):
        raise ValueError("pair source split must be preserved")
    if not source_id or not before_tokens or not after_tokens or round_index < 0:
        raise ValueError("a complete source and both token endpoints are required")
    if any(type(t) is not int or t < 0 for t in [*before_tokens, *after_tokens]):
        raise ValueError("responses must contain actual token IDs")
    decision = preference(Quality.from_score(before_score), Quality.from_score(after_score),
                          energy_margin=energy_margin)
    if list(before_tokens) == list(after_tokens):
        decision.update(chosen=None, reason="identical_token_responses")
    payload = {"schema": PAIR_SCHEMA, "source_id": source_id, "source_split": purpose,
               "round_index": round_index, "conditioning": dict(conditioning),
               "conditioning_sha256": fingerprint(conditioning),
               "before_tokens": list(before_tokens), "after_tokens": list(after_tokens),
               "before_identity": dict(before_identity), "after_identity": dict(after_identity),
               "before_score": dict(before_score), "after_score": dict(after_score),
               "preference": decision, "training_use_allowed": purpose == "train",
               "pair_usable": decision["chosen"] is not None,
               "DPO_training_performed": False}
    payload["pair_id"] = fingerprint(payload)
    return payload


def stage_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot report an empty evaluation cohort")
    result = {"requested": len(rows)}
    for metric in ("strict_sun", "meta_sun", "strict_stable", "meta_stable"):
        values = [strict_flag(row.get(metric), metric) for row in rows]
        positive = sum(value is True for value in values)
        unknown = sum(value is None for value in values)
        result[metric] = {"count": positive if not unknown else None, "confirmed": positive,
                          "unknown": unknown, "percent": 100 * positive / len(rows) if not unknown else None,
                          "bounds_percent": [100 * positive / len(rows), 100 * (positive + unknown) / len(rows)]}
    return result


def paired_change(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(before) != len(after) or not before:
        raise ValueError("stage comparisons require the same nonempty denominator")
    old = {row["sample_idx"]: row for row in before}
    new = {row["sample_idx"]: row for row in after}
    if len(old) != len(before) or len(new) != len(after) or set(old) != set(new):
        raise ValueError("paired stage source identities differ")
    result = {"requested": len(before)}
    for metric, name in (("strict_sun", "SUN"), ("meta_sun", "MSUN")):
        gain, loss, unknown = [], [], []
        for key in old:
            a, b = (strict_flag(row.get(metric), metric) for row in (old[key], new[key]))
            if a is None or b is None:
                unknown.append(key)
            elif not a and b:
                gain.append(key)
            elif a and not b:
                loss.append(key)
        result.update({name + "_gains": len(gain), name + "_losses": len(loss),
                       name + "_net_known": len(gain) - len(loss), name + "_unknown_pairs": len(unknown),
                       name + "_gain_sources": gain, name + "_loss_sources": loss,
                       name + "_unknown_sources": unknown})
    return result
