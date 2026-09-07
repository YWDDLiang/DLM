"""Bounded CPU timing on the pinned domain; no model or physics execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from crystal_dlm.r03_formula_bridge import C3FDFormulaOracle, R03C3FDFormulaLogitsProcessor


DOMAIN_SHA = "a2b4a49eff72790cabaa75977414dba9eba45812ad02944d46e023a9b1d804a4"


class ByteLevelFragmentProxy:
    """Decode a saved ByteLevel vocabulary without loading neural libraries.

    This is explicitly a vocabulary-workload proxy, not P0 tokenization. Input
    test prefixes are encoded as single ASCII characters so their text is exact.
    """
    def __init__(self, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["decoder"]["type"] == "ByteLevel"
        values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        characters = list(values)
        added = 0
        for byte in range(256):
            if byte not in values:
                values.append(byte)
                characters.append(256 + added)
                added += 1
        decoder = {chr(character): byte for byte, character in zip(values, characters)}
        self.pieces = {
            int(token_id): bytes(decoder[character] for character in token).decode("utf-8", errors="replace")
            for token, token_id in payload["model"]["vocab"].items()
        }
        self.eos_token_id = None
        for token in payload["added_tokens"]:
            token_id, text = int(token["id"]), token["content"]
            self.pieces[token_id] = "" if token["special"] else text
            if text == "<|endoftext|>":
                self.eos_token_id = token_id
        assert self.eos_token_id is not None
        self.characters = {}
        for token_id, piece in self.pieces.items():
            if len(piece) == 1 and piece.isascii():
                self.characters.setdefault(piece, token_id)
        assert set(self.pieces) == set(range(len(self.pieces)))

    def __len__(self):
        return len(self.pieces)

    def decode(self, ids, **_kwargs):
        return "".join(self.pieces[int(token_id)] for token_id in ids)

    def encode_characters(self, text):
        return [self.characters[character] for character in text]


def percentiles(rows):
    values = sorted(row["seconds"] for row in rows)
    if not values:
        return {"queries": 0}
    return {
        "queries": len(values), "total_seconds": sum(values),
        "p50_seconds": statistics.median(values),
        "p95_seconds": values[min(len(values) - 1, int(0.95 * len(values)))],
        "max_seconds": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=240)
    args = parser.parse_args()
    assert 1 <= args.budget_seconds <= 300
    raw = args.domain.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DOMAIN_SHA
    domain = json.loads(raw)
    kwargs = {key: domain[key] for key in (
        "nodes", "electronegativities", "metal_symbols", "max_atoms", "max_species", "allowed_strata"
    )}
    formulas = []
    for line in args.corpus.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        match = re.search(r"(?m)^formula: ([^\n]+)", row.get("raw_model_text", row.get("raw_plan_text", "")))
        if match and match.group(1).strip() not in formulas:
            formulas.append(match.group(1).strip())
        if len(formulas) == 64:
            break
    assert formulas
    prefixes = list(dict.fromkeys(text[:end] for text in formulas for end in range(len(text) + 1)))
    result = {
        "schema": "r03_formula_bridge_cpu_timing_v1", "status": "running", "pid": os.getpid(),
        "domain_sha256": DOMAIN_SHA, "elements": len(domain["nodes"]), "strata": len(domain["allowed_strata"]),
        "source_sha256": hashlib.sha256((ROOT / "src/crystal_dlm/r03_formula_bridge.py").read_bytes()).hexdigest(),
        "corpus": str(args.corpus), "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "formula_count": len(formulas), "unique_prefix_count": len(prefixes),
        "wrapper_tokenizer_scope": "legacy LLaDA ByteLevel vocabulary proxy; not P0 tokenizer or BPE trajectory",
        "tokenizer_json": str(args.tokenizer_json),
        "tokenizer_sha256": hashlib.sha256(args.tokenizer_json.read_bytes()).hexdigest(),
        "no_model_or_gpu_or_physics_calls": True,
        "budget_seconds": args.budget_seconds,
        "semantic_cold": [], "semantic_warm": [], "wrapper_cold": [], "wrapper_warm": [],
    }
    started = time.perf_counter()
    def save():
        result["elapsed_seconds"] = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    def expired():
        result["status"] = "budget_timeout"
        save()
        print(json.dumps({"event": "budget_timeout", "in_progress": result.get("in_progress")}), flush=True)
        os._exit(124)
    watchdog = threading.Timer(args.budget_seconds, expired)
    watchdog.daemon = True
    watchdog.start()
    try:
        start = time.perf_counter()
        oracle = C3FDFormulaOracle(**kwargs)
        result["semantic_constructor_seconds"] = time.perf_counter() - start
        for phase in ("semantic_cold", "semantic_warm"):
            for ordinal, prefix in enumerate(prefixes):
                result["in_progress"] = {"phase": phase, "prefix": prefix, "ordinal": ordinal}
                tick = time.perf_counter()
                value = oracle.is_prefix_viable(prefix)
                elapsed = time.perf_counter() - tick
                result[phase].append({"prefix": prefix, "seconds": elapsed, "result": value})
                if elapsed > 0.25:
                    print(json.dumps({"event": "slow_semantic_prefix", "phase": phase, "prefix": prefix, "seconds": elapsed}), flush=True)
            result[phase + "_summary"] = percentiles(result[phase])
            result[phase + "_cache"] = oracle.stats()
            save()
            print(json.dumps({"phase": phase, **result[phase + "_summary"]}), flush=True)
        start = time.perf_counter()
        tokenizer = ByteLevelFragmentProxy(args.tokenizer_json)
        result["proxy_decode_seconds"] = time.perf_counter() - start
        wrapper_oracle = C3FDFormulaOracle(**kwargs)
        start = time.perf_counter()
        processor = R03C3FDFormulaLogitsProcessor(
            tokenizer, oracle=wrapper_oracle, start_length=0, eos_token_id=tokenizer.eos_token_id
        )
        result["wrapper_constructor_seconds"] = time.perf_counter() - start
        result["wrapper_static_stats"] = processor.stats()
        print(json.dumps({"phase": "wrapper_ready", **processor.stats()}), flush=True)
        wrapper_prefixes = list(dict.fromkeys(
            "formula: " + text[:end] for text in formulas[:4] for end in range(len(text) + 1)
        ))
        for phase in ("wrapper_cold", "wrapper_warm"):
            for ordinal, prefix in enumerate(wrapper_prefixes):
                result["in_progress"] = {"phase": phase, "prefix": prefix, "ordinal": ordinal}
                ids = tokenizer.encode_characters(prefix)
                assert tokenizer.decode(ids) == prefix
                tick = time.perf_counter()
                allowed = processor.allowed_token_ids(ids)
                elapsed = time.perf_counter() - tick
                result[phase].append({"prefix": prefix, "seconds": elapsed,
                                      "allowed_count": None if allowed is None else len(allowed)})
                print(json.dumps({"phase": phase, "prefix": prefix, "seconds": elapsed,
                                  "allowed_count": None if allowed is None else len(allowed)}), flush=True)
            result[phase + "_summary"] = percentiles(result[phase])
            result[phase + "_stats"] = processor.stats()
            result[phase + "_cache"] = wrapper_oracle.stats()
            save()
        result["status"] = "complete"
        result["in_progress"] = None
        save()
        print(json.dumps({key: value for key, value in result.items() if key.endswith("_summary") or key in ("status", "elapsed_seconds")}), flush=True)
    finally:
        watchdog.cancel()


if __name__ == "__main__":
    main()
