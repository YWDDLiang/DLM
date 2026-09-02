# Track A: Plan-Conditioned LLM-Only Crystal Executor

Status: **system comparison; does not block priority Track B**

## 1. Role

Track A tests the same scientific Planner with an autoregressive structure
executor:

```text
C3FD–Llama Plan + Llama-pointer species program
        -> Plan-conditioned AR body Llama
        -> ordinary CrysLLMGen text
        -> parsed raw crystal
```

“LLM-only” means no masked DLM or continuous model produces the raw discrete
structure. C3FD still supplies scientific composition support.

## 2. Inputs

The AR prompt includes:

- exact N/elements/counts;
- soft LS/SG/VPA Plan;
- the species construction program.

The body uses native Llama text tokens. It never consumes DLM special tokens.

## 3. Training

Use the full MP20 train split with:

- MP20 teacher Compact-Plan prompts; predicted Plans are inference inputs;
- canonical CrysLLMGen body text;
- sites grouped in Planner program order;
- prompt tokens excluded from body CE;
- no energy, hull, CHGNet or continuous-refiner target.

A thin wrapper replaces the vendor trainer's mixed generation/infill behavior
and prevents accidental energy/hull attributes in prompts.

One body LoRA is sufficient for the first comparison. Track A uses at most two
A800 and runs in parallel only when B does not need those resources.

## 4. Inference

The Planner samples one valid Plan and predicts one program. The AR model generates one
crystal text trajectory in program order. The result is parsed into the
canonical state and evaluated without repair or replacement.

The common terminal continuous refiner is reported separately.

## 5. Comparison with B

Both routes share:

- Planner checkpoint;
- predicted Plan/program ledger;
- requested compositions;
- one trajectory and common sampling streams;
- final evaluator/refiner.

They do not share body tokenization. Track A establishes the AR system
baseline. Track B tests whether non-causal anchor generation and
suffix-preserving backfill add value.

## 6. Metrics

- Planner proposal composition validity;
- body composition retention and text/CIF parse;
- raw Direct, minimum-distance/collision and lattice diagnostics;
- raw stability surrogate;
- terminal Strict/Meta S.U.N.;
- wall time and model calls.
