# CCFD frozen-tokenizer interface audit

Same-tokenizer Phase 1 feasible: **True**

| Dataset | Formulas | Syntax | Round-trip | Incremental | Prefix-safe | UNK-free | Prefill boundary | Newline-crossing | Tokens mean/q90/max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 27136 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 9.69/12/16 |
| val | 9047 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 9.67/12/16 |
| test | 9046 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 9.68/12/16 |
| raw1000 | 1000 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 10.01/12/16 |

## Gates

- all_syntax_terminal: `True`
- all_roundtrip_exact: `True`
- all_incremental_prefix_exact: `True`
- all_formula_prefix_safe: `True`
- all_unk_free: `True`
- all_prefill_boundaries_exact: `True`
- same_tokenizer_phase1_feasible: `True`
