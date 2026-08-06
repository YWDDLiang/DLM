# Exact-tokenizer preflight V1 → V2 repair

V1 immutable report:

- status: `fail`
- report SHA-256:
  `88cb59739b7e347eba1f24fdfccbd20825a81e51062d68b81022f1b224c4c92f`
- sole failure: `all_metal_direct_alignment`
- exact-tokenizer trie/scalar support parity: pass
- V1 trie/scalar support time: 10.240856 / 119.766654 seconds
- V1 maximum trie support DP states: 76,267

Root cause: the audit fixture assumed `Fe-Pm` was an all-metal shortcut.
The frozen Direct evaluator excludes `Pm` from its metal set and therefore
classifies `Fe-Pm` as `oxidation_state_missing`. The CR-Plan V2 certificate
already matched Direct and correctly failed it closed; only the audit's expected
fixture label was wrong.

Repair: retain actual all-metal fixtures (`Fe-Cu`, `Na-Fr`, `Ba-Ra`) and add an
explicit precedence fixture requiring `Fe-Pm` to match Direct's
`oxidation_state_missing` outcome.

This is an audit-only engineering repair. It does not alter model weights,
tokenizer, prompt, seed roles, atom budget, oxidation states, endpoint policy,
sampling support, thresholds, or any scientific denominator.
