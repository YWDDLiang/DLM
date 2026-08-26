# Counterfactual-grounding DLM checkpoint sweep

This reports every frozen checkpoint arm; it does not select only the best checkpoint.
Steps 500/1000/1696 correspond to approximately 0.295/0.590/1.000 training epoch.

| Step | Arm | Body | Direct J | N∩U | Hull K/U | Strict | Meta | Strict stable→S.U.N. | Meta stable→S.U.N. |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | control | 250/256 | 213 | 230 | 243/7 | 25 | 127 | 83.33% | 86.39% |
| 500 | candidate | 250/256 | 215 | 228 | 242/8 | 28 | 126 | 87.50% | 85.14% |
| 1000 | control | 251/256 | 215 | 228 | 242/9 | 18 | 121 | 78.26% | 84.03% |
| 1000 | candidate | 252/256 | 216 | 232 | 244/8 | 21 | 125 | 84.00% | 86.21% |
| 1696 | control | 255/256 | 218 | 234 | 246/9 | 26 | 127 | 83.87% | 85.81% |
| 1696 | candidate | 255/256 | 218 | 232 | 246/9 | 21 | 125 | 80.77% | 84.46% |

## Paired comparisons

- step 500: ΔStrict `+1.17%`, ΔMeta `-0.39%`, ΔDirect-J `+0.78%`, ΔStrict stable→S.U.N. retention `+4.17%`; screen pass `False`.
- step 1000: ΔStrict `+1.17%`, ΔMeta `+1.56%`, ΔDirect-J `+0.39%`, ΔStrict stable→S.U.N. retention `+5.74%`; screen pass `False`.
- step 1696: ΔStrict `-1.95%`, ΔMeta `-0.78%`, ΔDirect-J `+0.00%`, ΔStrict stable→S.U.N. retention `-3.10%`; screen pass `False`.

Qualifying steps: `[]`.

The public 105/1000 Strict and 488/1000 Meta headline remains unchanged.
