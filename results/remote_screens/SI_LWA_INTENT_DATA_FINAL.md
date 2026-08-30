# SI-LWA-v1 structural-intent data terminal

Job `38240` completed in `00:01:16` with 48 CPUs and zero GPUs.

## Coverage and entropy

| Split | Rows | Valid | Coverage | VPA entropy (nats) | CN entropy (nats) |
|---|---:|---:|---:|---:|---:|
| train | 27,136 | 27,136 | 100% | 2.07944 | 1.86517 |
| validation | 9,047 | 9,047 | 100% | 2.07893 | 1.87346 |

No CIF or CrystalNN failure occurred. Train VPA class counts are exactly
balanced at 3,392 each by construction. Validation VPA counts remain close to
balanced. CN classes are non-collapsed but imbalanced; the largest train class
contains 7,401 rows and the smallest 1,184.

The data terminal establishes label coverage and diversity only. It does not
establish composition-only predictability or an effect on raw/refined energy.
VPA/CN remain audit candidates rather than authorized GPU conditions until the
intent-sufficiency audit is complete.

## Frozen identities

- manifest SHA: `a5872e1ffa3455c41f0179732dfddec92fc1f5d6ed31433c948ea62f2551b13c`
- train JSONL SHA: `819f0a9f3d4c7ee9ef44f94b247ef2f17ed016968abf9fb3f82d1d553b694975`
- validation JSONL SHA: `644ab2a1405e01ac9df597ed52b0b3f2e39ba4775cf33dd25795a60028426290`
- source MP20 train/validation SHAs remain `9b8031...` and `ae7b87...`.
- output: `$ROOT/data/si_lwa_intent_v1_20260830`

VPA train quantile edges are:

```text
1.7849504468, 2.4812110999, 2.6260504451, 2.7609274928,
2.9023934231, 3.0419029065, 3.1905731731, 3.3789909895,
4.8996962792
```

All eight CN representatives are distinct observed train histograms. The
manifest records their material IDs, source rows, histograms, fit seed 82000,
class counts, failures, source/code/output hashes, and zero test-outcome/GPU
usage.

