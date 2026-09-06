# SUN snapshot

Strict / Meta SUN; development rows each retain all 256 requests.

| Method | Raw | Tau800 | Both endpoints complete |
|---|---:|---:|---|
| Reference | 6/55 (2.34%/21.48%) | 19/121 (7.42%/47.27%) | True |
| K4 | 7/57 (2.73%/22.27%) | 19/126 (7.42%/49.22%) | True |
| K8 | 11/66 (4.30%/25.78%) | 17/123 (6.64%/48.05%) | True |
| V2 repair | 8/50 (3.12%/19.53%) | 16/113 (6.25%/44.14%) | True |
| V3 G | 9/61 (3.52%/23.83%) | 13/116 (5.08%/45.31%) | True |
| V2 construction | 12/44 (4.69%/17.19%) | 17/120 (6.64%/46.88%) | True |
| V3 T construction | 13/53 (5.08%/20.70%) | 17/111 (6.64%/43.36%) | True |
| K4 short-contact v1 (eval_40074) | 6/56 (2.34%/21.88%) | 20/122 (7.81%/47.66%) | True |
| K8 short-contact v1 (eval_40074) | 11/68 (4.30%/26.56%) | 15/123 (5.86%/48.05%) | True |
| K4/K8 equal probability mixture (eval_40076) | pending | pending | False |

Original K8 independent cohorts are separate from development:

| Requests | Raw | Tau800 |
|---:|---:|---:|
| 1000 | 39/244 (3.90%/24.40%) | 65/484 (6.50%/48.40%) |
| 1200 | 47/285 (3.92%/23.75%) | 79/570 (6.58%/47.50%) |

Source paths, original report hashes, verified counts, incomplete endpoints and target checks are in SUN_SNAPSHOT.json.
No metric from one method is combined with another method. No evaluation file or selection is changed.
