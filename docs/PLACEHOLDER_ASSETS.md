# Placeholder asset ledger

| Asset | Relative destination | Current state | Used by |
|---|---|---|---|
| A800 environment export | `environment/` | live environment verified; export pending | all stages |
| Planner checkpoint | `checkpoints/planner/` | verified on A800; transfer pending | Plan sampling |
| DLM checkpoint | `checkpoints/dlm/` | verified on A800; transfer pending | body generation |
| Diffusion checkpoint | `checkpoints/diffusion/model_494.pt` | verified on A800; transfer pending | refinement |
| CHGNet checkpoint | `checkpoints/chgnet/chgnet_0.3.0_e29f68s314m37.pth.tar` | verified in A800 environment; transfer pending | final evaluation |
| MP-20 train/val/test | `data/mp20/` | rows/files verified; transfer pending | training/evaluation |
| Historical H1-A2 parsed Plans | `data/plans/h1a2_parsed_1186.jsonl` | source file verified; transfer pending | full-route Planner fallback |
| Raw frozen Plans | `data/plans/r03_raw_256.jsonl` | source512 verified; first256 extraction pending | audit/resampling |
| Parsed frozen Plans | `data/plans/r03_parsed_256.jsonl` | source512 verified; first256 extraction pending | quick reproduction |
| Scientific seed ledger | `data/plans/r03_seed_ledger_256.jsonl` | verified 256-row ledger; transfer pending | paired body/refiner noise |

This public ledger intentionally contains no private cluster path. Download
locations will be added when release assets are uploaded.
