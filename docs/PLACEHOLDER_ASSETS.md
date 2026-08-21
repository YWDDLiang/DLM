# Placeholder asset ledger

| Asset | Relative destination | Current state | Used by |
|---|---|---|---|
| A800 environment export | `environment/` | pending A800 audit | all stages |
| Planner checkpoint | `checkpoints/planner/` | pending publication | Plan sampling |
| DLM checkpoint | `checkpoints/dlm/` | pending publication | body generation |
| Diffusion checkpoint | `checkpoints/diffusion/model_494.pt` | pending publication | refinement |
| MP-20 train/val/test | `data/mp20/` | pending A800 copy | training/evaluation |
| Historical H1-A2 parsed Plans | `data/plans/h1a2_parsed_1186.jsonl` | pending release copy | full-route Planner fallback |
| Raw frozen Plans | `data/plans/r03_raw_256.jsonl` | pending A800 copy | audit/resampling |
| Parsed frozen Plans | `data/plans/r03_parsed_256.jsonl` | pending A800 copy | quick reproduction |
| Scientific seed ledger | `data/plans/r03_seed_ledger_256.jsonl` | pending A800 copy | paired body/refiner noise |

This public ledger intentionally contains no private cluster path. Download
locations will be added when release assets are uploaded.
