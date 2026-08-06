# H1 CR-Plan exact-support preflight V4 runtime repair record

Status: `runtime_only_repair_not_a_scientific_change`

V4 replaces only the Python interpreter selected by the remote preflight
launcher. The scientific implementation, frozen tokenizer/table contract,
audit fixtures, test suite, and audit script are byte-identical to V3.

## Predecessor evidence

- V3 source manifest SHA-256:
  `9409b9ee8a45ff15448e495f249544f9547e54c36aeffc8c946e651d799e100d`
- V3 source archive SHA-256:
  `a425fa47892b08f6a5bb84c775c66d1bc85567a63557c17388d9cc2873425f75`
- sealed V3 failure-report SHA-256:
  `2629c49278fd83c6dc39f2df74446e5939554acc3542ab69fbca8ea90f1cf148`
- V3 failed before exact audit import because the launcher selected base
  Python 3.9, where `dataclass(slots=True)` is unsupported.
- V3 loaded no model, used no GPU or network, generated no sample, and
  triggered no downstream action.

## Bounded runtime probe

The sole runtime probe used:

`/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python`

It returned Python `3.10.18`, successfully constructed a slots dataclass, and
imported Transformers `4.54.0`. No model was loaded. CUDA was disabled and the
framework-selection variables disabled Torch, TensorFlow, Flax, and XLA.

## V4 execution contract

- Use the exact interpreter path above for both isolated unit tests and the
  CPU-only exact audit.
- Set `CUDA_VISIBLE_DEVICES` empty and run Transformers in local/offline mode.
- Set `USE_TORCH=0`, `USE_TF=0`, `USE_FLAX=0`, and `USE_TORCH_XLA=0`.
- Do not use Slurm, a GPU, network access, model weights, generation, retry,
  replacement, repair, filtering, reranking, or downstream actions.
- Refuse to overwrite the V4 source, run root, report, terminal marker, or
  success marker.
- A clean V4 report authorizes only the separately frozen same-node
  performance probe. It does not authorize four-arm 512.

Any V4 audit mismatch is an engineering terminal for this CR-Plan route.
