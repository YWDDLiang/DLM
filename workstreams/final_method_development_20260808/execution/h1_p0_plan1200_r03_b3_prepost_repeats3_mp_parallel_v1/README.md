# H1 Plan1200 MP cache parallel continuation

This source package continues the frozen V3 Materials Project cache completion
after an operator-approved interruption of its serial prefix.

The contract is intentionally narrow:

- validate and reuse only the contiguous, fully resolved serial prefix;
- query only the remaining frozen chemical systems;
- use six worker threads with a shared ten-request-per-second limiter;
- honor `Retry-After` and retain the frozen five-attempt transport ceiling;
- consume the credential from a private one-time file and unlink it before any
  request is dispatched;
- reconstruct the same sorted 2,464-row planner-union cache expected by the V3
  body preflight;
- record the serial checkpoint, this continuation source, worker/rate settings,
  and every per-chemsys transport outcome in the completion manifest;
- never run in Slurm and never perform sample retry, replacement, repair,
  filtering, reranking, training, promotion, or RL.

`install_parallel_source_once.sh` installs this package once beneath the frozen
run root. `parallel_complete.py` is then invoked once on the A800 login node.
