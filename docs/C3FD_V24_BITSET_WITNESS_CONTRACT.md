# C³FD-v2.4 Bitset-Witness Contract

Date: 2026-08-28

Status: frozen before v2.4 outcomes. Contribution point 1 and public
`105/488` remain unchanged.

## Prior stop

C³FD-v2.3 represented the right constructive charge/Pauling certificate but
cached individual charge states. Its frozen-data CPU audit exceeded the
preregistered 120-second limit and reached about 1.27 GB RSS; it was stopped
without GPU work or relaxed gates.

## Single v2.4 change

Keep the exact same constructive witness and legal action set, but represent
all reachable suffix charges as a bounded integer bitset. Adding a
species/count shifts the bitset; alternative suffixes combine by bitwise OR.
Family, exact arity, branch, required anion, all-metal/unary shortcuts, and
the strict Pauling boundary remain explicit cache dimensions.

No model, training data, loss, calibration, temperature, top-p, seed, or
scientific condition changes. There is still no BPE, repair, replacement,
reranking, RL, or outcome label.

## Gates

### CPU audit

- bitset and recursive constructive oracle agree on frozen synthetic cases;
- 100% train/validation teacher trajectories carry a constructive witness;
- all supported `(family,N,arity)` strata are reachable;
- one full-mask deterministic trajectory per stratum has zero dead ends and
  passes the independent benchmark at EOS;
- total runtime <=60 seconds and model weights/outcome labels are unused.

### requested-256 pilot

Only after the CPU gate passes, retrain the unchanged v2.1 head and set
`reachability_mode=pauling_bitset`. Require zero semantic dead ends, parse
noninferior within 1 pp, pooled/per-seed comp-valid gains with positive paired
CI lower bound, ionic gain, Novel × Unique noninferior within 1 pp, all-metal
within 3 pp of full train, and family/N/arity TVD no worse than P0 +0.01.

### requested-1000 confirmation

Run only after every requested-256 gate passes; report both seeds and pooled.

## Stop condition

No GPU is authorized if the 60-second CPU gate or any witness/certificate
invariant fails. Gates are not relaxed after observation.
