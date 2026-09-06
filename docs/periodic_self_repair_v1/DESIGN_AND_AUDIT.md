# Periodic interactions and numerical learning inside the retained DLM

Registered 2026-09-06, updated after the user's clarification that the DLM architecture is the priority. Budget is two A800 / eight CPU within the already allocated 06:49–18:49 window. The user subsequently authorized using currently free cards for immediate engineering work and a multi-agent mathematical audit.

**07:58 execution amendment:** the user requested immediate two-GPU launch. The completed K4 data and model now initialize scientific repair round0 immediately. Round1 continues that repair model using the complete K4+K8 target pool; its frozen conditional-policy reference is the completed original K8 model. Both stages use four declared passes with a fresh stage optimizer/schedule. This changes execution order, without changing the family-balanced objective, target admission, legal support, or original K8 training recipe.

## What the paper's core retains

The C³FD / typed Llama Planner, composition support, species program, exact atom counts and species, and native 7+4N canvas remain the chemical and program interface. The legal-set construction and its conditional sampling semantics remain the validity mechanism. The structure is still sampled from the same bidirectional DLM's discrete token probabilities.

Other implementation details may be adapted or removed when incompatible. In this first implementation, the previous construction, cooperative region, and closure remain usable; an additional full-cell repair is appended. This is a concrete initial implementation, not a claim that every old phase must be retained forever. No continuous coordinate generator is introduced. Frozen model494 results remain separate.

The existing learned LoRA and periodic state conditioner initially come from completed original K4 policy39892. The original embedding/output tables remain frozen. The new numeric and attention increments are zero-initialized in repair round0. Round1 preserves the trained repair modules and LoRA, while updating the frozen policy reference to final K8. The separate three-step engineering check discarded its temporary weights and cannot become a scientific checkpoint.

## Internal model

For a current integer canvas x and explicit old integer geometry x_old, the input still contains the old conditioner residual. It additionally receives task features:

1. The fraction of all native numeric positions currently masked.
2. Whether the old geometry is complete.
3. Whether the numeric-noise level is known.
4. Its supplied level when known.
5. A one-hot task: construction, cooperative revision, closure, full-cell repair, or structured denoising.

Unknown actual errors have noise sentinel -1, including already sampled construction prefixes. Visibility is not a cleanliness label. Only the controlled structured-corruption task supplies a known noise level. No target distance, future relaxed energy, or target force is used as model input.

Known old atom pairs supply finite-shell minimum-image distances, radial features, periodic displacement sin/cos, element embeddings, old lattice Gram features, and active-state flags. A zero-output MLP maps these to a shared additive attention bias:

    softmax(Q K^T / sqrt(d) + B_geom) V.

The bias is an explicit forward argument passed through the existing LLaDA Transformer layers and native activation checkpointing. There is no mutable attention hook. Unknown pairs produce zero bias; initial all-mask geometry does not produce guessed coordinates. The existing input conditioner is retained.

Pair distances use the declared centered 125-image shell, consistent with the inherited support machinery. This is not exact closest-image enumeration for every arbitrary unreduced lattice. Pair-only periodic translation/wrapping checks do not imply strict equivariance of the complete Transformer; absolute old-coordinate features, program order and token positions remain.

For each numeric family/axis, a trainable hidden projection is multiplied by a fixed numeric basis and added only to that family's output rows:

    logits_j = base_logits_j + W_family h_j · phi_family(value).

Fractional coordinates use eight Fourier modes and identical features at 0 and 1. Lengths use log-length and RBF features; angles use an ordered, nonperiodic RBF basis. The final projection begins at zero. All numeric geometry constants and new modules preserve FP32; increments are cast at the interface to the base model.

## Legal-set argument and reachable supervision

Let G be the declared finite-support set: exact N/species, legal numeric vocabulary, lattice angular determinant factor greater than the frozen min_lattice_rad, and the inherited bounded-image distance condition.

Starting from an accepted x_old in G, the repair transaction stages all six lattice values and all XYZ positions in the retained scalar conditional order. At scalar j it samples from the actual masked legal distribution

    pi_theta(a | s_j) = exp(ell_theta(a)/T) / sum_{v in S(s_j)} exp(ell_theta(v)/T).

Coordinate aliases are merged before normalization. Empty support restores the whole transaction. A completed proposal is accepted only if it is in G; otherwise the complete old state is restored. Thus an accepted input remains in G by induction over transactions. This proves preservation of the **declared support**, not low energy or universal crystal validity.

The full-cell action covers every lattice/coordinate difference between x and its target, while protected composition slots never move. Teacher forcing exposes the target prefix in the same conditional order and leaves all remaining numeric positions masked; the complete erroneous parent remains in old_body. Independent sampling of all coordinate marginals is not substituted for this joint factorization.

## Physical-feedback target provenance

Sources are the complete original train K4 and K8 requests, with failures retained. The saved R(x) is the same verified, fixed CHGNet/FIRE/FrechetCellFilter terminal used by the original pipeline. Sites are never deliberately reordered. Species order is checked, but same-species atom identity additionally relies on the frozen optimizer preserving site order.

Quantization Q is applied to that terminal with the existing vocabulary. Clipped lengths/angles, changed composition, missing identity, and unsupported target geometries are recorded as unavailable. Continuous terminal verification is not copied to Q(R(x)).

Every encodable target receives fresh labels using the same R protocol. The registered target admission is:

- The fresh R(Q(R(x))) passes the common terminal verification.
- Its quantized raw-to-relaxed gap is at most 0.05 eV/atom.
- Quantized raw maximum force is at most 1 eV/Å and stress at most 5 GPa.
- Quantized raw energy is no more than parent raw energy plus 0.001 eV/atom.

These are training-target admission limits, not revised SUN thresholds and not a certification that the raw quantized target is already a minimum. Body SHA256, occurrence identity, per-condition K4/K8 counts, frozen physics parameters, and package versions are checked. All rejects remain in ledgers.

Since R(Q(R(x))) can enter a different basin, admission does not prove A(target) < A(parent) or B(target) <= B(parent). The data report separately records actual target delta A, delta B, native-energy changes and adverse-change counts. No theorem about the teacher is presented as a student-performance guarantee.

## Training objective

Formal phase A uses four passes over the admitted pairs, with three sampled scalar states per pair per pass: one length, one angle, and one coordinate. For M pairs across C admitted reduced compositions, a pair from composition c with n_c occurrences has weight M/(C n_c). After the explicit padding correction, this estimates

    (1/C) sum_c (1/n_c) sum_{i in c} (1/3) sum_f
         (1/|J_if|) sum_{j in J_if} loss_ij.

It is a family-balanced conditional reconstruction risk, **not** the unweighted full-path likelihood. Each scalar loss is 90% exact-target CE plus 10% narrow numerical-target CE. The numerical Gaussian uses periodic distance only for fractional coordinates, is restricted and renormalized on the actual legal support, and narrows from 0.75 to 0.25 bins across the four passes.

Three quarters of the scheduled repair examples use actual ancestor-model errors. One quarter uses declared strain/displacement corruption around the admitted target:

    L_t = L_* exp(S), S = S^T;
    U_t = wrap(U_* + Xi L_t^(-1)).

Registered (strain operator bound, Cartesian component displacement bound) levels are (.01,.05 Å), (.03,.15 Å), and (.06,.30 Å). A single infeasible corruption falls back to that occurrence's actual parent error with an explicit counter; it is not repeatedly resampled until an attractive example appears.

On actual recorded ancestor-path states, the frozen original K4 policy in round0 and original final K8 policy in round1 supply

    0.01 * KL(pi_reference(.|s) || pi_new(.|s)).

Both policies use the same actual legal support, alias transform, and T=0.7. The original raw logits dtype is retained through alias merging; normalization for training is FP32. The term regularizes observed conditional policies, not the whole new trajectory distribution, and it does not cover every new repair state.

The original MP20 revision CE anchors remain at one update per four repair updates. No new MP20 construction supervision is introduced. Optimizer settings for this separate experiment are global batch 16 on two ranks, LoRA/conditioner LR 1e-5, new modules LR 1e-4, 50 warmup updates, cosine decay. The final registered endpoint is evaluated; intermediate development scores do not select a checkpoint.

Round0 is an explicitly identified intermediate repair model; its diagnostic scores cannot substitute for round1. The original mainline K8 optimizer still follows its unchanged continuation protocol.

## Evaluation and limits

The fixed hash split is by reduced composition before outcomes, with 10% held out from the **new repair supervision**. The K8 base and original MP20 anchors may already have seen these compositions; this is not a claim of completely unseen-composition generalization.

Unfiltered validation parent paths are exported separately from admitted validation targets. Fixed-start repair evaluation retains failed, unverified and unchanged outcomes and reports native energy, force/stress distributions and tails, generation validity, and paired A/B only where both endpoints are verified. Admitted-target CE is an auxiliary diagnostic, never the repair-effect denominator.

This implementation teaches old-error-conditioned joint masked reconstruction. It does not yet demonstrate a general all-visible token-to-token editor. The actual native sampler still uses staged conditional draws. Native and fixed model494 tau800 evaluation remain separate. Later basin generation is a separately registered phase and has not yet been implemented or run.

## Independent audits and corrections

Three explicitly user-authorized read-only agents reviewed mathematics/objectives, architecture/runtime, and data/protocols. They found and the root fixed:

- FP32 geometry values being rounded by parent BF16 conversions.
- Missing min_lattice_rad in full-state target admission.
- Construction prefixes being incorrectly described as known-zero numeric error.
- BF16 alias processing differing between training and sampling because training cast too early.
- Padding scaling the repair/anchor balance; padded/real correction now restores the objective.
- Repeated tokenizer vocabulary creation inside structured-corruption loops.
- Insufficient body hashes, per-condition occurrence checks, and physics-version checks.

36 relevant local CPU tests passed after these corrections. Separate CPU objective probes verified BF16 and FP32 legal vectors and gradients exactly against the full deployment transform, same-policy KL=0, positive perturbed KL, alias gradients, and <=1.18e-7 CE difference from the sampler's FP64 log recording. A real two-GPU test will check actual LLaDA initialization, gradients, memory, full-cell sampling and path replay before scientific training.

Job39930 completed the real two-GPU check: both ranks had zero initial logit difference and zero full-cell replay difference, with approximately34.73GiB peak memory per GPU. All new modules and the retained conditioner received finite gradients. Its full K4 target preparation retained4096 requests,972 encodable verified-parent targets and3124 unavailable targets; fresh quantized-target labels were695 verified and277 not converged. These695 are before the separately registered raw-target admission filters.

The local mathematical bounds remain conditional: near a verified local minimum with bounded Hessian, smaller geometric error bounds local excess energy. This does not cover arbitrary severe collisions or imply that training reaches that neighborhood. A subsequent energy-tilted finite-basin teacher can increase stable-set teacher mass on fixed support, but student fitting, novelty, uniqueness and SUN remain empirical questions.

## Primary sources

- [MaskGXT / HACO](https://arxiv.org/abs/2606.22866), [official code](https://github.com/kiyoung98/MaskGXT): periodic and ordered numerical mechanisms.
- [Crystalite](https://arxiv.org/abs/2604.02270): geometric attention bias.
- [Self-Generated Error Training](https://arxiv.org/abs/2606.17175): own-model errors for editing; the crystal/physical-feedback transfer is this experiment's hypothesis.
