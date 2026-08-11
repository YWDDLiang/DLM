# Frozen R03 refined256 current-S.U.N. replay, environment-repair V2

This is an evaluation-only, fail-closed replay of the four historical R03 refined256 candidate process realizations.

It freezes and verifies every historical generation JSONL and every per-repeat CHGNet relaxation cache. It does not run the planner, R03 body model, diffusion refiner, CHGNet, training, RL, retry, replacement, repair, filtering, or reranking.

The login-node preparation merges the registered MP caches from oldest to newest, then queries only the 92 chemical systems still missing from the 224-system historical cohort. The one-time credential carrier is read and destroyed before querying. The four Slurm jobs have no MP credential and use the completed cache offline.

The active seven-line h1_rich_plan_v1 prompt branch is byte-identical between H1A2 and the current Plan1200 study. The relevant sampling change is the RNG/cohort design, not prompt text: H1A2 used one global RNG stream at seed 17029; Plan1200 used stateless ordinal sampling with three independent base seeds and three disjoint first-1000 parse-success cohorts.

The final report compares the historical and current-cache strict S.U.N. and meta-S.U.N. attempt-by-attempt. Headline denominators are the 248 reconstructed structures per repeat; all-256 rates are secondary, and evaluated-only rates are diagnostic.

V1 is preserved as a failed-closed provenance run. It exited before any HTTP query and before any Slurm submission because the base login environment lacked pymatgen package metadata; its one-time key was destroyed. V2 changes only the login-node interpreter selection by activating the already registered diff_meets_diff environment.
