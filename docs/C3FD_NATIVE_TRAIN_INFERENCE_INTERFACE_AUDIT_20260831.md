# C3FD-native train/inference interface audit

The frozen teacher SFT data and the inference-only dual-C3FD predictions were
rendered through the same shared `build_native_body_prompt` path.

Across all `27,136` train and `9,047` validation compositions, for both C3FD
checkpoints (`seed17`, `seed18`):

- teacher prompt byte-replay mismatch: `0`;
- missing predicted Plan rows: `0`;
- instruction-prefix or body-suffix mismatch: `0`;
- JSON key-set mismatch: `0`;
- hard-field mismatch (`schema/N/elements/counts/family`): `0`;
- changes outside `lattice_system/spacegroup_bucket/volume_per_atom_bin`: `0`.

Renderer SHA-256:
`566830481cbfb990b2dd63d15d4ec0cf8db815759744b16a46762c5bfc8f219f`.

Audit report SHA-256:
`7ee6b85d04246151ee9a2920501aaa83deba54e523d072b094c0a95efacf9f95`.

The shared renderer is the default implementation and produced exact byte
parity in this audit. It is evidence, not a renderer-SHA hard gate: an equivalent
generation renderer is acceptable if, after the same runtime whitespace
normalization, it preserves the instruction semantics, Plan field set, hard
composition, and body label, with only LS/SG/VP values changing.
