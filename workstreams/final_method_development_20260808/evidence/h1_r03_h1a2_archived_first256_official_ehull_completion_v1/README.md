# Official E_hull completion evidence

This directory is the SHA-verified return of the stability-only completion for
the archived H1-A2 D1 versus R03 D2 safe-axis first256 pair. Generation,
model-494 refine800, CHGNet energies, novelty, and uniqueness were frozen.

Primary evidence is `terminal_report.json` and `RESULTS_COMPLETE.md`. The
query used official `MPRester.get_entries_in_chemsys()`,
`compatible_only=True`, and `GGA_GGA+U`. One of ten queried systems resolved;
nine Yb systems remain explicit hull-unknown.

No credential value is stored in this directory.
