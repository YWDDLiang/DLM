# Source freeze repair log

The first local source freeze on 2026-08-06 was not submitted or transferred.
Its own inventory passed, but the isolated test command stopped after 65 tests
because `tests/test_h1_crplan.py` imports the historical read-only
`evaluate_paired32.py` helper and that test-only dependency was absent from the
archive.

Source repair V2 adds exactly that historical evaluator helper to the source
selection.  No Planner, data, training, evaluator, threshold, seed, model,
tokenizer, ledger, or downstream contract changed.  The failed local V1
freeze remains preserved under its original local source-bundle directory.
