#!/usr/bin/env python3
"""Build the frozen MP20 exact-SMACT4 witness ledger on the local machine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.h1_local_smact4_ledger import (  # noqa: E402
    build_witness_ledger_payload,
    write_witness_bundle,
)
from crystal_dlm.h1_nocharge_ion_aux import (  # noqa: E402
    SMACT4_VERSION,
    load_smact4_icsd24_oxidation_map,
)
from scripts.build_h1_nocharge_ion_aux_sft_data import (  # noqa: E402
    attach_smact4_witnesses,
    load_legacy_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-snapshot-dir", type=Path, required=True)
    parser.add_argument("--source-inventory-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_rows, legacy_report = load_legacy_snapshot(
        args.legacy_snapshot_dir,
        splits=("train", "val"),
        require_frozen=True,
    )
    oxidation_map, contract = load_smact4_icsd24_oxidation_map()
    if contract.get("smact_version") != SMACT4_VERSION:
        raise SystemExit(
            f"exact local SMACT {SMACT4_VERSION} required, found "
            f"{contract.get('smact_version')!r}"
        )
    reports = {
        split: attach_smact4_witnesses(source_rows[split], oxidation_map)
        for split in ("train", "val")
    }
    payload = build_witness_ledger_payload(
        source_rows,
        source_inventory_sha256=args.source_inventory_sha256,
        legacy_report=legacy_report,
        smact4_contract=contract,
        witness_reports=reports,
    )
    manifest = write_witness_bundle(args.output_dir, payload)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
