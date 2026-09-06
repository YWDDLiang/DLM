"""Extract saved fields for local CPU auditing; no geometry or score is recomputed."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


LABEL_FIELDS = ("trajectory_id", "group_id", "status", "verified", "optimizer_converged", "actual_steps",
                "raw_energy", "terminal_energy", "gap", "terminal", "terminal_consistency")


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def geometry(structure):
    if not isinstance(structure, dict):
        return None
    return {"lattice": {"matrix": structure["lattice"]["matrix"]},
            "sites": [{"abc": site["abc"]} for site in structure["sites"]]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--paths", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    paths = {r["trajectory_id"]: {"trajectory_id": r["trajectory_id"], "structure": geometry(r.get("structure"))}
             for r in rows(args.paths)}
    count = 0
    ids = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for r in rows(args.labels):
            identity = r["trajectory_id"]
            if identity in ids:
                raise ValueError("Duplicate source label identity")
            ids.add(identity)
            label = {k: r.get(k) for k in LABEL_FIELDS}
            label["final_structure"] = geometry(r.get("final_structure"))
            stream.write(json.dumps({"label": label, "path": paths[identity]}, separators=(",", ":")) + "\n")
            count += 1
    if ids != set(paths):
        raise ValueError("Source label/path identity sets differ")
    manifest = {"schema": "saved_physics_field_export_v1", "records": count, "new_samples": 0,
                "scores_recomputed": False, "coordinates_recomputed": False,
                "omissions": "Fields not read by audit_saved_physics_lane.py, including chemical identity and event traces, are omitted.",
                "sources": {str(x): digest(x) for x in (args.labels, args.paths)},
                "output": str(args.output), "output_sha256": digest(args.output), "bytes": args.output.stat().st_size}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
