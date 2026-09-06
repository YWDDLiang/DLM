"""Read-only source inventory; does not infer scientific outcomes or delete files."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--since", default="2026-08-26")
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output_dir.resolve()
    if not out.is_relative_to(root) or out == root:
        raise ValueError("Inventory outputs must stay inside the explicit source tree")
    if Path(git(root, "rev-parse", "--show-toplevel").strip()).resolve() != root:
        raise ValueError("Root must be the exact Git worktree")
    files = git(root, "ls-files", "-z").split("\0")
    documents, failure_signals, hash_groups = [], [], defaultdict(list)
    for relative in files:
        if not relative or not relative.endswith(".md"):
            continue
        path = root / relative
        if not path.is_file() or path.is_relative_to(out.parent):
            continue
        raw = path.read_bytes()
        content = raw.decode("utf-8-sig", errors="replace")
        lines = content.splitlines()
        digest = hashlib.sha256(raw).hexdigest()
        hash_groups[digest].append(relative)
        headings = [{"line": i, "text": line} for i, line in enumerate(lines, 1)
                    if re.match(r"^#{1,3} ", line)]
        status = [{"line": i, "text": line[:400]} for i, line in enumerate(lines[:55], 1)
                  if re.search(r"current|active|status|supersed|当前|状态|过时|历史|暂停", line, re.I)]
        links = []
        for m in re.finditer(r"\]\(([^\n)]+)\)", content):
            target = m.group(1).strip().strip("<>")
            if re.match(r"^[a-zA-Z]+://|^#|^mailto:", target):
                continue
            target_no_anchor = target.split("#", 1)[0]
            if re.match(r"^[A-Za-z]:[/\\]", target_no_anchor):
                resolved = Path(re.sub(r":\d+$", "", target_no_anchor)).resolve()
            else:
                resolved = (path.parent / target_no_anchor).resolve()
            links.append({"target": target, "exists": resolved.exists(),
                          "inside_worktree": resolved.is_relative_to(root)})
        documents.append({"path": relative, "bytes": len(raw), "sha256": digest,
                          "headings": headings, "early_status_signals": status, "local_links": links})
        for i, line in enumerate(lines, 1):
            if re.search(r"\b(?:FAIL(?:ED|URE)?|negative|regress(?:ion|ed)?|rejected|undertrained)\b|失败|退化|无提升|不达标|未达|假提升|未复现", line, re.I):
                failure_signals.append({"path": relative, "line": i, "text": line[:650],
                                       "interpretation": "discovery_signal_not_adjudicated_failure"})
    commits = []
    for row in git(root, "log", "--first-parent", f"--since={args.since}",
                   "--format=%H%x09%aI%x09%s").splitlines():
        sha, date, subject = row.split("\t", 2)
        changed = git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", sha).splitlines()
        commits.append({"sha": sha, "authored_at": date, "subject": subject, "changed_paths": changed})
    duplicates = [{"sha256": sha, "paths": paths} for sha, paths in hash_groups.items() if len(paths) > 1]
    out.mkdir(parents=True, exist_ok=True)
    write = lambda name, value: (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "worktree": str(root),
                "head": git(root, "rev-parse", "HEAD").strip(), "since": args.since,
                "status": "inventory_only_manual_adjudication_required", "tracked_markdown_count": len(documents),
                "recent_first_parent_commits": len(commits), "duplicate_groups": len(duplicates),
                "failure_text_signals": len(failure_signals), "all_failures_adjudicated": False,
                "document_prefix_counts": dict(Counter(item["path"].split("/")[0] for item in documents))}
    write("DOCUMENT_INVENTORY.json", {"metadata": metadata, "documents": documents, "exact_duplicates": duplicates})
    write("RECENT_CHANGE_INVENTORY.json", {"metadata": metadata, "commits": commits})
    write("FAILURE_DISCOVERY_INDEX.json", {"metadata": metadata, "signals": failure_signals})
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
