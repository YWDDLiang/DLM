"""Update only parent-authorized documentation references after verified moves."""
from __future__ import annotations

import collections
import datetime as dt
import json
import os
from pathlib import Path
import re

from build_document_cleanup_plan import AUDIT, ROOT, resolve_target, sha
from reference_syntax import inside_code, markdown_protected_ranges

PLAN = ROOT / AUDIT / "DOCUMENT_CLEANUP_PLAN.json"
ROOT_EDIT = {"WORKFLOW.md", "PAPER_PIPELINE.md", "REPRODUCTION.md"}
LEGACY_EDIT = {"docs/paper/README.md", "docs/paper/METHOD_AT_A_GLANCE.md", "docs/PLACEHOLDER_ASSETS.md", "docs/SEEDS.md"}
PARENT_ROOT = {"README.md", "EXPERIMENT.md", "docs/CURRENT_STATE.md", "docs/CURRENT_ARCHITECTURE.md"}
MD_LINK = re.compile(r"\]\((<[^>]+>|[^\n)]*)\)")
MD_REFERENCE = re.compile(r"^\s*\[[^\]]+\]:\s*(<[^>]+>|\S+)", re.M)
DOC_LITERAL = re.compile(r"(?<![A-Za-z0-9_/\\])docs[/\\][A-Za-z0-9_./\\-]+\.(?:md|json|py|mmd)(?::\d+)?")


def records(text: str):
    """Yield exact reference spans. Text outside these spans is never modified."""
    found = []
    code_ranges = markdown_protected_ranges(text)
    for pattern, kind in ((MD_LINK, "markdown_link"), (MD_REFERENCE, "markdown_reference")):
        for match in pattern.finditer(text):
            captured = match.group(1)
            part = re.match(r"\s*(<[^>]+>|[^\s]+)", captured)
            if not part:
                continue
            start = match.start(1) + part.start(1)
            end = match.start(1) + part.end(1)
            if inside_code(start, end, code_ranges):
                continue
            raw = text[start:end]
            if raw.startswith("<") and raw.endswith(">"):
                start, end, raw = start + 1, end - 1, raw[1:-1]
            found.append({"start": start, "end": end, "target": raw, "kind": kind,
                          "line": text.count("\n", 0, start) + 1})
    for match in DOC_LITERAL.finditer(text):
        if any(match.start() < old["end"] and match.end() > old["start"] for old in found):
            continue
        found.append({"start": match.start(), "end": match.end(), "target": match.group(0),
                      "kind": "repository_path_literal", "line": text.count("\n", 0, match.start()) + 1})
    return sorted(found, key=lambda x: x["start"])


def target_after(source_old: str, source_new: str, ref: dict, mapping: dict) -> tuple[str, dict]:
    info = resolve_target(source_old, ref["target"], repository_relative=ref["kind"] == "repository_path_literal")
    if info["scope"] in {"external_or_app", "same_document_anchor"}:
        return ref["target"], info
    old_target = info["resolved_path"]
    if old_target not in mapping and not info.get("exists", False):
        # Unknown targets, including formula-like strings, are retained verbatim.
        return ref["target"], {**info, "unknown_target_preserved": True}
    if info["scope"] == "workspace":
        new_target = mapping.get(old_target, old_target)
        absolute = ROOT / new_target
    else:
        new_target = old_target
        absolute = Path(old_target)
    if ref["kind"] == "repository_path_literal":
        base = new_target
    elif re.match(r"^[A-Za-z]:[/\\]", ref["target"]):
        base = absolute.as_posix()
    elif ref["target"].startswith("/") and info["scope"] == "outside_workspace":
        return ref["target"], info
    else:
        try:
            base = os.path.relpath(absolute, (ROOT / source_new).parent).replace("\\", "/")
        except ValueError:
            base = absolute.as_posix()
    base += info.get("line_suffix", "")
    if info.get("anchor"):
        base += "#" + info["anchor"]
    return base, info


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    entries = [r for r in plan["entries"] if r["action"] in {"archive", "move_reference", "reuse_immutable_archive"}]
    mapping = {r["source_path"]: r["destination"] for r in entries}
    journal_files = sorted((ROOT / AUDIT / "evidence_cleanup").glob("MIGRATION_LOG_*.jsonl"))
    events = {}
    for path in journal_files:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            event = json.loads(line)
            events[event["source"].replace("\\", "/").lower()] = event
    original_map = []
    # All byte-for-byte post-move checks precede any link edit.
    for entry in entries:
        source, destination = ROOT / entry["source_path"], ROOT / entry["destination"]
        event = events.get(source.as_posix().lower())
        if not event or event.get("status") != "completed" or event["original_sha256"] != entry["sha256"]:
            raise RuntimeError(f"No completed matching migration receipt: {entry['source_path']}")
        if source.exists() or not destination.is_file() or sha(destination) != entry["sha256"]:
            raise RuntimeError(f"Post-move source/destination verification failed: {entry['source_path']}")
        original_map.append({"source_path": entry["source_path"], "destination": entry["destination"],
                             "action": entry["action"], "original_sha256": entry["sha256"],
                             "sha256_after_byte_preserving_move": sha(destination), "link_changes": []})

    allowed = {r["source_path"]: r["destination"] for r in entries if r["action"] != "reuse_immutable_archive" and r["destination"].endswith(".md")}
    allowed.update({p: p for p in ROOT_EDIT | LEGACY_EDIT if (ROOT / p).is_file()})
    edits, no_change = [], []
    original_bytes_by_source = {}
    for source_old, source_new in sorted(allowed.items()):
        path = ROOT / source_new
        raw = path.read_bytes()
        original_bytes_by_source[source_old] = raw
        text = raw.decode("utf-8")
        replacements = []
        for ref in records(text):
            replacement, resolved = target_after(source_old, source_new, ref, mapping)
            if replacement != ref["target"]:
                replacements.append({**ref, "new_target": replacement, "resolved_before_move": resolved.get("resolved_path")})
        if not replacements:
            no_change.append(source_new)
            continue
        changed = text
        for item in reversed(replacements):
            changed = changed[:item["start"]] + item["new_target"] + changed[item["end"]:]
        if changed.count("\n") != text.count("\n"):
            raise RuntimeError(f"A reference edit would alter line count: {source_new}")
        if path.read_bytes() != raw:
            raise RuntimeError(f"Document changed concurrently: {source_new}")
        before_sha = sha(path)
        path.write_bytes(changed.encode("utf-8"))
        edits.append({"source_original": source_old, "path": source_new, "before_sha256": before_sha,
                      "after_sha256": sha(path), "line_count_unchanged": True, "changes": replacements})

    edit_by_source = {r["source_original"]: r for r in edits}
    for row in original_map:
        row["final_sha256"] = sha(ROOT / row["destination"])
        if row["source_path"] in edit_by_source:
            row["link_changes"] = edit_by_source[row["source_path"]]["changes"]
        if row["action"] == "reuse_immutable_archive" and row["final_sha256"] != row["original_sha256"]:
            raise RuntimeError("Immutable duplicate original was modified")

    parent_pending, preserved = [], []
    allowed_current = set(allowed.values())
    active_paths = sorted({p for p in PARENT_ROOT | ROOT_EDIT if (ROOT / p).is_file()}
                          | {r["source_path"] for r in plan["entries"] if r["action"] == "retain" and r["source_path"].endswith(".md") and not r["source_path"].startswith(f"{AUDIT}/")}
                          | {p.relative_to(ROOT).as_posix() for p in (ROOT / AUDIT).glob("*.md") if p.name not in {"DOCUMENT_MIGRATION_REPORT.md", "DOCUMENT_CLEANUP_PLAN.md"}})
    active_checks = []
    for source in active_paths:
        text = (ROOT / source).read_bytes().decode("utf-8")
        for ref in records(text):
            info = resolve_target(source, ref["target"], repository_relative=ref["kind"] == "repository_path_literal")
            if info["scope"] != "workspace":
                continue
            target = info["resolved_path"]
            check = {"source_path": source, "line": ref["line"], "kind": ref["kind"], "target": ref["target"],
                     "resolved_path": target, "exists": (ROOT / target).exists()}
            if target in mapping:
                replacement, _ = target_after(source, source, ref, mapping)
                item = {**check, "replacement": replacement, "destination": mapping[target],
                        "source_sha256_at_check": sha(ROOT / source), "owner": "parent"}
                if source in allowed_current:
                    check["status"] = "unexpected_unupdated_authorized_reference"
                else:
                    check["status"] = "pending_parent_replacement"
                    parent_pending.append(item)
            else:
                check["status"] = "valid" if check["exists"] else "missing_not_in_migration_map"
            active_checks.append(check)

    historical_checks = []
    for source_old, source_new in allowed.items():
        if source_new in ROOT_EDIT | LEGACY_EDIT:
            continue
        before = original_bytes_by_source[source_old].decode("utf-8")
        before_refs = records(before)
        after_refs = records((ROOT / source_new).read_bytes().decode("utf-8"))
        if len(before_refs) != len(after_refs):
            raise RuntimeError(f"Reference record count changed unexpectedly: {source_new}")
        for old, new in zip(before_refs, after_refs):
            expected, old_info = target_after(source_old, source_new, old, mapping)
            current = resolve_target(source_new, new["target"], repository_relative=new["kind"] == "repository_path_literal")
            if current["scope"] != "workspace":
                continue
            expected_path = mapping.get(old_info["resolved_path"], old_info["resolved_path"])
            exists = (ROOT / current["resolved_path"]).exists()
            original_exists = old_info["resolved_path"] in mapping or (ROOT / old_info["resolved_path"]).exists()
            historical_checks.append({"source_original": source_old, "source_path": source_new, "line": new["line"],
                "target": new["target"], "resolved_path": current["resolved_path"], "exists": exists,
                "same_logical_target": current["resolved_path"] == expected_path if original_exists else None,
                "status": "valid" if exists else "preexisting_missing_target" if not original_exists else "new_missing_target"})

    # Source-code and provenance references are locators only; never edit them here.
    for entry in entries:
        for ref in entry["inbound_links"]:
            source = ref["source_path"]
            if source in allowed or source in active_paths or source in mapping or ref["kind"] == "basename_mention":
                continue
            if ref["kind"] == "repository_path_literal" and source == "src/crystal_dlm/d3po.py":
                parent_pending.append({**ref, "replacement": entry["destination"], "destination": entry["destination"],
                                       "owner": "parent_source_docstring", "source_sha256_at_check": sha(ROOT / source)})
            elif ref["owner_policy"].startswith("preserved_provenance"):
                preserved.append({**ref, "destination_in_migration_map": entry["destination"], "rewrite": False})

    invariant_paths = [p for p in plan["protected_exact_paths"] if p not in PARENT_ROOT and p not in LEGACY_EDIT]
    retained = []
    by_source = {r["source_path"]: r for r in plan["entries"]}
    for p in invariant_paths:
        if p in by_source:
            retained.append({"path": p, "exists": (ROOT / p).is_file(), "sha256": sha(ROOT / p),
                             "same_as_plan": sha(ROOT / p) == by_source[p]["sha256"]})
    problems = [r for r in historical_checks if r["same_logical_target"] is False or r["status"] == "new_missing_target"]
    problems += [r for r in active_checks if r["status"] == "unexpected_unupdated_authorized_reference"]
    report = {"schema": "document_migration_report_v1", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "workspace": ROOT.as_posix(), "status": "completed" if not problems else "requires_reference_repair",
              "scope": f"{len(entries)} approved sources; scientific text preserved, only authorized Markdown references/path literals updated.",
              "counts": {"sources_processed": len(entries), "byte_preserving_moves": sum(r["action"] != "reuse_immutable_archive" for r in entries),
                         "exact_duplicates_reused": sum(r["action"] == "reuse_immutable_archive" for r in entries),
                         "documents_with_reference_edits": len(edits), "reference_edits": sum(len(r["changes"]) for r in edits),
                         "active_documents_checked": len(active_paths), "active_references_checked": len(active_checks),
                         "active_statuses": dict(collections.Counter(r["status"] for r in active_checks)),
                         "historical_references_checked": len(historical_checks),
                         "historical_statuses": dict(collections.Counter(r["status"] for r in historical_checks)),
                         "pending_parent_replacements": len(parent_pending), "unexpected_migration_reference_problems": len(problems)},
              "journal_files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in journal_files],
              "mapping": original_map, "reference_edits": edits, "unchanged_authorized_documents": no_change,
              "protected_contract_verification": retained, "parent_replacements": parent_pending,
              "preserved_provenance_references": preserved, "active_link_checks": active_checks,
              "historical_link_checks": historical_checks, "unexpected_problems": problems,
              "claim_boundary": "Original reviewed-version SHA strings and scientific claims are unchanged. Different final file hashes record link-only edits. Project src/scripts/slurm code, models, run assets, raw data, immutable archives, and other projects were not modified; documentation migration tools were added separately."}
    output = ROOT / AUDIT / "DOCUMENT_MIGRATION_REPORT.json"
    dump_json(output, report)
    dump_json(ROOT / AUDIT / "DOCUMENT_MIGRATION_MAP.json", {"schema": "document_migration_map_v1", "mapping": original_map})
    dump_json(ROOT / AUDIT / "PARENT_REFERENCE_REPLACEMENTS.json", {"schema": "parent_document_reference_replacements_v1", "replacements": parent_pending,
              "preserved_provenance_references": preserved, "automation_prompts": "Parent updates separately using DOCUMENT_MIGRATION_MAP; not inspected or modified here."})
    lines = ["# 文档迁移与链接核验结果", "", f"已按主审协调范围处理 {len(entries)} 个源：{report['counts']['byte_preserving_moves']} 个逐字节移动、{report['counts']['exact_duplicates_reused']} 个已复核完全重复的 docs 副本复用既有 immutable archive。原 archive 未修改。", "",
             f"随后只在获准的 {len(edits)} 份文档内更新 {sum(len(r['changes']) for r in edits)} 处链接/文档路径，所有行数保持不变；科学正文、审稿记录中的被审版本 SHA 没有改写。原 SHA、移动后 SHA 和链接改写后 SHA 均在 [逐文件映射](DOCUMENT_MIGRATION_MAP.json)。", "",
             "| 核验 | 结果 |", "|---|---|", f"| 活跃文件/引用 | {len(active_paths)} / {len(active_checks)} |",
             f"| 活跃引用状态 | `{json.dumps(report['counts']['active_statuses'], ensure_ascii=False)}` |",
             f"| 已迁移历史文档引用 | {len(historical_checks)}；`{json.dumps(report['counts']['historical_statuses'], ensure_ascii=False)}` |",
             f"| 本次迁移意外引用问题 | {len(problems)} |", "",
             "当前状态/架构、root README/EXPERIMENT、V2 运行合同、在写 audit 正文和 d3po 源码 docstring 均按分工保留。需要主审替换的精确行和目标见 [替换清单](PARENT_REFERENCE_REPLACEMENTS.json)；不把这些已明确留给主审的项混报为已全部修完。自动化 prompt 由主审依映射处理，本子任务未访问或修改。", "",
             "## 主审剩余替换范围", "", "| 文件 | 待替换引用数 |", "|---|---|"]
    for source, count in sorted(collections.Counter(r["source_path"] for r in parent_pending).items()):
        lines.append(f"| `{source}` | {count} |")
    lines += ["", "原有缺失资源仍单列 `missing_not_in_migration_map` 或 `preexisting_missing_target`，没有擅自恢复旧数据/模型。全部逐引用结果、保留合同哈希与移动 journal 见 [完整报告](DOCUMENT_MIGRATION_REPORT.json)。", ""]
    (ROOT / AUDIT / "DOCUMENT_MIGRATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": output.as_posix(), "status": report["status"], **report["counts"]}, ensure_ascii=False))
    if problems:
        raise RuntimeError(f"{len(problems)} unexpected migration link problems remain; see report")


if __name__ == "__main__":
    main()
