"""Inventory-only documentation relocation plan; never move or delete inputs."""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote
from reference_syntax import inside_code, markdown_protected_ranges

ROOT = Path(__file__).resolve().parents[3]
AUDIT = "docs/v3_scientific_audit_20260906"
OUTPUTS = {f"{AUDIT}/DOCUMENT_CLEANUP_PLAN.json", f"{AUDIT}/DOCUMENT_CLEANUP_PLAN.md"}
INVENTORY = ROOT / AUDIT / "evidence/inventory_before_cleanup"
PROTECTED = {
    "docs/CURRENT_STATE.md": "当前状态入口；主审正在维护。",
    "docs/CURRENT_ARCHITECTURE.md": "当前架构入口；主审正在维护。",
    "docs/DATA_LICENSES.md": "数据许可证仍需直接可用。",
    "docs/MODEL_LICENSES.md": "模型许可证仍需直接可用。",
    "docs/paper/README.md": "test_paper_pipeline_layout 读取的历史 G2 兼容入口，不宣称当前方法。",
    "docs/paper/METHOD_AT_A_GLANCE.md": "test_paper_pipeline_layout 读取并校验历史 G2 术语，暂保留字节。",
    "docs/PLACEHOLDER_ASSETS.md": "现有环境/下载/提交脚本仍把此路径显示为帮助入口；是发布资产说明。",
    "docs/SEEDS.md": "slurm/_common.sh 的帮助文本仍指向此历史种子清单。",
}
LIVE_V2 = {
    "CHECKLIST.md", "V2_LAUNCH_MANIFEST.json", "V2_EXECUTION_PACKAGE_20260906.md",
    "V2_IMPLEMENTATION_REVIEW_20260906.md", "LOSS_GEOMETRY_AND_V2_PROPOSAL_20260906.md",
    "RAW_LLADA_DESIGN.md",
}
COMPONENT_REFERENCES = {
    "docs/ADR_C3FD_LLAMA_FUSED_TYPED_PLANNER_20260831.md":
        "冻结 typed Planner 的组成支持/软条件接口仍用于当前上游；执行预算与旧结果仅是历史上下文。",
    "docs/C3FD_RICH_FIELD_SEMANTICS_AUDIT_V2.md":
        "软 lattice/SG/volume 字段语义更正仍有组件参考价值；不作为当前 SUN 或执行状态。",
}
# Parent clarified scope: these two tracked historical documents may move;
# the separate PMTR worktree, its uncommitted files, models and runs may not.
PMTR_HOLD: set[str] = set()
TEXT_EXTENSIONS = {".md", ".mmd", ".py", ".ps1", ".sh", ".sbatch", ".toml", ".yaml", ".yml", ".json"}
MAX_SCAN_BYTES = 2_000_000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rg_files(*paths: str) -> list[str]:
    result = subprocess.run(["rg", "--files", *paths], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    return sorted({line.replace("\\", "/") for line in result.stdout.splitlines() if line})


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def owner_policy(path: str) -> str:
    if (path.startswith(("archives/", "results/", "runs/", f"{AUDIT}/evidence", f"{AUDIT}/historical/"))
            or (path.startswith("docs/") and path.endswith(".json"))):
        return "preserved_provenance_do_not_rewrite_automatically"
    if path.startswith(f"{AUDIT}/") or path in PROTECTED or path in {"README.md", "PAPER_PIPELINE.md"}:
        return "author_coordination_before_link_edits"
    return "link_edits_only_after_coordinated_move"


def resolve_target(source: str, raw: str, *, repository_relative: bool = False) -> dict:
    target = unquote(raw.strip().strip("<>"))
    target = re.split(r'\s+["\']', target, maxsplit=1)[0]
    if re.match(r"^(https?|mailto|codex|app|data):", target, re.I):
        return {"target": raw, "scope": "external_or_app", "resolved_path": None}
    if not target or target.startswith("#"):
        return {"target": raw, "scope": "same_document_anchor", "resolved_path": source}
    base, _, anchor = target.partition("#")
    match = re.match(r"^(.*?)(:\d+)?$", base)
    base, line_suffix = match.group(1), match.group(2) or ""
    base = base.replace("\\", "/")
    p = Path(base)
    if not p.is_absolute():
        p = ROOT / base if repository_relative else ROOT / Path(source).parent / base
    p = p.resolve(strict=False)
    try:
        relative = p.relative_to(ROOT).as_posix()
    except ValueError:
        return {"target": raw, "scope": "outside_workspace", "resolved_path": p.as_posix(), "exists": p.exists(), "line_suffix": line_suffix, "anchor": anchor}
    return {"target": raw, "scope": "workspace", "resolved_path": relative, "exists": p.exists(),
            "line_suffix": line_suffix, "anchor": anchor}


def classification(path: str, duplicates: dict) -> dict:
    name = Path(path).name
    if path.startswith(f"{AUDIT}/"):
        return dict(action="retain", category="protected_current_audit_or_evidence", reason="整个当前审计目录受保护；仅登记快照。", batch="protected", destination=path)
    if path in PROTECTED:
        return dict(action="retain", category="current_entry_license_or_compatibility", reason=PROTECTED[path], batch="protected", destination=path)
    if path.startswith("docs/periodic_self_repair_v1/") and name in LIVE_V2:
        return dict(action="retain", category="frozen_current_execution_contract", reason="V2 当前执行/设计/验收/初始化链仍在使用；状态段可过期，但本次保持路径与内容。", batch="protected", destination=path)
    if path in PMTR_HOLD:
        return dict(action="retain_pending_scope", category="pmtr_history_scope_hold", reason="本仓库的 PMTR 历史说明；按“不动 PMTR”暂不加入迁移批次，实际 PMTR 项目和运行资产始终范围外。", batch="protected", destination=path)
    if path in duplicates:
        return dict(action="reuse_immutable_archive", category="exact_duplicate_component_history", reason="已与既有 C³FD immutable archive 再次核对 SHA256 完全相同；引用改向既有原件后，可去掉 docs 中重复副本。", batch="reuse_immutable", destination=duplicates[path]["canonical_path"])
    if path in COMPONENT_REFERENCES:
        return dict(action="move_reference", category="still_relevant_component_reference", reason=COMPONENT_REFERENCES[path], batch="component_reference", destination=f"{AUDIT}/reference/component_specs/{name}")
    if path.endswith((".json", ".py", ".mmd")):
        category = "historical_supporting_data_or_tool"
        reason = "保留与同目录 Markdown 对应的原始统计、推导工具或图；不删除、不执行、不改科学内容。"
    elif re.search(r"HANDOFF|NEXT_CONVERSATION|GPT56", name):
        category, reason = "superseded_handoff", "旧交接与下一步指令已被当前审计/执行入口替代；保留作为时间线证据。"
    elif name.startswith(("28_FINAL_SPRINT", "29_PERIODIC_SELF_REPAIR")):
        category, reason = "superseded_status_or_execution_plan", "旧最终冲刺/12小时修复计划已被当前 V2 执行与审计入口替代；保留计划原文。"
    elif re.search(r"FINAL|RESULTS|FAILURE|FAILED|REGRESSION|REPRODUCIBILITY|MATCHED_EVALUATION|DIAGNOSIS", name):
        category, reason = "historical_result_or_failure_evidence", "历史结果、失败、复算或诊断；独有科学证据须保留，不能作为当前状态入口。"
    elif re.search(r"CHECKLIST|STATUS|WORKLOG|DECISION_LOG|EXECUTION|RESUMED|FINAL_SPRINT", name):
        category, reason = "superseded_status_or_execution_plan", "旧版本状态/预算/执行清单已过时；归档并保留原文，当前动作只由现行入口/manifest决定。"
    elif re.search(r"CONTRACT|DESIGN|PLAN|PROPOSED|ARCHITECTURE|METHOD", name):
        category, reason = "historical_design_or_component_contract", "记录已测试或未执行旧方法的设计/合同；不应与 V2 当前执行合同混读。"
    elif re.search(r"AUDIT|REVIEW|SEMANTIC|MATH|MECHANISM|FEASIBILITY|ASSESSMENT", name):
        category, reason = "historical_analysis", "保留历史论证和分析；最新裁决集中在当前科学审计目录。"
    else:
        category, reason = "historical_context_or_reader_entry", "旧论文叙事、入口、参考、提示词或上下文；保留供追溯，不充当当前执行说明。"
    if path.startswith("docs/paper/"):
        reason += " 此 paper 目录描述历史 G2 CLI，不能标为当前主线。"
    if path.startswith("docs/teacher_feedback_unified_v1/"):
        reason += " 属于旧 teacher-feedback 演进链，和当前原始 LLaDA/MP20 V2 输入分开。"
    return dict(action="archive", category=category, reason=reason, batch="historical_docs", destination=f"{AUDIT}/historical/{path}")


def main() -> None:
    baseline = read_json(INVENTORY / "DOCUMENT_INVENTORY.json")
    failures = read_json(INVENTORY / "FAILURE_DISCOVERY_INDEX.json")
    changes = read_json(INVENTORY / "RECENT_CHANGE_INVENTORY.json")
    baseline_docs = {d["path"]: d for d in baseline["documents"]}
    failure_by_path = collections.defaultdict(list)
    for row in failures["signals"]:
        failure_by_path[row["path"]].append({"line": row["line"], "text": row["text"]})
    last_change = {}
    for commit in changes["commits"]:
        for raw in commit["changed_paths"]:
            for path in raw.split("\t")[1:]:
                last_change.setdefault(path, {k: commit[k] for k in ("sha", "authored_at", "subject")})

    docs = [p for p in rg_files("-uu", "docs") if p not in OUTPUTS and not p.startswith(f"{AUDIT}/evidence_cleanup/MIGRATION_LOG_")]
    hashes = {p: sha(ROOT / p) for p in docs}
    duplicates, duplicate_receipts = {}, []
    for group in baseline["exact_duplicates"]:
        archive_paths = [p for p in group["paths"] if p.startswith("archives/") and (ROOT / p).is_file()]
        for source in group["paths"]:
            if not source.startswith("docs/") or source not in hashes:
                continue
            for canonical in archive_paths:
                archive_sha = sha(ROOT / canonical)
                receipt = {"source": source, "canonical_path": canonical, "inventory_sha256": group["sha256"],
                           "source_sha256": hashes[source], "archive_sha256": archive_sha,
                           "exact_match_now": hashes[source] == archive_sha == group["sha256"]}
                duplicate_receipts.append(receipt)
                if receipt["exact_match_now"]:
                    duplicates[source] = receipt
                    break

    records = {}
    for p in docs:
        stat = (ROOT / p).stat()
        initial = baseline_docs.get(p)
        info = classification(p, duplicates)
        records[p] = {"source_path": p, "source_absolute": (ROOT / p).as_posix(), "sha256": hashes[p],
                      "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, **info,
                      "destination_absolute": (ROOT / info["destination"]).as_posix(),
                      "baseline_inventory_sha256": initial.get("sha256") if initial else None,
                      "baseline_same_bytes": hashes[p] == initial["sha256"] if initial else None,
                      "last_change_in_inventory": last_change.get(p),
                      "failure_signal_count": len(failure_by_path[p]),
                      "failure_signal_examples": failure_by_path[p][:3],
                      "classification_evidence": {"headings": initial.get("headings", [])[:5], "early_status_signals": initial.get("early_status_signals", [])[:5]} if initial else {},
                      "inbound_links": [], "outbound_links": [], "runtime_test_dependencies": [], "intrinsic_path_assumptions": []}

    by_name = collections.defaultdict(list)
    for p in docs:
        if not p.startswith(f"{AUDIT}/"):
            by_name[Path(p).name].append(p)
    unique_names = {name: paths[0] for name, paths in by_name.items() if len(paths) == 1}
    basename_re = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(n) for n in sorted(unique_names, key=len, reverse=True)) + r")(?![A-Za-z0-9_])")
    md_link = re.compile(r"\]\((<[^>]+>|[^\n)]*)\)")
    md_reference = re.compile(r"^\s*\[[^\]]+\]:\s*(<[^>]+>|\S+)")
    explicit_docs = re.compile(r"(?<![A-Za-z0-9_])docs[/\\][A-Za-z0-9_./\\-]+\.(?:md|json|py|mmd)(?::\d+)?")
    links, skipped = [], []
    for source in rg_files():
        p = ROOT / source
        if source in OUTPUTS or p.suffix.lower() not in TEXT_EXTENSIONS or source.startswith("src/vendor/"):
            continue
        if source.startswith(f"{AUDIT}/evidence") or source.startswith(("checkpoints/", "data/", "runs/")):
            continue
        if source.startswith("archives/") and p.suffix.lower() != ".md":
            continue
        if p.stat().st_size > MAX_SCAN_BYTES:
            skipped.append({"source": source, "reason": "text_scan_size_limit", "bytes": p.stat().st_size})
            continue
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except UnicodeError:
            skipped.append({"source": source, "reason": "not_utf8_text"})
            continue
        per_source = {}
        code_ranges = markdown_protected_ranges("\n".join(lines))
        offsets, running = [], 0
        for line in lines:
            offsets.append(running)
            running += len(line) + 1
        for number, (line_offset, line) in enumerate(zip(offsets, lines), 1):
            if source in records and p.suffix.lower() == ".py":
                if "__file__" in line or "sys.argv[1]" in line:
                    records[source]["intrinsic_path_assumptions"].append({"line": number, "text": line.strip(),
                        "interpretation": "Adjacent output stays with the archived script; explicit argv run-root remains caller-supplied. Historical tools are not executed by cleanup."})
            found = [(m.group(1), "markdown_link", False) for m in md_link.finditer(line)
                     if not inside_code(line_offset + m.start(1), line_offset + m.end(1), code_ranges)]
            match = md_reference.match(line)
            if match and not inside_code(line_offset + match.start(1), line_offset + match.end(1), code_ranges):
                found.append((match.group(1), "markdown_reference", False))
            found.extend((m.group(0), "repository_path_literal", True) for m in explicit_docs.finditer(line))
            for raw, kind, repo_relative in found:
                item = {"source_path": source, "line": number, "kind": kind, "owner_policy": owner_policy(source),
                        **resolve_target(source, raw, repository_relative=repo_relative)}
                key = (number, item["resolved_path"] or item["target"], item.get("line_suffix", ""), item.get("anchor", ""), item["scope"])
                per_source.setdefault(key, item)
            for match in basename_re.finditer(line):
                target = unique_names[match.group(1)]
                if any(key[0] == number and key[1] == target for key in per_source):
                    continue
                key = (number, target, "", "", "workspace")
                per_source.setdefault(key, {"source_path": source, "line": number, "kind": "basename_mention",
                                            "owner_policy": owner_policy(source), "target": match.group(1),
                                            "resolved_path": target, "scope": "workspace", "exists": True})
        for item in per_source.values():
            links.append(item)
            if source in records:
                records[source]["outbound_links"].append(item)
            target = item["resolved_path"]
            if target in records:
                records[target]["inbound_links"].append(item)
                if source.startswith(("src/", "scripts/", "tests/", "operations/", "slurm/")) and p.suffix.lower() != ".md":
                    dep_kind = ("test_file_read" if source in {"tests/test_paper_pipeline_layout.py", "tests/test_periodic_v2_queue.py"}
                                else "docstring_reference" if source == "src/crystal_dlm/d3po.py"
                                else "help_text_reference" if source in {"slurm/_common.sh", "scripts/create_environment.sh", "scripts/download_checkpoints.sh", "scripts/submit_h1a2.sh"}
                                else "text_reference_review")
                    records[target]["runtime_test_dependencies"].append({**item, "dependency_kind": dep_kind})

    moved = {p: r["destination"] for p, r in records.items() if r["action"] in {"archive", "move_reference", "reuse_immutable_archive"}}
    link_updates = []
    for item in links:
        if item["scope"] != "workspace" or item["kind"] == "basename_mention":
            continue
        source, target = item["source_path"], item["resolved_path"]
        if source not in moved and target not in moved:
            continue
        if source in moved and records[source]["action"] == "reuse_immutable_archive":
            continue
        after_source, after_target = moved.get(source, source), moved.get(target, target)
        if item["kind"] == "repository_path_literal":
            replacement = after_target
        elif re.match(r"^[A-Za-z]:[/\\]", item["target"]):
            replacement = (ROOT / after_target).as_posix()
        else:
            replacement = os.path.relpath(ROOT / after_target, (ROOT / after_source).parent).replace("\\", "/")
        replacement += item.get("line_suffix", "")
        if item.get("anchor"):
            replacement += "#" + item["anchor"]
        if replacement != item["target"] or after_source != source:
            link_updates.append({**item, "source_after_move": after_source, "target_after_move": after_target,
                                 "suggested_target_after_move": replacement,
                                 "apply_now": False, "preserve_provenance_literal": item["owner_policy"].startswith("preserved_provenance")})

    action_counts = dict(collections.Counter(r["action"] for r in records.values()))
    non_audit = [r for r in records.values() if not r["source_path"].startswith(f"{AUDIT}/")]
    dependencies = [{"document": r["source_path"], "references": r["runtime_test_dependencies"]} for r in records.values() if r["runtime_test_dependencies"]]
    all_hashes = collections.defaultdict(list)
    for path, value in hashes.items():
        all_hashes[value].append(path)
    plan = {
        "schema": "document_cleanup_plan_v1", "status": "parent_coordinated_execution_approved_no_moves_by_builder",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "workspace": ROOT.as_posix(),
        "scope": "Only documentation under this worktree docs/; existing archives are read-only duplicate references.",
        "excluded_mutations": ["models", "raw data", "runs", "archives", "PMTR project/assets", "other projects", "root entrypoints owned by parent"],
        "scope_clarification": "Parent explicitly approved historical_docs, component_reference, six exact archive duplicates, and tracked teacher-feedback PMTR history 15/16. The separate PMTR worktree remains excluded.",
        "excluded_self_referential_products": sorted(OUTPUTS),
        "protected_exact_paths": sorted(PROTECTED) + [f"docs/periodic_self_repair_v1/{name}" for name in sorted(LIVE_V2)] + sorted(PMTR_HOLD),
        "protected_source_prefixes": [AUDIT + "/"],
        "source_inventories": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in sorted(INVENTORY.glob("*.json"))],
        "baseline_context": {"tracked_markdown_across_repo": len(baseline["documents"]), "baseline_docs_markdown": sum(p.startswith("docs/") for p in baseline_docs),
                             "failure_text_signals_not_independent_experiments": len(failures["signals"]), "recent_first_parent_commits": len(changes["commits"])},
        "summary": {"document_files": len(records), "non_audit_document_files": len(non_audit), "actions": action_counts,
                    "batches": dict(collections.Counter(r["batch"] for r in records.values())),
                    "planned_link_updates": len(link_updates), "documents_with_runtime_test_text_references": len(dependencies)},
        "archive_duplicate_receipts": duplicate_receipts,
        "other_current_exact_duplicate_groups_no_automatic_deletion": [{"sha256": value, "paths": paths} for value, paths in all_hashes.items() if len(paths) > 1],
        "runtime_test_dependency_index": dependencies, "scan_limits": {"max_text_bytes": MAX_SCAN_BYTES, "skipped": skipped,
            "not_scanned_as_runtime": ["vendor", "audit evidence snapshots", "checkpoints/data/runs", "non-Markdown archive files"],
            "interpretation": "Literal/path/link scan plus explicit read-site review; basename mentions are evidence locators, not proof of runtime reads."},
        "entries": list(records.values()), "link_updates": link_updates,
        "execution": {"script": f"{AUDIT}/evidence_cleanup/migrate_documents.ps1", "default": "read_only_preflight",
                      "apply_after": "Parent explicitly coordinates selected sources and reference owners.",
                      "reference_policy": "Moves preserve bytes; update only separately authorized links, recording original SHA. Never rewrite immutable archive/provenance snapshots.",
                      "dedup_policy": "Only six revalidated immutable-archive duplicates are eligible for single-file source removal after references are redirected; no recursive deletion."},
    }
    out_json, out_md = ROOT / AUDIT / "DOCUMENT_CLEANUP_PLAN.json", ROOT / AUDIT / "DOCUMENT_CLEANUP_PLAN.md"
    out_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = ["# 文档清理批准执行清单", "", "本清单只规划本工作树 docs。主审已协调阅读作者并明确批准 historical_docs、component_reference、六份重复复用及本树 PMTR 历史说明15/16；本清单构建工具本身不执行迁移，实际结果见随后生成的迁移报告。", "",
            f"本次登记 **{len(records)}** 个文件，其中 audit 外 **{len(non_audit)}** 个。动作计数：`{json.dumps(action_counts, ensure_ascii=False)}`。逐源原 SHA、原库存对照、分类原因、双向链接、代码/测试引用和建议替换均在 [JSON](DOCUMENT_CLEANUP_PLAN.json)。", "",
            "源库存的 209 是全仓库 Markdown 数，其中 docs 有 136；559 是失败文本线索数，不是独立失败实验数。当前还纳入 JSON 统计、推导脚本、图和新增审计证据，故总数不可直接与209相减解释为新增报告。", "",
            "## 已协调迁移的具体范围", "", "- `historical_docs`：除下表保护项外，旧 top-level 计划/状态/交接/结果、teacher_feedback 历史（含本树PMTR说明15/16）、旧 periodic review_notes 和历史 paper 说明，镜像保存在本审计目录 `historical/docs/`。", "- `component_reference`：两份仍有组件参考价值的 typed Planner/soft-field 语义说明，移到 `reference/component_specs/`；不把旧执行预算或结果改称当前事实。", "- `reuse_immutable`：六份已重新核 SHA 的 C³FD docs 副本复用既有 immutable archive。归档原件不动；重复源逐文件去重，待主审统一改其负责的入口/审计引用。原库存另五组 results 重复不在本任务范围。", "",
            "主审已完成旧文档读取并通知上游作者使用冻结 Git 版本或映射。JSON 保留逐引用边界；本子任务只改获准的历史文档、剩余 legacy 文档及稳定 root WORKFLOW/PAPER_PIPELINE/REPRODUCTION 的链接。", "",
            "## 保持原位的入口与执行依赖", "", "| 文件 | 原因 |", "|---|---|"]
    for r in non_audit:
        if r["action"].startswith("retain"):
            rows.append(f"| `{r['source_path']}` | {r['reason']} |")
    rows += ["", "当前 audit 目录整体保留，包含主审已存档的六个 root 入口。`PAPER_PIPELINE.md` 的旧 CLI 已明确是历史 G2；本任务不改父审正在维护的 root README 或入口。", "",
             "依赖核验：paper 两文件由 `tests/test_paper_pipeline_layout.py` 读取；V2 manifest 由 `tests/test_periodic_v2_queue.py` 读取。`PLACEHOLDER_ASSETS.md` 与 `SEEDS.md` 仍是脚本帮助链接；`src/crystal_dlm/d3po.py:5` 对 D3PO 合同是 docstring 引用，不是运行时读取。迁移该旧合同后需更新这个已定位的引用。", "",
             "两份旧分析脚本也已核路径假设：`continuous_math_checks.py` 写相邻同名 JSON，因此与 JSON 一起归档；`analyze_round0_gap_change.py` 从 argv 接受 run root，不依赖所在 docs 深度。它们只作为历史证据移动，不在清理中执行。JSON 台账内记录的历史来源路径保留原值，不当成 Markdown 链接批量替换。", "",
             "## 六份可复用的逐字节重复", "", "| 当前副本 | 既有原件 | SHA256 |", "|---|---|---|"]
    for receipt in duplicate_receipts:
        if receipt["exact_match_now"]:
            rows.append(f"| `{receipt['source']}` | `{receipt['canonical_path']}` | `{receipt['source_sha256']}` |")
    rows += ["", "## 逐文件迁移表", "", "JSON 保留所有受保护审计证据的逐文件登记。下表列出 audit 外所有建议迁移/复用项；原始科学证据和配套 JSON/推导脚本一起保存。", "",
             "| 源文件 | 动作 / 分类 | 目标或复用位置 | 入链 / 出链 / 代码测试引用 |", "|---|---|---|---|"]
    for r in non_audit:
        if r["action"].startswith("retain"):
            continue
        rows.append(f"| `{r['source_path']}` | `{r['action']}` / `{r['category']}` | `{r['destination']}` | {len(r['inbound_links'])} / {len(r['outbound_links'])} / {len(r['runtime_test_dependencies'])} |")
    rows += ["", "## 执行方案", "", "配置已经包含逐源 SHA 与目的路径；迁移工具默认只预检，默认不会改引用或文件。源内容在预检后有变更、目标已存在、路径越界或遇到 reparse point 时停止该批次。", "", "```powershell",
             f"& '{(ROOT / AUDIT / 'evidence_cleanup/migrate_documents.ps1').as_posix()}' -Batch historical_docs",
             "# 主审明确协调范围与引用作者后，才对同一选定批次加 -Apply。",
             "# 也可使用 -SourcePath 'docs/具体文件.md' 仅预检/执行选定源。", "```", "",
             "工具只用原生 PowerShell 的 `Move-Item -LiteralPath` 搬移文件；仅对 SHA 完全相同且归档原件再次验证的重复副本使用无递归 `Remove-Item -LiteralPath`。移动和去重均在写操作之前核对解析后的绝对路径属于本工作树 docs，目标位于指定 audit 子目录或既有只读 archive，并留下逐文件 journal。", "",
             f"本次发现 {len(link_updates)} 条迁移后需要检查的链接/路径记录。`link_updates` 给出原行、迁移后源/目标、建议新 target；basename 提及不自动改，immutable archive 与证据快照中的历史路径不自动改。移动后仅更新主审明确指定的引用；清单构建阶段不运行测试或迁移。", ""]
    out_md.write_text("\n".join(rows), encoding="utf-8")
    print(json.dumps({"plan_json": out_json.as_posix(), "plan_markdown": out_md.as_posix(), **plan["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
