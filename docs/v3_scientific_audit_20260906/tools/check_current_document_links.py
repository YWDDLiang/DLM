"""Check local links in current entrypoints and audit reports, without editing."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evidence_cleanup'))
from reference_syntax import markdown_protected_ranges, inside_code


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    audit = root / 'docs/v3_scientific_audit_20260906'
    sources = [root / name for name in ('README.md', 'EXPERIMENT.md', 'WORKFLOW.md', 'REPRODUCTION.md', 'PAPER_PIPELINE.md')]
    sources += list((root / 'docs').glob('*.md'))
    sources += list((root / 'docs/periodic_self_repair_v1').glob('*.md'))
    sources += list((root / 'docs/paper').glob('*.md'))
    sources += [f for f in audit.glob('*.md') if f.name not in {
        'DOCUMENT_CLEANUP_PLAN.md', 'DOCUMENT_MIGRATION_REPORT.md'}]
    mapping = {r['source_path']: r['destination'] for r in json.loads((audit / 'DOCUMENT_MIGRATION_MAP.json').read_text())['mapping']}
    checked, missing = [], []
    for path in sorted(set(sources)):
        text = path.read_text(encoding='utf-8')
        protected = markdown_protected_ranges(text)
        for match in re.finditer(r'\]\(([^\n)]+)\)', text):
            if inside_code(match.start(), match.end(), protected):
                continue
            target = match.group(1).strip().strip('<>')
            if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*://|^mailto:|^#', target):
                continue
            physical = unquote(target.split('#', 1)[0])
            physical = re.sub(r':\d+$', '', physical)
            destination = Path(physical) if re.match(r'^[A-Za-z]:[/\\]', physical) else path.parent / physical
            destination = destination.resolve()
            item = {'source': path.relative_to(root).as_posix(), 'line': text.count('\n', 0, match.start()) + 1,
                    'target': target, 'exists': destination.exists()}
            if not item['exists']:
                if destination.is_relative_to(root):
                    item['migration_destination'] = mapping.get(destination.relative_to(root).as_posix())
                missing.append(item)
            checked.append(item)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'documents': len(set(sources)),
              'local_links': len(checked), 'missing_count': len(missing), 'missing': missing,
              'scope': 'Current entrypoints, retained contracts and top-level audit reports. Historical snapshots and third-party citations are not rewritten.'}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    if missing:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
