"""Exclude code and LaTeX/math when recognizing rendered Markdown links."""
from __future__ import annotations

import re


def markdown_code_ranges(text: str) -> list[tuple[int, int]]:
    ranges = []
    fence = None
    offset = 0
    for line in text.splitlines(keepends=True):
        match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence is None and match:
            fence = (match.group(1)[0], len(match.group(1)), offset)
        elif fence is not None and re.match(r"^ {0,3}" + re.escape(fence[0]) + "{" + str(fence[1]) + r",}\s*$", line):
            ranges.append((fence[2], offset + len(line)))
            fence = None
        offset += len(line)
    if fence is not None:
        ranges.append((fence[2], len(text)))
    for match in re.finditer(r"(?<!`)(`+)(?!`)(.*?)(?<!`)\1(?!`)", text, re.S):
        if not any(start <= match.start() < end for start, end in ranges):
            ranges.append(match.span())
    return sorted(ranges)


def inside_code(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start < stop and end > begin for begin, stop in ranges)


def markdown_protected_ranges(text: str) -> list[tuple[int, int]]:
    ranges = markdown_code_ranges(text)
    for pattern in (r"\\\[[\s\S]*?\\\]", r"\\\([\s\S]*?\\\)",
                    r"(?<!\\)\$\$[\s\S]*?(?<!\\)\$\$",
                    r"(?<![\\$])\$(?!\$)(?:\\.|[^$\\])*?\$(?!\$)"):
        ranges.extend(match.span() for match in re.finditer(pattern, text))
    return sorted(ranges)
