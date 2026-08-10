#!/usr/bin/env python3
"""Credential-free deterministic checks for the parallel continuation driver."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import parallel_complete as subject


class FakeCompletionError(RuntimeError):
    pass


class FakeSession:
    def get(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


class FakeClient:
    def __init__(self, _api_key: str) -> None:
        self.session = FakeSession()

    def get_entries_in_chemsys(
        self, chemsys: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self.session.get("mock://thermo")
        time.sleep(0.002 * (1 + len(chemsys) % 3))
        return [{"chemsys": chemsys}], {"api_pages": 1}


class FakeModule:
    CompletionError = FakeCompletionError
    CurrentMPThermoClient = FakeClient

    @staticmethod
    def slim_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chemsys = str(rows[0]["chemsys"])
        return [
            {
                "entry_id": f"fake-{chemsys}",
                "composition": {element: 1.0 for element in chemsys.split("-")},
                "energy": -1.0,
            }
        ]

    @staticmethod
    def sanitized_query_error(exc: BaseException) -> dict[str, Any]:
        return {
            "type": type(exc).__name__,
            "http_status": None,
            "message_serialized": False,
        }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    missing = [f"A-{index:02d}" for index in range(1, 13)]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint = root / "serial"
        checkpoint.mkdir()
        serial_fragments = [
            {
                "chemsys": chemsys,
                "entries": [
                    {
                        "entry_id": f"serial-{chemsys}",
                        "composition": {"A": 1.0},
                        "energy": -1.0,
                    }
                ],
            }
            for chemsys in missing[:3]
        ]
        serial_progress = [
            {
                "schema": subject.SERIAL_SCHEMA,
                "query_index": index,
                "query_total": len(missing),
                "chemsys": chemsys,
                "status": "resolved",
                "entry_count": 1,
                "transport_attempts": 1,
                "transport_retries": 0,
            }
            for index, chemsys in enumerate(missing[:3], start=1)
        ]
        write_jsonl(checkpoint / "mp_query_fragment.jsonl", serial_fragments)
        write_jsonl(checkpoint / "mp_query_progress.jsonl", serial_progress)
        queried, progress, report = subject.validate_serial_prefix(checkpoint, missing)
        assert set(queried) == set(missing[:3])
        assert len(progress) == 3
        assert report["resolved_prefix_count"] == 3

        spool = root / "spool"
        spool.mkdir()
        remaining = list(enumerate(missing[3:], start=4))
        parallel_queried, parallel_progress = subject.query_remaining(
            module=FakeModule,
            api_key="x" * 32,
            remaining=remaining,
            total=len(missing),
            maximum_attempts=5,
            spool=spool,
        )
        assert set(parallel_queried) == set(missing[3:])
        assert [row["query_index"] for row in parallel_progress] == list(range(4, 13))
        assert len(list(spool.glob("query_*.json"))) == 9
        assert all(row["status"] == "resolved" for row in parallel_progress)

    print(
        json.dumps(
            {
                "schema": "h1_plan1200_mp_parallel_self_test_v1",
                "status": "pass",
                "workers": subject.WORKERS,
                "max_requests_per_second": subject.MAX_REQUESTS_PER_SECOND,
                "credential_used": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
