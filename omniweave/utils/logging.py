"""JSONL logger and canonical experiment hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    """Append-only JSONL file logger — one valid JSON record per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        """Write one JSON record as a single line."""
        self._handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def experiment_id(
    config: dict[str, Any],
    source_revision: str,
    dataset_hash: str,
) -> str:
    """Generate a canonical 16-character experiment hash.

    Deterministic: same inputs always produce the same ID.
    """
    payload = json.dumps(
        {
            "config": config,
            "source_revision": source_revision,
            "dataset_hash": dataset_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]
