"""Tamper-evident audit log.

Each entry stores the SHA-256 hash of the previous entry, so editing,
deleting or reordering any line breaks the chain from that point on.
`verify()` re-computes the chain and reports the first broken entry.
Free text is redacted (account numbers masked) before it is written.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dorsal.security.guard import redact

GENESIS = "0" * 64


def _clean(o):
    if isinstance(o, str):
        return redact(o)
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return o


def _hash(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: Optional[Path | str] = None):
        self.path = Path(path) if path else None
        self.entries: list[dict] = []
        if self.path and self.path.exists():
            self.entries = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def append(self, actor: str, event: str, data: dict | None = None) -> dict:
        prev = self.entries[-1]["hash"] if self.entries else GENESIS
        entry = {"seq": len(self.entries) + 1, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "actor": actor, "event": event, "data": _clean(data or {}), "prev": prev}
        entry["hash"] = _hash(entry)
        self.entries.append(entry)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        return entry

    @staticmethod
    def verify_entries(entries: list[dict]) -> dict:
        prev = GENESIS
        for i, e in enumerate(entries):
            if e.get("prev") != prev or e.get("hash") != _hash(e) or e.get("seq") != i + 1:
                return {"ok": False, "first_bad_seq": e.get("seq", i + 1), "checked": i}
            prev = e["hash"]
        return {"ok": True, "checked": len(entries), "head": prev}

    def verify(self) -> dict:
        return self.verify_entries(self.entries)

    @classmethod
    def verify_file(cls, path: Path | str) -> dict:
        entries = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        return cls.verify_entries(entries)
