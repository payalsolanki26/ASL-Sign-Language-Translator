"""
Tamper-evident audit log  (Data Security)

Append-only JSON-lines file. Each entry stores the hash of the previous entry, so editing
or deleting a past line breaks the chain and `verify_chain` reports it. Log *events*
(who/what/when) - never images, keys or raw personal data.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return "0" * 64
        last = self.path.read_text().strip().splitlines()[-1]
        return json.loads(last)["hash"]

    def log(self, event: str, **details) -> dict:
        entry = {"ts": round(time.time(), 3), "event": event, "details": details, "prev": self._last_hash()}
        entry["hash"] = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        return entry

    def verify_chain(self) -> tuple[bool, int]:
        prev, n = "0" * 64, 0
        if not self.path.exists():
            return True, 0
        for line in self.path.read_text().splitlines():
            e = json.loads(line)
            h = e.pop("hash")
            if e["prev"] != prev or hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest() != h:
                return False, n
            prev, n = h, n + 1
        return True, n
