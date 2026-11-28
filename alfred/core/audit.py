"""Audit log. In-memory for now; swap in a SQLite version behind the same Protocol."""
from __future__ import annotations

from alfred.core.interfaces import AuditEntry


class InMemoryAuditLog:
    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def record(self, entry: AuditEntry) -> None:
        self._entries.append(entry)

    def recent(self, limit: int = 50) -> list[AuditEntry]:
        """Newest first."""
        return list(reversed(self._entries[-limit:]))
