"""
Small persistent state: which emails Alfred has already handled, plus items
waiting for the next digest.

Plain JSON, written atomically (temp file + rename) so a crash or power cut
can't leave a half-written file. If the file is ever unreadable it is moved
aside as <name>.corrupt and Alfred starts fresh. For the email monitor that
is safe: a fresh start re-baselines the inbox instead of re-announcing
everything.

Privacy note: pending digest entries contain sender, subject and a summary of
real emails. Keep this file out of git (see .gitignore) and out of cloud-synced
folders if you can.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

log = logging.getLogger("alfred.state")

MAX_SEEN_PER_MAILBOX = 2000


class StateStore:
    def __init__(self, path: str | None = None) -> None:
        """path=None keeps everything in memory (tests, dry runs)."""
        self._path = path
        self._d: dict[str, Any] = self._empty()
        if path and os.path.exists(path):
            self._load()

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"mailboxes": {}, "pending": [], "junk_ignored": 0, "last_digest_date": None}

    def _load(self) -> None:
        assert self._path
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            if not (
                isinstance(data, dict)
                and isinstance(data.get("mailboxes", {}), dict)
                and isinstance(data.get("pending", []), list)
            ):
                raise ValueError("unexpected structure")
        except (OSError, ValueError) as exc:
            log.warning("state file %s unreadable (%s); starting fresh", self._path, exc)
            try:
                os.replace(self._path, self._path + ".corrupt")
            except OSError:
                pass
            return
        for key in self._d:
            if key in data:
                self._d[key] = data[key]

    def save(self) -> None:
        if not self._path:
            return
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._d, f, indent=2)
        os.replace(tmp, self._path)

    # -- seen emails ------------------------------------------------------
    def mailbox(self, mailbox_id: str) -> dict[str, Any] | None:
        return self._d["mailboxes"].get(mailbox_id)

    def reset_mailbox(self, mailbox_id: str, uidvalidity: str | None) -> None:
        self._d["mailboxes"][mailbox_id] = {"uidvalidity": uidvalidity, "seen": []}

    def is_seen(self, mailbox_id: str, uid: str) -> bool:
        box = self.mailbox(mailbox_id)
        return bool(box) and uid in box["seen"]

    def mark_seen(self, mailbox_id: str, uid: str) -> None:
        box = self.mailbox(mailbox_id)
        if box is None:
            raise KeyError(f"unknown mailbox {mailbox_id!r}; call reset_mailbox first")
        if uid not in box["seen"]:
            box["seen"].append(uid)
            overflow = len(box["seen"]) - MAX_SEEN_PER_MAILBOX
            if overflow > 0:
                del box["seen"][:overflow]

    # -- digest -----------------------------------------------------------
    def add_pending(self, entry: dict[str, Any]) -> None:
        self._d["pending"].append(entry)

    @property
    def pending(self) -> list[dict[str, Any]]:
        return list(self._d["pending"])

    def add_junk(self) -> None:
        self._d["junk_ignored"] += 1

    @property
    def junk_ignored(self) -> int:
        return self._d["junk_ignored"]

    @property
    def last_digest_date(self) -> str | None:
        return self._d["last_digest_date"]

    def digest_sent(self, date_iso: str) -> None:
        """Call only after the digest was actually delivered."""
        self._d["pending"] = []
        self._d["junk_ignored"] = 0
        self._d["last_digest_date"] = date_iso
