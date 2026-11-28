"""Tool wrappers for the read-only email skill. Everything returned is untrusted."""
from __future__ import annotations

import asyncio
import imaplib
from typing import Any

from alfred.core.interfaces import RiskTier, ToolResult, ToolSpec
from alfred.skills.email_imap.client import ImapClient, MailError

_MAIL_ERRORS = (MailError, imaplib.IMAP4.error, OSError)  # OSError covers network and TLS


class ListInboxTool:
    def __init__(self, client: ImapClient, max_list: int = 20) -> None:
        self._client = client
        self._max = max_list
        self.spec = ToolSpec(
            name="email.list_inbox",
            description=(
                "List recent emails, newest first: uid, sender, subject, date, unread. "
                "Does not include message bodies; use email.read with a uid for that."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": f"How many emails (1-{max_list}). Default 10."},
                    "unread_only": {"type": "boolean", "description": "Only unread emails."},
                },
            },
            risk=RiskTier.READ_ONLY,
            domain="email",
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        limit = max(1, min(int(arguments.get("limit", 10)), self._max))
        unread = bool(arguments.get("unread_only", False))
        try:
            emails = await asyncio.to_thread(self._client.list_recent, limit, unread)
        except _MAIL_ERRORS as exc:
            return ToolResult(ok=False, error=f"mail server problem: {exc}")
        return ToolResult(ok=True, data={"emails": emails}, untrusted=True)


class ReadEmailTool:
    def __init__(self, client: ImapClient) -> None:
        self._client = client
        self.spec = ToolSpec(
            name="email.read",
            description=(
                "Read one email by uid (from email.list_inbox): headers, text body "
                "(truncated if long) and attachment names. Never opens attachments."
            ),
            parameters={
                "type": "object",
                "properties": {"uid": {"type": "string", "description": "The uid of the email."}},
                "required": ["uid"],
            },
            risk=RiskTier.READ_ONLY,
            domain="email",
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            message = await asyncio.to_thread(self._client.read, str(arguments["uid"]))
        except _MAIL_ERRORS as exc:
            return ToolResult(ok=False, error=f"mail server problem: {exc}")
        return ToolResult(ok=True, data=message, untrusted=True)
