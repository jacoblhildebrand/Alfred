"""Read-only email skill (IMAP). Usage:

    from alfred.skills.email_imap import register, EmailConfig
    register(registry, EmailConfig.from_env())
"""
from __future__ import annotations

from typing import Any, Callable

from alfred.skills.email_imap.client import EmailConfig, ImapClient, MailError
from alfred.skills.email_imap.tools import ListInboxTool, ReadEmailTool

__all__ = ["register", "EmailConfig", "ImapClient", "MailError"]


def register(registry: Any, config: EmailConfig, connect: Callable[[EmailConfig], Any] | None = None) -> None:
    client = ImapClient(config, connect)
    registry.register(ListInboxTool(client, config.max_list))
    registry.register(ReadEmailTool(client))
