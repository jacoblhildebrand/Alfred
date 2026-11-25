"""Terminal channel: handy for development, and a fallback for users with no chat app."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable

from alfred.core.interfaces import ApprovalRequest, Inbound, Urgency


class CliChannel:
    name = "cli"

    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self._in = input_fn
        self._out = output_fn

    async def send_text(self, text: str, urgency: Urgency = Urgency.NORMAL) -> None:
        tag = "!" if urgency is Urgency.URGENT else ""
        self._out(f"[alfred{tag}] {text}")

    async def send_voice(self, audio: bytes, mime: str) -> None:
        self._out(f"[alfred] (voice message: {len(audio)} bytes, {mime})")

    async def request_approval(self, req: ApprovalRequest) -> str:
        self._out(f"\n[alfred] Approval needed: {req.summary}")
        for i, option in enumerate(req.options, 1):
            self._out(f"  {i}) {option}")
        try:
            raw = await asyncio.to_thread(self._in, "choose> ")
        except EOFError:
            return "Deny"
        return self._parse(raw, req.options)

    @staticmethod
    def _parse(raw: str, options: tuple[str, ...]) -> str:
        """Accept a number or an option name; anything else is a denial."""
        raw = raw.strip().lower()
        for i, option in enumerate(options, 1):
            if raw == str(i) or raw == option.lower():
                return option
        return "Deny"

    async def incoming(self) -> AsyncIterator[Inbound]:
        while True:
            try:
                line = await asyncio.to_thread(self._in, "you> ")
            except EOFError:
                return
            yield Inbound(text=line)
