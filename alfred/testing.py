"""Fakes for tests and offline demos: scripted provider, scripted channel, simple tool."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from alfred.core.interfaces import (
    ApprovalRequest,
    Completion,
    Inbound,
    Message,
    RiskTier,
    ToolCall,
    ToolResult,
    ToolSpec,
    Urgency,
)


def say(text: str) -> Completion:
    return Completion(Message("assistant", text))


def use(tool: str, reason: str = "", **arguments: Any) -> Completion:
    return Completion(Message("assistant", "", [ToolCall(tool, arguments, reason)]))


class FakeProvider:
    name = "fake"
    is_local = True

    def __init__(self, script: list[Completion]) -> None:
        self._script = list(script)
        self.seen: list[list[Message]] = []   # snapshot of messages at each call
        self.calls: list[dict] = []           # {"tools": ..., "json_schema": ...} per call

    async def generate(self, messages, tools=None, json_schema=None) -> Completion:
        self.seen.append(list(messages))
        self.calls.append({"tools": tools, "json_schema": json_schema})
        if not self._script:
            raise AssertionError("FakeProvider script exhausted")
        return self._script.pop(0)

    def stream(self, messages) -> AsyncIterator[str]:
        raise NotImplementedError


class FakeChannel:
    name = "fake"

    def __init__(self, answers: list[str] | None = None, delay: float = 0.0) -> None:
        self._answers = list(answers or [])
        self._delay = delay
        self.texts: list[tuple[str, Urgency]] = []
        self.approvals: list[ApprovalRequest] = []

    async def send_text(self, text: str, urgency: Urgency = Urgency.NORMAL) -> None:
        self.texts.append((text, urgency))

    async def send_voice(self, audio: bytes, mime: str) -> None:
        pass

    async def request_approval(self, req: ApprovalRequest) -> str:
        self.approvals.append(req)
        if self._delay:
            await asyncio.sleep(self._delay)
        if not self._answers:
            raise AssertionError("FakeChannel has no scripted answer left")
        return self._answers.pop(0)

    async def incoming(self) -> AsyncIterator[Inbound]:
        return
        yield  # pragma: no cover  (makes this an async generator)


class FakeTool:
    def __init__(
        self,
        name: str,
        risk: RiskTier = RiskTier.READ_ONLY,
        result: ToolResult | Exception | None = None,
        domain: str = "general",
        properties: dict[str, Any] | None = None,
        required: list[str] | None = None,
    ) -> None:
        self.spec = ToolSpec(
            name=name,
            description=f"fake tool {name}",
            parameters={
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
            risk=risk,
            domain=domain,
        )
        self._result = result if result is not None else ToolResult(ok=True, data="done")
        self.calls: list[dict[str, Any]] = []

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append(arguments)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result
