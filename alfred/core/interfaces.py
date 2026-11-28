"""
Alfred core interfaces (sketch).

Everything else in the project (email skill, Telegram adapter, Ollama
provider, Whisper STT, ...) is an implementation of one of these.
Suggested location: alfred/core/interfaces.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, AsyncIterator, Protocol, runtime_checkable


# --------------------------------------------------------------------------
# Risk and autonomy
# --------------------------------------------------------------------------
class RiskTier(IntEnum):
    """Every tool action declares one of these. Higher = more dangerous."""
    READ_ONLY = 0      # check calendar, read email
    REVERSIBLE = 1     # create draft, add calendar event
    EXTERNAL = 2       # send email, post message, anything others can see
    IRREVERSIBLE = 3   # spend money, delete data


class Autonomy(str, Enum):
    """User-facing presets. Each maps to the highest tier allowed without asking."""
    ASK_EVERYTHING = "ask_everything"        # approve every action
    ASK_EXTERNAL = "ask_external"            # free up to REVERSIBLE
    ACT_THEN_TELL = "act_then_tell"          # free up to EXTERNAL, summary after
    FULL_AUTO = "full_auto"                  # free up to IRREVERSIBLE, within hard limits


class Decision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


# --------------------------------------------------------------------------
# Untrusted content (emails, web pages, anything from outside)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Untrusted:
    """Wrap external text so it can never be mistaken for an instruction."""
    text: str
    source: str  # e.g. "email:inbox:msg-1234"


# --------------------------------------------------------------------------
# Tools / skills
# --------------------------------------------------------------------------
@dataclass
class ToolSpec:
    name: str                       # "email.read_inbox"
    description: str
    parameters: dict[str, Any]      # JSON Schema; also used to validate model output
    risk: RiskTier
    domain: str = "general"         # lets users override autonomy per domain


@dataclass
class ToolCall:
    tool: str
    arguments: dict[str, Any]
    reason: str = ""                # model's stated rationale, shown in approvals
    id: str = ""                    # provider-assigned call id; tool results echo it back


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None
    untrusted: bool = False         # True if data came from outside (emails, web)


@runtime_checkable
class Tool(Protocol):
    spec: ToolSpec

    async def run(self, arguments: dict[str, Any]) -> ToolResult: ...


class SkillRegistry(Protocol):
    def register(self, tool: Tool) -> None: ...
    def get(self, name: str) -> Tool: ...
    def specs(self) -> list[ToolSpec]: ...


# A skill is a module exposing: def register(registry: SkillRegistry) -> None


# --------------------------------------------------------------------------
# LLM providers (Ollama, llama.cpp, OpenAI-compatible, Anthropic, Gemini, ...)
# --------------------------------------------------------------------------
@dataclass
class Message:
    role: str                       # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set on role="tool" messages: which call this answers


@dataclass
class Completion:
    message: Message
    input_tokens: int = 0
    output_tokens: int = 0          # feeds the budget tracker for paid providers


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    is_local: bool                  # lets privacy rules block cloud routing

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        json_schema: dict[str, Any] | None = None,   # constrained output
    ) -> Completion: ...

    def stream(self, messages: list[Message]) -> AsyncIterator[str]: ...


# --------------------------------------------------------------------------
# Speech
# --------------------------------------------------------------------------
@runtime_checkable
class SpeechToText(Protocol):
    async def transcribe(self, audio: bytes, mime: str) -> str: ...


@runtime_checkable
class TextToSpeech(Protocol):
    async def synthesize(self, text: str) -> tuple[bytes, str]:
        """Return (audio_bytes, mime_type)."""
        ...


# --------------------------------------------------------------------------
# Channels (Telegram, Discord, SMS, voice call, CLI, ...)
# --------------------------------------------------------------------------
class Urgency(IntEnum):
    LOW = 0        # batch into a digest
    NORMAL = 1     # push/chat message
    URGENT = 2     # escalate: SMS, then call


@dataclass
class Inbound:
    text: str | None = None
    audio: bytes | None = None
    audio_mime: str | None = None
    reply_to: str | None = None     # id of an approval request, if answering one


@dataclass
class ApprovalRequest:
    id: str
    summary: str                    # plain-language description of the action
    call: ToolCall
    options: tuple[str, ...] = ("Approve", "Deny", "Edit")


@runtime_checkable
class Channel(Protocol):
    name: str

    async def send_text(self, text: str, urgency: Urgency = Urgency.NORMAL) -> None: ...
    async def send_voice(self, audio: bytes, mime: str) -> None: ...
    async def request_approval(self, req: ApprovalRequest) -> str:
        """Block until the user answers; return the chosen option."""
        ...
    def incoming(self) -> AsyncIterator[Inbound]: ...


# --------------------------------------------------------------------------
# Policy: the one gate every tool call passes through (enforced in code,
# never by the prompt)
# --------------------------------------------------------------------------
@dataclass
class PolicyContext:
    autonomy: Autonomy
    domain_overrides: dict[str, Autonomy] = field(default_factory=dict)
    blocked_targets: set[str] = field(default_factory=set)    # contacts, paths
    spend_cap_remaining: float | None = None
    triggered_by_untrusted: bool = False   # call originated from email/web content


class PolicyEngine(Protocol):
    def decide(self, call: ToolCall, spec: ToolSpec, ctx: PolicyContext) -> Decision:
        """Pure function of its inputs: easy to unit test. Hard limits -> DENY first."""
        ...


# --------------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------------
@dataclass
class AuditEntry:
    timestamp: float
    call: ToolCall
    decision: Decision
    approved_by: str | None         # "policy" or channel name
    result: ToolResult | None = None


class AuditLog(Protocol):
    def record(self, entry: AuditEntry) -> None: ...
    def recent(self, limit: int = 50) -> list[AuditEntry]: ...
