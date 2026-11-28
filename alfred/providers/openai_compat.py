"""
Provider for any server that speaks the OpenAI /chat/completions format:
Ollama (default), llama.cpp's llama-server, LM Studio, vLLM, OpenRouter, ...

Standard library only (urllib in a worker thread), so Alfred has no HTTP
dependency to install. Non-streaming for now; stream() comes later.

Things this adapter handles so the rest of Alfred doesn't have to:
  * Tool names: Alfred uses "email.send"; OpenAI-style APIs only allow
    letters, digits, "_" and "-", so dots are sent as "__" and mapped back.
  * Reasoning models (Qwen3.x etc.): reasoning fields are ignored and any
    <think>...</think> block that leaks into the text is stripped. Reasoning
    is never shown to the user or fed back into the conversation.
  * Bad tool-call JSON from small models: the call is returned with a marker
    argument that fails validation, so it is rejected instead of run with
    guessed or empty arguments.
"""
from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.request
import uuid
from typing import Any, AsyncIterator

from alfred.core.interfaces import Completion, Message, ToolCall, ToolSpec

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
UNPARSEABLE_KEY = "_unparseable_arguments"


class ProviderError(RuntimeError):
    pass


def _wire_name(name: str) -> str:
    return name.replace(".", "__")


def _local_name(name: str) -> str:
    return name.replace("__", ".")


def _wire_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": _wire_name(t.name),
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _wire_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or "",
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": _wire_name(c.tool),
                                "arguments": json.dumps(c.arguments),
                            },
                        }
                        for c in m.tool_calls
                    ],
                }
            )
        elif m.role == "tool":
            msg: dict[str, Any] = {"role": "tool", "content": m.content}
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            out.append(msg)
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {UNPARSEABLE_KEY: str(raw)[:200]}
    if not isinstance(parsed, dict):
        return {UNPARSEABLE_KEY: str(raw)[:200]}
    return parsed


def _parse_tool_calls(raw_calls: list[dict[str, Any]] | None) -> list[ToolCall]:
    calls = []
    for rc in raw_calls or []:
        fn = rc.get("function", {})
        calls.append(
            ToolCall(
                tool=_local_name(fn.get("name", "")),
                arguments=_parse_arguments(fn.get("arguments")),
                id=rc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            )
        )
    return calls


class OpenAICompatProvider:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",   # Ollama's default
        api_key: str | None = None,
        is_local: bool = True,
        timeout_s: float = 300.0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        """extra_body is merged into every request, for server-specific options
        (e.g. a flag that turns reasoning down). Check your server's docs."""
        self.name = f"openai-compat:{model}"
        self.is_local = is_local
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._timeout = timeout_s
        self._extra = extra_body or {}

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": _wire_messages(messages),
            "stream": False,
            **self._extra,
        }
        if tools:
            body["tools"] = _wire_tools(tools)
        if json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "alfred", "schema": json_schema, "strict": True},
            }
        data = await asyncio.to_thread(self._post, body)
        return self._parse(data)

    def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        raise NotImplementedError("streaming is not implemented yet")

    # ------------------------------------------------------------------
    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(
            self._url, data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise ProviderError(f"{self.name}: HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise ProviderError(f"{self.name}: request failed: {exc}") from exc

    @staticmethod
    def _parse(data: dict[str, Any]) -> Completion:
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {str(data)[:200]}") from exc
        text = _THINK_BLOCK.sub("", msg.get("content") or "").strip()
        usage = data.get("usage") or {}
        return Completion(
            message=Message("assistant", text, _parse_tool_calls(msg.get("tool_calls"))),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
