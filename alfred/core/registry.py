"""Minimal tool registry. Skills call register() on it at startup."""
from __future__ import annotations

from alfred.core.interfaces import Tool, ToolSpec


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if name in self._tools:
            raise ValueError(f"tool {name!r} is already registered")
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]  # KeyError if unknown; the orchestrator handles it

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]
