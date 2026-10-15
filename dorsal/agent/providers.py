"""Language-model providers.

* ``offline``   - no model, no network. A deterministic playbook drives the
                  same tools, so judges can run the full system with no key.
* ``anthropic`` - Claude via the Anthropic Messages API (tool use).
* ``openai``    - any OpenAI-compatible endpoint (e.g. DeepSeek, Qwen,
                  OpenAI) via base_url, for deployments where Anthropic is
                  not reachable.

Select with DORSAL_LLM=offline|anthropic|openai. Keys come only from the
environment (ANTHROPIC_API_KEY / OPENAI_API_KEY); they are never logged.
"""
from __future__ import annotations

import json
import os
from typing import Any


class Provider:
    name = "base"

    def start(self, system: str, user: str) -> dict:
        raise NotImplementedError

    def step(self, state: dict, tools: list) -> tuple[str, list[dict]]:
        raise NotImplementedError

    def add_results(self, state: dict, calls: list[dict], results: list[str]) -> None:
        raise NotImplementedError


class OfflineProvider(Provider):
    """Marker provider: the agent uses its deterministic playbook instead."""
    name = "offline"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str | None = None):
        import anthropic  # imported lazily so offline mode needs no SDK
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("DORSAL_ANTHROPIC_MODEL", "claude-sonnet-5-5")

    def start(self, system: str, user: str) -> dict:
        return {"system": system, "messages": [{"role": "user", "content": user}]}

    def step(self, state: dict, tools: list) -> tuple[str, list[dict]]:
        resp = self.client.messages.create(
            model=self.model, max_tokens=2000, system=state["system"], messages=state["messages"],
            tools=[{"name": t.name, "description": t.description, "input_schema": t.schema} for t in tools])
        content, text, calls = [], [], []
        for b in resp.content:
            if b.type == "text":
                content.append({"type": "text", "text": b.text})
                text.append(b.text)
            elif b.type == "tool_use":
                content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
                calls.append({"id": b.id, "name": b.name, "input": b.input or {}})
        state["messages"].append({"role": "assistant", "content": content})
        return "\n".join(text), calls

    def add_results(self, state: dict, calls: list[dict], results: list[str]) -> None:
        state["messages"].append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": c["id"], "content": r} for c, r in zip(calls, results)]})


class OpenAICompatProvider(Provider):
    name = "openai"

    def __init__(self, model: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(base_url=os.environ.get("DORSAL_OPENAI_BASE_URL") or None)
        self.model = model or os.environ.get("DORSAL_OPENAI_MODEL", "gpt-4o-mini")

    def start(self, system: str, user: str) -> dict:
        return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}

    def step(self, state: dict, tools: list) -> tuple[str, list[dict]]:
        resp = self.client.chat.completions.create(
            model=self.model, messages=state["messages"],
            tools=[{"type": "function", "function": {"name": t.name, "description": t.description,
                                                     "parameters": t.schema}} for t in tools])
        msg = resp.choices[0].message
        calls = []
        entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = [{"id": tc.id, "type": "function",
                                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                   for tc in msg.tool_calls]
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append({"id": tc.id, "name": tc.function.name, "input": args})
        state["messages"].append(entry)
        return msg.content or "", calls

    def add_results(self, state: dict, calls: list[dict], results: list[str]) -> None:
        for c, r in zip(calls, results):
            state["messages"].append({"role": "tool", "tool_call_id": c["id"], "content": r})


def get_provider(name: str | None = None) -> Provider:
    name = (name or os.environ.get("DORSAL_LLM", "offline")).lower()
    if name == "anthropic":
        return AnthropicProvider()
    if name in ("openai", "deepseek", "qwen"):
        return OpenAICompatProvider()
    return OfflineProvider()
