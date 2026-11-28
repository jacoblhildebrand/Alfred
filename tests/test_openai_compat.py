"""Provider tests against a throwaway local HTTP server (no Ollama needed)."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from alfred.core.audit import InMemoryAuditLog
from alfred.core.interfaces import Message, RiskTier, ToolCall, ToolResult
from alfred.core.orchestrator import Orchestrator, OrchestratorConfig
from alfred.core.policy import DefaultPolicyEngine
from alfred.core.registry import Registry
from alfred.providers.openai_compat import (
    UNPARSEABLE_KEY,
    OpenAICompatProvider,
    ProviderError,
)
from alfred.testing import FakeChannel, FakeTool


class FakeServer:
    """Serves scripted JSON responses and records every request it receives."""

    def __init__(self, responses, status=200):
        self.responses = list(responses)
        self.requests = []   # (headers, body)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                outer.requests.append((dict(self.headers), json.loads(self.rfile.read(length))))
                payload = json.dumps(outer.responses.pop(0)).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


def reply(content="", tool_calls=None, **extra):
    msg = {"role": "assistant", "content": content, **extra}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg}], "usage": {"prompt_tokens": 11, "completion_tokens": 7}}


def wire_call(name, args, call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


SPEC_TOOL = FakeTool("email.send", RiskTier.EXTERNAL,
                     properties={"to": {"type": "string"}}, required=["to"])


class RequestShape(unittest.IsolatedAsyncioTestCase):
    async def test_request_and_response_mapping(self):
        with FakeServer([reply("hi")]) as srv:
            p = OpenAICompatProvider("qwen3.8", base_url=srv.url, api_key="secret")
            out = await p.generate(
                [Message("system", "s"), Message("user", "u")],
                tools=[SPEC_TOOL.spec],
                json_schema={"type": "object"},
            )
        headers, body = srv.requests[0]
        self.assertEqual(body["model"], "qwen3.8")
        self.assertFalse(body["stream"])
        self.assertEqual(headers["Authorization"], "Bearer secret")
        self.assertEqual(body["tools"][0]["function"]["name"], "email__send")  # dots mapped
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(out.message.content, "hi")
        self.assertEqual((out.input_tokens, out.output_tokens), (11, 7))

    async def test_extra_body_is_merged(self):
        with FakeServer([reply("ok")]) as srv:
            p = OpenAICompatProvider("m", base_url=srv.url, extra_body={"temperature": 0.1})
            await p.generate([Message("user", "u")])
        self.assertEqual(srv.requests[0][1]["temperature"], 0.1)

    async def test_tool_history_is_serialised_with_ids(self):
        history = [
            Message("user", "go"),
            Message("assistant", "", [ToolCall("email.send", {"to": "a@x.com"}, id="call_9")]),
            Message("tool", "sent", tool_call_id="call_9"),
        ]
        with FakeServer([reply("done")]) as srv:
            await OpenAICompatProvider("m", base_url=srv.url).generate(history)
        sent = srv.requests[0][1]["messages"]
        self.assertEqual(sent[1]["tool_calls"][0]["id"], "call_9")
        self.assertEqual(sent[1]["tool_calls"][0]["function"]["name"], "email__send")
        self.assertEqual(json.loads(sent[1]["tool_calls"][0]["function"]["arguments"]), {"to": "a@x.com"})
        self.assertEqual(sent[2], {"role": "tool", "content": "sent", "tool_call_id": "call_9"})


class ResponseParsing(unittest.IsolatedAsyncioTestCase):
    async def _gen(self, response):
        with FakeServer([response]) as srv:
            return await OpenAICompatProvider("m", base_url=srv.url).generate([Message("user", "u")])

    async def test_tool_call_name_mapped_back_and_args_parsed(self):
        out = await self._gen(reply("", [wire_call("email__send", '{"to": "a@x.com"}', "call_7")]))
        call = out.message.tool_calls[0]
        self.assertEqual((call.tool, call.arguments, call.id), ("email.send", {"to": "a@x.com"}, "call_7"))

    async def test_arguments_may_already_be_a_dict(self):
        out = await self._gen(reply("", [wire_call("email__send", {"to": "a@x.com"})]))
        self.assertEqual(out.message.tool_calls[0].arguments, {"to": "a@x.com"})

    async def test_missing_call_id_is_filled_in(self):
        raw = {"type": "function", "function": {"name": "email__send", "arguments": "{}"}}
        out = await self._gen(reply("", [raw]))
        self.assertTrue(out.message.tool_calls[0].id.startswith("call_"))

    async def test_malformed_arguments_are_marked_not_guessed(self):
        for bad in ['{"to": "a@x.com"', "not json", "[1, 2]"]:
            with self.subTest(bad=bad):
                out = await self._gen(reply("", [wire_call("email__send", bad)]))
                self.assertIn(UNPARSEABLE_KEY, out.message.tool_calls[0].arguments)

    async def test_think_blocks_and_reasoning_fields_are_dropped(self):
        out = await self._gen(
            reply("<think>plan the reply\nsecretly</think>Here you go.", reasoning="more thinking",
                  reasoning_content="and more")
        )
        self.assertEqual(out.message.content, "Here you go.")

    async def test_odd_response_shape_raises(self):
        with self.assertRaises(ProviderError):
            await self._gen({"unexpected": True})


class Errors(unittest.IsolatedAsyncioTestCase):
    async def test_http_error_becomes_provider_error(self):
        with FakeServer([{"error": "model not found"}], status=404) as srv:
            p = OpenAICompatProvider("missing", base_url=srv.url)
            with self.assertRaises(ProviderError) as cm:
                await p.generate([Message("user", "u")])
        self.assertIn("404", str(cm.exception))

    async def test_connection_refused_becomes_provider_error(self):
        p = OpenAICompatProvider("m", base_url="http://127.0.0.1:1/v1", timeout_s=2)
        with self.assertRaises(ProviderError):
            await p.generate([Message("user", "u")])


class EndToEnd(unittest.IsolatedAsyncioTestCase):
    async def test_orchestrator_with_real_provider_round_trips_call_ids(self):
        cal = FakeTool("calendar.read", result=ToolResult(True, data="2pm dentist"))
        registry = Registry()
        registry.register(cal)
        responses = [
            reply("", [wire_call("calendar__read", "{}", "call_abc")]),
            reply("<think>ok</think>You have a dentist visit at 2pm."),
        ]
        with FakeServer(responses) as srv:
            orch = Orchestrator(
                OpenAICompatProvider("qwen3.8", base_url=srv.url),
                FakeChannel(), registry, DefaultPolicyEngine(), InMemoryAuditLog(),
                OrchestratorConfig(),
            )
            answer = await orch.run_task("what's on my calendar?")
        self.assertEqual(answer, "You have a dentist visit at 2pm.")
        self.assertEqual(len(cal.calls), 1)
        second_request = srv.requests[1][1]["messages"]
        tool_msg = [m for m in second_request if m["role"] == "tool"][0]
        self.assertEqual(tool_msg["tool_call_id"], "call_abc")
        self.assertEqual(tool_msg["content"], "2pm dentist")


if __name__ == "__main__":
    unittest.main()
