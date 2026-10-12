"""Actual OpenAI SDK with a scripted raw DS4 transport and real node runtime."""

import json
from html import escape
from uuid import uuid4

import httpx2 as httpx
from openai import OpenAI

from exmekh_lib.core.agents.deep_agent.stream_event import DeepAgentEvent
from exmekh_lib.core.agents.node import DefaultCapabilities, Node
from exmekh_lib.interfaces.event_handler import DeepAgentEventHandler
from exmekh_lib.providers.local import LocalProvider
from tests.live.model_support import CacheTestTokenizer


def response(*calls, text=None, continuation=None):
    output = [
        {
            "type": "function_call",
            "id": f"item-{call_id}",
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(arguments),
            "status": "completed",
        }
        for call_id, name, arguments in calls
    ]
    if text is not None:
        output.append(
            {
                "type": "message",
                "id": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    value = {
        "id": f"response-{uuid4().hex}",
        "object": "response",
        "created_at": 0,
        "model": "model",
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "output": output,
        "parallel_tool_calls": False,
        "temperature": 1,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    if continuation is not None:
        value["interactive"] = {"version": 2, "continuation_id": continuation}
    return value


def pause(
    path="notes.txt",
    *,
    name="edit-file",
    call_id="edit",
    continuation="cont-edit",
    exchange="exchange-edit",
    fields=None,
    edit_index=1,
):
    return {
        "type": "interactive.await_input",
        "version": 2,
        "status": "awaiting_input",
        "continuation_id": continuation,
        "exchange_id": exchange,
        "call_id": call_id,
        "tool_name": name,
        "behavior": "edit_file.v2",
        "request": {
            "kind": "file_snapshot",
            "edit_index": edit_index,
            "values": {"path": path} if fields is None else fields,
        },
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


CAPABILITIES = {
    "interactive": {
        "version": 2,
        "apis": ["responses"],
        "behaviors": ["edit_file.v2"],
        "model_syntax": "qwen27",
        "syntaxes": ["qwen27"],
        "model_supported": True,
    }
}


class ScriptedServer:
    def __init__(self, steps, *, streaming=False):
        self.steps = list(steps)
        self.requests = []
        self.streaming = streaming
        self.capabilities = CAPABILITIES
        self.client = OpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(self.handle)),
        )

    def handle(self, request):
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "model",
                            "object": "model",
                            "created": 0,
                            "owned_by": "test",
                        }
                    ],
                },
            )
        if request.url.path == "/v1/agentic/capabilities":
            return httpx.Response(200, json=self.capabilities)
        assert request.url.path == "/v1/responses"
        body = json.loads(request.content)
        self.requests.append(body)
        assert self.steps, f"Unexpected inference/control: {body}"
        step = self.steps.pop(0)
        value = step(body) if callable(step) else step
        if isinstance(value, Exception):
            raise value
        if isinstance(value, httpx.Response):
            return value
        if self.streaming and body.get("stream", True):
            events = []
            if "interactive" in value:
                events.append({"type": "interactive.accepted", **value["interactive"]})
                events.append({"type": "response.completed", "response": value})
            elif value.get("object") == "response":
                events.append({"type": "response.completed", "response": value})
            else:
                events.append(value)
            wire = "".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                for event in events
            )
            return httpx.Response(
                200,
                text=wire + "data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=value)


def make_sdk_node(tmp_path, monkeypatch, steps, *, streaming=False, hooks=None):
    server = ScriptedServer(steps, streaming=streaming)
    monkeypatch.setattr(LocalProvider, "_build_client", lambda self: server.client)
    provider = LocalProvider(
        "model",
        api="responses",
        base_url="http://test/v1",
        tokenizer=CacheTestTokenizer("interactive-test", tmp_path / "cache"),
    )
    hds = tmp_path / "capabilities" / "hds"
    hds.mkdir(parents=True)
    (hds / "main.md").write_text(
        "---\ndescription: Edit files.\nexit-tool: exit_agent\nsub-skills:\n  - file-processing\n---\nUse {{!file-processing}}."
    )
    node = Node(
        [hds.parent],
        name_hds="main",
        agent_context_id=f"test-{uuid4().hex}",
        default_capabilities=DefaultCapabilities(
            harness=False,
            include={
                "file-processing",
                "inspect-file",
                "read-file",
                "edit-file",
                "exit_agent",
            },
            tool_hooks=hooks,
        ),
    )
    if streaming:

        class Events(DeepAgentEventHandler):
            def handle(self, event: DeepAgentEvent) -> None:
                pass

        node.initialize(provider, work_root=tmp_path, on_stream=Events())
    else:
        node.initialize(provider, work_root=tmp_path)
    return node, server, provider


def enter():
    return response(("open", "file-processing", {}))


def inspect():
    return response(
        ("inspect", "inspect-file", {"path": "notes.txt"}), continuation="cont-inspect"
    )


def read(start=1, end=1):
    return response(
        (
            "read",
            "read-file",
            {"path": "notes.txt", "start_line": start, "end_line": end},
        ),
        continuation="cont-read",
    )


def exit_skill():
    return response(
        ("exit", "exit_agent", {"text": "done", "items": []}), continuation="cont-exit"
    )


def line_tag(number):
    return f'<line number="{number}"/>'


def edit_fragment(path, old, new):
    old_body = (
        old
        if isinstance(old, str) and old.startswith("<")
        else escape(old)
        if isinstance(old, str)
        else str(old)
    )
    new_body = "<delete/>" if new is None else escape(new)
    return f'<edit path="{escape(path, quote=True)}"><old_text>{old_body}</old_text><new_text>{new_body}</new_text></edit>'


def edit_arguments(path, old, new):
    return {"edits": edit_fragment(path, old, new)}
