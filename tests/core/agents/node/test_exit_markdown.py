from pathlib import Path

import pytest

from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.cache import SessionCache
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.tool_message import ToolMessage
from tests.core.agents.node.test_node import (
    SequenceProvider,
    _write_nested_capabilities,
)
from tests.core.agents.node.test_tool_executions import tool_message
from tests.event_handler import as_agent_handler


@pytest.mark.parametrize("streaming", [False, True], ids=["ordinary", "streaming"])
def test_exit_markdown_reaches_only_the_active_skill(tmp_path: Path, streaming):
    root = _write_nested_capabilities(tmp_path)
    token = "EXIT-MARKDOWN-7319"
    (root / "exit-tool" / "finish" / "instructions.md").write_text(
        f"Finish by calling {{{{!exit_agent}}}} with text='{token}' and items=[].",
        encoding="utf-8",
    )
    provider = SequenceProvider(
        [
            tool_message("mock-skill", {}, "open"),
            tool_message("exit_agent", {"text": token, "items": []}, "exit"),
            AssistantMessage("done"),
        ]
    )
    with SessionCache(tmp_path / "cache") as session:
        provider.get_session = lambda: session
        provider.generate_id = lambda: "skill-output"
        node = Node([root], name_hds="main", persistent_history=False)
        node.initialize(
            provider,
            work_root=tmp_path,
            on_stream=as_agent_handler((lambda event: None) if streaming else None),
        )
        assert (
            node.run("Open the skill and follow its exit instructions.").get_message()
            == "done"
        )
        assert node.depth == 0
        record = node._runtime._tool_cache.get("skill-output")
        assert record.result == {"text": token, "items": []}
        assert record.skill_path == "/mock-skill"

    assert len(provider.requests) == 3
    for index, request in enumerate(provider.requests):
        allowed = request.get_extra()["agentic"]["allowed_tools"]
        assert ("exit_agent" in allowed) == (index == 1)
        assert "finish" not in request.get_tools().list_tools()
        assert request.get_streaming() == streaming
    instructions = [
        message.get_message()
        for message in provider.requests[1].get_messages()
        if isinstance(message, ToolMessage) and "<exit-tool" in message.get_message()
    ]
    assert instructions
    assert all(token in text and "{{!" not in text for text in instructions)
    assert token not in provider.requests[0].get_messages()[0].get_message()
