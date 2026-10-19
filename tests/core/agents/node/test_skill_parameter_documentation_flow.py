import pytest

from exmekh_lib.core.agents.deep_agent.deep_agent import DeepAgent
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.providers.openai_response_tool_mapper import (
    OpenAIResponseToolMapper,
)
from tests.core.agents.node.steering_support import call, tool_reply
from tests.core.agents.node.test_node import (
    SequenceProvider,
    _write_nested_capabilities,
)


@pytest.mark.parametrize("parameter_descriptions_in_schema", [False, True])
def test_skill_entry_loads_docs_before_tool_calls_and_exit_removes_them(
    tmp_path, parameter_descriptions_in_schema
):
    path = _write_nested_capabilities(tmp_path)
    for name, documentation in (
        ("mock-sub-tool", "Parent parameter documentation."),
        ("nested-sub-tool", "Child parameter documentation."),
    ):
        (path / "tools" / name / "main.py").write_text(
            "from typing import Annotated\n"
            "from exmekh_lib.core.agents.tools.decorator import deep_tool\n"
            f"@deep_tool(name={name!r}, description='Scoped work')\n"
            f"def run(work_root, text: Annotated[str, {documentation!r}]) -> str:\n"
            "    return text\n"
        )
    provider = SequenceProvider(
        [
            tool_reply(call("mock-skill", call_id="parent")),
            tool_reply(call("mock-sub-tool", {"text": "one"}, "work-parent")),
            tool_reply(call("nested-skill", call_id="child")),
            tool_reply(call("nested-sub-tool", {"text": "two"}, "work-child")),
            tool_reply(
                call("exit_agent", {"text": "child done", "items": []}, "exit-child")
            ),
            tool_reply(
                call("exit_agent", {"text": "parent done", "items": []}, "exit-parent")
            ),
            tool_reply(call("mock-skill", call_id="parent-again")),
            tool_reply(
                call("exit_agent", {"text": "done again", "items": []}, "exit-again")
            ),
            AssistantMessage("done"),
        ]
    )
    agent = DeepAgent(
        provider=provider,
        node=Node([path], name_hds="main"),
        work_root=tmp_path,
        parameter_descriptions_in_schema=parameter_descriptions_in_schema,
    )
    assert agent.run("start").get_message() == "done"

    def skill_instructions(index):
        return "\n".join(
            message.get_message()
            for message in provider.requests[index].get_messages()
            if isinstance(message, ToolMessage)
            and message.get_name() in {"mock-skill", "nested-skill"}
        )

    if parameter_descriptions_in_schema:
        for index in range(len(provider.requests)):
            assert "Parent parameter documentation." not in skill_instructions(index)
            assert "Child parameter documentation." not in skill_instructions(index)
    else:
        assert "Parent parameter documentation." in skill_instructions(1)
        assert "Child parameter documentation." not in skill_instructions(1)
        assert "Child parameter documentation." in skill_instructions(3)
        assert "Child parameter documentation." not in skill_instructions(5)
        assert "Parent parameter documentation." in skill_instructions(5)
        assert "Parent parameter documentation." not in skill_instructions(6)
        assert "Parent parameter documentation." in skill_instructions(7)
    for request in provider.requests:
        definitions = OpenAIResponseToolMapper.map_tools(request.get_tools())
        tool = next(d for d in definitions if d["name"] == "mock-sub-tool")
        assert (
            "description" in tool["parameters"]["properties"]["text"]
        ) is parameter_descriptions_in_schema
        assert (
            "description" in tool["parameters"]["properties"]["execution_mode"]
        ) is parameter_descriptions_in_schema
        assert (
            request.get_messages()[0]
            .get_message()
            .count("Tools exposing execution_mode accept")
            == 1
        )
