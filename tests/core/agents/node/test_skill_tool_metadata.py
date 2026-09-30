from xml.etree import ElementTree as ET

import pytest

from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.tool import Tool
from tests.core.agents.node.steering_support import call, tool_reply
from tests.core.agents.node.test_node import SequenceProvider
from tests.tool_result_support import parse_used_tools


def exit_call(call_id):
    return call("exit_agent", {"text": "done", "items": []}, call_id)


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("cache_result", [False, True])
def test_skill_returns_list_own_and_descendant_tool_outputs(
    make_agent, managed, cache_result
):
    provider = SequenceProvider(
        [
            tool_reply(call("mock-tool", {"text": "root"}, "root")),
            tool_reply(call("mock-skill", call_id="parent")),
            tool_reply(call("shared-tool", {"text": "before"}, "work")),
            tool_reply(call("nested-skill", call_id="child")),
            tool_reply(call("shared-tool", {"text": "first child"}, "work")),
            tool_reply(exit_call("child-exit")),
            tool_reply(call("shared-tool", {"text": "after"}, "work")),
            tool_reply(call("nested-skill", call_id="child")),
            tool_reply(call("shared-tool", {"text": "second child"}, "work")),
            tool_reply(exit_call("child-exit")),
            tool_reply(exit_call("parent-exit")),
            AssistantMessage("final"),
        ]
    )
    agent = make_agent(
        provider,
        Tool(
            lambda text: text,
            name="shared-tool",
            background_enabled=managed,
            cache_result=cache_result,
        ),
    )

    response = agent.run("start")

    parent = response.skill_executions[0]
    first, second = parent.children
    assert parse_used_tools(first.output) == [{"name": "shared-tool", "id": "output-2"}]
    assert parse_used_tools(second.output) == [
        {"name": "shared-tool", "id": "output-5"}
    ]
    expected = [
        {"name": "shared-tool", "id": identifier}
        for identifier in ("output-1", "output-2", "output-4", "output-5")
    ]
    assert parse_used_tools(parent.output) == expected
    assert (
        parse_used_tools(provider.requests[-1].get_messages()[-1].get_message())
        == expected
    )
    assert parent.output.split("</header>\n", 1)[1] == "done"
    output_id = ET.fromstring(
        parent.output.split("</header>", 1)[0] + "</header>"
    ).findtext("id")
    assert (
        parse_used_tools(agent.root_node.read_tool_output(output_id).content)
        == expected
    )


def test_skill_with_no_tool_outputs_has_an_empty_list(make_agent):
    provider = SequenceProvider(
        [
            tool_reply(call("mock-skill", call_id="parent")),
            tool_reply(exit_call("exit")),
            AssistantMessage("final"),
        ]
    )

    agent = make_agent(provider)
    response = agent.run("start")

    assert parse_used_tools(response.skill_executions[0].output) == []
    output_id = ET.fromstring(
        response.skill_executions[0].output.split("</header>", 1)[0] + "</header>"
    ).findtext("id")
    assert parse_used_tools(agent.root_node.read_tool_output(output_id).content) == []


def test_output_reads_do_not_add_duplicate_or_ancestor_outputs(make_agent):
    provider = SequenceProvider(
        [
            tool_reply(call("mock-tool", {"text": "root"}, "root")),
            tool_reply(call("mock-skill", call_id="parent")),
            tool_reply(call("shared-tool", {"text": "parent"}, "work")),
            tool_reply(call("nested-skill", call_id="child")),
            tool_reply(
                call("read_tool_output", {"output_id": "output-0"}, "root-read")
            ),
            tool_reply(
                call("read_tool_output", {"output_id": "output-1"}, "work-read")
            ),
            tool_reply(exit_call("child-exit")),
            tool_reply(exit_call("parent-exit")),
            AssistantMessage("final"),
        ]
    )

    response = make_agent(provider).run("start")

    parent = response.skill_executions[0]
    assert parse_used_tools(parent.children[0].output) == []
    assert parse_used_tools(parent.output) == [
        {"name": "shared-tool", "id": "output-1"}
    ]
