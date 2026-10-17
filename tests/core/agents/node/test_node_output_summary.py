from xml.etree import ElementTree as ET

import pytest

from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.tool import Tool
from tests.core.agents.node.steering_support import call, tool_reply
from tests.core.agents.node.test_node import SequenceProvider


@pytest.mark.parametrize("mode", ["brief", "detailed"])
def test_model_and_public_node_reads_use_isolated_summary_client(make_agent, mode):
    class SummaryProvider(SequenceProvider):
        def __init__(self, responses):
            super().__init__(responses)
            self.summary_requests = []

        def call(self, request):
            if request.get_reasoning_effort_override() == "none":
                self.summary_requests.append(request)
                return ProviderResponse(AssistantMessage("essential facts"))
            return super().call(request)

    provider = SummaryProvider(
        [
            tool_reply(call()),
            tool_reply(
                call(
                    "read_tool_output",
                    {"output_id": "output-0", "summary_mode": mode},
                    "read",
                )
            ),
            AssistantMessage("done"),
        ]
    )
    agent = make_agent(provider, Tool(lambda: "x" * 300, name="mock-tool"))
    assert agent.run("start").get_message() == "done"
    result = provider.requests[-1].get_messages()[-1]
    header, body = result.get_message().split("</header>\n", 1)
    root = ET.fromstring(header + "</header>")
    assert root.findtext("id") == "output-0"
    assert root.findtext("summary-mode") == mode
    assert body == "essential facts"
    summary_result = agent.root_node.read_tool_output("output-0", summary_mode=mode)
    assert summary_result.content.endswith("essential facts")
    assert len(provider.summary_requests) == 2
    assert all(
        len(request.get_messages()) == 2 for request in provider.summary_requests
    )
    assert all(request.get_tools() is None for request in provider.summary_requests)
    original = agent.root_node.read_tool_output("output-0")
    assert original.content.endswith("x" * 300)
    assert original.content.split("</header>")[0].count("<elapsed>") == 1
