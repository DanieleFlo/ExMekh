from PIL import Image

from exmekh_lib.core.agents.deep_agent.deep_agent import DeepAgent
from exmekh_lib.core.agents.node import DefaultCapabilities, Node
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.user_message import ImagesUserMessage, UserMessage
from tests.core.agents.node.steering_support import call, tool_reply
from tests.core.agents.node.test_node import SequenceProvider
from tests.core.agents.node.test_steering_history import assert_complete_tool_batches


def test_image_prompt_survives_tools_skill_return_and_followup(tmp_path):
    provider = SequenceProvider(
        [
            tool_reply(call("mock-tool", {"text": "TOKEN_VIOLET_731"}, "before")),
            AssistantMessage("saved"),
            tool_reply(call("mock-skill", {}, "skill")),
            tool_reply(call("mock-sub-tool", {"text": "inspect"}, "inside")),
            tool_reply(
                call("exit_agent", {"text": "image reviewed", "items": []}, "leave")
            ),
            AssistantMessage("reviewed"),
            tool_reply(call("mock-tool", {"text": "after"}, "after")),
            AssistantMessage("done"),
        ]
    )
    node = Node(name_hds="main", default_capabilities=DefaultCapabilities(mocks=True))
    agent = DeepAgent(provider, node, tmp_path)
    agent.run("Remember TOKEN_VIOLET_731")
    pixels = Image.new("RGB", (3, 2), "red")
    content = ["Inspect; preserve the previous token", ImagesUserMessage([pixels])]
    agent.run(content)
    agent.run("Call a tool and recall the token and image")
    for request in provider.requests[2:]:
        messages = request.get_messages()
        assert any(
            isinstance(m, UserMessage)
            and m.get_message() == "Remember TOKEN_VIOLET_731"
            for m in messages
        )
        images = [
            m.get_message()
            for m in messages
            if isinstance(m, UserMessage) and isinstance(m.get_message(), list)
        ]
        assert len(images) == 1
        assert images[0][0] == "Inspect; preserve the previous token"
        assert images[0][1].images[0].tobytes() == pixels.tobytes()
        assert_complete_tool_batches(messages)
