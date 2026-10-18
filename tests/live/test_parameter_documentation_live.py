"""Verify argument comprehension with parameter documentation in either location."""

import json
import os
from uuid import uuid4

import pytest
from dotenv import load_dotenv

from exmekh_lib.config.provider import DEFAULT_PROVIDER_SETTINGS
from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.deep_agent.deep_agent import DeepAgent
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMH
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from tests.core.agents.node.test_node import _write_nested_capabilities
from tests.live.model_support import CacheTestTokenizer, LiveModelProvider

pytestmark = pytest.mark.live


@pytest.mark.parametrize("api", ["responses", "chat_completions"])
@pytest.mark.parametrize("parameter_descriptions_in_schema", [False, True])
def test_live_model_uses_configured_parameter_documentation(
    tmp_path, api, parameter_descriptions_in_schema
):
    load_dotenv()
    token = f"DOCS-{uuid4().hex}"
    path = _write_nested_capabilities(tmp_path)
    documentation = f"Set this field to exactly {token!r}. Copy it literally."
    (path / "tools" / "nested-sub-tool" / "main.py").write_text(
        "from typing import Annotated, TypedDict\n"
        "from exmekh_lib.core.agents.tools.decorator import deep_tool\n"
        "class Payload(TypedDict):\n"
        f"    text: Annotated[str, {documentation!r}]\n"
        "@deep_tool(name='nested-sub-tool', description='Check the documented payload.')\n"
        "def run(work_root, payload: Payload) -> dict[str, object]:\n"
        f"    return {{'accepted': payload['text'] == {token!r}, 'text': payload['text']}}\n"
    )
    with LiveModelProvider(
        model=os.getenv("EXMEKH_LIB_LIVE_MODEL", DEFAULT_PROVIDER_SETTINGS.model),
        reasoning_effort=ReasoningEffortLMH.LOW,
        api=api,
        tokenizer=CacheTestTokenizer(
            "live-parameter-documentation", tmp_path / "cache"
        ),
    ) as provider:
        node = Node([path], name_hds="main")
        agent = DeepAgent(
            provider=provider,
            node=node,
            work_root=tmp_path,
            parameter_descriptions_in_schema=parameter_descriptions_in_schema,
        )
        data = agent.capabilities_by_node[node]
        original = next(
            tool for tool in data.tools if tool.get_name() == "nested-sub-tool"
        )
        contextual = next(
            tool
            for tool in data.get_contextual_tools(data.get_hds("main"))
            if tool.get_name() == original.get_name()
        )
        skill = next(
            skill for skill in data.deep_skills if skill.get_name() == "nested-skill"
        )
        tools = Tools([contextual])
        instructions = skill.render_instructions("")
        messages = [
            SystemMessage(f"{RUNTIME.harness_output_guidance}\n\n{instructions}"),
            UserMessage(
                "Call nested-sub-tool once, following its parameter documentation."
            ),
        ]
        definitions = json.loads(provider.get_tools_text(tools))
        assert (token in json.dumps(definitions)) is parameter_descriptions_in_schema
        assert (token in instructions) is not parameter_descriptions_in_schema
        response = provider.call(
            ProviderRequest(messages, tools=tools, tool_choice="required")
        )
        calls = response.get_tool_calls() or []
        trace = {
            "parameter_descriptions_in_schema": parameter_descriptions_in_schema,
            "tools": definitions,
            "messages": [message.message_dump() for message in messages],
            "calls": [
                {"name": call.get_name(), "arguments": call.get_arguments()}
                for call in calls
            ],
        }
        (tmp_path / "model-trace.json").write_text(json.dumps(trace, indent=2))
        assert len(calls) == 1
        assert calls[0].get_name() == "nested-sub-tool"
        assert calls[0].get_arguments()["payload"] == {"text": token}
        result = original.execute(
            **original.get_work_arguments(calls[0].get_arguments())
        )
        assert result == {"accepted": True, "text": token}
