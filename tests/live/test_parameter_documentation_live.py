"""Verify argument comprehension with documentation outside the function schema."""

import json
import os
from uuid import uuid4

import pytest
from dotenv import load_dotenv

from exmekh_lib.config.provider import DEFAULT_PROVIDER_SETTINGS
from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.capabilities.capabilities import Capabilities
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMH
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from tests.core.agents.node.test_node import _write_nested_capabilities
from tests.live.model_support import CacheTestTokenizer, LiveModelProvider

pytestmark = pytest.mark.live


@pytest.mark.parametrize("api", ["responses", "chat_completions"])
def test_live_model_uses_parameter_documentation_outside_the_schema(tmp_path, api):
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
    data = Capabilities([path]).load(work_root=tmp_path)
    original = next(tool for tool in data.tools if tool.name == "nested-sub-tool")
    contextual = next(
        tool
        for tool in data.get_contextual_tools(data.get_hds("main"))
        if tool.name == original.name
    )
    skill = next(skill for skill in data.deep_skills if skill.name == "nested-skill")
    tools = Tools([contextual])
    messages = [
        SystemMessage(
            f"{RUNTIME.harness_output_guidance}\n\n{skill.render_instructions('')}"
        ),
        UserMessage(
            "Call nested-sub-tool once, following its parameter documentation."
        ),
    ]
    with LiveModelProvider(
        model=os.getenv("EXMEKH_LIB_LIVE_MODEL", DEFAULT_PROVIDER_SETTINGS.model),
        reasoning_effort=ReasoningEffortLMH.LOW,
        api=api,
        tokenizer=CacheTestTokenizer(
            "live-parameter-documentation", tmp_path / "cache"
        ),
    ) as provider:
        definitions = json.loads(provider.get_tools_text(tools))
        assert token not in json.dumps(definitions)
        response = provider.call(
            ProviderRequest(messages, tools=tools, tool_choice="required")
        )
        calls = response.get_tool_calls() or []
        trace = {
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
