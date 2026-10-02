"""The bundled runtime tools expose only model-supplied arguments."""

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

from exmekh_lib.capabilities.tools.bash.bash_tool import BashTool
from exmekh_lib.capabilities.tools.bash.main import run as run_bash
from exmekh_lib.capabilities.tools.call_sub_agent.main import run as run_sub_agent
from exmekh_lib.capabilities.tools.call_sub_agent.sub_agent_tool import SubAgentTool
from exmekh_lib.capabilities.tools.control_tool_execution.execution_control_tool import (
    ExecutionControlTool,
)
from exmekh_lib.capabilities.tools.control_tool_execution.main import run as run_control
from exmekh_lib.capabilities.tools.get_execution_history.main import run as run_history
from exmekh_lib.capabilities.tools.read_tool_output.main import run as run_read
from exmekh_lib.capabilities.tools.search_tool_outputs.main import run as run_search
from exmekh_lib.capabilities.tools.search_tool_outputs.raw_search import (
    run as run_raw_search,
)
from exmekh_lib.core.agents.tools.decorator import get_tool_definition
from exmekh_lib.core.providers.openai_chat_completion_annotation_schema_builder import (
    OpenAIChatCompletionAnnotationSchemaBuilder,
)
from exmekh_lib.core.providers.openai_function_tool_definition_mapper import (
    OpenAIFunctionToolDefinitionMapper,
)
from exmekh_lib.core.tool_calls.execution_history import ExecutionHistory
from exmekh_lib.core.tool_calls.execution_manager import ToolExecutionManager
from exmekh_lib.core.tool_calls.harness_memory_tools import HarnessMemoryTools


def test_runtime_entrypoints_are_decorated_and_start_with_work_root() -> None:
    for function in (
        run_bash,
        run_sub_agent,
        run_control,
        run_history,
        run_read,
        run_search,
        run_raw_search,
    ):
        assert get_tool_definition(function) is not None
        assert next(iter(inspect.signature(function).parameters)) == "work_root"
        assert get_type_hints(function)


def test_runtime_state_is_absent_from_model_schemas(tmp_path: Path) -> None:
    memory = HarnessMemoryTools(ExecutionHistory(), lambda: None, work_root=tmp_path)
    child = SimpleNamespace(name_hds="writer")
    tools = (
        *memory.tools,
        ExecutionControlTool(ToolExecutionManager(), tmp_path),
        BashTool(work_root=tmp_path),
        SubAgentTool([child], tmp_path),
    )
    mapper = OpenAIFunctionToolDefinitionMapper(
        OpenAIChatCompletionAnnotationSchemaBuilder()
    )
    schemas = {
        tool.name: set(mapper.map(tool, strict=True).get_parameters()["properties"])
        for tool in tools
    }

    assert schemas == {
        "get_execution_history": {"start_position", "limit"},
        "search_tool_outputs": {"query", "tool_name", "skill_path", "offset", "limit"},
        "read_tool_output": {"output_id", "cursor"},
        "control_tool_execution": {"tool_call_id", "action", "wait_seconds"},
        "bash": {"command", "execution_mode", "wait_seconds"},
        "call_sub_agent": {"sub_agent", "text"},
    }
