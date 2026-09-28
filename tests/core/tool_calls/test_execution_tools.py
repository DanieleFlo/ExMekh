import pytest

from qs.capabilities.tools.control_tool_execution.main import ExecutionControlTool
from qs.core.tool_calls.execution_manager import ToolExecutionManager


def test_execution_control_rejects_unknown_actions_before_looking_up_work():
    tool = ExecutionControlTool(ToolExecutionManager())

    with pytest.raises(ValueError, match="action must be 'wait' or 'cancel'"):
        tool.control_tool_execution("missing", action="unknown")
