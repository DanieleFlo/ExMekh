"""The actor's execution tool supports programs split across workspace files."""

from evaluations.benchmark.accounting import MeteredProvider
from evaluations.benchmark.drivers import run_native
from evaluations.benchmark.models import Criterion, Limits, Task, Variant
from exmekh_lib.core.models.assistant_message import AssistantMessage
from tests.evaluations.benchmark.test_accounting import ScriptedProvider, call
from tests.tool_result_support import parse_tool_result


def test_actor_can_run_a_script_importing_another_workspace_module(tmp_path):
    provider = ScriptedProvider(
        [
            call("file-processing", {}),
            call("run-python", {"path": "check.py"}),
            call("exit_agent", {"text": "Verified 42", "items": []}),
            AssistantMessage("42"),
        ]
    )
    task = Task(
        id="modules",
        domain="code",
        prompt="Run check.py",
        files={
            "solution.py": "def answer():\n    return 42\n",
            "check.py": "from solution import answer\nprint(answer())\n",
        },
        criteria=(
            Criterion(
                id="result",
                description="Script runs",
                verification="exact",
                verification_reason="Execution",
                checker="answer",
            ),
        ),
    )
    result = run_native(
        task,
        Variant(id="current"),
        tmp_path,
        MeteredProvider(provider, Limits(max_context_tokens=100_000)),
    )
    assert result.status == "completed"
    output = parse_tool_result(provider.requests[2].get_messages()[-1].get_message())
    assert output["exit_code"] == 0, output
    assert output["stdout"].strip() == "42"
