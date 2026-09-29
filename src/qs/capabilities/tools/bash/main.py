"""Expose cancellable Bash work as a tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qs.config.runtime import RUNTIME
from qs.core.agents.tools.decorator import deep_tool

from .bash_process import _BashProcess


@deep_tool(
    name=RUNTIME.bash_tool_name,
    description=(
        "Run a command in Bash from the node workroot. Each call starts an "
        "independent process. Read incremental stdout/stderr with "
        "read_tool_output(output_id, cursor); wait or cancel with "
        "control_tool_execution(tool_call_id). Requires Bash on PATH."
    ),
)
def run(work_root: Path, executable: str | None, command: str) -> dict[str, Any]:
    return _BashProcess(command, work_root, executable).run()
