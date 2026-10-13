"""Expose cancellable Bash work as a tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.tools.decorator import deep_tool

from .bash_process import BashProcess


@deep_tool(
    name=RUNTIME.bash_tool_name,
    description=(RUNTIME.bash_description),
)
def run(work_root: Path, executable: str | None, command: str) -> dict[str, Any]:
    return BashProcess(command, work_root, executable).run()
