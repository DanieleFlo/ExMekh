from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.tools.decorator import deep_tool


@deep_tool(
    name=RUNTIME.sub_agent_tool_name,
    description="Send a user message to one direct sub-agent and return its final response.",
    background_enabled=False,
)
def run(
    work_root: Path,
    delegate: Callable[..., str],
    sub_agent: str,
    text: str,
) -> str:
    del work_root
    return delegate(sub_agent=sub_agent, text=text)
