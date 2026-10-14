from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from exmekh_lib.config.agent_text import AGENT_TEXT
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.interfaces.tool_hook import AfterTool, BeforeTool

from .decorator import ToolFunction, get_tool_definition


class DeepTool(Tool):
    """Bind a decorated entrypoint to its node's runtime dependencies."""

    def __init__(
        self,
        function: ToolFunction,
        work_root: Path,
        *runtime_arguments: Any,
        name: str | None = None,
        description: str | None = None,
        cache_result: bool = True,
        before: BeforeTool | None = None,
        after: AfterTool | None = None,
    ) -> None:
        definition = get_tool_definition(function)
        if definition is None or definition.description is None:
            raise ValueError(
                AGENT_TEXT.tool_function_must_have_deep_tool_with_a_description
            )

        bound_function = partial(function, work_root, *runtime_arguments)
        super().__init__(
            bound_function,
            name=(name if name is not None else definition.name or function.__name__),
            description=(
                description if description is not None else definition.description
            ),
            cache_result=cache_result,
            before=before if before is not None else definition.before,
            after=after if after is not None else definition.after,
            background_enabled=definition.background_enabled,
            argument_guidance=definition.argument_guidance,
            interactive=definition.interactive,
        )
