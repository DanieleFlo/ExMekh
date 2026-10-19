from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from exmekh_lib.config.agent_text import AGENT_TEXT
from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.core.agents.capability.conditions import (
    CapabilityPromptContext,
    has_conditions,
    render_conditions,
)
from exmekh_lib.core.agents.capability.references import parse_references
from exmekh_lib.core.agents.exit_tool.data import ExitToolData, ExitToolResult
from exmekh_lib.core.agents.exit_tool.decorator import ExitToolFunction
from exmekh_lib.core.models.assistant_message import ToolCall
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.interfaces.message import Message
from exmekh_lib.interfaces.tool_hook import AfterTool, BeforeTool, ToolContext
from exmekh_lib.utils.xml import create_text_element, render_xml


class ExitTool(Tool):
    def __init__(
        self,
        function: ExitToolFunction,
        *,
        name: str,
        description: str,
        instructions_path: Path,
        instructions: str,
        folder_path: Path,
        before: BeforeTool | None = None,
        after: AfterTool | None = None,
    ) -> None:
        if not isinstance(instructions_path, Path):
            raise TypeError(AGENT_TEXT.instructions_path_must_be_a_path)
        if not isinstance(instructions, str):
            raise TypeError(AGENT_TEXT.instructions_must_be_a_string)
        if not instructions.strip():
            raise ValueError(
                AGENT_TEXT.exit_tool_instructions_cannot_be_blank.format(
                    instructions_path=instructions_path
                )
            )
        self.instructions_path = instructions_path
        self.instructions, self.references = (
            (instructions.strip(), ())
            if has_conditions(instructions)
            else parse_references(instructions.strip())
        )
        self.folder_path = folder_path
        self._prompt_context: CapabilityPromptContext | None = None
        super().__init__(
            function,
            name=name,
            description=description,
            background_enabled=False,
            before=before,
            after=after,
        )

    def render_prompt_section(
        self, *, context: CapabilityPromptContext | None = None
    ) -> str:
        context = context or self._prompt_context
        instructions = self.instructions
        if context is not None:
            instructions, references = parse_references(
                render_conditions(instructions, context)
            )
            context.validate_references(references, self.references)
        return render_xml(
            create_text_element(
                "exit-tool", instructions, attributes={"name": self.name}
            )
        )

    def attach_prompt_context(self, context: CapabilityPromptContext) -> None:
        self._prompt_context = context

    def copy(self) -> Self:
        copied = super().copy()
        copied._prompt_context = None
        return copied

    def execute(self, *args: Any, **kwargs: Any) -> ExitToolResult:
        result = self.tool(*args, **kwargs)
        if not isinstance(result, ExitToolData):
            raise TypeError(AGENT_TEXT.exit_tool_must_return_exittooldata)

        return result.dump()

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        super().validate_arguments(arguments)
        ExitToolData(**arguments)

    def _finalize_result(self, context: ToolContext, result: Any) -> ExitToolResult:
        result = super()._finalize_result(context, result)
        if not isinstance(result, dict):
            raise TypeError(AGENT_TEXT.exit_tool_result_must_be_a_dictionary)
        return ExitToolData(**result).dump()

    def __str__(self) -> str:
        return AGENT_TEXT.name_type_exittool_instructions.format(
            name=self.name,
            instructions=self.instructions[
                : CAPABILITY_FORMAT.representation_preview_chars
            ],
        )


class CalledExitTool:
    def __init__(self, exit_tool: ExitTool, tool_call: ToolCall) -> None:
        if not isinstance(exit_tool, ExitTool):
            raise TypeError(AGENT_TEXT.exit_tool_must_be_an_exittool)
        if not isinstance(tool_call, ToolCall):
            raise TypeError(AGENT_TEXT.tool_call_must_be_a_toolcall)

        self._exit_tool = exit_tool
        self._tool_call = tool_call
        self._validate_matching()

    def _validate_matching(self) -> None:
        call_name = self._tool_call.get_name()
        if call_name != self.name:
            raise ValueError(
                AGENT_TEXT.tool_call_name_does_not_match_exit_tool_name.format(
                    call_name=call_name, name=self.name
                )
            )

    @property
    def name(self) -> str:
        return self._exit_tool.get_name()

    @property
    def call_id(self) -> str:
        return self._tool_call.get_id()

    @property
    def arguments(self) -> dict[str, Any]:
        return self._tool_call.get_arguments()

    @property
    def tool_call(self) -> ToolCall:
        return self._tool_call

    def validate_arguments(self) -> None:
        self._exit_tool.validate_arguments(self.arguments)

    def execute(
        self,
        *,
        work_root: Path | None = None,
        history: tuple[Message[object], ...] = (),
    ) -> ExitToolResult:
        return self._exit_tool.execute_with_hooks(
            self.arguments,
            call_id=self.call_id,
            work_root=work_root if work_root is not None else Path.cwd(),
            history=history,
        )
