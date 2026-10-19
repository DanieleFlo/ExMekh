from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import get_type_hints

from exmekh_lib.config.agent_text import AGENT_TEXT
from exmekh_lib.core.agents.capability.loader import MarkdownLoader
from exmekh_lib.core.agents.exit_tool.data import ExitToolData
from exmekh_lib.core.agents.exit_tool.decorator import (
    ExitToolDefinition,
    ExitToolFunction,
    get_exit_tool_definition,
)
from exmekh_lib.core.agents.exit_tool.exit_tool import ExitTool
from exmekh_lib.core.agents.tools.deep_tools import DeepTools


class ExitTools(DeepTools):
    def __init__(self, tool_paths: Sequence[Path]) -> None:
        super().__init__(tool_paths, work_root=Path.cwd())

    def _load_tool(self, tool_path: Path) -> ExitTool:
        entrypoint_path = self._validate_tool_path(tool_path)
        module = self._module_importer.import_module(tool_path, entrypoint_path)
        try:
            function, definition = self._find_exit_entrypoint(
                module,
                entrypoint_path,
            )
            self._validate_return_type(function, entrypoint_path)
            if definition.description is None:
                raise ValueError(
                    AGENT_TEXT.exit_tool_entrypoint_must_have_a_description.format(
                        entrypoint_path=entrypoint_path
                    )
                )
            instructions_path = tool_path / definition.instructions_path
            instructions = MarkdownLoader(instructions_path).load()
            return ExitTool(
                function,
                name=definition.name or tool_path.name,
                description=definition.description,
                instructions_path=instructions_path,
                instructions=instructions,
                folder_path=tool_path,
                before=definition.before,
                after=definition.after,
            )
        except Exception:
            self._module_importer.unload_module(tool_path)
            raise

    @staticmethod
    def _find_exit_entrypoint(
        module: ModuleType,
        entrypoint_path: Path,
    ) -> tuple[ExitToolFunction, ExitToolDefinition]:
        entrypoints: list[tuple[ExitToolFunction, ExitToolDefinition]] = []
        for _, function in inspect.getmembers(module, inspect.isfunction):
            if function.__module__ != module.__name__:
                continue
            definition = get_exit_tool_definition(function)
            if definition is not None:
                entrypoints.append((function, definition))

        if len(entrypoints) != 1:
            raise ValueError(
                AGENT_TEXT.exit_tool_entrypoint_must_define_exactly_one_exit_tool.format(
                    entrypoint_path=entrypoint_path
                )
            )
        return entrypoints[0]

    @staticmethod
    def _validate_return_type(
        function: ExitToolFunction,
        entrypoint_path: Path,
    ) -> None:
        try:
            return_type = get_type_hints(function).get("return")
        except (NameError, TypeError) as error:
            raise ValueError(
                AGENT_TEXT.exit_tool_entrypoint_must_return_exittooldata.format(
                    entrypoint_path=entrypoint_path
                )
            ) from error
        if return_type is not ExitToolData:
            raise ValueError(
                AGENT_TEXT.exit_tool_entrypoint_must_return_exittooldata.format(
                    entrypoint_path=entrypoint_path
                )
            )

    def load(self) -> tuple[ExitTool, ...]:
        return self._load_tools(
            self._load_tool, AGENT_TEXT.exittools_can_load_only_once
        )

    def load_exit_tools(self) -> tuple[ExitTool, ...]:
        """Compatibility alias for the common tool-loading interface."""
        return self.load()
