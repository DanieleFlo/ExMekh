from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import get_type_hints

from qs.core.agents.exit_tool.data import ExitToolData
from qs.core.agents.exit_tool.decorator import (
    ExitToolDefinition,
    ExitToolFunction,
    get_exit_tool_definition,
)
from qs.core.agents.exit_tool.exit_tool import ExitTool
from qs.core.agents.tools.deep_tools import DeepTools


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
                    f"Exit-tool entrypoint '{entrypoint_path}' must have a description"
                )
            return ExitTool(
                function,
                name=definition.name or tool_path.name,
                description=definition.description,
                exit_skill_name=tool_path.name,
                folder_path=tool_path,
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
                f"Exit-tool entrypoint '{entrypoint_path}' must define "
                "exactly one @exit_tool entrypoint"
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
                f"Exit-tool entrypoint '{entrypoint_path}' must return ExitToolData"
            ) from error
        if return_type is not ExitToolData:
            raise ValueError(
                f"Exit-tool entrypoint '{entrypoint_path}' must return ExitToolData"
            )

    def load_exit_tools(self) -> tuple[ExitTool, ...]:
        if self._load_attempted:
            raise RuntimeError("ExitTools can load only once")

        self._load_attempted = True
        tools: list[ExitTool] = []
        loaded_paths: list[Path] = []

        try:
            for tool_path in self._tool_paths:
                tools.append(self._load_tool(tool_path))
                loaded_paths.append(tool_path)
        except Exception:
            for loaded_path in loaded_paths:
                self._module_importer.unload_module(loaded_path)
            raise

        return tuple(tools)
