from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Final

from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.core.agents.tools.decorator import (
    ToolDefinition,
    ToolFunction,
    get_tool_definition,
)
from exmekh_lib.core.agents.tools.deep_tool import DeepTool
from exmekh_lib.core.agents.tools.module_importer import ToolModuleImporter
from exmekh_lib.core.models.tool import Tool


class DeepTools:
    ENTRYPOINT_FILENAME: Final[str] = CAPABILITY_FORMAT.tool_entrypoint_filename

    def __init__(
        self,
        tool_paths: Sequence[Path],
        *,
        work_root: Path,
    ) -> None:
        self._tool_paths = self._validate_tool_paths(tool_paths)
        self._work_root = work_root
        self._module_importer = ToolModuleImporter()
        self._load_attempted = False

    @staticmethod
    def _validate_tool_paths(
        tool_paths: Sequence[Path],
    ) -> tuple[Path, ...]:
        if isinstance(tool_paths, (str, bytes)) or not isinstance(
            tool_paths,
            Sequence,
        ):
            raise TypeError("tool_paths must be a sequence of Path")

        paths = tuple(tool_paths)

        if not all(isinstance(path, Path) for path in paths):
            raise TypeError("tool_paths must contain only Path instances")

        canonical_paths = tuple(path.resolve() for path in paths)

        if len(canonical_paths) != len(set(canonical_paths)):
            raise ValueError("Duplicate tool path")

        return paths

    def load(self) -> tuple[Tool, ...]:
        if self._load_attempted:
            raise RuntimeError("DeepTools can load only once")

        self._load_attempted = True
        tools: list[Tool] = []
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

    def unload(self) -> None:
        for tool_path in self._tool_paths:
            self._module_importer.unload_module(tool_path)

    def _load_tool(self, tool_path: Path) -> Tool:
        entrypoint_path = self._validate_tool_path(tool_path)
        module = self._module_importer.import_module(
            tool_path,
            entrypoint_path,
        )
        try:
            function, definition = self._find_entrypoint(
                module,
                entrypoint_path,
            )
            description = definition.description

            if description is None:
                raise ValueError(
                    f"Tool entrypoint '{entrypoint_path}' must have a description"
                )

            return DeepTool(
                function, self._work_root, name=definition.name or tool_path.name
            )
        except Exception:
            self._module_importer.unload_module(tool_path)
            raise

    def _validate_tool_path(self, tool_path: Path) -> Path:
        if tool_path.is_symlink():
            raise ValueError(f"Tool path must not be a symbolic link: {tool_path}")

        if not tool_path.exists():
            raise FileNotFoundError(f"Tool path does not exist: {tool_path}")

        if not tool_path.is_dir():
            raise ValueError(f"Tool path must be a directory: {tool_path}")

        entrypoint_path = tool_path / self.ENTRYPOINT_FILENAME

        if entrypoint_path.is_symlink():
            raise ValueError(
                f"Tool entrypoint must not be a symbolic link: {entrypoint_path}"
            )

        if not entrypoint_path.exists():
            raise FileNotFoundError(
                f"Tool entrypoint does not exist: {entrypoint_path}"
            )

        if not entrypoint_path.is_file():
            raise ValueError(f"Tool entrypoint must be a file: {entrypoint_path}")

        return entrypoint_path

    def _find_entrypoint(
        self,
        module: ModuleType,
        entrypoint_path: Path,
    ) -> tuple[ToolFunction, ToolDefinition]:
        entrypoints: list[tuple[ToolFunction, ToolDefinition]] = []

        for _, function in inspect.getmembers(module, inspect.isfunction):
            if function.__module__ != module.__name__:
                continue

            definition = get_tool_definition(function)

            if definition is not None:
                entrypoints.append((function, definition))

        if len(entrypoints) != 1:
            raise ValueError(
                f"Tool entrypoint '{entrypoint_path}' must define exactly "
                "one @deep_tool entrypoint"
            )

        return entrypoints[0]
