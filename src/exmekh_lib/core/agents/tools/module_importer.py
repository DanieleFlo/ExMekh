from __future__ import annotations

import sys
from hashlib import sha256
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from uuid import uuid4

from exmekh_lib.config.agent_text import AGENT_TEXT


class ToolModuleImporter:
    def __init__(self) -> None:
        self._namespace = uuid4().hex

    def import_module(
        self,
        tool_path: Path,
        entrypoint_path: Path,
    ) -> ModuleType:
        self._validate_paths(tool_path, entrypoint_path)
        package_name = self._package_name(tool_path)
        self._remove_package_modules(package_name)
        self._register_package(package_name, tool_path)

        module_name = f"{package_name}.main"
        specification = spec_from_file_location(
            module_name,
            entrypoint_path,
        )

        if specification is None or specification.loader is None:
            self._remove_package_modules(package_name)
            raise ImportError(
                AGENT_TEXT.cannot_import_tool_entrypoint.format(
                    entrypoint_path=entrypoint_path
                )
            )

        try:
            module = module_from_spec(specification)
            sys.modules[module_name] = module
            specification.loader.exec_module(module)
        except Exception:
            self._remove_package_modules(package_name)
            raise

        return module

    def unload_module(self, tool_path: Path) -> None:
        if not isinstance(tool_path, Path):
            raise TypeError(AGENT_TEXT.tool_path_must_be_a_path)

        self._remove_package_modules(self._package_name(tool_path))

    @staticmethod
    def _validate_paths(tool_path: Path, entrypoint_path: Path) -> None:
        if not isinstance(tool_path, Path):
            raise TypeError(AGENT_TEXT.tool_path_must_be_a_path)

        if not isinstance(entrypoint_path, Path):
            raise TypeError(AGENT_TEXT.entrypoint_path_must_be_a_path)

        if tool_path.is_symlink():
            raise ValueError(AGENT_TEXT.tool_path_must_not_be_a_symbolic_link_2)

        if entrypoint_path.is_symlink():
            raise ValueError(AGENT_TEXT.entrypoint_path_must_not_be_a_symbolic_link)

        if not tool_path.is_dir():
            raise ValueError(AGENT_TEXT.tool_path_must_be_an_existing_directory)

        if not entrypoint_path.is_file():
            raise ValueError(AGENT_TEXT.entrypoint_path_must_be_an_existing_file)

        canonical_tool_path = tool_path.resolve()
        canonical_entrypoint_path = entrypoint_path.resolve()

        if canonical_tool_path not in canonical_entrypoint_path.parents:
            raise ValueError(AGENT_TEXT.entrypoint_path_must_be_inside_tool_path)

    @staticmethod
    def _register_package(package_name: str, tool_path: Path) -> None:
        package = ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [str(tool_path)]
        package.__spec__ = ModuleSpec(
            package_name,
            loader=None,
            is_package=True,
        )
        sys.modules[package_name] = package

    def _package_name(self, tool_path: Path) -> str:
        canonical_path = str(tool_path.resolve()).encode()
        path_hash = sha256(canonical_path).hexdigest()
        return f"_exmekh_lib_deep_tool_{path_hash}_{self._namespace}"

    @staticmethod
    def _remove_package_modules(package_name: str) -> None:
        package_prefix = f"{package_name}."
        loaded_names = tuple(
            module_name
            for module_name in sys.modules
            if module_name == package_name or module_name.startswith(package_prefix)
        )

        for module_name in loaded_names:
            del sys.modules[module_name]
