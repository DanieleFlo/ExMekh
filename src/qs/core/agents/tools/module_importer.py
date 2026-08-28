from __future__ import annotations

import sys
from hashlib import sha256
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from uuid import uuid4


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
            raise ImportError(f"Cannot import tool entrypoint: {entrypoint_path}")

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
            raise TypeError("tool_path must be a Path")

        self._remove_package_modules(self._package_name(tool_path))

    @staticmethod
    def _validate_paths(tool_path: Path, entrypoint_path: Path) -> None:
        if not isinstance(tool_path, Path):
            raise TypeError("tool_path must be a Path")

        if not isinstance(entrypoint_path, Path):
            raise TypeError("entrypoint_path must be a Path")

        if tool_path.is_symlink():
            raise ValueError("tool_path must not be a symbolic link")

        if entrypoint_path.is_symlink():
            raise ValueError("entrypoint_path must not be a symbolic link")

        if not tool_path.is_dir():
            raise ValueError("tool_path must be an existing directory")

        if not entrypoint_path.is_file():
            raise ValueError("entrypoint_path must be an existing file")

        canonical_tool_path = tool_path.resolve()
        canonical_entrypoint_path = entrypoint_path.resolve()

        if canonical_tool_path not in canonical_entrypoint_path.parents:
            raise ValueError("entrypoint_path must be inside tool_path")

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
        return f"_qs_deep_tool_{path_hash}_{self._namespace}"

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
