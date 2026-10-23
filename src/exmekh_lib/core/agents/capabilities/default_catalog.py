"""The bundled resource catalog, independent of any node's selection."""

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Literal

from exmekh_lib.config.default_capabilities import DEFAULT_CAPABILITIES
from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.deep_skill.deep_skill import DeepSkill
from exmekh_lib.core.agents.exit_tool.exit_tool import ExitTool
from exmekh_lib.core.agents.hds.hds import HDS
from exmekh_lib.core.models.tool import Tool

CapabilityKind = Literal["tool", "skill", "hds", "exit-tool"]


@dataclass(frozen=True, slots=True)
class DefaultCapability:
    name: str
    kind: CapabilityKind
    resource_type: type[Tool]
    dependencies: tuple[str, ...] = ()
    runtime: bool = False


def _catalog() -> MappingProxyType[str, DefaultCapability]:
    names = DEFAULT_CAPABILITIES
    special: dict[str, tuple[CapabilityKind, type[Tool]]] = {
        names.main: ("hds", HDS),
        names.quick: ("hds", HDS),
        names.deep_thinking: ("skill", DeepSkill),
        names.file_processing: ("skill", DeepSkill),
        names.exit_agent: ("exit-tool", ExitTool),
    }
    dependencies = {
        names.read_file: (names.inspect_file,),
        names.edit_file: (names.inspect_file, names.read_file),
    }
    catalog = {}
    for field in fields(names):
        name = getattr(names, field.name)
        kind, resource_type = special.get(name, ("tool", Tool))
        catalog[name] = DefaultCapability(
            name,
            kind,
            resource_type,
            dependencies.get(name, ()),
            name in RUNTIME.runtime_tool_names,
        )
    return MappingProxyType(catalog)


DEFAULT_CAPABILITY_CATALOG = _catalog()
