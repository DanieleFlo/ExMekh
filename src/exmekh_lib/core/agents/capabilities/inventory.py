"""Public, structured inspection of the capabilities resolved for one node."""

from dataclasses import dataclass

from exmekh_lib.core.agents.capabilities.data import CapabilitiesData
from exmekh_lib.core.agents.capabilities.default_catalog import (
    DEFAULT_CAPABILITY_CATALOG,
    CapabilityKind,
)
from exmekh_lib.core.agents.capability.conditions import CapabilityPromptContext
from exmekh_lib.core.models.tool import Tool


@dataclass(frozen=True, slots=True)
class ResolvedCapability:
    name: str
    kind: CapabilityKind
    dependencies: tuple[str, ...]
    direct: bool
    reachable: bool
    replacement: bool


@dataclass(frozen=True, slots=True)
class CapabilityInventory:
    entries: tuple[ResolvedCapability, ...]

    @property
    def selected_names(self) -> frozenset[str]:
        return frozenset(entry.name for entry in self.entries)


def build_capability_inventory(
    data: CapabilitiesData,
    runtime_tools: tuple[Tool, ...],
    context: CapabilityPromptContext,
    override_names: frozenset[str],
) -> CapabilityInventory:
    entries = []
    groups: tuple[tuple[CapabilityKind, tuple[Tool, ...]], ...] = (
        ("hds", data.hds),
        ("skill", data.deep_skills),
        ("tool", (*data.tools, *runtime_tools)),
        ("exit-tool", data.exit_tools),
    )
    for kind, resources in groups:
        for tool in resources:
            name = tool.get_name()
            definition = DEFAULT_CAPABILITY_CATALOG.get(name)
            entries.append(
                ResolvedCapability(
                    name,
                    kind,
                    definition.dependencies
                    if definition and name not in override_names
                    else (),
                    name in context.direct_tools or name in context.direct_skills,
                    name in context.reachable_tools or name in context.reachable_skills,
                    name in override_names,
                )
            )
    return CapabilityInventory(tuple(entries))
