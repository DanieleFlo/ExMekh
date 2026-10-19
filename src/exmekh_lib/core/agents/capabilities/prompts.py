"""Build instruction contexts from the same declarations used for execution scopes."""

from collections.abc import Iterable
from dataclasses import dataclass

from exmekh_lib.core.agents.capabilities.catalog_attacher import SkillCatalogAttacher
from exmekh_lib.core.agents.capabilities.data import CapabilitiesData
from exmekh_lib.core.agents.capabilities.default_catalog import (
    DEFAULT_CAPABILITY_CATALOG,
)
from exmekh_lib.core.agents.capabilities.index import CapabilitiesIndex
from exmekh_lib.core.agents.capabilities.validator import CapabilitiesValidator
from exmekh_lib.core.agents.capability.conditions import (
    CapabilityPromptContext,
    render_conditions,
)
from exmekh_lib.core.agents.capability.references import parse_references
from exmekh_lib.core.agents.deep_skill.deep_skill import DeepSkill
from exmekh_lib.core.agents.hds.hds import HDS
from exmekh_lib.core.models.tool import Tool


class CapabilityPrompts:
    def __init__(
        self,
        data: CapabilitiesData,
        runtime_names: Iterable[str],
        *,
        exit_tool_name: str | None = None,
    ) -> None:
        self._data = data
        self._runtime = frozenset(runtime_names)
        self._exit_tool_name = exit_tool_name
        self._skills = {skill.get_name(): skill for skill in data.deep_skills}
        self._tools = (
            frozenset(tool.get_name() for tool in (*data.tools, *data.exit_tools))
            | self._runtime
        )
        self._known_tools = (
            frozenset(
                name
                for name, entry in DEFAULT_CAPABILITY_CATALOG.items()
                if entry.kind in {"tool", "exit-tool"}
            )
            | self._tools
        )
        self._known_skills = (
            frozenset(
                name
                for name, entry in DEFAULT_CAPABILITY_CATALOG.items()
                if entry.kind == "skill"
            )
            | self._skills.keys()
        )
        for resource in (*data.hds, *data.deep_skills):
            self._known_tools |= frozenset(resource.get_optional_sub_tool_names())
            self._known_skills |= frozenset(resource.get_optional_sub_skill_names())

    def context(
        self, resource: HDS | DeepSkill, *, direct_tools: Iterable[str] | None = None
    ) -> CapabilityPromptContext:
        direct = (
            frozenset(
                resource.get_sub_tool_names() if direct_tools is None else direct_tools
            )
            | self._runtime
        )
        exit_tool_name = (
            resource.get_exit_tool_name()
            if isinstance(resource, HDS)
            else self._exit_tool_name
        )
        if isinstance(resource, DeepSkill) and exit_tool_name is not None:
            direct |= {exit_tool_name}
        skills = frozenset(resource.get_sub_skill_names())
        reachable_tools = set(direct)
        reachable_skills: set[str] = set()
        pending = list(skills)
        while pending:
            name = pending.pop()
            if name in reachable_skills:
                continue
            reachable_skills.add(name)
            if skill := self._skills.get(name):
                reachable_tools.update(skill.get_sub_tool_names())
                pending.extend(skill.get_sub_skill_names())
        if reachable_skills and exit_tool_name is not None:
            reachable_tools.add(exit_tool_name)
        return CapabilityPromptContext(
            self._known_tools,
            frozenset(self._known_skills),
            self._tools,
            frozenset(self._skills),
            direct & self._tools,
            skills & self._skills.keys(),
            frozenset(reachable_tools) & self._tools,
            frozenset(reachable_skills) & self._skills.keys(),
        )

    def attach(self) -> None:
        for resource in (*self._data.hds, *self._data.deep_skills):
            self.attach_resource(resource, self.context(resource))
        resources = (*self._data.hds, *self._data.deep_skills)
        if resources:
            for tool in (*self._data.tools, *self._data.exit_tools):
                owner = next(
                    (
                        resource
                        for resource in resources
                        if tool.get_name() in resource.get_sub_tool_names()
                    ),
                    resources[0],
                )
                context = self.context(owner)
                tool.render_description(lambda text: self.render(text, context))
            for tool in self._data.exit_tools:
                tool.attach_prompt_context(self.context(resources[0]))

    def attach_resource(
        self, resource: HDS | DeepSkill, context: CapabilityPromptContext
    ) -> None:
        """Bind both body instructions and description to the same callable scope."""
        resource.attach_prompt_context(context)
        resource.render_description(lambda text: self.render(text, context))

    def render(self, text: str, context: CapabilityPromptContext) -> str:
        rendered, references = parse_references(render_conditions(text, context))
        context.validate_references(
            references, (hds.get_name() for hds in self._data.hds)
        )
        return rendered

    def skill_context(
        self, name: str, direct_tools: Iterable[str]
    ) -> CapabilityPromptContext:
        return self.context(self._skills[name], direct_tools=direct_tools)


@dataclass(frozen=True, slots=True)
class PreparedCapabilityPrompts:
    capabilities: CapabilitiesData
    hds: HDS
    prompts: CapabilityPrompts


def prepare_node_prompts(
    data: CapabilitiesData,
    hds_name: str,
    runtime_tools: tuple[Tool, ...],
    initial_depth: int,
) -> PreparedCapabilityPrompts:
    capabilities = data.copy()
    hds = capabilities.get_hds(hds_name)
    runtime_names = frozenset(tool.get_name() for tool in runtime_tools)
    prompts = CapabilityPrompts(
        capabilities, runtime_names, exit_tool_name=hds.get_exit_tool_name()
    )
    prompts.attach()
    direct = (
        *hds.get_sub_tool_names(),
        *((hds.get_exit_tool_name(),) if initial_depth > 0 else ()),
    )
    context = prompts.context(hds, direct_tools=direct)
    prompts.attach_resource(hds, context)
    index = CapabilitiesIndex.build(capabilities)
    CapabilitiesValidator(index, runtime_names).validate(
        (*capabilities.hds, *capabilities.deep_skills)
    )
    SkillCatalogAttacher(
        index,
        parameter_descriptions_in_schema=capabilities.parameter_descriptions_in_schema,
    ).attach_to_all(capabilities.deep_skills)
    for tool in runtime_tools:
        tool.render_description(lambda text: prompts.render(text, context))
    return PreparedCapabilityPrompts(capabilities, hds, prompts)
