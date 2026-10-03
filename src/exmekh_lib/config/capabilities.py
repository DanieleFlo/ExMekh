from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CapabilityFormat:
    """Fixed names used to discover and render bundled capabilities."""

    root_folder: str = "capabilities"
    deep_skill_folder: str = "deep-skill"
    tools_folder: str = "tools"
    exit_tool_folder: str = "exit-tool"
    hds_folder: str = "hds"
    markdown_suffix: str = ".md"
    tool_entrypoint_filename: str = "main.py"
    exit_tool_instructions_filename: str = "instructions.md"
    deep_tool_attribute: str = "__exmekh_lib_deep_tool__"
    exit_tool_attribute: str = "__exmekh_lib_exit_tool__"
    catalog_tag: str = "catalog"
    description_field: str = "description"
    sub_skills_field: str = "sub-skills"
    sub_tools_field: str = "sub-tools"
    inherits_parent_tools_field: str = "inherits-parent-tools"
    exit_tool_field: str = "exit-tool"
    agent_context_id_field: str = "agent-context-id"

    @property
    def bundled_root(self) -> Path:
        return Path(__file__).resolve().parents[1] / self.root_folder


CAPABILITY_FORMAT = CapabilityFormat()


@dataclass(frozen=True, slots=True)
class CapabilityText:
    """Fixed text presented by bundled capabilities."""

    skill_only_availability: str = "Available in every skill."
    exit_tool_description: str = (
        "Finishes the current skill and returns its result to the caller."
    )


CAPABILITY_TEXT = CapabilityText()
