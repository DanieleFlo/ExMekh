from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any
from xml.etree.ElementTree import Element, SubElement

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.tools.deep_tool import DeepTool
from exmekh_lib.utils.xml import append_text_element, render_xml

from .main import run

if TYPE_CHECKING:
    from exmekh_lib.core.agents.node.node import Node


class SubAgentTool(DeepTool):
    """Delegate a user message to one of a node's direct children."""

    NAME = RUNTIME.sub_agent_tool_name
    _ARGUMENT_NAMES = RUNTIME.sub_agent_argument_names

    def __init__(self, children: Sequence[Node], work_root: Path | None = None) -> None:
        child_names = tuple(child.name_hds for child in children)
        if not child_names:
            raise ValueError(RUNTIME.sub_agent_children_required)
        if len(child_names) != len(set(child_names)):
            duplicates = sorted(
                name for name in set(child_names) if child_names.count(name) > 1
            )
            raise ValueError(
                RUNTIME.sub_agent_duplicate_names.format(names=", ".join(duplicates))
            )

        self._children = dict(zip(child_names, children, strict=True))
        self._child_names = child_names
        super().__init__(
            run,
            work_root or Path.cwd(),
            self._call_sub_agent,
            description=self._describe_children({}),
        )

    def set_child_descriptions(self, descriptions: dict[str, str]) -> None:
        self.description = self._describe_children(descriptions)

    def _describe_children(self, descriptions: dict[str, str]) -> str:
        agents = Element("sub_agents")
        for name in self._child_names:
            agent = SubElement(agents, "sub_agent")
            append_text_element(agent, "name", name)
            append_text_element(agent, "description", descriptions.get(name, ""))
        xml = render_xml(agents)
        return RUNTIME.sub_agent_catalog_description.format(
            description=RUNTIME.call_sub_agent_description,
            names=", ".join(self._child_names),
            catalog=xml,
        )

    @property
    def child_names(self) -> tuple[str, ...]:
        return self._child_names

    def get_type_hints(self) -> dict[str, Any]:
        return {
            "sub_agent": Annotated[
                str,
                RUNTIME.sub_agent_name_guidance.format(
                    names=", ".join(self._child_names)
                ),
                {"enum": list(self._child_names)},
            ],
            "text": Annotated[
                str,
                RUNTIME.sub_agent_text_guidance,
                {"minLength": 1, "pattern": r"\S"},
            ],
            "return": str,
        }

    def execute(self, **arguments: Any) -> str:
        unexpected = sorted(set(arguments).difference(self._ARGUMENT_NAMES))
        if unexpected:
            return RUNTIME.sub_agent_argument_error.format(
                tool_name=self.get_name(),
                unexpected=", ".join(unexpected),
            )

        sub_agent = arguments.get("sub_agent")
        if not isinstance(sub_agent, str) or sub_agent not in self._children:
            return self._invalid_sub_agent_message()

        text = arguments.get("text")
        if not isinstance(text, str) or not text.strip():
            return RUNTIME.sub_agent_text_error

        return super().execute(**arguments)

    def _call_sub_agent(self, sub_agent: str, text: str) -> str:
        response = self._children[sub_agent].run(text)
        content = response.get_message()
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(RUNTIME.sub_agent_empty_response.format(name=sub_agent))
        return content

    def _invalid_sub_agent_message(self) -> str:
        return RUNTIME.sub_agent_selector_error.format(
            names=", ".join(self._child_names)
        )
