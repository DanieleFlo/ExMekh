"""One output's identity, companions and text needed at the presentation boundary."""

from dataclasses import dataclass

from exmekh_lib.core.models.tool import PreformattedToolResult
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.core.models.xml_header import XmlHeader


@dataclass(frozen=True, slots=True)
class ToolOutputPresentation:
    identity: ToolOutputReference
    result: PreformattedToolResult
    budget_text: str | None = None

    @property
    def body(self) -> str:
        body = XmlHeader.body_from_content(self.result.content)
        return self.result.content if body is None else body

    def replace_body(self, body: str) -> str:
        content = self.result.content
        if XmlHeader.has_leading_header(content):
            return XmlHeader.replace_body_in_content(content, body)
        return (
            XmlHeader("Tool", self.identity.name, self.identity.id)
            .add_field("response-type", "final")
            .prepend_to(body)
        )
