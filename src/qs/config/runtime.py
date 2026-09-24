from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Fixed strings and limits required by the harness protocol."""

    required_tool_choice: str = "required"
    xml_header_tag: str = "header"
    empty_response_message: str = (
        "A response without a tool call and without content is not allowed. "
        "Respond with either a tool call or non-empty content."
    )
    default_agent_context_id: str = "default"
    history_tool_name: str = "get_execution_history"
    search_tool_name: str = "search_saved_tool_outputs"
    read_tool_name: str = "read_saved_tool_output"
    exit_tool_name: str = "exit_agent"
    sub_agent_tool_name: str = "call_sub_agent"
    sub_agent_argument_names: frozenset[str] = frozenset({"sub_agent", "text"})
    memory_preview_limit: int = 240

    @property
    def harness_output_guidance(self) -> str:
        return (
            "<harness-output-guidance>\n"
            "When finishing a skill, put in items only the IDs of complete outputs "
            "that the caller needs to read immediately. Prefer the saved-output-id shown "
            "in the history; a call ID is accepted if it is unambiguous. "
            "If a later skill needs the result, provide only its ID in text "
            f"and ask the caller to retrieve it with {self.read_tool_name}: repeating the output "
            "in the message consumes context. Use items=[] when text alone is enough.\n"
            "</harness-output-guidance>"
        )


RUNTIME = RuntimeSettings()
