"""Discoverable, immutable names of the capabilities shipped with the library."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DefaultCapabilityNames:
    main: str = "main"
    deep_thinking: str = "deep-thinking"
    file_processing: str = "file-processing"
    search_web: str = "search-web"
    read_web_page: str = "read-web-page"
    interact_web_page: str = "interact-web-page"
    calculate: str = "calculate"
    date_time: str = "date-time"
    inspect_file: str = "inspect-file"
    read_file: str = "read-file"
    edit_file: str = "edit-file"
    new_file: str = "new-file"
    delete_file: str = "delete-file"
    logic_check: str = "logic-check"
    logic_consistency: str = "logic-consistency"
    logic_counterexample: str = "logic-counterexample"
    exit_agent: str = "exit_agent"
    get_execution_history: str = "get_execution_history"
    search_tool_outputs: str = "search_tool_outputs"
    read_tool_output: str = "read_tool_output"
    control_tool_execution: str = "control_tool_execution"
    bash: str = "bash"
    call_sub_agent: str = "call_sub_agent"


DEFAULT_CAPABILITIES = DefaultCapabilityNames()
