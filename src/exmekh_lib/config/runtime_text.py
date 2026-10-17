"""Fixed diagnostic and log messages for the runtime domain."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeText:
    ambiguous_search_tokenizer: str = (
        "Supply provider or tokenizer to HarnessMemoryTools, not both"
    )
    timestamp_timezone_required: str = "Runtime timestamps must be timezone-aware"
    invalid_duration: str = "Runtime durations must be finite nonnegative numbers"
    search_previews_require_a_tokenizer: str = "Search previews require a tokenizer; supply provider or tokenizer to HarnessMemoryTools."
    execution_manager_required: str = (
        "Publishing execution output requires an execution manager"
    )
    final_output_required: str = "Only a finished execution can publish final output"
    unknown_output_id: str = "Unknown output ID '{output_id}'"
    a_published_execution_must_have_an_output_id: str = (
        "A published execution must have an output ID"
    )
    agent_context_id_cannot_be_empty: str = "agent_context_id cannot be empty"
    agent_context_id_must_be_a_string: str = "agent_context_id must be a string"
    agentic_scope_path_depth_scope: str = "Agentic scope: path=%s depth=%d scope=%s"
    ambiguous_tool_call_id: str = "Ambiguous tool call ID '{call_id}'"
    call_sub_agent_requires_at_least_one_child_node: str = (
        "call_sub_agent requires at least one child node"
    )
    cancellation_callback_must_return_a_bool: str = (
        "Cancellation callback must return a bool"
    )
    cancellation_failed: str = "Cancellation failed: {error}"
    cannot_restore_history_for_a_non_persistent_node: str = (
        "Cannot restore history for a non-persistent node"
    )
    capability_is_not_allowed_at_allowed_tools_allowed_skills: str = "Capability '{call_get_name}' is not allowed at {skill_path}; allowed_tools={allowed_tools}, allowed_skills={allowed_skills}. Choose a listed tool or enter an allowed skill before using its tools; the rejected batch was not executed."
    children_must_be_a_sequence_of_node: str = "children must be a sequence of Node"
    children_must_contain_only_node_instances: str = (
        "children must contain only Node instances"
    )
    closed_deep_skill_name_opening_call_id_start_index: str = "Closed deep skill: name=%s opening_call_id=%s start_index=%d end_index=%d transcript_messages=%d depth=%d"
    completed_node_run_node_depth_root_executions: str = (
        "Completed node run: node=%s depth=%d root_executions=%d"
    )
    deep_thinking_summary_unavailable: str = "Deep-thinking summary unavailable"
    deep_thinking_token_count_unavailable: str = "Deep-thinking token count unavailable"
    depth_cannot_be_negative: str = "depth cannot be negative"
    depth_must_be_an_integer: str = "depth must be an integer"
    directory_must_be_a_path_or_none: str = "directory must be a Path or None"
    either_directory_tokenizer_identity_or_tokenizer_must_be_provided: str = (
        "Either 'directory', 'tokenizer_identity', or 'tokenizer' must be provided"
    )
    error_correct_the_error_and_call_again: str = "Error {called_tool_name}: {message} Correct the error and call {called_tool_name_2} again."
    executing_exit_tool_name_call_id_active_skills_depth: str = (
        "Executing exit tool: name=%s call_id=%s active_skills=%d depth=%d"
    )
    executing_tool_name_call_id_depth: str = (
        "Executing tool: name=%s call_id=%s depth=%d"
    )
    execution_history_position_is_out_of_range: str = (
        "execution history position is out of range"
    )
    exit_tool_executed_without_an_active_deep_skill_call: str = (
        "Exit tool executed without an active deep skill: call_id=%s depth=%d"
    )
    folder_name_must_be_a_non_empty_string: str = (
        "folder_name must be a non-empty string"
    )
    folder_name_must_be_a_portable_directory_name: str = (
        "folder_name must be a portable directory name"
    )
    harness_runtime_tool_names_are_reserved: str = (
        "Harness runtime tool names are reserved: {names}"
    )
    harness_runtime_tool_names_must_be_unique: str = (
        "Harness runtime tool names must be unique: {names}"
    )
    invalid_arguments: str = "Invalid arguments: {error}"
    limit_must_be_between_and: str = "limit must be between {runtime_history_min_page_limit} and {runtime_memory_max_page_limit}"
    managed_tool_execution_is_missing_its_output_id: str = (
        "Managed tool execution is missing its output ID"
    )
    name_hds_cannot_be_empty: str = "name_hds cannot be empty"
    name_hds_must_be_a_string: str = "name_hds must be a string"
    no_tool_call_recorded_for_id_at: str = (
        "No tool call recorded for ID '{call_id}' at {directory}"
    )
    node_must_be_initialized_before_accessing_its_hds: str = (
        "Node must be initialized before accessing its HDS"
    )
    node_must_be_initialized_before_accessing_tools: str = (
        "Node must be initialized before accessing tools"
    )
    node_must_be_initialized_before_controlling_tool_work: str = (
        "Node must be initialized before controlling tool work"
    )
    node_must_be_initialized_before_reading_tool_output: str = (
        "Node must be initialized before reading tool output"
    )
    node_must_be_initialized_before_restoring_history: str = (
        "Node must be initialized before restoring history"
    )
    node_must_be_initialized_before_run: str = "Node must be initialized before run"
    node_must_be_initialized_before_updating_sub_agents: str = (
        "Node must be initialized before updating sub-agents"
    )
    node_supports_only_deep_skills_exit_tools_and_tools: str = (
        "Node supports only deep skills, exit tools, and tools"
    )
    opened_deep_skill_name_call_id_start_index_depth: str = (
        "Opened deep skill: name=%s call_id=%s start_index=%d depth=%d"
    )
    output_id_unavailable_using_a_uuid: str = "Output ID unavailable; using a UUID: %s"
    partial_output_must_contain_text_and_a_non_decreasing: str = (
        "Partial output must contain text and a non-decreasing integer cursor"
    )
    persistent_history_must_be_a_bool: str = "persistent_history must be a bool"
    preview_limit_must_be_a_positive_integer: str = (
        "preview limit must be a positive integer"
    )
    provider_generated_an_invalid_output_id: str = (
        "Provider generated an invalid output ID"
    )
    provider_must_be_a_provider: str = "provider must be a Provider"
    skill_path_must_be_an_absolute_skill_path: str = (
        "skill_path must be an absolute skill path"
    )
    start_position_must_be_a_positive_integer_or_none: str = (
        "start_position must be a positive integer or None"
    )
    starting_node_run_node_depth: str = "Starting node run: node=%s depth=%d"
    starting_tool_execution_name_call_id_path: str = (
        "Starting tool execution: name=%s call_id=%s path=%s"
    )
    thinking_annotation_unavailable: str = "Thinking annotation unavailable: %s"
    too_many_active_tool_executions: str = "Too many active tool executions"
    tool_call_id_already_has_active_work: str = (
        "Tool call ID '{call_id}' already has active work"
    )
    tool_calls: str = "Tool calls: {names}"
    tool_execution_failed_without_a_diagnostic: str = (
        "Tool execution failed without a diagnostic."
    )
    tool_execution_is_still_running: str = "Tool execution is still running."
    tool_execution_was_cancelled: str = "Tool execution was cancelled."
    tool_is_not_a_valid_deep_skill_exit_tool: str = "Tool '{tool_name}' is not a valid deep skill, exit tool, or tool. Choose an exact registered name from: {available}. Availability still depends on the current skill."
    tool_is_not_a_valid_deep_skill_exit_tool_2: str = (
        "Tool '{tool_name}' is not a valid deep skill, exit tool, or tool instance"
    )
    tool_output_summary_unavailable_returning_full_output: str = (
        "Tool output summary unavailable; returning full output"
    )
    unknown_tool_call_id: str = "Unknown tool call ID '{call_id}'"
    use_text_result_items_or_text_items_id_output: str = '{diagnostic} Use {tool_name}(text="<result>", items=[]) or {tool_name_2}(text="… {{{{<id-output>}}}} …", items=["<id-output>"].'


RUNTIME_TEXT = RuntimeText()
