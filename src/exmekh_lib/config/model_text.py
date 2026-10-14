"""Fixed diagnostic and log messages for the model domain."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelText:
    image_hash: str = "Image hash: {hash}"
    all_keys_in_args_must_be_strings: str = "all keys in args must be strings"
    all_keys_in_fields_must_be_strings: str = "all keys in fields must be strings"
    all_messages_must_be_message_instances: str = (
        "All messages must be Message instances"
    )
    args_must_be_a_dictionary_or_none: str = "args must be a dictionary or None"
    argument_guidance_must_be_a_string: str = "argument_guidance must be a string"
    assistant_content_must_be_a_string: str = "Assistant content must be a string"
    assistant_tool_calls_must_be_a_list_of_toolcall: str = (
        "Assistant tool_calls must be a list of ToolCall objects"
    )
    attached_tool_results_require_a_usermessage: str = (
        "Attached tool results require a UserMessage"
    )
    background_enabled_must_be_a_bool: str = "background_enabled must be a bool"
    beforetool_handle_must_return_toolarguments_toolresponse_or_none: str = (
        "BeforeTool.handle must return ToolArguments, ToolResponse, or None"
    )
    cache_result_must_be_a_bool: str = "cache_result must be a bool"
    calling_provider_messages_tool_choice: str = (
        "Calling provider: messages=%d tool_choice=%s"
    )
    compacted_message_history_start_index_previous_size_replacement_size: str = "Compacted message history: start_index=%d previous_size=%d replacement_size=%d new_size=%d"
    content_must_be_a_subclass_of_basemodel_got: str = (
        "content must be a subclass of BaseModel, got {content_type!r}"
    )
    delivery_failed_must_be_a_boolean: str = "delivery_failed must be a boolean"
    document_extension_mime_type_hash: str = "Document: {part_name}, extension: {part_extension}, mime_type: {part_mime_type}, Hash: {hash}"
    execution_mode_must_be_wait_or_background: str = (
        "execution_mode must be 'wait' or 'background'"
    )
    expected_a_clientrequest_instance: str = "Expected a ClientRequest instance"
    expected_a_message_instance: str = "Expected a Message instance"
    expected_a_non_empty_list_of_messages: str = "Expected a non-empty list of Messages"
    expected_a_provider_instance: str = "Expected a Provider instance"
    expected_a_systemmessage_instance: str = "Expected a SystemMessage instance"
    expected_a_toolmessage_instance: str = "Expected a ToolMessage instance"
    expected_a_user_message: str = "Expected a user message"
    external_output_must_be_externaltooloutput_or_none: str = (
        "external_output must be ExternalToolOutput or None"
    )
    external_output_must_contain_output_text_files: str = (
        "External output must contain output text files"
    )
    extra_provider_data_must_be_a_mapping: str = "Extra provider data must be a mapping"
    fields_must_be_a_dictionary: str = "fields must be a dictionary"
    first_item_in_history_must_have_the_system_role: str = (
        "First item in history must have the system role"
    )
    function_content_must_be_a_string: str = "Function content must be a string"
    function_name_must_be_a_string: str = "Function name must be a string"
    history_cannot_be_empty: str = "History cannot be empty"
    hooks_must_be_a_toolhooks_instance: str = "hooks must be a ToolHooks instance"
    id_cannot_be_empty: str = "id cannot be empty"
    id_must_be_a_string: str = "id must be a string"
    initial_message_must_be_a_usermessage: str = "Initial message must be a UserMessage"
    interactive_events_require_structured_control_metadata: str = (
        "Interactive events require structured control metadata"
    )
    interactive_must_be_an_interactivedefinition_or_none: str = (
        "interactive must be an InteractiveDefinition or None"
    )
    interactive_tools_must_execute_synchronously: str = (
        "Interactive tools must execute synchronously"
    )
    interactive_tools_require_a_node_runtime_handler: str = (
        "Interactive tools require a node runtime handler"
    )
    item_must_be_an_instance_of_message: str = "Item must be an instance of Message"
    item_must_be_an_instance_of_message_or_a: str = (
        "Item must be an instance of Message or a list of Message instances"
    )
    modelthinking_thoughts_must_be_a_string_or_none: str = (
        "ModelThinking thoughts must be a string or None"
    )
    modelusage_completion_tokens_must_be_an_int_or_none: str = (
        "ModelUsage completion_tokens must be an int or None"
    )
    modelusage_prompt_tokens_must_be_an_int_or_none: str = (
        "ModelUsage prompt_tokens must be an int or None"
    )
    modelusage_total_tokens_must_be_an_int_or_none: str = (
        "ModelUsage total_tokens must be an int or None"
    )
    name_cannot_be_empty: str = "name cannot be empty"
    name_must_be_a_string: str = "name must be a string"
    name_type_tool_description: str = (
        "Name: {name}\nType: Tool\nDescription: {description}..."
    )
    only_interactive_events_can_contain_control_metadata: str = (
        "Only interactive events can contain control metadata"
    )
    only_reasoning_and_response_events_can_contain_content: str = (
        "Only reasoning and response events can contain content"
    )
    only_steering_events_can_contain_a_message: str = (
        "Only steering events can contain a message"
    )
    only_totals_events_can_contain_usage: str = "Only totals events can contain usage"
    output_counts_must_be_non_negative_integers: str = (
        "Output counts must be non-negative integers"
    )
    output_file_counts_must_be_non_negative_integers: str = (
        "Output file counts must be non-negative integers"
    )
    output_file_path_must_be_non_empty: str = "Output file path must be non-empty"
    output_limit_must_be_positive: str = "Output limit must be positive"
    preformatted_tool_attachments_must_be_a_usermessage_or_none: str = (
        "Preformatted tool attachments must be a UserMessage or None"
    )
    preformatted_tool_content_must_be_a_string: str = (
        "Preformatted tool content must be a string"
    )
    provider_response_recorded_tool_calls_history_size: str = (
        "Provider response recorded: tool_calls=%d history_size=%d"
    )
    reasoning_and_response_events_require_content: str = (
        "Reasoning and response events require content"
    )
    reasoning_effort_override_must_be_a_non_empty_string: str = (
        "Reasoning effort override must be a non-empty string"
    )
    response_content_must_be_a_pydantic_basemodel: str = (
        "Response content must be a Pydantic BaseModel"
    )
    response_model_must_be_a_pydantic_basemodel_type: str = (
        "Response model must be a Pydantic BaseModel type"
    )
    response_model_must_be_a_subclass_of_basemodel_got: str = (
        "response_model must be a subclass of BaseModel, got {response_model!r}"
    )
    response_must_be_a_structuredassistantmessage: str = (
        "Response must be a StructuredAssistantMessage"
    )
    response_must_be_an_assistantmessage: str = "Response must be an AssistantMessage"
    source_must_be_a_string: str = "source must be a string"
    source_must_be_tool_deepskill_or_usermessage_got: str = (
        "source must be 'Tool', 'DeepSkill', or 'UserMessage', got '{source}'"
    )
    start_index_is_outside_the_history: str = "start_index is outside the history"
    start_index_must_be_an_integer: str = "start_index must be an integer"
    steering_events_require_a_usermessage: str = "Steering events require a UserMessage"
    stream_event_content_must_be_a_string_or_none: str = (
        "Stream event content must be a string or None"
    )
    stream_event_type_must_be_a_providerstreameventtype: str = (
        "Stream event type must be a ProviderStreamEventType"
    )
    stream_event_usage_must_be_a_modelusage_or_none: str = (
        "Stream event usage must be a ModelUsage or None"
    )
    streaming_must_be_a_boolean: str = "Streaming must be a boolean"
    strict_must_be_a_boolean: str = "Strict must be a boolean"
    structured_assistant_content_must_be_a_pydantic_basemodel: str = (
        "Structured assistant content must be a Pydantic BaseModel"
    )
    structured_assistant_tool_calls_must_be_a_list_of: str = (
        "Structured assistant tool_calls must be a list of ToolCall objects"
    )
    system_content_must_be_a_string: str = "System content must be a string"
    system_name_must_be_a_string: str = "System name must be a string"
    the_assistant_message_does_not_contain_the_batch_call: str = (
        "The assistant message does not contain the batch call"
    )
    the_content_must_not_be_empty: str = "The content must not be empty!"
    the_final_tool_response_does_not_belong_to_this: str = (
        "The final tool response does not belong to this batch"
    )
    the_last_message_must_be_a_tool_response: str = (
        "The last message must be a tool response"
    )
    the_last_message_must_be_an_assistant_message: str = (
        "The last message must be an assistant message"
    )
    the_latest_assistant_message_has_no_tool_calls: str = (
        "The latest assistant message has no tool calls"
    )
    the_summary_response_contains_no_text: str = "The summary response contains no text"
    the_summary_response_must_not_contain_tool_calls: str = (
        "The summary response must not contain tool calls"
    )
    the_tool_batch_has_no_assistant_message: str = (
        "The tool batch has no assistant message"
    )
    thinking_must_be_a_modelthinking_instance: str = (
        "Thinking must be a ModelThinking instance"
    )
    tool_attachments_must_be_a_usermessage_or_none: str = (
        "Tool attachments must be a UserMessage or None"
    )
    tool_call_id_must_be_a_string: str = "Tool call ID must be a string"
    tool_call_must_be_a_toolcall_instance: str = "tool_call must be a ToolCall instance"
    tool_call_name_does_not_match_tool_name: str = (
        "Tool call name '{call_name}' does not match tool name '{name}'"
    )
    tool_choice_must_be_a_string: str = "Tool choice must be a string"
    tool_content_must_be_a_string: str = "Tool content must be a string"
    tool_description_must_be_a_string: str = "Tool description must be a string"
    tool_must_be_a_tool_instance: str = "tool must be a Tool instance"
    tool_must_be_callable: str = "Tool must be callable"
    tool_name_must_be_a_string: str = "Tool name must be a string"
    tool_name_must_contain_to_letters_digits_underscores_or: str = (
        "Tool name must contain 1 to 64 letters, digits, underscores, or hyphens"
    )
    tool_names_must_be_unique: str = "Tool names must be unique"
    tool_output_reference_cannot_be_empty: str = (
        "Tool output reference {field_name} cannot be empty"
    )
    tool_output_reference_must_be_a_dictionary: str = (
        "Tool output reference must be a dictionary"
    )
    tool_output_reference_must_be_a_string: str = (
        "Tool output reference {field_name} must be a string"
    )
    tool_output_reference_must_contain_only_name_and_id: str = (
        "Tool output reference must contain only name and id"
    )
    tool_parameters_cannot_use_reserved_execution_names: str = (
        "Tool parameters cannot use reserved execution names"
    )
    tool_response_recorded_name_call_id_history_size: str = (
        "Tool response recorded: name=%s call_id=%s history_size=%d"
    )
    tools_must_be_a_list_of_tool_instances: str = (
        "Tools must be a list of Tool instances"
    )
    tools_must_be_a_tools_instance: str = "Tools must be a Tools instance"
    totals_events_require_usage: str = "Totals events require usage"
    unsupported_user_content_type: str = (
        "Unsupported user content type: {part_type_name}"
    )
    usage_must_be_a_modelusage_instance: str = "Usage must be a ModelUsage instance"
    user_content_must_be_a_string_or_an_instance: str = "User content must be a string or an instance of TextsUserMessage, ImagesUserMessage, or FilesUserMessage, or a list of these types"
    user_name_must_be_a_string: str = "User name must be a string"
    wait_seconds_must_be_a_number: str = "wait_seconds must be a number"
    wait_seconds_must_be_between_and: str = (
        "wait_seconds must be between 0 and {runtime_max_tool_wait_seconds}"
    )
    wrong_message_sequence: str = (
        "Wrong message sequence: {previous_get_role_name} -> {current_get_role_name}"
    )


MODEL_TEXT = ModelText()
