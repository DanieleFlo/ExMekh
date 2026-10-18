"""Fixed diagnostic and log messages for the provider domain."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderText:
    api_key_environment_variable_must_be_a_non_empty: str = (
        "API key environment variable must be a non-empty string"
    )
    api_key_environment_variable_must_be_a_string: str = (
        "API key environment variable must be a string"
    )
    api_key_must_be_a_string_or_none: str = "API key must be a string or None"
    api_must_be_chat_completions_or_responses: str = (
        "API must be 'chat_completions' or 'responses'"
    )
    argument_does_not_match_any_allowed_type: str = (
        "Argument '{path}' does not match any allowed type"
    )
    argument_has_an_invalid_value: str = "Argument '{path}' has an invalid value"
    argument_has_the_wrong_array_length: str = (
        "Argument '{path}' has the wrong array length"
    )
    argument_has_unknown_fields: str = "Argument '{path}' has unknown fields: {names}"
    argument_is_invalid: str = "Argument '{path}' is invalid"
    argument_is_missing_fields: str = "Argument '{path}' is missing fields: {names}"
    argument_is_not_a_valid_enum_value: str = (
        "Argument '{path}' is not a valid enum value"
    )
    argument_is_not_an_allowed_literal: str = (
        "Argument '{path}' is not an allowed literal"
    )
    argument_must_be_a_boolean: str = "Argument '{path}' must be a boolean"
    argument_must_be_a_number: str = "Argument '{path}' must be a number"
    argument_must_be_a_string: str = "Argument '{path}' must be a string"
    argument_must_be_an_array: str = "Argument '{path}' must be an array"
    argument_must_be_an_integer: str = "Argument '{path}' must be an integer"
    argument_must_be_an_object: str = "Argument '{path}' must be an object"
    argument_must_be_null: str = "Argument '{path}' must be null"
    base_url_environment_variable_must_be_a_non_empty: str = (
        "Base URL environment variable must be a non-empty string"
    )
    base_url_environment_variable_must_be_a_string: str = (
        "Base URL environment variable must be a string"
    )
    base_url_must_be_a_string_or_none: str = "Base URL must be a string or None"
    base_url_must_be_a_valid_http_s_url: str = "Base URL must be a valid HTTP(S) URL"
    cancellation_was_not_confirmed: str = (
        "Cancellation was not confirmed: {cleanup_error}"
    )
    client_must_be_an_instance_of_openai: str = "Client must be an instance of OpenAI"
    complete_interactive_call_has_no_matching_frozen_authorizations: str = (
        "Complete interactive call has no matching frozen authorizations"
    )
    completed_turn_omitted_a_suspended_interactive_call: str = (
        "Completed turn omitted a suspended interactive call"
    )
    continuation_id_changed_during_the_logical_call: str = (
        "Continuation ID changed during the logical call"
    )
    control_operation_returned_no_receipt: str = "Control operation returned no receipt"
    custom_tool_calls_are_deprecated_and_will_be_removed: str = (
        "Custom tool calls are deprecated and will be removed in a future version"
    )
    duplicate_call_id_in_interactive_result: str = (
        "Duplicate call ID in interactive result"
    )
    duplicate_consumed_interactive_exchange: str = (
        "Duplicate/consumed interactive exchange"
    )
    duplicate_json_property: str = "Duplicate JSON property: {key}"
    edit_indices_must_be_consecutive_within_the_same_call: str = (
        "Edit indices must be consecutive within the same call"
    )
    environment_variable_is_not_set: str = (
        "Environment variable '{base_url_env_var}' is not set."
    )
    expected_a_chatcompletionchunk_instance: str = (
        "Expected a ChatCompletionChunk instance"
    )
    expected_a_providerrequest_instance: str = "Expected a ProviderRequest instance"
    expected_a_tool_instance: str = "Expected a Tool instance"
    expected_a_tools_instance: str = "Expected a Tools instance"
    expected_an_openai_response_instance: str = "Expected an OpenAI Response instance"
    expected_role_assistant_got: str = "Expected role 'assistant', got '{role}'"
    expected_role_function_got: str = "Expected role 'function', got '{role}'"
    expected_role_system_or_developer_got: str = (
        "Expected role 'system' or 'developer', got '{role}'"
    )
    expected_role_tool_got: str = "Expected role 'tool', got '{role}'"
    expected_role_user_got: str = "Expected role 'user', got '{role}'"
    interactive_arguments_must_be_a_json_object: str = (
        "Interactive arguments must be a JSON object"
    )
    interactive_arguments_must_be_encoded_json: str = (
        "Interactive arguments must be encoded JSON"
    )
    interactive_call_id_must_be_a_nonempty_string: str = (
        "Interactive call ID must be a nonempty string"
    )
    interactive_document_must_be_the_only_exact_string_argument: str = (
        "Interactive document must be the only exact string argument"
    )
    interactive_exchange_budget_exhausted: str = "Interactive exchange budget exhausted"
    interactive_http_output_exceeds_the_byte_budget: str = (
        "Interactive HTTP output exceeds the byte budget"
    )
    interactive_logical_call_timeout: str = "Interactive logical-call timeout"
    interactive_logical_call_timeout_while_reading_http_output: str = (
        "Interactive logical-call timeout while reading HTTP output"
    )
    interactive_output_must_be_a_list_of_json_objects: str = (
        "Interactive output must be a list of JSON objects"
    )
    interactive_response_must_be_a_json_object: str = (
        "Interactive response must be a JSON object"
    )
    interactive_tools_require_the_ds4_responses_v2_api: str = (
        "Interactive tools require the DS4 Responses v2 API"
    )
    invalid_agentic_scope: str = "Invalid agentic scope"
    invalid_arguments_for_tool: str = (
        "Invalid arguments for tool '{tool_get_name}': {error}"
    )
    invalid_interactive_document: str = "Invalid interactive document: {error}"
    invalid_pause: str = "Invalid pause {field}"
    invalid_server_snapshot_byte_limit: str = "Invalid server snapshot byte limit"
    json_schema_reference_is_not_an_object: str = (
        "JSON schema reference is not an object: {reference}"
    )
    local_provider_request_failed: str = "Local provider request failed"
    malformed_interactive_argument_json: str = "Malformed interactive argument JSON"
    malformed_interactive_capabilities: str = "Malformed interactive capabilities"
    malformed_interactive_capability_lists: str = (
        "Malformed interactive capability lists"
    )
    malformed_interactive_limits: str = "Malformed interactive limits"
    malformed_interactive_response_encoding_or_json: str = (
        "Malformed interactive response encoding or JSON"
    )
    malformed_server_capabilities: str = "Malformed server capabilities"
    missing_interactive_acceptance_object: str = "Missing interactive acceptance object"
    missing_or_invalid_interactive_acceptance: str = (
        "Missing or invalid interactive acceptance"
    )
    model_must_be_a_non_empty_string: str = "Model must be a non-empty string"
    model_must_be_a_string: str = "Model must be a string"
    model_name_is_not_set: str = "Model name is not set."
    model_not_found_available_models: str = (
        "Model '{model}' not found. Available models: {names}"
    )
    no_choices_returned_from_the_provider: str = "No choices returned from the provider"
    no_chunks_returned_from_the_provider: str = "No chunks returned from the provider"
    no_completed_response_returned_from_the_provider: str = (
        "No completed response returned from the provider"
    )
    no_interactive_continuation_was_accepted: str = (
        "No interactive continuation was accepted"
    )
    no_models_are_available_from_the_provider: str = (
        "No models are available from the provider."
    )
    no_terminal_receipt_before_the_logical_call_timeout: str = (
        "No terminal receipt before the logical-call timeout"
    )
    null_uses_the_default: str = "Null uses the default: {parameter_default}."
    pause_request_must_be_a_json_object: str = "Pause request must be a JSON object"
    pause_requires_a_path_and_positive_edit_index: str = (
        "Pause requires a path and positive edit_index"
    )
    pause_tool_behavior_is_outside_the_configured_scope: str = (
        "Pause tool/behavior is outside the configured scope"
    )
    provide_requires_exact_unicode_text: str = "Provide requires exact Unicode text"
    provider_model_catalog_data_must_be_a_list: str = (
        "Provider model catalog data must be a list"
    )
    provider_response_did_not_complete: str = (
        "Provider response did not complete: {response_status}"
    )
    provider_response_failed: str = "Provider response failed: {response_error_message}"
    provider_returned_a_malformed_model_catalog: str = (
        "Provider returned a malformed model catalog"
    )
    provider_returned_a_malformed_model_identifier: str = (
        "Provider returned a malformed model identifier"
    )
    reasoning_effort_must_be_a_reasoningeffort_instance: str = (
        "Reasoning effort must be a ReasoningEffort instance"
    )
    reasoning_effort_must_be_a_reasoningeffort_or_none: str = (
        "Reasoning effort must be a ReasoningEffort or None"
    )
    reasoning_effort_must_be_a_string_or_an_instance: str = (
        "Reasoning effort must be a string or an instance of ReasoningEffort"
    )
    rejected_call_does_not_match_the_suspended_edit: str = (
        "Rejected call does not match the suspended edit"
    )
    rejected_result_does_not_match_the_rejected_exchange: str = (
        "Rejected result does not match the rejected exchange"
    )
    rejected_turn_omitted_an_earlier_complete_interactive_call: str = (
        "Rejected turn omitted an earlier complete interactive call"
    )
    response_model_must_be_a_pydantic_basemodel_type: str = (
        "Response model must be a Pydantic BaseModel type"
    )
    responses_api_does_not_support_legacy_function_messages: str = (
        "Responses API does not support legacy function messages"
    )
    server_continued_a_call_after_its_resource_was_rejected: str = (
        "Server continued a call after its resource was rejected"
    )
    server_model_does_not_support_interactive_responses_v2_no: str = "Server/model does not support interactive Responses v2; no ordinary-call fallback is permitted"
    snapshot_exceeds_the_negotiated_byte_limit: str = (
        "Snapshot exceeds the negotiated byte limit"
    )
    sse_data_must_be_a_json_object: str = "SSE data must be a JSON object"
    steering_invalidated_this_inference_before_dispatch: str = (
        "Steering invalidated this inference before dispatch"
    )
    stream_ended_without_acceptance_or_a_confirmed_result: str = (
        "Stream ended without acceptance or a confirmed result"
    )
    strict_must_be_a_boolean: str = "Strict must be a boolean"
    structured_assistant_content_must_be_a_pydantic_basemodel_got: str = "Structured assistant content must be a Pydantic BaseModel, got {message_content_type!r}"
    structured_assistant_content_must_be_a_pydantic_model: str = (
        "Structured assistant content must be a Pydantic model"
    )
    structured_responses_do_not_support_interactive_tool_continuations: str = (
        "Structured responses do not support interactive tool continuations"
    )
    tool_arguments_are_not_valid_json: str = (
        "Tool '{call_get_name}' arguments are not valid JSON"
    )
    tool_arguments_must_be_a_json_object: str = (
        "Tool '{call_get_name}' arguments must be a JSON object"
    )
    tool_call_must_be_an_openai_function_tool_call: str = (
        "tool_call must be an OpenAI function tool call"
    )
    tool_call_must_be_an_openai_response_function_tool: str = (
        "tool_call must be an OpenAI Response function tool call"
    )
    tool_has_unsupported_parameter: str = (
        "Tool '{tool_get_name}' has unsupported parameter '{parameter_name}'"
    )
    tool_is_not_registered: str = "Tool '{name}' is not registered"
    tool_parameter_must_have_a_type_annotation: str = (
        "Tool parameter '{parameter_name}' must have a type annotation"
    )
    tools_are_required_to_resolve_a_function_tool_call: str = (
        "Tools are required to resolve a function tool call"
    )
    unexpected_interactive_arguments: str = "Unexpected interactive arguments"
    unresolvable_json_schema_reference: str = (
        "Unresolvable JSON schema reference: {reference}"
    )
    unsupported_json_schema_reference: str = (
        "Unsupported JSON schema reference: {reference}"
    )
    unsupported_message_type: str = "Unsupported message type: {message_type}"
    unsupported_openai_tool_call_type: str = (
        "Unsupported OpenAI tool call type: {tool_call_type_name}"
    )
    unsupported_scalar_annotation: str = "Unsupported scalar annotation"
    unsupported_tool_call_type: str = (
        "Unsupported tool call type: {tool_call_type_name}"
    )
    unsupported_type_for_argument: str = (
        "Unsupported type for argument '{path}': {annotation!r}"
    )
    unsupported_user_content_type: str = (
        "Unsupported user content type: {part_type_name}"
    )


PROVIDER_TEXT = ProviderText()
