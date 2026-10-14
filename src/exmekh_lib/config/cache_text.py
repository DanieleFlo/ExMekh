"""Fixed diagnostic and log messages for the cache domain."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CacheText:
    output_root_must_not_redirect_to_another_location: str = (
        "Output root must not redirect to another location"
    )
    external_output_excludes_original_result: str = (
        "An external output record must not retain the original result"
    )
    cache_directory_must_be_a_path_or_none: str = (
        "cache_directory must be a Path or None"
    )
    cache_identifier_must_be_a_non_empty_string: str = (
        "cache identifier must be a non-empty string"
    )
    cached_images_must_be_png_data_urls: str = "Cached images must be PNG data URLs"
    cached_interactive_record_must_be_an_object: str = (
        "Cached interactive record must be an object"
    )
    cached_user_message_must_be_a_string: str = (
        "Cached user message {name} must be a string"
    )
    cached_user_message_parts_must_be_a_list: str = (
        "Cached user message parts must be a list"
    )
    cannot_read_tool_call_record: str = "Cannot read tool call record %s: %s"
    capacity_must_be_an_integer: str = "capacity must be an integer"
    capacity_must_be_positive: str = "capacity must be positive"
    directory_must_be_a_path: str = "directory must be a Path"
    execution_must_be_a_dictionary_or_none: str = (
        "execution must be a dictionary or None"
    )
    external_output_must_be_externaltooloutput_or_none: str = (
        "external_output must be ExternalToolOutput or None"
    )
    fingerprint_must_be_a_string: str = "fingerprint must be a string"
    identity_must_be_a_non_empty_string: str = "identity must be a non-empty string"
    identity_must_be_a_string: str = "identity must be a string"
    invalid_cached_user_message: str = "Invalid cached user message: {error}"
    missing_required_keys_in_tool_call_record: str = (
        "Missing required keys in tool call record: {missing}"
    )
    node_cache_directory_must_not_redirect_to_another_location: str = (
        "node cache directory must not redirect to another location"
    )
    node_id_must_be_a_non_empty_string: str = "node_id must be a non-empty string"
    output_directory_must_not_redirect_to_another_location: str = (
        "Output directory must not redirect to another location"
    )
    output_file_session_is_closed: str = "Output file session is closed"
    output_lease_must_not_be_a_symlink: str = "Output lease must not be a symlink"
    output_root_must_not_be_a_symlink: str = "Output root must not be a symlink"
    previously_exported_output_files_are_unavailable: str = (
        "Previously exported output files are unavailable"
    )
    record_dict_must_be_a_dictionary: str = "record_dict must be a dictionary"
    record_must_be_a_toolcallrecord_instance: str = (
        "record must be a ToolCallRecord instance"
    )
    serialized_used_tools_must_be_a_list_or_none: str = (
        "Serialized used_tools must be a list or None"
    )
    session_is_closed: str = "Session is closed"
    session_key_does_not_match_directory_or_process: str = (
        "session key does not match directory or process"
    )
    skill_path_must_be_a_string: str = "skill_path must be a string"
    skill_path_must_be_an_absolute_skill_path: str = (
        "skill_path must be an absolute skill path"
    )
    source_must_be_tool_deepskill_or_usermessage: str = (
        "source must be 'Tool', 'DeepSkill', or 'UserMessage'"
    )
    thinking_must_be_a_string_or_none: str = "thinking must be a string or None"
    tokenizer_id_space_exhausted_for_this_session: str = (
        "Tokenizer ID space exhausted for this session"
    )
    tool_call_id_must_be_a_non_empty_string: str = (
        "tool_call_id must be a non-empty string or None"
    )
    tool_call_must_be_toolcall_or_implement_calledtoolmetadata: str = (
        "tool_call must be ToolCall or implement CalledToolMetadata"
    )
    toolcallrecord_args_must_be_a_dictionary: str = (
        "ToolCallRecord 'args' must be a dictionary"
    )
    toolcallrecord_id_cannot_be_empty: str = "ToolCallRecord 'id' cannot be empty"
    toolcallrecord_id_must_be_a_string: str = "ToolCallRecord 'id' must be a string"
    toolcallrecord_name_cannot_be_empty: str = "ToolCallRecord 'name' cannot be empty"
    toolcallrecord_name_must_be_a_string: str = "ToolCallRecord 'name' must be a string"
    unsupported_cached_user_content_type: str = (
        "Unsupported cached user content type: {unknown}"
    )
    used_tools_must_be_a_tuple_of_tooloutputreference_or: str = (
        "used_tools must be a tuple of ToolOutputReference or None"
    )
    user_message_must_be_a_dictionary_or_none: str = (
        "user_message must be a dictionary or None"
    )


CACHE_TEXT = CacheText()
