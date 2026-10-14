"""Fixed diagnostic and log messages for the agent domain."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentText:
    a_running_tool_cannot_have_a_terminal_failure: str = (
        "A running tool cannot have a terminal failure"
    )
    argument_guidance_must_be_a_string: str = "argument_guidance must be a string"
    available_in_skill: str = "Available in skill '{skill_names}'."
    available_in_skills: str = "Available in skills {names}."
    available_tools: str = "Available tools: %r"
    background_enabled_must_be_a_bool: str = "background_enabled must be a bool"
    body_must_be_a_string: str = "body must be a string"
    call_id_identifies_multiple_outputs_use_a_specific_output: str = 'Call ID {identifier!r} identifies multiple outputs: {ids}. Use a specific output ID in items=["<id-output>"].'
    cannot_add_a_child_to_a_closed_deep_skill: str = (
        "Cannot add a child to a closed deep skill"
    )
    cannot_be_empty: str = "{label} cannot be empty"
    cannot_close_a_deep_skill_when_none_is_active: str = (
        "Cannot close a deep skill when none is active"
    )
    cannot_import_tool_entrypoint: str = (
        "Cannot import tool entrypoint: {entrypoint_path}"
    )
    cannot_record_a_tool_output_in_a_closed_deep: str = (
        "Cannot record a tool output in a closed deep skill"
    )
    capabilities_entry_must_be_a_directory: str = (
        "Capabilities entry '{capability_type_value}' must be a directory: {entry}"
    )
    capabilities_entry_must_not_be_a_symbolic_link: str = "Capabilities entry '{capability_type_value}' must not be a symbolic link: {entry}"
    capabilities_folder_must_be_named: str = "Capabilities folder must be named '{capabilities_folder_name}': {capabilities_path}"
    capabilities_path_does_not_exist: str = (
        "Capabilities path does not exist: {capabilities_path}"
    )
    capabilities_path_must_be_a_directory: str = (
        "Capabilities path must be a directory: {capabilities_path}"
    )
    capabilities_path_must_not_be_a_symbolic_link: str = (
        "Capabilities path must not be a symbolic link: {capabilities_path}"
    )
    capabilities_paths_cannot_be_empty: str = "capabilities_paths cannot be empty"
    capabilities_paths_must_be_a_sequence_of_path: str = (
        "capabilities_paths must be a sequence of Path"
    )
    capabilities_paths_must_contain_only_path_instances: str = (
        "capabilities_paths must contain only Path instances"
    )
    capability_cannot_be_included_or_overridden_and_excluded: str = (
        "Capability cannot be included or overridden and excluded: "
    )
    capability_file_must_be_a_markdown_file: str = (
        "Capability file must be a Markdown file: {markdown_path}"
    )
    catalog_already_attached_to_deep_skill: str = (
        "Catalog already attached to deep skill '{name}'"
    )
    catalog_must_be_skillcatalog: str = "catalog must be SkillCatalog"
    contextual_capability_hds_type_name_description: str = (
        "Contextual capability: hds=%s type=%s name=%s description=%s"
    )
    data_must_be_a_dictionary: str = "data must be a dictionary"
    data_must_be_capabilitiesdata: str = "data must be CapabilitiesData"
    deep_skill_execution_is_already_closed: str = (
        "Deep skill execution is already closed"
    )
    deep_skill_must_be_deepskill: str = "deep_skill must be DeepSkill"
    deep_tool_can_decorate_only_functions: str = (
        "@deep_tool can decorate only functions"
    )
    deep_tool_entrypoint_must_define_work_root_as_its: str = (
        "@deep_tool entrypoint must define work_root as its first positional parameter"
    )
    deeptools_can_load_only_once: str = "DeepTools can load only once"
    default_capabilities_must_be_defaultcapabilities_or_none: str = (
        "default_capabilities must be DefaultCapabilities or None"
    )
    depth_cannot_be_negative: str = "Depth cannot be negative"
    depth_must_be_an_integer: str = "Depth must be an integer"
    distinct_nodes_in_hierarchy_must_have_unique_agent_context: str = (
        "Distinct nodes in hierarchy must have unique agent_context_id: "
    )
    duplicate_capabilities_path: str = "Duplicate capabilities path"
    duplicate_folder_path: str = "Duplicate {duplicate_label} folder path"
    duplicate_name: str = "Duplicate {resource_type} name: '{name}'"
    duplicate_override_for: str = "Duplicate override for '{name}'"
    duplicate_tool_path: str = "Duplicate tool path"
    empty_placeholder_use_text_and_items_id_output: str = (
        'Empty placeholder. Use text="… {{<id-output>}} …" and items=["<id-output>"].'
    )
    entrypoint_path_must_be_a_path: str = "entrypoint_path must be a Path"
    entrypoint_path_must_be_an_existing_file: str = (
        "entrypoint_path must be an existing file"
    )
    entrypoint_path_must_be_inside_tool_path: str = (
        "entrypoint_path must be inside tool_path"
    )
    entrypoint_path_must_not_be_a_symbolic_link: str = (
        "entrypoint_path must not be a symbolic link"
    )
    event_type_must_be_a_deepagentexecutioneventtype: str = (
        "Event type must be a DeepAgentExecutionEventType"
    )
    execution_events_require_a_tool_call_id: str = (
        "Execution events require a tool call id"
    )
    execution_events_require_a_tool_name: str = "Execution events require a tool name"
    execution_failure_must_be_a_boolean: str = "Execution failure must be a boolean"
    exit_tool_can_decorate_only_functions: str = (
        "@exit_tool can decorate only functions"
    )
    exit_tool_entrypoint_must_define_exactly_one_exit_tool: str = "Exit-tool entrypoint '{entrypoint_path}' must define exactly one @exit_tool entrypoint"
    exit_tool_entrypoint_must_have_a_description: str = (
        "Exit-tool entrypoint '{entrypoint_path}' must have a description"
    )
    exit_tool_entrypoint_must_return_exittooldata: str = (
        "Exit-tool entrypoint '{entrypoint_path}' must return ExitToolData"
    )
    exit_tool_instructions_cannot_be_blank: str = (
        "Exit-tool instructions cannot be blank: {instructions_path}"
    )
    exit_tool_must_be_an_exittool: str = "exit_tool must be an ExitTool"
    exit_tool_must_return_exittooldata: str = "exit tool must return ExitToolData"
    exit_tool_not_found: str = "Exit tool '{name}' not found"
    exit_tool_result_must_be_a_dictionary: str = "Exit tool result must be a dictionary"
    exittools_can_load_only_once: str = "ExitTools can load only once"
    failed_to_load_markdown: str = "Failed to load Markdown: %s"
    file_must_have_extension: str = (
        "{resource_label} file must have '{markdown_suffix}' extension: {entry}"
    )
    file_must_not_be_a_symbolic_link: str = (
        "{resource_label} file must not be a symbolic link: {entry}"
    )
    folder_can_contain_only_markdown_files: str = (
        "{resource_label} folder can contain only Markdown files: {entry}"
    )
    folder_does_not_exist: str = "{resource_label} folder does not exist: {folder_path}"
    folder_must_not_be_a_symbolic_link: str = (
        "{resource_label} folder must not be a symbolic link: {folder_path}"
    )
    folder_paths_must_be_a_sequence_of_path: str = (
        "folder_paths must be a sequence of Path"
    )
    folder_paths_must_contain_only_path_instances: str = (
        "folder_paths must contain only Path instances"
    )
    function_is_already_decorated_with_deep_tool: str = (
        "Function '{function_name}' is already decorated with @deep_tool"
    )
    function_is_already_decorated_with_exit_tool: str = (
        "Function '{function_name}' is already decorated with @exit_tool"
    )
    has_invalid: str = "{label} has invalid {group}: {invalid}"
    hds_deepskills_tools_exittools: str = "\n        HDS: {hds}\n        DeepSkills: {deep_skills}\n        Tools: {tools}\n        ExitTools: {exit_tools}\n        "
    hds_not_found_available_hds: str = (
        "HDS '{name}' not found. Available HDS: {available_hds}"
    )
    history_must_be_a_list: str = "History must be a list"
    history_tree_must_be_a_list: str = "History tree must be a list"
    id_has_no_saved_retrievable_output_use_get_execution: str = 'ID {identifier!r} has no saved, retrievable output. Use get_execution_history to find a saved-output-id, then items=["<id-output>"].'
    id_is_not_present_in_this_node_s_history: str = 'ID {identifier!r} is not present in this node\'s history. Use get_execution_history to find a saved-output-id, then items=["<id-output>"].'
    incomplete_placeholder_use_text_and_items_id_output_or: str = 'Incomplete placeholder. Use text="… {{<id-output>}} …" and items=["<id-output>"], or remove the double braces.'
    instructions_must_be_a_string: str = "instructions must be a string"
    instructions_path_cannot_be_blank: str = "instructions_path cannot be blank"
    instructions_path_must_be_a_path: str = "instructions_path must be a Path"
    instructions_path_must_be_a_string_or_path: str = (
        "instructions_path must be a string or Path"
    )
    invalid_yaml_metadata: str = "Invalid YAML metadata: {error}"
    items_cannot_contain_blank_strings: str = "items cannot contain blank strings"
    items_must_be_a_list: str = "items must be a list"
    items_must_contain_only_strings: str = "items must contain only strings"
    markdown_file_does_not_exist: str = "Markdown file does not exist: {markdown_path}"
    markdown_must_be_a_string: str = "markdown must be a string"
    markdown_must_contain_a_valid_yaml_front_matter_section: str = (
        "Markdown must contain a valid YAML front matter section delimited by '---'."
    )
    markdown_path_must_be_a_file: str = "Markdown path must be a file: {markdown_path}"
    markdown_path_must_be_a_path: str = "markdown_path must be a Path"
    markdown_path_must_not_be_a_symbolic_link: str = (
        "Markdown path must not be a symbolic link: {markdown_path}"
    )
    metadata_field_cannot_be_empty: str = (
        "Metadata field '{field_key}' cannot be empty."
    )
    metadata_field_cannot_contain_duplicate_values: str = (
        "Metadata field '{field_key}' cannot contain duplicate values."
    )
    metadata_field_cannot_contain_empty_values: str = (
        "Metadata field '{field_key}' cannot contain empty values."
    )
    metadata_field_must_be_a_boolean: str = (
        "Metadata field '{field_key}' must be a boolean."
    )
    metadata_field_must_be_a_string: str = (
        "Metadata field '{field_key}' must be a string."
    )
    metadata_field_must_be_a_string_or_a_list: str = (
        "Metadata field '{field_key}' must be a string or a list of strings."
    )
    metadata_field_must_contain_only_strings: str = (
        "Metadata field '{field_key}' must contain only strings."
    )
    metadata_field_names_must_be_strings: str = "Metadata field names must be strings"
    metadata_must_be_capabilitymetadata: str = "metadata must be CapabilityMetadata"
    metadata_must_be_deepskillmetadata: str = "metadata must be DeepSkillMetadata"
    metadata_must_be_hdsmetadata: str = "metadata must be HDSMetadata"
    metadata_section_cannot_be_empty: str = "Metadata section cannot be empty."
    metadata_section_must_be_a_yaml_mapping: str = (
        "Metadata section must be a YAML mapping."
    )
    missing_required_metadata_fields: str = "Missing required metadata fields: {fields}"
    mocks_and_harness_must_be_bool: str = "mocks and harness must be bool"
    must_be_a_string: str = "{label} must be a string"
    must_be_a_tuple: str = "{field_name} must be a tuple"
    must_be_an_iterable_of_capability_names: str = (
        "{label} must be an iterable of capability names"
    )
    must_contain_only_instances: str = (
        "{field_name} must contain only {item_type_name} instances"
    )
    must_contain_only_strings: str = "{label} must contain only strings"
    name_type_deepskill_body_description: str = (
        "Name: {name}\nType: DeepSkill\nBody: {body}...\nDescription: {description}..."
    )
    name_type_exittool_instructions: str = (
        "Name: {name}\nType: ExitTool\nInstructions: {instructions}..."
    )
    name_type_hds_body_description: str = (
        "Name: {name}\nType: HDS\nBody: {body}...\nDescription: {description}..."
    )
    names_shared_by_deep_skills_and_tools: str = (
        "Names shared by deep skills and tools: {duplicates}"
    )
    names_shared_by_exit_tools_and_other_capabilities: str = (
        "Names shared by exit tools and other capabilities: {duplicates}"
    )
    names_shared_by_hds_and_other_capabilities: str = (
        "Names shared by HDS and other capabilities: {duplicates}"
    )
    no_deep_skill_execution_is_active: str = "No deep skill execution is active"
    node_hierarchy_cannot_contain_cycles: str = "Node hierarchy cannot contain cycles"
    node_must_be_a_node: str = "node must be a Node"
    output_appears_more_than_once_in_items_use_only: str = 'Output {record_id!r} appears more than once in items. Use only one entry: items=["<id-output>"].'
    output_belongs_to_another_skill_or_invocation_find_an: str = 'Output {identifier!r} belongs to another skill or invocation. Find an output from the current skill or a descendant and use items=["<id-output>"].'
    override_deep_skills_not_found_in_capabilities_paths: str = (
        "Override deep skills not found in capabilities_paths: "
    )
    override_for_is_not_a_resolved_tool: str = (
        "Override for '{name}' is not a resolved Tool"
    )
    override_for_must_be_a: str = "Override for '{name}' must be a {expected_name}"
    override_for_must_be_resolved_before_composition: str = (
        "Override for '{name}' must be resolved before composition"
    )
    override_for_must_have_name: str = "Override for '{name}' must have name '{name_2}'"
    override_for_must_match: str = "Override for '{name}' must match {expected_name}"
    override_for_must_target: str = "Override for '{name}' must target '{name_2}'"
    override_name_is_ambiguous: str = "Override {label} name is ambiguous: {name}"
    override_not_found_in_capabilities_paths: str = (
        "Override {label} not found in capabilities_paths: {name}"
    )
    override_tools_not_found_in_capabilities_paths: str = (
        "Override tools not found in capabilities_paths: "
    )
    overrides_must_be_a_mapping_or_iterable_of_overrides: str = (
        "overrides must be a mapping or iterable of overrides"
    )
    overrides_must_contain_only_override_descriptors: str = (
        "overrides must contain only override descriptors"
    )
    path_must_be_a_directory: str = (
        "{resource_label} path must be a directory: {folder_path}"
    )
    placeholder_is_not_listed_in_items_use_items_or: str = 'Placeholder {{{{{identifier}}}}} is not listed in items. Use items=["{identifier_2}"] or remove the placeholder.'
    placeholder_requires_its_id_in_items_use_items_or: str = 'Placeholder {{{{{identifier}}}}} requires its ID in items. Use items=["{identifier_2}"] or remove the placeholder.'
    provider_event_must_be_a_providerstreamevent: str = (
        "provider_event must be a ProviderStreamEvent"
    )
    provider_must_be_a_provider: str = "provider must be a Provider"
    references_cannot_contain_duplicates: str = "references cannot contain duplicates"
    references_must_be_a_tuple: str = "references must be a tuple"
    references_must_contain_only_strings: str = "references must contain only strings"
    selected_default_capabilities_are_missing: str = (
        "Selected default capabilities are missing: "
    )
    skill_path_must_start_with: str = "Skill path must start with '/'"
    text_cannot_be_blank: str = "text cannot be blank"
    text_must_be_a_string: str = "text must be a string"
    the_exit_skill_directory_is_no_longer_supported_move: str = "The exit-skill directory is no longer supported; move instructions into exit-tool and set instructions_path on @exit_tool"
    tool_call_must_be_a_toolcall: str = "tool_call must be a ToolCall"
    tool_call_must_be_toolcall: str = "tool_call must be ToolCall"
    tool_call_name_does_not_match_deep_skill_name: str = "Tool call name '{tool_call_get_name}' does not match deep skill name '{deep_skill_get_name}'"
    tool_call_name_does_not_match_exit_tool_name: str = (
        "Tool call name '{call_name}' does not match exit tool name '{name}'"
    )
    tool_description_cannot_be_blank: str = "Tool description cannot be blank"
    tool_description_must_be_a_string_or_none: str = (
        "Tool description must be a string or None"
    )
    tool_entrypoint_does_not_exist: str = (
        "Tool entrypoint does not exist: {entrypoint_path}"
    )
    tool_entrypoint_must_be_a_file: str = (
        "Tool entrypoint must be a file: {entrypoint_path}"
    )
    tool_entrypoint_must_define_exactly_one_deep_tool_entrypoint: str = "Tool entrypoint '{entrypoint_path}' must define exactly one @deep_tool entrypoint"
    tool_entrypoint_must_have_a_description: str = (
        "Tool entrypoint '{entrypoint_path}' must have a description"
    )
    tool_entrypoint_must_not_be_a_symbolic_link: str = (
        "Tool entrypoint must not be a symbolic link: {entrypoint_path}"
    )
    tool_function_must_have_deep_tool_with_a_description: str = (
        "Tool function must have @deep_tool with a description"
    )
    tool_hooks_for_must_be_a_toolhooks_instance: str = (
        "Tool hooks for '{name}' must be a ToolHooks instance"
    )
    tool_hooks_for_require_a_tool_or_exit_tool: str = (
        "Tool hooks for '{name}' require a tool or exit tool"
    )
    tool_hooks_must_be_a_mapping: str = "tool_hooks must be a mapping"
    tool_hooks_require_selected_capabilities: str = (
        "Tool hooks require selected capabilities: "
    )
    tool_name_must_be_a_string_or_none: str = "Tool name must be a string or None"
    tool_path_does_not_exist: str = "Tool path does not exist: {tool_path}"
    tool_path_must_be_a_directory: str = "Tool path must be a directory: {tool_path}"
    tool_path_must_be_a_path: str = "tool_path must be a Path"
    tool_path_must_be_an_existing_directory: str = (
        "tool_path must be an existing directory"
    )
    tool_path_must_not_be_a_symbolic_link: str = (
        "Tool path must not be a symbolic link: {tool_path}"
    )
    tool_path_must_not_be_a_symbolic_link_2: str = (
        "tool_path must not be a symbolic link"
    )
    tool_paths_must_be_a_sequence_of_path: str = "tool_paths must be a sequence of Path"
    tool_paths_must_contain_only_path_instances: str = (
        "tool_paths must contain only Path instances"
    )
    tools_must_contain_only_tool_instances: str = (
        "tools must contain only Tool instances"
    )
    tracked_deep_skill_closure_name_remaining_active_messages: str = (
        "Tracked deep skill closure: name=%s remaining_active=%d messages=%d"
    )
    tracked_deep_skill_opening_name_parent_active_roots: str = (
        "Tracked deep skill opening: name=%s parent=%s active=%d roots=%d"
    )
    unexpected_entry_inside_capabilities_folder_allowed_directories: str = "Unexpected entry inside capabilities folder: {entry}. Allowed directories: {allowed}"
    unknown_default_capabilities: str = "Unknown default capabilities: {names}"
    unknown_metadata_fields: str = "Unknown metadata fields: {fields}"
    unknown_or_non_scalar_configuration_reference: str = (
        "Unknown or non-scalar configuration reference: {name}.{field}"
    )
    work_root_does_not_exist: str = "work_root does not exist: {work_root}"
    work_root_must_be_a_directory: str = "work_root must be a directory: {work_root}"
    work_root_must_be_a_path: str = "work_root must be a Path"
    work_root_must_be_readable_writable_and_traversable: str = (
        "work_root must be readable, writable, and traversable: {work_root}"
    )


AGENT_TEXT = AgentText()
