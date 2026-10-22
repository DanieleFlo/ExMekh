"""Fixed diagnostic and log messages for the web domain."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WebText:
    browser_cleanup_failed: str = "Browser cleanup failed: {error_type}: {error}."
    browser_cleanup_failed_log: str = (
        "Browser cleanup failed while preserving the original failure"
    )
    playwright_reported: str = " Playwright reported: {detail}"
    content_truncated: str = "\n\n[Content truncated]"
    action_delta_y_must_be_numeric: str = "Action {index} delta_y must be numeric."
    action_has_unexpected_fields: str = "Action {index} has unexpected fields: {names}."
    action_has_unsupported_kind: str = "Action {index} has unsupported kind: {value}."
    action_milliseconds_must_be_an_integer: str = (
        "Action {index} milliseconds must be an integer."
    )
    action_must_be_an_object: str = "Action {index} must be an object."
    action_requires_string_field: str = (
        "Action {index} requires string field '{field}'."
    )
    action_wait_requires_exactly_one_of_selector_or_milliseconds: str = (
        "Action {index} wait requires exactly one of selector or milliseconds."
    )
    actions_must_be_a_json_array: str = "Actions must be a JSON array."
    actions_must_contain_between_and_items: str = (
        "Actions must contain between 1 and {max_actions} items."
    )
    animated_images_are_not_supported: str = "Animated images are not supported."
    browser_crawl_failed: str = "Browser crawl failed: {error_type_name}."
    chromium_is_unavailable_run_python_m_playwright_install_chromium: str = "Chromium is unavailable. Run `python -m playwright install chromium --only-shell` in this environment."
    could_not_install_chromium_automatically_run_python_m_playwright: str = "Could not install Chromium automatically. Run `python -m playwright install chromium --only-shell` in this environment."
    could_not_resolve_url_hostname: str = "Could not resolve URL hostname: {host}."
    could_not_resolve_url_hostname_2: str = (
        "Could not resolve URL hostname: {hostname}."
    )
    credentials_in_urls_are_not_supported: str = (
        "Credentials in URLs are not supported."
    )
    embedding_backend_returned_an_invalid_batch_shape: str = (
        "Embedding backend returned an invalid batch shape."
    )
    embedding_cache_capacity_must_be_positive: str = (
        "Embedding cache capacity must be positive."
    )
    executable_doesn_t_exist: str = "Executable doesn't exist"
    image_dimensions_exceed_the_megapixel_limit: str = (
        "Image dimensions exceed the {max_image_pixels:g} megapixel limit."
    )
    image_path_must_be_a_non_empty_path: str = "image_path must be a non-empty path."
    image_path_must_be_an_image_file_no_larger: str = (
        "image_path must be an image file no larger than {max_local_image_bytes:g} MB."
    )
    image_path_must_resolve_to_a_file_inside_workroot: str = (
        "image_path must resolve to a file inside workroot."
    )
    image_path_requires_an_injected_workroot: str = (
        "image_path requires an injected workroot."
    )
    main_frame_navigation_exceeded_hops: str = (
        "Main-frame navigation exceeded {max_navigations} hops."
    )
    main_frame_navigation_exceeded_the_redirect_budget: str = (
        "Main-frame navigation exceeded the redirect budget."
    )
    main_frame_navigation_left_the_initial_origin: str = (
        "Main-frame navigation left the initial origin: {final_url}."
    )
    main_frame_navigation_left_the_initial_origin_2: str = (
        "Main-frame navigation left the initial origin: {url}."
    )
    max_navigations_must_not_be_negative: str = "max_navigations must not be negative."
    max_results_must_be_between_and: str = (
        "max_results must be between 1 and {max_search_results}."
    )
    navigation_action_left_the_initial_origin: str = (
        "Navigation action left the initial origin: {value}."
    )
    only_absolute_http_and_https_urls_are_supported: str = (
        "Only absolute http and https URLs are supported."
    )
    only_credential_free_absolute_http_s_urls_are_supported: str = (
        "Only credential-free absolute HTTP(S) URLs are supported."
    )
    only_jpeg_png_and_webp_images_are_supported: str = (
        "Only JPEG, PNG, and WebP images are supported."
    )
    native_semantic_ranking_unavailable: str = "Native semantic ranking unavailable: %s"
    output_mode_must_be_markdown_ast_or_hybrid: str = (
        "output_mode must be markdown, ast, or hybrid."
    )
    public_web_search_failed_across_the_available_providers: str = (
        "Public web search failed across the available providers."
    )
    remote_image_discovery_exceeded_the_redirect_limit: str = (
        "Remote image discovery exceeded the redirect limit."
    )
    remote_redirect_omitted_its_destination: str = (
        "Remote redirect omitted its destination."
    )
    remote_resource_exceeds_its_size_limit: str = (
        "Remote resource exceeds its size limit."
    )
    remote_resource_returned_http: str = (
        "Remote resource returned HTTP {response_status}."
    )
    rendered_page_html_could_not_be_parsed: str = (
        "Rendered page HTML could not be parsed."
    )
    rrf_k_must_be_positive: str = "RRF k must be positive."
    search_query_must_be_a_non_empty_string: str = (
        "Search query must be a non-empty string."
    )
    native_visual_ranking_unavailable: str = "Native visual ranking unavailable: %s"
    the_file_does_not_contain_a_valid_supported_image: str = (
        "The file does not contain a valid supported image."
    )
    the_local_image_could_not_be_read: str = "The local image could not be read."
    timeout_seconds_must_be_between_and: str = (
        "timeout_seconds must be between 1 and {web_navigation_max_timeout_seconds}."
    )
    title_domain_snippet: str = (
        "Title: {result_get_strip}\nDomain: {domain}\nSnippet: {result_get_strip_2}"
    )
    unknown_crawl4ai_failure: str = "Unknown Crawl4AI failure."
    unsupported_remote_content_type: str = "Unsupported remote content type: {value}."
    url_contains_an_invalid_port: str = "URL contains an invalid port."
    url_hostname_is_not_publicly_routable: str = (
        "URL hostname is not publicly routable: {host}."
    )
    url_hostname_is_not_publicly_routable_2: str = (
        "URL hostname is not publicly routable: {hostname}."
    )


WEB_TEXT = WebText()
