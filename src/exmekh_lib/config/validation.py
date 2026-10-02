from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValidationPatterns:
    """Fixed formats that protect names, XML fields, and cache paths."""

    tool_name: str = r"^[A-Za-z0-9_-]{1,64}$"
    xml_name: str = r"^[A-Za-z_][A-Za-z0-9_.-]*$"
    portable_cache_component: str = r"[a-z0-9_-]{1,100}\Z"
    windows_device_name: str = r"(?:con|prn|aux|nul|com[1-9]|lpt[1-9])\Z"
    ascii_vocabulary_token: str = r"[a-zA-Z0-9]+"
    capability_reference: str = r"{{\s*!([A-Za-z0-9][A-Za-z0-9_-]*)\s*}}"
    capability_front_matter: str = (
        r"\A"
        r"---[ \t]*\r?\n"
        r"(?P<head>.*?)"
        r"\r?\n---[ \t]*"
        r"(?:\r?\n|\Z)"
        r"(?P<body>.*)"
        r"\Z"
    )
    exit_item_placeholder: str = r"{{([^{}]*)}}"


VALIDATION_PATTERNS = ValidationPatterns()
