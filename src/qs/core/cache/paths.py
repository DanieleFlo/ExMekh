from __future__ import annotations

import re
from hashlib import sha256

from qs.config.validation import VALIDATION_PATTERNS

_PORTABLE_COMPONENT = re.compile(VALIDATION_PATTERNS.portable_cache_component)
_WINDOWS_DEVICE = re.compile(VALIDATION_PATTERNS.windows_device_name)


def cache_component(identifier: str) -> str:
    """Encode an opaque ID without traversal, case aliases or lossy replacement.

    Simple lowercase IDs retain readable filenames. The hash prefix is excluded
    from that alphabet, so encoded IDs cannot collide with readable IDs.
    """
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("cache identifier must be a non-empty string")
    if _PORTABLE_COMPONENT.fullmatch(identifier) and not _WINDOWS_DEVICE.fullmatch(
        identifier
    ):
        return identifier
    return "~" + sha256(identifier.encode("utf-8")).hexdigest()
