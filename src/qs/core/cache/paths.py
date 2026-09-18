from __future__ import annotations

import re
from hashlib import sha256

_PORTABLE_COMPONENT = re.compile(r"[a-z0-9_-]{1,100}\Z")
_WINDOWS_DEVICE = re.compile(r"(?:con|prn|aux|nul|com[1-9]|lpt[1-9])\Z")


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
