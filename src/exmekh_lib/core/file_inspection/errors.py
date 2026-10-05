"""Model-facing validation errors for workspace file inspection."""


class FileInspectionError(ValueError):
    """An invalid path, selection, or unreadable workspace file."""
