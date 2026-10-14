"""Model-facing validation errors for workspace file inspection."""

from exmekh_lib.utils.tool_errors import ToolInputError


class FileInspectionError(ToolInputError):
    """An invalid path, selection, or unreadable workspace file."""
