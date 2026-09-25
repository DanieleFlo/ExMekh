"""Errors owned by the web-navigation capability implementation."""


class WebNavigationError(RuntimeError):
    """Describe one safe, model-facing web navigation failure."""

    def __init__(self, message: str, *, kind: str = "web_navigation") -> None:
        """Store a stable error kind and a concise public message."""
        super().__init__(message)
        self.kind = kind
