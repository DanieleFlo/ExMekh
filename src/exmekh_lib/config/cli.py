from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CliPresentation:
    """Text and logging format shown by the example CLI."""

    log_format: str = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    thinking_label: str = "Thinking"
    response_stream_label: str = "Response"
    user_prompt: str = "User: "
    exit_commands: tuple[str, ...] = ("exit", "quit")
    exit_message: str = "Exiting..."
    response_label: str = "\nResponse:"
    history_label: str = "history:"
    skill_history_label: str = "Deep skill history:\n"
    tools_label: str = "# Tools text"
    separator: str = "----------"


@dataclass(frozen=True, slots=True)
class CliSettings:
    """User-facing capability selection for the example CLI."""

    root_hds: str = "main"
    child_hds: str | None = None
    presentation: CliPresentation = field(default_factory=CliPresentation)

    @classmethod
    def from_environment(cls) -> CliSettings:
        defaults = cls()
        return cls(
            root_hds=os.getenv("EXMEKH_LIB_ROOT_HDS", defaults.root_hds),
            child_hds=os.getenv("EXMEKH_LIB_CHILD_HDS", defaults.child_hds),
        )
