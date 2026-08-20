from __future__ import annotations

import enum

class SupportedProviders(enum.Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
