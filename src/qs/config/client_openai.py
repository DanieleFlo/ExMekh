from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class ClientOpenAI:
    model: str = "gpt-4o"
    max_retries: int = 3
    timeout: float = 120.