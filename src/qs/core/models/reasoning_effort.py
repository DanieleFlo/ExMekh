from __future__ import annotations

import enum


class ReasoningEffortLMX(enum.Enum):
    OFF = "none"
    LOW = "low"
    MEDIUM = "medium"
    XHIGH = "xhigh"


class ReasoningEffortLMH(enum.Enum):
    OFF = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReasoningEffortLMHXM(enum.Enum):
    OFF = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    Max = "max"


ReasoningEffort = str | ReasoningEffortLMX | ReasoningEffortLMH | ReasoningEffortLMHXM
