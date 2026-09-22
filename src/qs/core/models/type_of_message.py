from .assistant_message import AssistantMessage
from .system_message import SystemMessage
from .tool_message import ToolMessage
from .user_message import UserMessage

TypeOfMessage = SystemMessage | ToolMessage | UserMessage | AssistantMessage
