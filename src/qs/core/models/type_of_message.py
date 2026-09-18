from .system_message import SystemMessage
from .tool_message import ToolMessage
from .user_message import UserMessage
from .assistant_message import AssistantMessage

TypeOfMessage = SystemMessage | ToolMessage | UserMessage | AssistantMessage
