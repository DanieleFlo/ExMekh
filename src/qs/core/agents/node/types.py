from qs.core.models.user_message import (
    FilesUserMessage,
    ImagesUserMessage,
    TextsUserMessage,
)

type NodeUserMessage = str | TextsUserMessage | ImagesUserMessage | FilesUserMessage
