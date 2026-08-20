from __future__ import annotations

from qs.config.provider import SupportedProviders
from qs.core.providers.chat_completion_openai import ChatCompletionOpenAI

class ProviderMapper:
    @staticmethod
    def provider(provider: SupportedProviders, *args, **kwargs):
        if provider == SupportedProviders.OPENAI:
            raise NotImplementedError("OpenAI provider is not implemented yet.")
        elif provider == SupportedProviders.OLLAMA:
            return ChatCompletionOpenAI(*args, **kwargs)
        elif provider == SupportedProviders.LMSTUDIO:
            return ChatCompletionOpenAI(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    