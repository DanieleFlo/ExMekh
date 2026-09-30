"""Shared HTTP settings for tests against the configured model endpoint."""

import math
import os

from openai import OpenAI

from qs.providers.local import LocalProvider

DEFAULT_LIVE_REQUEST_TIMEOUT_SECONDS = 900.0


class LiveModelProvider(LocalProvider):
    def _build_client(self) -> OpenAI:
        value = os.getenv(
            "QS_LIVE_REQUEST_TIMEOUT_SECONDS",
            str(DEFAULT_LIVE_REQUEST_TIMEOUT_SECONDS),
        )
        message = (
            "QS_LIVE_REQUEST_TIMEOUT_SECONDS must be a finite number of seconds "
            "greater than zero"
        )
        try:
            timeout = float(value)
        except ValueError as error:
            raise ValueError(message) from error
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(message)
        return super()._build_client().with_options(timeout=timeout, max_retries=0)
