"""Native DS4 service identities and tool presentation settings."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class NativeSuiteSettings:
    image_model: Literal["Qwen/Qwen-Image-2.1-Turbo"] = "Qwen/Qwen-Image-2.1-Turbo"
    embedding_model: Literal["google/embeddinggemma-2"] = "google/embeddinggemma-2"
    dimensions: int = 768
    include_image_in_result: bool = True
    preview_scale: float = 1.0
    timeout_seconds: float = 600
    image_description: str = (
        "Generate or edit images using the configured native image model. Supply an "
        "ordered list of reference paths for editing. Returns media IDs, local PNG "
        "paths, seeds, recipe and dimensions. include_image_in_result attaches "
        "labeled previews to the next model request; preview_scale changes previews only."
    )
    image_result_context: str = (
        "These images are previews from the completed generate-image tool call. "
        "The original PNGs, file IDs, seeds and recipe are in its tool result. "
        "Continue the current request using that completed result."
    )
    unsupported_provider: str = "This provider does not expose the native suite."
    invalid_response: str = "The native service returned an invalid response."
    invalid_scale: str = "preview_scale must be greater than zero and at most one."
    invalid_path: str = "Media paths must stay inside the workspace."


NATIVE_SUITE = NativeSuiteSettings()
