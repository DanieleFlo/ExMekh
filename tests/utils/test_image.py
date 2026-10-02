import base64
from io import BytesIO
from typing import cast

import pytest
from PIL import Image

from exmekh_lib.utils.image import pil_image_to_data_url


@pytest.mark.parametrize(
    ("image_format", "expected_mime_type", "expected_pillow_format"),
    [
        ("PNG", "image/png", "PNG"),
        ("JPEG", "image/jpeg", "JPEG"),
        ("JPG", "image/jpeg", "JPEG"),
        ("WEBP", "image/webp", "WEBP"),
    ],
)
def test_pil_image_to_data_url_encodes_requested_image_format(
    image_format, expected_mime_type, expected_pillow_format
) -> None:
    image = Image.new("RGB", (2, 3), color=(20, 40, 60))

    result = pil_image_to_data_url(image, format=image_format.lower())

    prefix, encoded = result.split(",", maxsplit=1)
    decoded = base64.b64decode(encoded)
    with Image.open(BytesIO(decoded)) as decoded_image:
        assert prefix == f"data:{expected_mime_type};base64"
        assert decoded_image.format == expected_pillow_format
        assert decoded_image.size == (2, 3)


def test_pil_image_to_data_url_rejects_unsupported_format() -> None:
    image = Image.new("RGB", (1, 1))

    with pytest.raises(KeyError, match="GIF"):
        pil_image_to_data_url(image, format="GIF")


def test_pil_image_to_data_url_rejects_non_image() -> None:
    with pytest.raises(TypeError, match="Input must be a PIL Image"):
        pil_image_to_data_url(cast(Image.Image, object()))


def test_pil_image_to_data_url_rejects_non_string_format() -> None:
    with pytest.raises(TypeError, match="Format must be a string"):
        pil_image_to_data_url(Image.new("RGB", (1, 1)), format=cast(str, 1))
