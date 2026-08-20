import base64
from io import BytesIO

import pytest
from PIL import Image

from qs.utils.image import pil_image_to_data_url


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
