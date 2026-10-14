import base64
from io import BytesIO

from PIL import Image

from exmekh_lib.config.utility_text import UTILITY_TEXT


def pil_image_to_data_url(
    image: Image.Image,
    format: str = "PNG",
) -> str:
    if not isinstance(image, Image.Image):
        raise TypeError(UTILITY_TEXT.input_must_be_a_pil_image)
    if not isinstance(format, str):
        raise TypeError(UTILITY_TEXT.format_must_be_a_string)
    normalized_format = format.upper()
    pillow_format = "JPEG" if normalized_format == "JPG" else normalized_format
    mime_type = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "WEBP": "image/webp",
    }[normalized_format]

    buffer = BytesIO()
    image.save(buffer, format=pillow_format)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"
