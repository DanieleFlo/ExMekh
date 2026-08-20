import base64


def convert_bytes_to_base64(file_bytes: bytes) -> str:
    base64_data = base64.b64encode(file_bytes).decode("utf-8")
    return base64_data
