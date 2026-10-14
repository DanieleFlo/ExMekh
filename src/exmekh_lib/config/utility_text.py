"""Fixed diagnostic and log messages for the utility domain."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UtilityText:
    circular_reference: str = "<circular reference>"
    an_error_occurred_while_reading_the_file: str = (
        "An error occurred while reading the file: {error}"
    )
    file_path_does_not_exist: str = "File path {file_path} does not exist."
    file_path_is_not_a_file: str = "File path {file_path} is not a file."
    file_path_must_be_a_pathlib_path_object: str = (
        "File path must be a pathlib.Path object"
    )
    format_must_be_a_string: str = "Format must be a string"
    input_must_be_a_pil_image: str = "Input must be a PIL Image"
    input_must_be_of_type_bytes: str = "Input must be of type 'bytes'"


UTILITY_TEXT = UtilityText()
