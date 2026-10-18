"""Synthetic tool contracts and independently specified argument-compilation tasks."""

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.utils.json import JsonValue


@dataclass(frozen=True)
class ArgumentCase:
    name: str
    category: str
    prompt: str
    tool: Tool
    expected: dict[str, JsonValue]

    def expected_arguments(self) -> dict[str, JsonValue]:
        arguments = deepcopy(self.expected)
        if self.tool.background_enabled:
            arguments.setdefault("execution_mode", "wait")
            arguments.setdefault("wait_seconds", RUNTIME.default_tool_wait_seconds)
        return arguments


def lookup_customer(
    account_id: Annotated[str, "Use the customer account ID, never the order ID."],
    include_archived: Annotated[
        bool, "True only if archived customers were requested."
    ],
) -> None:
    """Look up a customer account."""


def query_period(
    start: Annotated[str, "Inclusive start date, in YYYY-MM-DD format."],
    stop: Annotated[str, "Exclusive end date, in YYYY-MM-DD format."],
) -> None:
    """Query records in a date interval."""


def prepare_payment(
    amount_minor: Annotated[
        int, "Amount in cents: multiply the stated euro amount by 100."
    ],
    currency: Literal["EUR", "USD"],
) -> None:
    """Prepare a payment preview without submitting a payment."""


class OrderLine(TypedDict):
    sku: Annotated[str, "Use the product code, not its display name."]
    quantity: Annotated[int, "Number of units ordered."]
    unit_price_minor: Annotated[int, "Unit price in cents, not the total line price."]


def prepare_order(
    lines: list[OrderLine],
    shipping: Annotated[
        Literal["std", "exp"], "Use exp for express shipping and std otherwise."
    ],
) -> None:
    """Prepare an order preview."""


@dataclass
class ShippingAddress:
    recipient: Annotated[str, "Exact recipient name, preserving accents."]
    street: Annotated[str, "Street and house number only."]
    postal_code: Annotated[str, "Postal code as text; preserve leading zeroes."]
    country: Annotated[
        Literal["IT", "FR"], "ISO country code: IT for Italy, FR for France."
    ]


def prepare_delivery(address: ShippingAddress) -> None:
    """Prepare a shipping address."""


def search_documents(
    query: str,
    folder: Annotated[
        str | None, "Use null when all folders are requested; never the string all."
    ],
    limit: int,
) -> None:
    """Search indexed documents."""


def classify_transfer(
    direction: Annotated[
        Literal["in", "out"],
        "Use in for incoming transfers and out for outgoing transfers.",
    ],
    reference: Annotated[
        str, "Use the transfer reference, not the customer reference."
    ],
) -> None:
    """Classify a transfer record."""


def label_samples(
    labels: Annotated[
        dict[
            str,
            Annotated[
                str, "Encode high as H and low as L. Do not copy the input word."
            ],
        ],
        "Keep the supplied keys literally, including a key named description.",
    ],
) -> None:
    """Assign coded labels to named samples."""


def select_reference(
    reference: Annotated[
        str | int,
        "Supplier codes are opaque strings, preserving leading zeroes; internal IDs are integers.",
    ],
) -> None:
    """Select a supplier code or internal identifier."""


def locate_point(
    coordinates: Annotated[
        list[float], "Exactly [longitude, latitude]. West and south are negative."
    ],
) -> None:
    """Locate a point using signed geographic coordinates."""


def configure_cache(
    disabled: Annotated[bool, "True disables caching; false enables it."],
    ttl_seconds: Annotated[
        int, "Cache lifetime in seconds, converting minutes to seconds."
    ],
) -> None:
    """Prepare cache configuration."""


def prepare_document(
    document: Annotated[
        str,
        'Emit exactly <new-exmekh-file path="PATH"/>BODY with no added whitespace. XML-escape the path attribute; keep BODY literal, including angle brackets and ampersands.',
    ],
) -> None:
    """Prepare a document creation protocol without writing files."""


def queue_job(job_id: str) -> None:
    """Prepare a job invocation."""


def _verification_tool(code: str) -> Tool:
    def verify(
        code: Annotated[str, f"Use the exact verification literal {code!r}."],
    ) -> None:
        """Verify the documented literal."""

    return Tool(verify)


def build_cases(seed: int = 0) -> tuple[ArgumentCase, ...]:
    """Keep paired variants identical; vary the verification literal across repeats."""
    code = f"VERIFY-{random.Random(seed).getrandbits(48):012x}"
    return (
        ArgumentCase(
            "customer_identifier",
            "identifiers",
            "Look up account C-1042 for order O-9002. Include active customers only.",
            Tool(lookup_customer),
            {"account_id": "C-1042", "include_archived": False},
        ),
        ArgumentCase(
            "exclusive_date_boundary",
            "dates",
            "Query every day from 17 March 2027 through 19 March 2027, inclusive.",
            Tool(query_period),
            {"start": "2027-03-17", "stop": "2027-03-20"},
        ),
        ArgumentCase(
            "money_in_minor_units",
            "units",
            "Prepare a preview for a payment of 12.34 euros.",
            Tool(prepare_payment),
            {"amount_minor": 1234, "currency": "EUR"},
        ),
        ArgumentCase(
            "nested_order_lines",
            "nested_arrays",
            "Prepare an express order: 2 units of product B-17 (display name Blue pen) at EUR 1.25 each, then 3 units of R-04 (display name Red pen) at EUR 2.10 each. Keep this order.",
            Tool(prepare_order),
            {
                "lines": [
                    {"sku": "B-17", "quantity": 2, "unit_price_minor": 125},
                    {"sku": "R-04", "quantity": 3, "unit_price_minor": 210},
                ],
                "shipping": "exp",
            },
        ),
        ArgumentCase(
            "dataclass_postal_address",
            "dataclasses",
            "Prepare delivery for Chloé Martin, Via Roma 12, postal code 00144, Italy.",
            Tool(prepare_delivery),
            {
                "address": {
                    "recipient": "Chloé Martin",
                    "street": "Via Roma 12",
                    "postal_code": "00144",
                    "country": "IT",
                }
            },
        ),
        ArgumentCase(
            "null_folder",
            "nullable",
            "Search all folders for quarterly report and return at most 3 results.",
            Tool(search_documents),
            {"query": "quarterly report", "folder": None, "limit": 3},
        ),
        ArgumentCase(
            "enum_direction_mapping",
            "enums",
            "Classify incoming transfer TR-82 for customer CU-19.",
            Tool(classify_transfer),
            {"direction": "in", "reference": "TR-82"},
        ),
        ArgumentCase(
            "dictionary_description_key",
            "mappings",
            "Label sample description as high and sample priority as low.",
            Tool(label_samples),
            {"labels": {"description": "H", "priority": "L"}},
        ),
        ArgumentCase(
            "union_opaque_code",
            "unions",
            "Select supplier code 0017, not an internal identifier.",
            Tool(select_reference),
            {"reference": "0017"},
        ),
        ArgumentCase(
            "signed_coordinate_order",
            "arrays",
            "Locate longitude 73.5 degrees west and latitude 40.25 degrees north.",
            Tool(locate_point),
            {"coordinates": [-73.5, 40.25]},
        ),
        ArgumentCase(
            "boolean_polarity_and_duration",
            "booleans",
            "Enable caching for 5 minutes.",
            Tool(configure_cache),
            {"disabled": False, "ttl_seconds": 300},
        ),
        ArgumentCase(
            "literal_document_protocol",
            "text_protocol",
            "Prepare file drafts/ab&cd.txt with the exact body <hello>&done.",
            Tool(prepare_document),
            {"document": '<new-exmekh-file path="drafts/ab&amp;cd.txt"/><hello>&done'},
        ),
        ArgumentCase(
            "documented_verification_literal",
            "documentation_lookup",
            "Verify the literal specified by the tool documentation.",
            _verification_tool(code),
            {"code": code},
        ),
        ArgumentCase(
            "background_execution",
            "execution_options",
            "Start job JOB-42 in background without waiting. Leave the initial wait limit at its default.",
            Tool(queue_job),
            {
                "job_id": "JOB-42",
                "execution_mode": "background",
                "wait_seconds": RUNTIME.default_tool_wait_seconds,
            },
        ),
    )
