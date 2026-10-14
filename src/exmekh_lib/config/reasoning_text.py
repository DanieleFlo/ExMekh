"""Fixed diagnostic and log messages for the reasoning domain."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReasoningText:
    invalid_function_arguments: str = (
        "{function} received {count} arguments; expected {expected}."
    )
    matrix_has_no_inverse: str = "The matrix is singular and has no inverse."
    undefined_math_result: str = (
        "The expression produces an undefined or non-finite numeric result."
    )
    amount_must_be_a_finite_number: str = "amount must be a finite number."
    conclusion_is_required_for_this_task: str = "conclusion is required for this task."
    constants_must_be_bounded_booleans_or_numbers: str = (
        "Constants must be bounded booleans or numbers."
    )
    date_time_or_duration_is_outside_the_supported_range: str = (
        "Date/time or duration is outside the supported range."
    )
    declare_up_to_bool_int_or_real_variables_with: str = "Declare up to {formal_logic_max_variables} Bool, Int, or Real variables with simple names."
    division_requires_a_nonzero_literal_divisor: str = (
        "Division requires a nonzero literal divisor."
    )
    each_formula_node_must_be_an_object: str = "Each formula node must be an object."
    equality_operands_have_incompatible_types: str = (
        "Equality operands have incompatible types."
    )
    evaluate_requires_an_expression_without_variables: str = (
        "evaluate requires an expression without variables"
    )
    expected_a_numeric_expression: str = "Expected a numeric expression."
    expected_received: str = "Expected {sort}, received {term_sort}."
    exponents_must_be_numeric_and_at_most: str = (
        "exponents must be numeric and at most {math_max_exponent}"
    )
    expression_contains_unsupported_syntax: str = (
        "expression contains unsupported syntax"
    )
    expression_is_too_complex: str = "expression is too complex"
    expression_is_too_long: str = "expression is too long"
    expression_must_be_non_empty: str = "expression must be non-empty"
    formula_exceeds_its_size_or_depth_limit: str = (
        "Formula exceeds its size or depth limit."
    )
    formula_node_needs_an_operator: str = "Formula node needs an operator."
    formulas_must_be_json_serializable: str = "Formulas must be JSON serializable."
    install_the_exmekh_lib_wheel_with_its_declared_dependencies: str = (
        "Install the ExMekh-Lib wheel with its declared dependencies."
    )
    integrate_requires_both_bounds_or_neither: str = (
        "integrate requires both bounds or neither"
    )
    invalid_expression_syntax: str = "invalid expression syntax"
    local_time_does_not_exist_in_this_timezone_supply: str = "{parameter} local time does not exist in this timezone. Supply an explicit UTC offset."
    local_time_is_ambiguous_in_this_timezone_supply_an: str = "{parameter} local time is ambiguous in this timezone. Supply an explicit UTC offset."
    matrix_entries_must_be_finite: str = "matrix entries must be finite"
    matrix_entries_must_be_numbers: str = "matrix entries must be numbers"
    matrix_must_be_square_and_at_most_by: str = "matrix must be square and at most {math_max_matrix_dimension} by {math_max_matrix_dimension_2}"
    missing_exmekh_lib_runtime_module: str = (
        "Missing exmekh_lib runtime module: {error_name}."
    )
    must_be_a_non_empty_iso_date_or_datetime: str = (
        "{parameter} must be a non-empty ISO 8601 date or datetime."
    )
    must_be_an_iso_date_or_datetime: str = (
        "{parameter} must be an ISO 8601 date or datetime."
    )
    needs_operands: str = "{op} needs 2-{formal_logic_max_operands} operands."
    nonlinear_multiplication_is_unsupported: str = (
        "Nonlinear multiplication is unsupported."
    )
    number_is_too_large: str = "number is too large"
    numbers_must_be_finite: str = "numbers must be finite"
    operator_node_must_have_keys: str = "Operator node must have keys {expected}."
    precision_must_be_between_and: str = (
        "precision must be between {math_min_precision} and {math_max_precision}"
    )
    premises_and_conclusions_must_be_boolean: str = (
        "Premises and conclusions must be Boolean."
    )
    premises_must_be_a_list_of_at_most_formulas: str = (
        "premises must be a list of at most {formal_logic_max_premises} formulas."
    )
    request_deadline_exceeded: str = "request deadline exceeded"
    request_exceeds_the_character_limit: str = (
        "Request exceeds the {formal_logic_max_request_chars} character limit."
    )
    solve_accepts_at_most_one_equality_sign: str = (
        "solve accepts at most one equality sign"
    )
    the_iana_timezone_database_is_unavailable_install_the_declared: str = "The IANA timezone database is unavailable. Install the declared tzdata dependency."
    the_z3_check_failed_unexpectedly: str = "The Z3 check failed unexpectedly."
    the_z3_memory_limit_was_reached: str = "The Z3 memory limit was reached."
    timeout_seconds_must_be_between_and: str = (
        "timeout_seconds must be between 1 and {formal_logic_max_timeout_seconds}."
    )
    timezone_must_be_a_non_empty_iana_name_such: str = (
        "timezone must be a non-empty IANA name, such as Europe/Rome."
    )
    timezone_must_be_a_valid_iana_name_such_as: str = (
        "timezone must be a valid IANA name, such as Europe/Rome."
    )
    undeclared_variable: str = "Undeclared variable: {name}."
    unhandled_validated_operator: str = "Unhandled validated operator: {op}"
    unit_must_be_seconds_minutes_hours_or_days: str = (
        "unit must be seconds, minutes, hours, or days."
    )
    unknown_logic_task: str = "Unknown logic task."
    unknown_timezone_use_an_iana_name_such_as_europe: str = (
        "Unknown timezone {name!r}. Use an IANA name such as Europe/Rome."
    )
    unsupported_date_time_operation: str = "unsupported date/time operation"
    unsupported_mathematical_operation: str = "unsupported mathematical operation"
    unsupported_operator: str = "Unsupported operator: {op}."
    variable_must_be_a_simple_identifier: str = "variable must be a simple identifier"
    z3_logic_tool_failed: str = "Z3 logic tool failed"


REASONING_TEXT = ReasoningText()
