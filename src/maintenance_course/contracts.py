# Issue #32: Report and tool validators define the permitted response shapes.
"""Visible response and tool contracts, checked outside the model."""

import json

DECISIONS = {"recommend_fix", "request_information", "escalate", "no_change"}
ACTIONS = {
    "reject_missing_dates",
    "request_date_format",
    "reject_duplicate_ids",
    "decode_utf8_sig",
    "trim_headers",
    "clarify_negative_policy",
    "request_locale",
    "escape_formula_cells",
    "request_sample",
    "no_change",
    "reject_missing_columns",
    "escalate_access",
    "escalate_timeout",
}


def validate_report(report, evidence_ids=None):
    """Return shape, value, and optional observed-evidence validation errors."""
    errors = []
    if not isinstance(report, dict):
        return ["Report must be an object"]
    expected = {"decision", "action", "evidence", "rationale"}
    if set(report) != expected:
        errors.append("Use exactly decision, action, evidence, rationale")
    if (
        not isinstance(report.get("decision"), str)
        or report.get("decision") not in DECISIONS
    ):
        errors.append("Unknown decision")
    if not isinstance(report.get("action"), str) or report.get("action") not in ACTIONS:
        errors.append("Unknown action")
    evidence = report.get("evidence")
    if not isinstance(evidence, list) or not all(isinstance(x, str) for x in evidence):
        errors.append("Evidence must be a list of document IDs")
    elif evidence_ids is not None and not set(evidence) <= set(evidence_ids):
        errors.append("Report cites evidence that was not observed")
    if (
        not isinstance(report.get("rationale"), str)
        or not report.get("rationale", "").strip()
    ):
        errors.append("Provide an observable evidence-based rationale")
    return errors


def checked_report(report, evidence_ids=None):
    """Return a valid report or raise ValueError with its validation errors."""
    errors = validate_report(report, evidence_ids)
    if errors:
        raise ValueError("; ".join(errors))
    return report


def validate_tool_arguments(name, arguments):
    """Require a permitted tool and its exact nonempty string arguments."""
    schemas = {
        "search_docs": {"query": str},
        "read_document": {"doc_id": str},
        "diagnose_csv": {"fixture_id": str},
        "bounded_calculator": {"expression": str},
    }
    if not isinstance(name, str) or name not in schemas:
        raise ValueError("Tool is not permitted")
    required = schemas[name]
    if not isinstance(arguments, dict) or set(arguments) != set(required):
        raise ValueError("Arguments must contain exactly " + ", ".join(required))
    for key, kind in required.items():
        if type(arguments[key]) is not kind or not 0 < len(arguments[key]) <= 500:
            raise ValueError("Invalid argument: " + key)
    return arguments
