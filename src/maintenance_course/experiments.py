"""Instrumentation shared by student and instructor implementations."""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .contracts import validate_report
from .models import make_model
from .tools import ToolRegistry


# Issue #32: Experiment contracts distinguish captured workflow errors from setup failures.
def payload(case, registry, **extra):
    """Build context from observed evidence, allowing extra fields to override it."""
    return {
        "case": case,
        "documents": list(registry.observed.values()),
        "observations": registry.events,
        **extra,
    }


def gather(registry):
    """Record fixture diagnosis and request the seven version-2 policy documents."""
    registry.call("diagnose_csv", {"fixture_id": registry.case["fixture_id"]})
    for name in [
        "dates-v2",
        "ids-v2",
        "encoding-v2",
        "amounts-v2",
        "security-v2",
        "limits-v2",
        "operations-v2",
    ]:
        registry.call("read_document", {"doc_id": name})


def run_case(workflow, case, mode="fixture", fixture_root=None, **kwargs):
    """Return workflow output, validation findings, and measured usage.

    Workflow exceptions become error records; model and registry setup errors propagate.
    """
    model = make_model(mode)
    registry = ToolRegistry(case, fixture_root=fixture_root)
    started = time.perf_counter()
    error = None
    try:
        report = workflow(model, registry, case, **kwargs)
    except Exception as exc:
        report = None
        error = {"type": type(exc).__name__, "message": str(exc)}
    return {
        "run_id": str(uuid.uuid4()),
        "case_id": case["id"],
        "mode": mode,
        "model": model.model_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "workflow": workflow.__name__,
        "report": report,
        "error": error,
        "validation_errors": validate_report(report, registry.observed),
        "model_calls": model.calls,
        "input_tokens": model.input_tokens,
        "output_tokens": model.output_tokens,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "model_events": model.events,
        "tool_events": registry.events,
    }


def save_runs(path, runs):
    """Write indented UTF-8 JSON run records, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(runs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
