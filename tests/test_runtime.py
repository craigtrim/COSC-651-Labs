# Issue #32: Runtime tests exercise report, tool, and approval boundaries.
import json
import subprocess
import sys
from pathlib import Path
import pytest
from maintenance_course.contracts import validate_report, validate_tool_arguments
from maintenance_course.data import case_by_id, DATA
from maintenance_course.tools import ToolRegistry
from maintenance_course.state import StateStore
from maintenance_course.experiments import run_case, save_runs
from maintenance_course.baselines import fixed
from maintenance_course.models import FixtureModel, GeminiModel


@pytest.mark.parametrize(
    "bad",
    [None, [], {}, {"decision": [], "action": {}, "evidence": [1], "rationale": ""}],
)
def test_malformed_reports_return_errors(bad):
    assert validate_report(bad)


def test_scope_schema_stale_and_approval_are_enforced():
    tools = ToolRegistry(case_by_id("D01"))
    assert not tools.call("diagnose_csv", {"fixture_id": "../valid"})["ok"]
    assert not tools.call("read_document", {"doc_id": "dates-v1"})["ok"]
    assert not tools.call("read_document", {"doc_id": "dates-v2", "extra": 1})["ok"]
    assert not tools.call("approve", {})["ok"]
    assert tools.call("read_document", {"doc_id": "dates-v2"})["ok"]


@pytest.mark.parametrize(
    "expression", ['__import__("os")', "(1).__class__", "2**100000", "1/0", "1000001"]
)
def test_bounded_expression_rejects_escape_and_limits(expression):
    assert not ToolRegistry(case_by_id("D01")).call(
        "bounded_calculator", {"expression": expression}
    )["ok"]


def test_valid_calculator_and_diagnostic():
    registry = ToolRegistry(case_by_id("D01"))
    assert (
        registry.call("bounded_calculator", {"expression": "(3+5)/2"})["result"][
            "value"
        ]
        == 4
    )
    assert (
        "MISSING_DATE"
        in registry.call("diagnose_csv", {"fixture_id": "missing-date"})["result"][
            "codes"
        ]
    )


def test_killable_worker_timeout(monkeypatch):
    def expired(*args, **kwargs):
        raise subprocess.TimeoutExpired("worker", kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", expired)
    assert (
        ToolRegistry(case_by_id("D01")).call(
            "diagnose_csv", {"fixture_id": "missing-date"}
        )["error_type"]
        == "TimeoutExpired"
    )


def test_state_isolation_restart_exact_approval_and_idempotence(tmp_path):
    path = tmp_path / "state.db"
    store = StateStore(path)
    payload = {"action": "x"}
    store.save("s", "D01", {"step": 1})
    with pytest.raises(PermissionError):
        store.load("s", "D02")
    with pytest.raises(PermissionError):
        store.apply_simulated("s", payload)
    store.approve("s", payload, "reviewer")
    with pytest.raises(PermissionError):
        store.apply_simulated("s", {"action": "y"})
    assert store.apply_simulated("s", payload)["status"] == "recorded"
    store.close()
    store = StateStore(path)
    assert store.load("s", "D01")["step"] == 1
    assert store.apply_simulated("s", payload)["status"] == "already_recorded"
    store.clear("s")
    assert store.load("s", "D01") is None
    assert store.action_count() == 1
    store.close()


def test_trace_round_trip_and_mode(tmp_path):
    run = run_case(fixed, case_by_id("D01"))
    save_runs(tmp_path / "runs.json", [run])
    recovered = json.loads((tmp_path / "runs.json").read_text())
    assert recovered[0]["mode"] == "fixture"
    assert recovered[0]["model_calls"] == 1
    assert recovered[0]["report"]["action"] == "reject_missing_dates"


def test_live_adapter_contract_without_network(monkeypatch):
    import io
    import urllib.request

    def fake(request, timeout):
        assert request.get_header("X-goog-api-key") == "test-only-key"
        assert timeout == 45
        body = json.loads(request.data)
        assert "systemInstruction" in body
        return io.BytesIO(
            json.dumps(
                {
                    "candidates": [{"content": {"parts": [{"text": '{"score":1}'}]}}],
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 3,
                    },
                }
            ).encode()
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    model = GeminiModel(model="test-model", key="test-only-key", max_calls=1)
    assert model.complete("judge", {"case": case_by_id("D01")}) == {"score": 1}
    assert model.input_tokens == 10
    assert model.output_tokens == 3
    assert "test-only-key" not in json.dumps(model.events)
    with pytest.raises(RuntimeError):
        model.complete("judge", {})


def test_search_filters_archived_policy():
    response = ToolRegistry(case_by_id("D01")).call(
        "search_docs", {"query": "missing date"}
    )
    assert response["ok"]
    ids = [hit["id"] for hit in response["result"]["hits"]]
    assert "dates-v2" in ids
    assert "dates-v1" not in ids
