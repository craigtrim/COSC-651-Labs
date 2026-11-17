"""Verify the three-prompt experiment without model-service calls."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import urllib.error
import urllib.request

import pytest

from maintenance_course.claude_model import ClaudeCLIModel
from maintenance_course.data import case_by_id, documents
from maintenance_course.models import FixtureModel, GeminiModel, SYSTEM
from maintenance_course.prompt_comparison import (
    TEXT_SYSTEM,
    build_three_prompts,
    import_browser_comparison,
    prepare_browser_comparison,
    run_prompt_comparison,
)


def inputs():
    return (
        case_by_id("D01"),
        {"ok": True, "result": {"codes": ["MISSING_DATE"], "rows": 1}},
        next(d for d in documents() if d["id"] == "dates-v2"),
    )


def test_only_version_three_receives_observations_and_policies():
    case, diagnostic, policy = inputs()
    prompts = build_three_prompts(case, diagnostic, policy)
    assert list(prompts) == ["version1", "version2", "version3"]
    for version in prompts:
        assert case["report"] in prompts[version]
    assert "1. Classify the issue" not in prompts["version1"]
    assert "6. Write the internal escalation note" in prompts["version2"]
    for version in ("version1", "version2"):
        assert policy["text"] not in prompts[version]
        assert "INVOICE FINDINGS" not in prompts[version]
        assert "CURRENT COMPANY RULE" not in prompts[version]
    assert prompts["version3"].startswith(prompts["version2"] + "\n\n")
    assert "The supplied sample has a blank invoice_date field." in prompts["version3"]
    assert policy["text"] in prompts["version3"]
    assert "recommend_fix" not in prompts["version3"]


def test_unavailable_diagnostic_stays_a_failure_observation():
    case, _, policy = inputs()
    diagnostic = {"ok": False, "error_type": "TimeoutError", "error": "Unavailable"}
    prompts = build_three_prompts(case, diagnostic, policy)
    assert "Unavailable" in prompts["version3"]
    assert "sample has a blank" not in prompts["version3"]
    with pytest.raises(ValueError, match="version"):
        build_three_prompts(case, diagnostic, {**policy, "version": 1})


def test_multiple_diagnostic_findings_are_not_silently_dropped():
    case, diagnostic, policy = inputs()
    diagnostic["result"]["codes"].append("DUPLICATE_ID")
    prompts = build_three_prompts(case, diagnostic, policy)
    assert "DUPLICATE_ID" in prompts["version3"]
    assert "MISSING_DATE" in prompts["version3"]


def test_fixture_records_are_isolated_and_cannot_be_mistaken_for_model_answers():
    created = []

    def factory(mode):
        assert mode == "fixture"
        model = FixtureModel()
        created.append(model)
        return model

    records = run_prompt_comparison(*inputs(), model_factory=factory)
    assert len(created) == len(records) == 3
    assert len({id(m) for m in created}) == 3
    assert all(r["model_calls"] == 1 and len(r["model_events"]) == 1 for r in records)
    assert all("FIXTURE ONLY" in r["response_text"] for r in records)
    assert all(r["input_tokens"] is None and not r["usage_available"] for r in records)
    assert [r["supplied_policy_ids"] for r in records] == [[], [], ["dates-v2"]]
    assert records[0]["supplied_diagnostic"] is None
    assert records[2]["supplied_diagnostic"] == inputs()[1]
    for record in records:
        assert "report" not in record and "validation_errors" not in record
        assert (
            record["prompt_hash"]
            == hashlib.sha256(record["prompt"].encode()).hexdigest()
        )
        assert record["system_prompt"] == TEXT_SYSTEM


def test_pilot_can_be_retained_while_only_remaining_versions_run():
    pilot = run_prompt_comparison(*inputs(), variants=["version1"])
    remainder = run_prompt_comparison(*inputs(), variants=["version2", "version3"])
    assert [r["variant"] for r in pilot + remainder] == [
        "version1",
        "version2",
        "version3",
    ]
    assert sum(r["model_calls"] for r in pilot + remainder) == 3
    for bad in ([], ["version1", "version1"], ["unknown"]):
        with pytest.raises(ValueError, match="distinct versions"):
            run_prompt_comparison(*inputs(), variants=bad)


def test_request_failures_remain_records_in_the_denominator():
    class Failed(FixtureModel):
        def complete_text(self, prompt):
            self.calls += 1
            raise TimeoutError("No response")

    records = run_prompt_comparison(*inputs(), model_factory=lambda mode: Failed())
    assert len(records) == 3
    assert all(r["error"]["type"] == "TimeoutError" for r in records)
    assert all(r["response_text"] is None and r["model_calls"] == 1 for r in records)


def test_claude_text_is_exact_and_json_report_calls_remain_strict(monkeypatch):
    requests = []

    def run(command, **kwargs):
        requests.append(kwargs["input"])
        assert command[command.index("--tools") + 1] == ""
        assert command[command.index("--allowedTools") + 1] == ""
        assert "--strict-mcp-config" in command
        assert "--no-session-persistence" in command
        assert "--setting-sources" in command
        assert not list(Path(kwargs["cwd"]).iterdir())
        expected_system = TEXT_SYSTEM if len(requests) == 1 else SYSTEM
        assert command[command.index("--system-prompt") + 1] == expected_system
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "result": "A plain support reply.",
                    "usage": {"input_tokens": 8, "output_tokens": 4},
                }
            ),
            "",
        )

    monkeypatch.setattr(subprocess, "run", run)
    model = ClaudeCLIModel(system=SYSTEM, executable="mock-claude", max_calls=2)
    prompt = "Give a reply.\n\nComplaint:\nNo date."
    assert model.complete_text(prompt) == "A plain support reply."
    assert requests[0] == prompt
    assert model.events[0]["response_text"] == "A plain support reply."
    assert model.events[0]["usage_available"]
    with pytest.raises(RuntimeError, match="non-JSON"):
        model.complete("proposal", {"case": {"id": "D01"}})
    assert json.loads(requests[1])["task"] == "proposal"
    with pytest.raises(RuntimeError, match="budget exhausted"):
        model.complete_text(prompt)
    assert len(requests) == model.calls == 2


def test_gemini_text_is_not_forced_to_json(monkeypatch):
    prompt = "Respond to this complaint."

    def urlopen(request, timeout):
        body = json.loads(request.data)
        assert body["systemInstruction"]["parts"] == [{"text": TEXT_SYSTEM}]
        assert body["contents"][0]["parts"] == [{"text": prompt}]
        assert body["generationConfig"]["responseMimeType"] == "text/plain"
        return io.BytesIO(
            json.dumps(
                {
                    "candidates": [
                        {"content": {"parts": [{"text": "A plain answer."}]}}
                    ],
                    "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3},
                }
            ).encode()
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    model = GeminiModel(model="test-model", key="test-key", max_calls=1)
    assert model.complete_text(prompt) == "A plain answer."
    assert model.input_tokens == 7 and model.output_tokens == 3
    assert model.events[0]["usage_available"]
    assert "test-key" not in json.dumps(model.events)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        model.complete_text(prompt)


def test_gemini_text_failure_records_attempt_without_retry(monkeypatch):
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        raise urllib.error.HTTPError(request.full_url, 429, "Quota", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    model = GeminiModel(model="test-model", key="test-key", max_calls=1)
    with pytest.raises(RuntimeError, match="HTTP 429"):
        model.complete_text("Answer this complaint.")
    assert len(requests) == model.calls == len(model.events) == 1
    assert model.events[0]["error"] and not model.events[0]["usage_available"]


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: FixtureModel(max_calls=1),
        lambda: ClaudeCLIModel(system=SYSTEM, executable="mock-claude", max_calls=1),
        lambda: GeminiModel(model="test-model", key="test-key", max_calls=1),
    ],
)
def test_empty_text_prompt_never_consumes_a_request(constructor):
    model = constructor()
    with pytest.raises(ValueError, match="nonempty"):
        model.complete_text(" ")
    assert model.calls == 0


@pytest.fixture
def browser_capture(tmp_path, monkeypatch):
    """Create only marked mock replies; browser importing must use no service."""

    def no_request(*args, **kwargs):
        raise AssertionError("Browser import must not make an automated model request")

    monkeypatch.setattr(subprocess, "run", no_request)
    monkeypatch.setattr(urllib.request, "urlopen", no_request)
    folder = prepare_browser_comparison(*inputs(), tmp_path / "capture")
    capture = json.loads((folder / "capture.json").read_text(encoding="utf-8"))
    capture.update(
        provider="Test browser provider",
        model=None,
        same_model_and_settings_attested=True,
    )
    for run in capture["runs"]:
        run.update(
            timestamp="2026-11-09",
            fresh_conversation_attested=True,
            prompt_used_as_exported_attested=True,
        )
        (folder / f"{run['variant']}-response.txt").write_bytes(
            f"  MOCK browser reply: {run['variant']}.\r\nNo model was called.\r\n".encode()
        )

    def save_capture():
        (folder / "capture.json").write_bytes(json.dumps(capture).encode("utf-8"))

    save_capture()
    return folder, capture, save_capture


def test_browser_capture_exports_exact_prompts_without_overwriting_source_files(
    browser_capture,
):
    folder, _, _ = browser_capture
    original_files = {path.name: path.read_bytes() for path in folder.iterdir()}
    expected = build_three_prompts(*inputs())
    assert all(
        (folder / f"{version}-prompt.txt").read_bytes() == prompt.encode("utf-8")
        for version, prompt in expected.items()
    )
    with pytest.raises(FileExistsError):
        prepare_browser_comparison(*inputs(), folder)
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == original_files


def test_browser_import_keeps_verbatim_replies_and_reports_unknown_metadata(
    browser_capture,
):
    folder, capture, save_capture = browser_capture
    capture["runs"][1]["timestamp"] = "2026-11-09T13:25:00-06:00"
    save_capture()
    original_files = {path.name: path.read_bytes() for path in folder.iterdir()}
    records = import_browser_comparison(*inputs(), folder)
    assert [r["variant"] for r in records] == ["version1", "version2", "version3"]
    assert len({r["run_id"] for r in records}) == 3
    assert sum(r["model_calls"] for r in records) == 3
    assert sum(r["automated_model_calls"] for r in records) == 0
    assert [r["timestamp_precision"] for r in records] == ["date", "datetime", "date"]
    for record in records:
        assert record["mode"] == "live"
        assert record["backend"] == "browser-chat"
        assert record["transport"] == "manual_browser_import"
        assert record["model_call_count_source"] == "student_reported"
        assert record["model"] is record["requested_model"] is None
        assert record["system_prompt"] is record["system_hash"] is None
        assert record["input_tokens"] is record["output_tokens"] is None
        assert record["elapsed_ms"] is None and not record["usage_available"]
        assert record["model_events"] == []
        reply_bytes = (folder / f"{record['variant']}-response.txt").read_bytes()
        assert record["response_text"] == reply_bytes.decode("utf-8")
        assert (
            record["prompt_hash"]
            == hashlib.sha256(record["prompt"].encode()).hexdigest()
        )
        assert (
            record["provenance"]["response_sha256"]
            == hashlib.sha256(reply_bytes).hexdigest()
        )
        assert record["provenance"]["attestations_verified_by_importer"] is False
        assert record["settings"]["reasoning_mode"] is None
    assert [r["supplied_policy_ids"] for r in records] == [[], [], ["dates-v2"]]
    assert records[2]["supplied_diagnostic"] == inputs()[1]
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == original_files


def test_browser_failure_is_retained_with_original_error_text(browser_capture):
    folder, capture, save_capture = browser_capture
    capture["runs"][1]["error"] = {
        "type": "BrowserLimit",
        "message": "Request unavailable",
    }
    save_capture()
    records = import_browser_comparison(*inputs(), folder)
    assert len(records) == 3
    assert records[1]["response_text"] is None
    assert records[1]["error"]["type"] == "BrowserLimit"
    assert "MOCK browser reply: version2" in records[1]["browser_error_text"]


@pytest.mark.parametrize(
    "change",
    [
        "prompt",
        "diagnostic",
        "settings_attestation",
        "fresh_chat",
        "timestamp",
        "reply",
    ],
)
def test_browser_import_rejects_changed_inputs_or_incomplete_capture(
    browser_capture, change
):
    folder, capture, save_capture = browser_capture
    case, diagnostic, policy = inputs()
    if change == "prompt":
        (folder / "version2-prompt.txt").write_text(
            "An edited request.", encoding="utf-8"
        )
    elif change == "diagnostic":
        # This detail does not alter the one-line prompt finding. Input provenance
        # must still reject importing an older run against different observations.
        diagnostic["result"]["rows"] = 200
    elif change == "settings_attestation":
        capture["same_model_and_settings_attested"] = False
    elif change == "fresh_chat":
        capture["runs"][2]["fresh_conversation_attested"] = False
    elif change == "timestamp":
        capture["runs"][0]["timestamp"] = "2026-11-09T13:25:00"
    else:
        (folder / "version3-response.txt").write_text("  ", encoding="utf-8")
    save_capture()
    with pytest.raises(ValueError):
        import_browser_comparison(case, diagnostic, policy, folder)
