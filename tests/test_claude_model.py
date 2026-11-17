"""Exercise the Claude adapter without invoking a model service or reading auth."""

import json
import subprocess
from pathlib import Path

import pytest

from maintenance_course.claude_model import ClaudeCLIModel
from maintenance_course.models import FixtureModel, GeminiModel, SYSTEM, make_model


REPORT = {
    "decision": "no_change",
    "action": "no_change",
    "evidence": [],
    "rationale": "No observation was supplied.",
}


def completed(envelope, returncode=0, stderr=""):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=json.dumps(envelope), stderr=stderr
    )


def model(**kwargs):
    return ClaudeCLIModel(system=SYSTEM, executable="mock-claude", **kwargs)


def test_request_is_isolated_and_does_not_use_token_overrides(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-auth-token")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "test-oauth-token")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ACCESS_TOKEN", "test-session-token")
    monkeypatch.setenv("OPENAI_API_KEY", "test-other-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://invalid.example")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "existing-cli-config")
    invocations = []

    def run(command, **kwargs):
        invocations.append(command)
        assert command[0] == "mock-claude"
        for flag in (
            "--safe-mode",
            "--restricted",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--disable-slash-commands",
        ):
            assert flag in command
        for flag, value in (
            ("--tools", ""),
            ("--allowedTools", ""),
            ("--setting-sources", ""),
            ("--permission-prompts", "none"),
            ("--model", "sonnet"),
            ("--effort", "medium"),
            ("--system-prompt", SYSTEM),
        ):
            assert command[command.index(flag) + 1] == value
        assert "--bare" not in command
        assert "--fallback-model" not in command
        assert "--continue" not in command
        assert "--resume" not in command
        assert not list(Path(kwargs["cwd"]).iterdir())
        config = Path(command[command.index("--mcp-config") + 1])
        assert json.loads(config.read_text()) == {"mcpServers": {}}
        assert config.parent != Path(kwargs["cwd"])
        assert not any(
            marker in key.upper()
            for key in kwargs["env"]
            for marker in ("API_KEY", "AUTH_TOKEN", "OAUTH_TOKEN")
        )
        assert "CLAUDE_CODE_USE_BEDROCK" not in kwargs["env"]
        assert "ANTHROPIC_BASE_URL" not in kwargs["env"]
        assert kwargs["env"]["CLAUDE_CONFIG_DIR"] == "existing-cli-config"
        assert kwargs["shell"] is False
        assert kwargs["timeout"] == 120
        request = json.loads(kwargs["input"])
        assert request == {"task": "proposal", "case": {"id": "D01"}}
        assert kwargs["input"] not in command
        return completed(
            {
                "result": json.dumps(REPORT),
                "is_error": False,
                "num_turns": 1,
                "usage": {
                    "input_tokens": 10,
                    "cache_creation_input_tokens": 20,
                    "cache_read_input_tokens": 30,
                    "output_tokens": 8,
                },
                "modelUsage": {"claude-sonnet-resolved": {}},
            }
        )

    monkeypatch.setattr(subprocess, "run", run)
    adapter = model(model="sonnet", effort="medium")
    assert adapter.complete("proposal", {"case": {"id": "D01"}}) == REPORT
    assert len(invocations) == adapter.calls == 1
    assert adapter.input_tokens == 60
    assert adapter.output_tokens == 8
    assert adapter.model_id == "claude-sonnet-resolved"
    event = adapter.events[0]
    assert event["backend"] == "claude-cli"
    assert event["effort"] == "medium"
    assert event["requested_model"] == "sonnet"
    assert event["model"] == "claude-sonnet-resolved"
    assert event["cli_turns"] == 1
    assert event["usage_available"] is True
    assert event["response"] == REPORT
    assert len(event["system_hash"]) == len(event["input_hash"]) == 64


def test_twenty_logical_calls_have_no_extra_probes_or_retries(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(json.loads(kwargs["input"]))
        return completed({"result": json.dumps(REPORT)})

    monkeypatch.setattr(subprocess, "run", run)
    adapter = model(max_calls=20)
    for number in range(20):
        adapter.complete("proposal", {"case": {"id": f"case-{number}"}})
    with pytest.raises(RuntimeError, match="budget exhausted"):
        adapter.complete("proposal", {"case": {"id": "extra"}})
    assert len(calls) == len(adapter.events) == adapter.calls == 20
    assert all(event["usage_available"] is False for event in adapter.events)


def test_timeout_consumes_budget_and_records_failure_without_retry(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", run)
    adapter = model(max_calls=1, timeout=2)
    with pytest.raises(TimeoutError, match="timed out"):
        adapter.complete("proposal", {})
    with pytest.raises(RuntimeError, match="budget exhausted"):
        adapter.complete("proposal", {})
    assert len(calls) == adapter.calls == len(adapter.events) == 1
    assert adapter.events[0]["error"]["type"] == "TimeoutError"
    assert adapter.events[0]["usage_available"] is False


def test_cli_error_preserves_usage_without_echoing_auth_or_stderr(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: completed(
            {
                "is_error": True,
                "result": "test-sensitive-auth-detail",
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
            returncode=1,
            stderr="test-sensitive-stderr",
        ),
    )
    adapter = model()
    with pytest.raises(RuntimeError, match="exit status 1") as error:
        adapter.complete("proposal", {})
    assert adapter.calls == 1
    assert adapter.input_tokens == 5
    assert adapter.output_tokens == 2
    assert "test-sensitive" not in str(error.value)
    assert "test-sensitive" not in json.dumps(adapter.events)


def test_malformed_model_json_is_failed_but_usage_is_retained(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: completed(
            {"result": "not JSON", "usage": {"input_tokens": 7, "output_tokens": 3}}
        ),
    )
    adapter = model()
    with pytest.raises(RuntimeError, match="non-JSON model output"):
        adapter.complete("proposal", {})
    assert adapter.calls == 1
    assert adapter.input_tokens == 7
    assert adapter.events[0]["response_text"] == "not JSON"
    assert adapter.events[0]["error"]


@pytest.mark.parametrize(
    "envelope",
    [
        {"output": json.dumps(REPORT)},
        {"structured_output": REPORT},
        {"result": "```json\n" + json.dumps(REPORT) + "\n```"},
    ],
)
def test_supported_output_envelopes(monkeypatch, envelope):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed(envelope))
    assert model().complete("proposal", {}) == REPORT


def test_model_usage_fallback_identifies_model_and_counts_cache(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: completed(
            {
                "result": json.dumps(REPORT),
                "modelUsage": {
                    "claude-actual": {
                        "inputTokens": 2,
                        "cacheReadInputTokens": 3,
                        "cacheWriteInputTokens": 4,
                        "outputTokens": 5,
                    }
                },
            }
        ),
    )
    adapter = model()
    adapter.complete("proposal", {})
    assert adapter.input_tokens == 9
    assert adapter.output_tokens == 5
    assert adapter.model_id == "claude-actual"


@pytest.mark.parametrize("stdout", ["invalid envelope", "[]", "{}"])
def test_bad_envelope_is_one_recorded_failure(monkeypatch, stdout):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, stdout, ""),
    )
    adapter = model()
    with pytest.raises(RuntimeError):
        adapter.complete("proposal", {})
    assert adapter.calls == len(adapter.events) == 1
    assert adapter.events[0]["error"]


def test_start_failure_does_not_disclose_exception_detail(monkeypatch):
    def run(*args, **kwargs):
        raise OSError("test-sensitive-environment-detail")

    monkeypatch.setattr(subprocess, "run", run)
    adapter = model()
    with pytest.raises(RuntimeError, match="OSError"):
        adapter.complete("proposal", {})
    assert adapter.calls == 1
    assert "test-sensitive" not in json.dumps(adapter.events)


def test_live_backend_selection_preserves_fixture_and_gemini(monkeypatch):
    monkeypatch.setenv("COURSE_BACKEND", "claude-cli")
    monkeypatch.setenv("COURSE_MODEL", "sonnet")
    monkeypatch.setenv("COURSE_EFFORT", "medium")
    monkeypatch.setattr(
        "maintenance_course.claude_model.shutil.which", lambda _: "mock"
    )
    adapter = make_model("live")
    assert isinstance(adapter, ClaudeCLIModel)
    assert adapter.system == SYSTEM
    assert adapter.requested_model == "sonnet"
    assert adapter.effort == "medium"
    assert isinstance(make_model("fixture"), FixtureModel)
    monkeypatch.delenv("COURSE_BACKEND")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert isinstance(make_model("live"), GeminiModel)
    monkeypatch.setenv("COURSE_BACKEND", "unknown")
    with pytest.raises(ValueError, match="COURSE_BACKEND"):
        make_model("live")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_calls": 0},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"effort": "invalid"},
    ],
)
def test_invalid_configuration_fails_before_process_start(kwargs):
    with pytest.raises(ValueError):
        model(**kwargs)
