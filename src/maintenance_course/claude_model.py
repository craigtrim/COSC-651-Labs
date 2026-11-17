"""Use the signed-in Claude CLI for one isolated course model request at a time."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from .prompt_comparison import TEXT_SYSTEM


def _child_environment():
    """Keep the CLI's existing login while excluding API-token overrides."""
    excluded = {
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_CUSTOM_HEADERS",
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_SESSION_ACCESS_TOKEN",
        "CLAUDE_CODE_SIMPLE",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
    }
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in excluded
        and not any(
            marker in key.upper() for marker in ("API_KEY", "AUTH_TOKEN", "OAUTH_TOKEN")
        )
    }


def _count(value):
    return value if type(value) is int and value >= 0 else 0


class ClaudeCLIModel:
    """Call Claude Code without tools, repository context, or persisted sessions.

    Calls count logical subprocess attempts, including failures. CLI-reported
    turns and usage remain separate; the adapter makes no probes or retries.
    """

    mode = "live"
    backend = "claude-cli"

    def __init__(
        self,
        system,
        model=None,
        effort=None,
        max_calls=80,
        timeout=120,
        executable=None,
    ):
        self.requested_model = model or os.getenv("COURSE_MODEL", "sonnet")
        self.effort = effort or os.getenv("COURSE_EFFORT", "medium")
        if not re.fullmatch(r"[a-zA-Z0-9._-]+", self.requested_model):
            raise ValueError("Invalid Claude model identifier")
        if self.effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Invalid Claude effort")
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("Model call budget must be a positive integer")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Claude timeout must be positive and finite")
        self.executable = executable or shutil.which("claude")
        if not self.executable:
            raise ValueError("Install the Claude CLI and sign in before live runs")
        self.system = system
        self.model_id = self.requested_model
        self.max_calls = max_calls
        self.timeout = timeout
        self.calls = 0
        self.events = []
        self.input_tokens = 0
        self.output_tokens = 0

    def _usage(self, envelope, event):
        usage = envelope.get("usage")
        models = envelope.get("modelUsage")
        models = models if isinstance(models, dict) else {}
        resolved = envelope.get("model")
        if not isinstance(resolved, str) and len(models) == 1:
            resolved = next(iter(models))
        if isinstance(resolved, str) and resolved:
            self.model_id = resolved
        event["model"] = self.model_id
        event["resolved_models"] = list(models)
        event["cli_turns"] = envelope.get("num_turns")
        if isinstance(usage, dict) and any(
            key in usage
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        ):
            incoming = sum(
                _count(usage.get(key))
                for key in (
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            )
            outgoing = _count(usage.get("output_tokens"))
            event["usage"] = usage
            event["usage_available"] = True
        elif models:
            incoming = sum(
                _count(row.get(key))
                for row in models.values()
                if isinstance(row, dict)
                for key in (
                    "inputTokens",
                    "cacheReadInputTokens",
                    "cacheWriteInputTokens",
                )
            )
            outgoing = sum(
                _count(row.get("outputTokens"))
                for row in models.values()
                if isinstance(row, dict)
            )
            event["usage"] = {"modelUsage": models}
            event["usage_available"] = True
        else:
            incoming = outgoing = 0
        event["input_tokens"] = incoming
        event["output_tokens"] = outgoing
        self.input_tokens += incoming
        self.output_tokens += outgoing

    @staticmethod
    def _answer(envelope):
        for key in ("structured_output", "result", "output"):
            if key in envelope and envelope[key] is not None:
                answer = envelope[key]
                break
        else:
            raise RuntimeError("Claude CLI response has no result or output")
        if not isinstance(answer, str):
            return answer
        match = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", answer, re.S)
        if match:
            answer = match.group(1)
        try:
            return json.loads(answer)
        except json.JSONDecodeError:
            raise RuntimeError("Claude CLI returned non-JSON model output") from None

    def complete(self, task, payload):
        """Submit task JSON on stdin and retain success or failure accounting."""
        return self._complete(task, payload)

    def complete_text(self, prompt):
        """Send the exact plain prompt and return text without report parsing."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Supply a nonempty text prompt")
        return self._complete("prompt_comparison", {}, text_prompt=prompt)

    def _complete(self, task, payload, text_prompt=None):
        """Share isolated CLI transport while keeping text and JSON tasks distinct."""
        if self.calls >= self.max_calls:
            raise RuntimeError("Model call budget exhausted")
        self.calls += 1
        started = time.perf_counter()
        system = self.system if text_prompt is None else TEXT_SYSTEM
        event = {
            "task": task,
            "mode": self.mode,
            "backend": self.backend,
            "model": self.model_id,
            "requested_model": self.requested_model,
            "effort": self.effort,
            "system_hash": hashlib.sha256(system.encode("utf-8")).hexdigest(),
            "usage_available": False,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        try:
            prompt = (
                json.dumps({"task": task, **payload}, ensure_ascii=False)
                if text_prompt is None
                else text_prompt
            )
            event["input_hash"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            with tempfile.TemporaryDirectory(prefix="course-claude-") as folder:
                root = Path(folder)
                work = root / "work"
                work.mkdir()
                config = root / "mcp.json"
                config.write_text('{"mcpServers":{}}', encoding="utf-8")
                command = [
                    self.executable,
                    "--print",
                    "--output-format",
                    "json",
                    "--model",
                    self.requested_model,
                    "--effort",
                    self.effort,
                    "--system-prompt",
                    system,
                    "--safe-mode",
                    "--restricted",
                    "--tools",
                    "",
                    "--allowedTools",
                    "",
                    "--strict-mcp-config",
                    "--mcp-config",
                    str(config),
                    "--setting-sources",
                    "",
                    "--no-session-persistence",
                    "--disable-slash-commands",
                    "--permission-mode",
                    "dontAsk",
                    "--permission-prompts",
                    "none",
                ]
                result = subprocess.run(
                    command,
                    input=prompt,
                    cwd=work,
                    env=_child_environment(),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=self.timeout,
                    check=False,
                    shell=False,
                )
            event["exit_code"] = result.returncode
            try:
                envelope = json.loads(result.stdout)
            except json.JSONDecodeError:
                if result.returncode:
                    raise RuntimeError(
                        f"Claude CLI exited with status {result.returncode}"
                    ) from None
                raise RuntimeError(
                    "Claude CLI returned an invalid JSON envelope"
                ) from None
            if not isinstance(envelope, dict):
                raise RuntimeError("Claude CLI returned an invalid JSON envelope")
            self._usage(envelope, event)
            if result.returncode or envelope.get("is_error"):
                raise RuntimeError(
                    f"Claude CLI request failed (exit status {result.returncode}); no retry was made"
                )
            for key in ("result", "output"):
                if isinstance(envelope.get(key), str):
                    event["response_text"] = envelope[key]
                    break
            if text_prompt is None:
                answer = self._answer(envelope)
            else:
                answer = event.get("response_text")
                if not isinstance(answer, str) or not answer.strip():
                    raise RuntimeError("Claude CLI returned no text response")
            event["response"] = answer
            return answer
        except subprocess.TimeoutExpired:
            error = TimeoutError(f"Claude CLI request timed out after {self.timeout}s")
            event["error"] = {"type": type(error).__name__, "message": str(error)}
            raise error from None
        except Exception as exc:
            message = (
                str(exc)
                if isinstance(exc, RuntimeError)
                else f"Claude CLI request failed ({type(exc).__name__})"
            )
            event["error"] = {"type": type(exc).__name__, "message": message}
            raise RuntimeError(message) from None
        finally:
            event["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
            self.events.append(event)
