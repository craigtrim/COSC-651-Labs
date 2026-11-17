"""Three isolated, raw-text prompt runs for the Week 1 support exercise."""

from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

from .data import DATA


TEXT_SYSTEM = "You are a helpful assistant. Answer the user's request."
VERSIONS = ("version1", "version2", "version3")


def prepare_browser_comparison(case, diagnostic, policy, folder):
    """Export exact prompts and a capture form without making model requests.

    The new folder is exclusive: rerunning this operation cannot replace a
    student's source prompts, capture metadata, or saved replies.
    """
    prompts = build_three_prompts(case, diagnostic, policy)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    for version, prompt in prompts.items():
        (folder / f"{version}-prompt.txt").write_bytes(prompt.encode("utf-8"))
    source_inputs = {
        "case": case,
        "diagnostic": diagnostic,
        "policy": policy,
        "prompt_hashes": {
            version: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for version, prompt in prompts.items()
        },
    }
    (folder / "inputs.json").write_bytes(
        (json.dumps(source_inputs, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    capture = {
        "provider": "",
        "model": None,
        "visible_settings": {
            "reasoning_mode": None,
            "web_search": None,
            "memory": None,
            "custom_instructions": None,
        },
        "same_model_and_settings_attested": False,
        "runs": [
            {
                "variant": version,
                "timestamp": "",
                "fresh_conversation_attested": False,
                "prompt_used_as_exported_attested": False,
                "conversation_reference": None,
                "error": None,
            }
            for version in VERSIONS
        ],
    }
    (folder / "capture.json").write_bytes(
        (json.dumps(capture, indent=2) + "\n").encode("utf-8")
    )
    return folder


def _browser_timestamp(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Record each browser request's date or timestamp")
    try:
        if len(value) == 10:
            date.fromisoformat(value)
            return "date"
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("A time needs its timezone")
        return "datetime"
    except ValueError as exc:
        raise ValueError(
            "Use YYYY-MM-DD or an ISO timestamp with its timezone; do not guess a time"
        ) from exc


def import_browser_comparison(case, diagnostic, policy, folder):
    """Import student-captured browser replies, with no API or CLI requests.

    Exact prompt files and reply bytes are checked and retained. Browser
    metadata and session controls are student attestations, not observations
    of the provider. Unknown hidden settings, system text and usage stay
    unknown rather than borrowing the automated adapter's system prompt.
    """
    prompts = build_three_prompts(case, diagnostic, policy)
    folder = Path(folder)
    capture_bytes = (folder / "capture.json").read_bytes()
    capture = json.loads(capture_bytes.decode("utf-8"))
    if not isinstance(capture, dict):
        raise ValueError("capture.json must contain the supplied capture form")
    inputs_bytes = (folder / "inputs.json").read_bytes()
    source_inputs = json.loads(inputs_bytes.decode("utf-8"))
    expected_inputs = {
        "case": case,
        "diagnostic": diagnostic,
        "policy": policy,
        "prompt_hashes": {
            version: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for version, prompt in prompts.items()
        },
    }
    if source_inputs != expected_inputs:
        raise ValueError(
            "Comparison inputs changed; preserve this capture and start a new one"
        )
    provider = capture.get("provider")
    model = capture.get("model")
    settings = capture.get("visible_settings")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("Fill in the browser provider in capture.json")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("Use the displayed model label, or null if it is not shown")
    if not isinstance(settings, dict):
        raise ValueError("Record visible_settings; use null for unavailable values")
    json.dumps(settings, allow_nan=False)
    if capture.get("same_model_and_settings_attested") is not True:
        raise ValueError("Attest that the same displayed model and settings were used")
    runs = capture.get("runs")
    if not isinstance(runs, list) or not all(isinstance(run, dict) for run in runs):
        raise ValueError("Supply one capture entry for each of the three versions")
    if [run.get("variant") for run in runs] != list(VERSIONS):
        raise ValueError("Capture entries must be version1, version2 and version3")
    imported_at = datetime.now(timezone.utc).isoformat()
    records = []
    for run in runs:
        version = run["variant"]
        if run.get("fresh_conversation_attested") is not True:
            raise ValueError(f"Attest that {version} used a fresh conversation")
        if run.get("prompt_used_as_exported_attested") is not True:
            raise ValueError(
                f"Attest that {version} used the exported prompt unchanged"
            )
        precision = _browser_timestamp(run.get("timestamp"))
        reference = run.get("conversation_reference")
        if reference is not None and not isinstance(reference, str):
            raise ValueError("conversation_reference must be text or null")
        prompt_file = folder / f"{version}-prompt.txt"
        prompt_bytes = prompt_file.read_bytes()
        prompt = prompt_bytes.decode("utf-8")
        if prompt != prompts[version]:
            raise ValueError(
                f"The exported {version} prompt changed; preserve this run"
            )
        response_file = folder / f"{version}-response.txt"
        response_bytes = response_file.read_bytes()
        response = response_bytes.decode("utf-8")
        if not response.strip():
            raise ValueError(f"Save the verbatim {version} reply or browser error text")
        error = run.get("error")
        if error is not None and (
            not isinstance(error, dict)
            or not all(
                isinstance(error.get(key), str) and error[key].strip()
                for key in ("type", "message")
            )
        ):
            raise ValueError("A browser failure needs an error type and message")
        limitations = [
            "Provider, model label, visible settings, date and session controls are student reported.",
            "The browser system prompt, hidden state, internal call count, usage and elapsed time are unobserved.",
        ]
        if model is None:
            limitations.append("The browser did not display a model identifier.")
        records.append(
            {
                "run_id": str(uuid.uuid4()),
                "experiment": "week01_prompt_comparison",
                "case_id": case["id"],
                "variant": version,
                "mode": "live",
                "backend": "browser-chat",
                "transport": "manual_browser_import",
                "provider": provider.strip(),
                "model": model,
                "requested_model": model,
                "settings": settings.copy(),
                "timestamp": run["timestamp"],
                "timestamp_precision": precision,
                "system_prompt": None,
                "system_hash": None,
                "prompt": prompt,
                "prompt_hash": hashlib.sha256(prompt_bytes).hexdigest(),
                "response_text": response if error is None else None,
                "browser_error_text": response if error is not None else None,
                "error": error,
                "supplied_policy_ids": [policy["id"]] if version == "version3" else [],
                "supplied_policy": policy if version == "version3" else None,
                "supplied_diagnostic": diagnostic if version == "version3" else None,
                "model_calls": 1,
                "automated_model_calls": 0,
                "model_call_count_source": "student_reported",
                "usage_available": False,
                "input_tokens": None,
                "output_tokens": None,
                "elapsed_ms": None,
                "model_events": [],
                "metadata_limitations": limitations,
                "provenance": {
                    "source": "student_browser_chat",
                    "imported_at": imported_at,
                    "capture_file": "capture.json",
                    "capture_sha256": hashlib.sha256(capture_bytes).hexdigest(),
                    "inputs_file": "inputs.json",
                    "inputs_sha256": hashlib.sha256(inputs_bytes).hexdigest(),
                    "prompt_file": prompt_file.name,
                    "response_file": response_file.name,
                    "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
                    "conversation_reference": reference,
                    "fresh_conversation_attested": True,
                    "prompt_used_as_exported_attested": True,
                    "same_model_and_settings_attested": True,
                    "attestations_verified_by_importer": False,
                },
            }
        )
    return records


def build_three_prompts(case, diagnostic, policy):
    """Keep Version 2's instructions intact when appending Version 3 evidence.

    ``diagnostic`` is the registry's complete diagnostic response. ``policy``
    is the document returned inside a successful read_document result.
    Public answers are not inputs to this function.
    """
    if not isinstance(case.get("report"), str) or not case["report"].strip():
        raise ValueError("The case needs a nonempty customer report")
    if not isinstance(diagnostic, dict) or type(diagnostic.get("ok")) is not bool:
        raise ValueError("Supply the recorded diagnostic response")
    if not isinstance(policy, dict) or not all(
        isinstance(policy.get(key), str) and policy[key].strip()
        for key in ("id", "text")
    ):
        raise ValueError("Supply the current policy document with its ID and text")
    if policy.get("version") != case.get("app_version"):
        raise ValueError("Policy version does not match the case")
    folder = DATA / "prompts" / "week01"
    prompts = {
        version: (folder / f"{version}.txt")
        .read_text(encoding="utf-8")
        .replace("{{COMPLAINT}}", case["report"])
        .rstrip()
        for version in VERSIONS[:2]
    }
    result = diagnostic.get("result", {})
    if diagnostic["ok"] and not isinstance(result, dict):
        raise ValueError("A successful diagnostic needs its recorded result")
    if diagnostic["ok"] and result.get("codes") == ["MISSING_DATE"]:
        finding = "The supplied sample has a blank invoice_date field."
    else:
        finding = json.dumps(diagnostic, ensure_ascii=False, sort_keys=True)
    prompts["version3"] = (
        prompts["version2"]
        + "\n\nINVOICE FINDINGS\n"
        + finding
        + "\n\nCURRENT COMPANY RULE: "
        + policy["id"]
        + "\n"
        + policy["text"]
    )
    return prompts


def run_prompt_comparison(
    case, diagnostic, policy, mode="fixture", variants=None, model_factory=None
):
    """Return one raw-text record per selected version, retaining request failures.

    Each version receives a fresh model and an independent request. Select one
    version for a live pilot and the other versions afterward to avoid rerunning
    the pilot. Factory/configuration errors propagate before that version runs.
    These records are not four-field course reports and are not label-scored.
    """
    from .models import make_model

    if mode not in {"fixture", "live"}:
        raise ValueError("Mode must be fixture or live")
    selected = list(VERSIONS if variants is None else variants)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(version not in VERSIONS for version in selected)
    ):
        raise ValueError("Select distinct versions from version1, version2, version3")
    prompts = build_three_prompts(case, diagnostic, policy)
    factory = model_factory or make_model
    records = []
    for version in selected:
        model = factory(mode)
        if not callable(getattr(model, "complete_text", None)):
            raise ValueError("The model adapter must support complete_text(prompt)")
        prompt = prompts[version]
        started = time.perf_counter()
        response = None
        error = None
        try:
            response = model.complete_text(prompt)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        events = list(model.events)
        event = events[-1] if events else {}
        usage_available = bool(event.get("usage_available", False))
        records.append(
            {
                "run_id": str(uuid.uuid4()),
                "experiment": "week01_prompt_comparison",
                "case_id": case["id"],
                "variant": version,
                "mode": mode,
                "backend": getattr(model, "backend", "fixture"),
                "model": model.model_id,
                "requested_model": getattr(model, "requested_model", model.model_id),
                "settings": {
                    "effort": getattr(model, "effort", None),
                    "temperature": getattr(model, "temperature", None),
                    "max_output_tokens": getattr(model, "max_output_tokens", None),
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system_prompt": TEXT_SYSTEM,
                "system_hash": hashlib.sha256(TEXT_SYSTEM.encode()).hexdigest(),
                "prompt": prompt,
                "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "response_text": response,
                "error": error,
                "supplied_policy_ids": [policy["id"]] if version == "version3" else [],
                "supplied_policy": policy if version == "version3" else None,
                "supplied_diagnostic": diagnostic if version == "version3" else None,
                "model_calls": model.calls,
                "usage_available": usage_available,
                "input_tokens": model.input_tokens if usage_available else None,
                "output_tokens": model.output_tokens if usage_available else None,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "model_events": events,
            }
        )
    return records
