"""Student-mediated browser transport for the Colab text-review exercise."""

import json
from pathlib import Path
import time


CAPTURE_JS = """(prompt) => new Promise((resolve) => {
  const panel = document.createElement('section');
  panel.style.cssText = 'padding:16px;border:1px solid #aaa;background:#fff;color:#222';
  const instructions = document.createElement('p');
  instructions.textContent = 'Send this exact prompt in a fresh chat with your selected model. Paste the complete reply below, or save an access error.';
  const request = document.createElement('textarea');
  request.value = prompt; request.readOnly = true;
  request.setAttribute('aria-label', 'Exact model prompt');
  request.style.cssText = 'width:100%;height:180px';
  const response = document.createElement('textarea');
  response.setAttribute('aria-label', 'Complete model reply or access error');
  response.style.cssText = 'width:100%;height:180px';
  const label = document.createElement('label');
  const attest = document.createElement('input'); attest.type = 'checkbox';
  label.append(attest, ' I used a fresh chat, the exact prompt, and the same model and visible settings.');
  const save = document.createElement('button'); save.textContent = 'Save reply';
  const fail = document.createElement('button'); fail.textContent = 'Save request failure';
  const finish = (error) => {
    if (!response.value.trim() || (!error && !attest.checked)) return;
    save.disabled = true; fail.disabled = true; response.readOnly = true;
    resolve({text: response.value, error, attested: attest.checked});
  };
  save.onclick = () => finish(false); fail.onclick = () => finish(true);
  panel.append(instructions, request, response, label, document.createElement('br'), save, fail);
  document.body.append(panel);
  globalThis.google?.colab?.output?.setIframeHeight?.(document.documentElement.scrollHeight, true);
})"""


def colab_capture(prompt):
    try:
        from google.colab import output
    except ImportError:
        print(prompt)
        reply_file = Path(
            input("Path to the complete UTF-8 reply or access-error file: ")
        )
        failed = input(
            "Type ERROR for an access error, or REPLY for an answer: "
        ).strip()
        if failed not in {"ERROR", "REPLY"}:
            raise ValueError("Choose ERROR or REPLY")
        attested = (
            input(
                "Fresh chat, exact prompt, unchanged model/settings? Type YES to attest: "
            ).strip()
            == "YES"
        )
        return {
            "text": reply_file.read_text(encoding="utf-8"),
            "error": failed == "ERROR",
            "attested": attested,
        }

    from IPython.display import Javascript, display

    display(Javascript("window.coscCaptureReply = " + CAPTURE_JS + ";"))
    return output.eval_js(f"window.coscCaptureReply({json.dumps(prompt)})")


class BrowserTextModel:
    """The program selects requests; the student transfers them through chat."""

    mode = "live"
    backend = "browser-chat"
    transport = "manual_browser_transfer"

    def __init__(self, provider, model, settings, capture=None):
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("Record the browser provider")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError("Record the displayed model, or None if unavailable")
        self.provider = provider
        self.model_id = model
        self.visible_settings = dict(settings)
        self.capture = capture or colab_capture
        self.events = []
        self.calls = 0

    def complete_text(self, prompt):
        self.calls += 1
        started = time.perf_counter()
        event = {
            "mode": "live",
            "transport": self.transport,
            "provider": self.provider,
            "model": self.model_id,
            "visible_settings": self.visible_settings.copy(),
            "metadata_provenance": "student_reported",
            "usage_available": False,
            "automated_provider_calls": 0,
        }
        try:
            result = self.capture(prompt)
            if not isinstance(result, dict) or (
                result.get("error") is not True and result.get("attested") is not True
            ):
                raise ValueError(
                    "The browser capture needs an honest session/settings attestation"
                )
            text = result.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Paste the complete reply or error")
            event.update(response_text=text, attested=result.get("attested") is True)
            if result.get("error") is True:
                event["browser_error_text"] = text
                raise RuntimeError(
                    "Browser request failed; inspect the saved access error"
                )
            return text
        except Exception as exc:
            event["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            event["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
            event["timing_scope"] = (
                "includes student transfer and waiting; not provider latency"
            )
            self.events.append(event)
