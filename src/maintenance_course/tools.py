# Issue #32: The dispatcher enforces case scope and application document versions.
"""A scoped dispatcher with version checks and a killable diagnostic process."""

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from .data import DATA, documents
from .contracts import validate_tool_arguments


class ToolRegistry:
    """Dispatch permitted tools within one case and record observed evidence."""

    def __init__(self, case, fixture_root=None, timeout=2.0):
        """Bind a case, resolved fixture root, and worker timeout to this registry."""
        self.case = dict(case)
        self.fixture_root = Path(fixture_root or DATA / "fixtures").resolve()
        self.timeout = timeout
        self.observed = {}
        self.events = []

    def call(self, name, arguments):
        """Validate and dispatch a tool, recording its response or handled failure."""
        start = time.perf_counter()
        try:
            validate_tool_arguments(name, arguments)
            if name == "search_docs":
                terms = set(re.findall(r"[a-z]+", arguments["query"].lower()))
                candidates = []
                for doc in documents():
                    if doc["version"] != self.case["app_version"]:
                        continue
                    words = set(
                        re.findall(
                            r"[a-z]+",
                            (
                                doc["title"] + " " + doc["tags"] + " " + doc["text"]
                            ).lower(),
                        )
                    )
                    score = len(terms & words)
                    if score:
                        candidates.append((score, doc["id"], doc["title"]))
                candidates.sort(key=lambda row: (-row[0], row[1]))
                result = {
                    "hits": [
                        {"id": id_, "title": title, "score": score}
                        for score, id_, title in candidates[:4]
                    ]
                }
            elif name == "read_document":
                doc = next(
                    (d for d in documents() if d["id"] == arguments["doc_id"]), None
                )
                if not doc:
                    raise ValueError("Unknown document ID")
                if doc["version"] != self.case["app_version"]:
                    raise ValueError("Document version does not match application")
                self.observed[doc["id"]] = doc
                result = doc
            elif name == "diagnose_csv":
                fixture = arguments["fixture_id"]
                if fixture != self.case["fixture_id"] or not re.fullmatch(
                    r"[a-z0-9-]+", fixture
                ):
                    raise PermissionError("Fixture is outside this task's scope")
                fault = self.case.get("fault")
                if fault == "permission":
                    raise PermissionError("Diagnostic access denied")
                if fault == "timeout":
                    raise TimeoutError("Injected diagnostic timeout")
                path = (self.fixture_root / (fixture + ".csv")).resolve()
                if path.parent != self.fixture_root:
                    raise PermissionError("Path is outside fixture root")
                if path.stat().st_size > 100000:
                    raise ValueError("Fixture exceeds diagnostic size limit")
                result = self._worker(
                    {"task": "diagnose", "text": path.read_text(encoding="utf-8")}
                )
            else:
                result = self._worker(
                    {"task": "calculate", "expression": arguments["expression"]}
                )
            response = {"ok": True, "result": result}
        except (
            ValueError,
            PermissionError,
            TimeoutError,
            OSError,
            StopIteration,
            subprocess.TimeoutExpired,
        ) as exc:
            response = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.events.append(
            {
                "tool": name,
                "arguments": arguments,
                "response": response,
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
            }
        )
        return response

    def _worker(self, payload):
        worker = Path(__file__).with_name("worker.py")
        completed = subprocess.run(
            [sys.executable, "-I", str(worker)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=self.timeout,
            check=True,
        )
        result = json.loads(completed.stdout)
        if not result["ok"]:
            raise ValueError(result["error"])
        return result["result"]
