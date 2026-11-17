"""Local, synthetic inputs. Evaluation answers are deliberately not imported by models."""

import json
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[2]
DATA = LAB_ROOT / "data"


# Issue #32: Loader contracts identify the local source and missing-case behavior.
def cases(path=None):
    """Decode a UTF-8 JSON file, using development.json when no path is supplied."""
    return json.loads(
        Path(path or DATA / "development.json").read_text(encoding="utf-8")
    )


def documents():
    """Return decoded local document JSON files in filename order."""
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((DATA / "documents").glob("*.json"))
    ]


def case_by_id(case_id):
    """Return the first matching development case or raise StopIteration."""
    return next(c for c in cases() if c["id"] == case_id)
