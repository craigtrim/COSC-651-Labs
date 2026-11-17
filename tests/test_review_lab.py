"""Recording and comparison boundaries; these tests do not implement reflect."""

from copy import deepcopy
import csv
import json

import pytest

from maintenance_course.review_browser import BrowserTextModel
from maintenance_course.review_lab import (
    ReviewFixture,
    ReviewSession,
    check_answer,
    development_tasks,
    evaluate,
    example_answer,
    export_comparison,
    prepare_task,
    resource_rows,
    run_comparison,
)


@pytest.fixture(scope="module")
def task():
    return prepare_task("missing-date")


def one_fixed_exchange(session, initial, max_rounds=1):
    """A fixed request sequence for testing session recording."""
    review = session.review(initial)
    revised = session.revise(initial, review)
    return session.finish(revised, "revision_limit")


def test_prepare_supplies_observations_before_generation_without_answer_keys(task):
    assert len(task["source_events"]) == 8
    assert len(task["policy_ids"]) == 7
    assert "MISSING_DATE" in task["prompt"]
    assert "expected_action" not in task["prompt"]
    assert "1. Classification" in task["prompt"]


def test_code_checks_detect_specific_defects_without_claiming_semantic_correctness(
    task,
):
    checks = check_answer(task, example_answer(flawed=True))
    assert checks[0]["passed"]
    assert (
        not checks[1]["passed"]
        and checks[1]["detail"]["whitespace_separated_words"] == 155
    )
    assert not checks[2]["passed"]
    misleading = example_answer().replace(
        "Fixture citation: dates-v2.", "Fill missing dates with today; dates-v2."
    )
    assert all(c["passed"] for c in check_answer(task, misleading))


def test_generation_is_shared_and_only_external_review_gets_code_feedback(
    task, tmp_path
):
    models = []

    def factory():
        model = ReviewFixture()
        models.append(model)
        return model

    records = run_comparison(
        one_fixed_exchange, task, factory, tmp_path / "group", max_rounds=1
    )
    assert [m.calls for m in models] == [1, 2, 2]
    assert len({r["initial_hash"] for r in records}) == 1
    assert len({r["task_hash"] for r in records}) == 1
    assert records[1]["rounds"][0]["checks"] is None
    assert records[2]["rounds"][0]["checks"][1]["passed"] is False
    self_input = json.loads(records[1]["requests"][0]["prompt"].split("\n\n", 1)[1])
    external_input = json.loads(records[2]["requests"][0]["prompt"].split("\n\n", 1)[1])
    assert "external_checks" not in self_input
    assert "external_checks" in external_input
    assert self_input["answer"] == external_input["answer"] == records[0]["answer"]
    assert (
        self_input["original_request"]
        == external_input["original_request"]
        == task["prompt"]
    )
    assert all(
        (tmp_path / "group" / r["variant"] / "run.json").exists() for r in records
    )
    assert all(r["input_tokens"] is None for r in resource_rows(records))


def test_initial_failure_retains_three_attempted_paths_without_regeneration(
    task, tmp_path
):
    class Broken(ReviewFixture):
        def complete_text(self, prompt):
            self.calls += 1
            raise RuntimeError("Access unavailable")

    models = []

    def factory():
        models.append(Broken())
        return models[-1]

    records = run_comparison(one_fixed_exchange, task, factory, tmp_path / "failed")
    assert len(models) == 1 and models[0].calls == 1
    assert len(records) == 3
    assert all(
        r["stop_reason"] == "initial_failed" and r["answer"] is None for r in records
    )
    assert records[0]["requests"][0]["status"] == "failed"
    assert records[1]["requests"] == records[2]["requests"] == []


def test_malformed_review_is_saved_as_failure_not_approval(task, tmp_path):
    class BadReview(ReviewFixture):
        def complete_text(self, prompt):
            if prompt.startswith("Review the answer"):
                self.calls += 1
                self.events.append(
                    {"response_text": "Not JSON", "usage_available": False}
                )
                return "Not JSON"
            return super().complete_text(prompt)

    records = run_comparison(
        one_fixed_exchange, task, BadReview, tmp_path / "bad-review", max_rounds=1
    )
    for record in records[1:]:
        assert record["stop_reason"] == "execution_error"
        assert record["rounds"][0]["raw_review"] == "Not JSON"
        assert record["answer"] == records[0]["answer"]
        assert not any(r["kind"] == "revision" for r in record["requests"])


@pytest.mark.parametrize("budget", [-1, 3, True, 1.5])
def test_invalid_budgets_cannot_start_requests(task, tmp_path, budget):
    def forbidden():
        raise AssertionError("No request should be made")

    with pytest.raises(ValueError):
        run_comparison(
            one_fixed_exchange, task, forbidden, tmp_path / "invalid", max_rounds=budget
        )


def test_recorded_stop_and_declared_budget_are_enforced(task, tmp_path):
    session = ReviewSession(
        task,
        ReviewFixture(),
        "external_feedback",
        tmp_path / "session",
        example_answer(True),
        max_rounds=1,
    )
    with pytest.raises(ValueError):
        session.finish(example_answer(True), "review_complete")
    review = session.review(example_answer(True))
    revision = session.revise(example_answer(True), review)
    with pytest.raises(ValueError):
        session.review(revision)
    with pytest.raises(ValueError):
        session.revise(revision, review)
    with pytest.raises(ValueError):
        session.finish("Unrecorded answer", "revision_limit")
    session.finish(revision, "revision_limit")


def test_existing_comparison_is_not_overwritten_or_rerun(task, tmp_path):
    folder = tmp_path / "existing"
    run_comparison(one_fixed_exchange, task, ReviewFixture, folder, max_rounds=1)
    before = (folder / "comparison.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_comparison(
            one_fixed_exchange,
            task,
            lambda: pytest.fail("Unexpected model request"),
            folder,
        )
    assert (folder / "comparison.json").read_bytes() == before


def test_changed_model_does_not_silently_enter_the_comparison(task, tmp_path):
    count = 0

    def factory():
        nonlocal count
        model = ReviewFixture()
        if count:
            model.model_id = "different-model"
        count += 1
        return model

    records = run_comparison(
        one_fixed_exchange, task, factory, tmp_path / "model-change", max_rounds=1
    )
    assert all(
        r["stop_reason"] == "execution_error" and not r["requests"] for r in records[1:]
    )


def test_paired_scores_keep_regressions_and_require_human_evidence(task, tmp_path):
    records = run_comparison(
        one_fixed_exchange, task, ReviewFixture, tmp_path / "ratings", max_rounds=1
    )
    for record in records:
        record["answer"] = example_answer()
        record["final_checks"] = check_answer(task, record["answer"])
    ratings = [
        {
            "run_id": r["run_id"],
            "supported_facts": r["variant"] != "self_review",
            "follows_rules": True,
            "completes_task": True,
            "evidence_note": "A test judgment with an explicit source note.",
        }
        for r in records
    ]
    result = evaluate(records, ratings)
    assert result["counts"]["self_review"]["regressed"] == 1
    assert result["counts"]["external_feedback"]["unchanged_success"] == 1
    broken = deepcopy(ratings)
    broken[0]["supported_facts"] = None
    with pytest.raises(ValueError):
        evaluate(records, broken)
    with pytest.raises(ValueError):
        evaluate(records + [records[0]], ratings)
    export_comparison(tmp_path / "export", records, ratings)
    with (tmp_path / "export" / "metrics.csv").open(newline="") as source:
        exported = list(csv.DictReader(source))
    assert len(exported) == 3
    assert {row["mode"] for row in exported} == {"fixture"}
    assert {row["transport"] for row in exported} == {"scripted_fixture"}
    with pytest.raises(FileExistsError):
        export_comparison(tmp_path / "export", records, ratings)


def test_browser_capture_is_labeled_and_retains_failures():
    good = BrowserTextModel(
        "Claude",
        None,
        {"memory": None},
        capture=lambda p: {
            "text": "actual pasted text\nsecond line",
            "error": False,
            "attested": True,
        },
    )
    assert good.complete_text("Exact request") == "actual pasted text\nsecond line"
    assert good.events[0]["automated_provider_calls"] == 0
    assert good.events[0]["metadata_provenance"] == "student_reported"
    assert not good.events[0]["usage_available"]
    bad = BrowserTextModel(
        "Claude",
        None,
        {},
        capture=lambda p: {"text": "Access error", "error": True, "attested": True},
    )
    with pytest.raises(RuntimeError):
        bad.complete_text("Exact request")
    assert bad.events[0]["browser_error_text"] == "Access error"


def test_resolved_model_name_does_not_conflict_with_unchanged_requested_alias(
    task, tmp_path
):
    class Resolving(ReviewFixture):
        requested_model = "stable-alias"

        def complete_text(self, prompt):
            answer = super().complete_text(prompt)
            self.model_id = "resolved-model-version"
            self.events[-1]["resolved_models"] = [self.model_id]
            return answer

    records = run_comparison(
        one_fixed_exchange, task, Resolving, tmp_path / "resolved", max_rounds=1
    )
    assert all(r["error"] is None for r in records)
    assert all(r["model"] == "stable-alias" for r in records)
    assert all(r["resolved_models"] == ["resolved-model-version"] for r in records)


def test_failed_code_check_cannot_be_overruled_by_model_approval(task, tmp_path):
    class Approving(ReviewFixture):
        def complete_text(self, prompt):
            self.calls += 1
            return json.dumps({"decision": "no_supported_defect", "feedback": []})

    session = ReviewSession(
        task,
        Approving(),
        "external_feedback",
        tmp_path / "blocking",
        example_answer(True),
    )
    finding = session.review(example_answer(True))
    assert finding["decision"] == "revise"
    assert (
        session.record["rounds"][0]["model_review"]["decision"] == "no_supported_defect"
    )
    assert session.record["rounds"][0]["checks"][1]["passed"] is False


def test_ten_sample_group_preserves_all_thirty_records_and_shared_costs(tmp_path):
    tasks = development_tasks()
    assert len(tasks) == len({t["sample"] for t in tasks}) == 10
    records = []
    for current in tasks:
        records.extend(
            run_comparison(
                one_fixed_exchange,
                current,
                ReviewFixture,
                tmp_path / current["sample"],
                max_rounds=1,
            )
        )
    ratings = [
        {
            "run_id": r["run_id"],
            "supported_facts": False,
            "follows_rules": False,
            "completes_task": False,
            "evidence_note": "Scripted recorder integration test, not model-quality evidence.",
        }
        for r in records
    ]
    assessed = evaluate(records, ratings)
    assert len(records) == 30
    assert all(
        assessed["counts"][v]["unchanged_failure"] == 10 for v in assessed["counts"]
    )
    resources = resource_rows(records)
    assert sum(r["request_count"] for r in resources) == 50
    assert all(r["elapsed_ms"] >= r["request_elapsed_ms"] for r in resources)
    assert all(
        r["shared_preparation_ms"] == 0
        for r in resources
        if r["variant"] != "no_review"
    )
