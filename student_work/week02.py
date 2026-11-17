"""Week 2: control the review loop; supplied sessions record model exchanges."""


def reflect(session, initial_answer, max_rounds=2):
    """Return session.finish(answer, reason), retaining all supplied records.

    session.review(answer) returns decision and feedback. The decision is
    revise, no_supported_defect, or needs_information. session.revise(answer,
    review) returns a new answer. The session determines which path receives
    external code checks; keep the same control logic for both paths.
    """
    # TODO 2.1: validate an integer revision budget from zero to two.
    # TODO 2.2: review, route supported findings, and request permitted revisions.
    # TODO 2.3: stop on review_complete, needs_information, unchanged, or revision_limit.
    # Compare unchanged answers with .strip(), but retain the original text.
    raise NotImplementedError("TODO 2.1: reflect")
