# Issue #32: Baseline workflows separate direct proposals from gathered evidence.
"""Provided comparison workflows. Students implement their own weekly functions."""

from .experiments import payload, gather


def direct(model, registry, case):
    """Request one proposal using only the evidence already in the registry."""
    return model.complete("proposal", payload(case, registry))


def fixed(model, registry, case):
    """Gather the diagnostic and current policies before requesting one proposal."""
    gather(registry)
    return model.complete("proposal", payload(case, registry))
