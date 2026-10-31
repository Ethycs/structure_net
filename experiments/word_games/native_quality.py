"""Match synthetic capability IDs across native quality arms, never production."""
from contextlib import nullcontext
from unittest.mock import patch

from .common import read


def matched_setup(original, root, case):
    engine, context = original(case)
    saved = read(root / "quality/cases" / case["id"] / "context.json")
    if {k: v for k, v in saved.items() if k != "context_id"} != {
            k: v for k, v in context.items() if k != "context_id"}:
        raise ValueError("paired fixture authorization differs")
    old, new = context["context_id"], saved["context_id"]
    if new in engine.contexts and old != new:
        raise ValueError("paired synthetic context ID collision")
    engine.contexts[new] = engine.contexts.pop(old)
    engine._context_ids[new] = engine._context_ids.pop(old)
    context["context_id"] = new
    return engine, context


def run_quality(out, identity, client, paired_root=None, **campaign_options):
    from . import campaign
    original = campaign.setup
    manager = nullcontext()
    if paired_root is not None:
        prior = read(paired_root / "quality/campaign_results.json")
        if prior["completed"] != 12:
            raise ValueError("completed baseline fixture set required")
        manager = patch.object(campaign, "setup", lambda case: matched_setup(original, paired_root, case))
    with manager:
        return campaign.run(out, identity, client=client, **campaign_options)
