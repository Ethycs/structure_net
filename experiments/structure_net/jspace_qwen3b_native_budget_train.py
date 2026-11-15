"""Continue the frozen 3B recovery recipe under the user's native accuracy budget.

Only acceptance changes. Immediate-parent regression remains explicit evidence;
it is not an additional scientific stop rule. Original v3 results stay intact.
"""
from . import jspace_qwen3b_progressive_train as prior

CONFIG = prior.CONFIG
POLICY = "original_native_only"


def gates(correct, *, native_correct, parent_correct, count, maximum_drop=.05):
    """Apply the fixed five-percentage-point budget to original native accuracy.

    Reuse all frozen integer/range/budget validation and both diagnostic
    comparisons. Never rewrite a failed immediate-parent comparison as passed.
    """
    result = prior.gates(correct, native_correct=native_correct,
                         parent_correct=parent_correct, count=count,
                         maximum_drop=maximum_drop)
    return {**result, "policy": POLICY, "parent_gate_required": False,
            "passed": result["native"]["passed"],
            "minimum_correct": result["native"]["minimum_correct"]}


class NativeBudgetTrainer(prior.ProgressiveTrainer):
    """Same initialization, objective, optimizer, selection and update budget.

    The inherited fit calls this quality method only after selecting and
    strictly reloading its best packed development checkpoint.
    """

    def quality(self, correct):
        return gates(correct, native_correct=self.native_correct,
                     parent_correct=self.parent_correct,
                     count=self.development_count)
