"""Typed one-campaign record; three local-distillation arms are not replications."""
from dataclasses import asdict, dataclass
import math


ARMS = ("endpoint", "euclidean", "jacobian")
CLASSIFICATION = "exploratory_local_distillation_inconclusive"


@dataclass(frozen=True)
class JSpaceQwenLayerKDRecord:
    study_id: str
    scientific_fingerprint: str
    native_correct: int
    correct_by_arm: dict
    selected_epoch_by_arm: dict
    graph_tokens_per_second_by_arm: dict
    tensor_bytes: int
    count: int = 512
    completed_epochs_per_arm: int = 12
    updates_per_arm: int = 1536
    trained_arms: int = 3
    independent_confirmations: int = 0
    confirmed: bool = False
    classification: str = CLASSIFICATION
    scope: str = "one_seed_single_final_block_auxiliary_distillation_no_staged_joint_recovery"

    @classmethod
    def from_aggregate(cls, aggregate):
        result, training, reload = aggregate["result"], aggregate["training"], aggregate["reload"]
        if result["study_id"] != "JSPACE-QWEN-LAYERKD-01" or result["status"] != "completed":
            raise ValueError("Wrong study or incomplete result")
        if result["confirmed"] is not False or aggregate["classification"] != CLASSIFICATION:
            raise ValueError("Exploratory evidence cannot be confirmed or relabelled")
        if result["fresh_evaluation_count"] != 512 or set(training) != set(ARMS):
            raise ValueError("Wrong evaluation or arm population")
        if len({training[a]["initialized_checkpoint_sha256"] for a in ARMS}) != 1:
            raise ValueError("Initialization differs between arms")
        results = result["results"]
        counts, selected, rates, sizes = {}, {}, {}, set()
        for arm in ("native", *ARMS):
            accuracy = results[arm]["accuracy"]
            if accuracy["count"] != 512 or not 0 <= accuracy["correct"] <= 512:
                raise ValueError("Incomplete or invalid accuracy")
            if accuracy["overall"] != accuracy["correct"] / 512:
                raise ValueError("Accuracy count/rate mismatch")
            if arm == "native":
                continue
            t = training[arm]
            if t["completed_epochs"] != 12 or t["updates"] != 1536:
                raise ValueError("Recovery budgets differ or are incomplete")
            if not 0 <= t["selected_epoch"] <= 12 or reload[arm]["records_exact"] is not True:
                raise ValueError("Invalid checkpoint selection or missing strict reload")
            if reload[arm]["accuracy"] != accuracy:
                raise ValueError("Reload accuracy changed")
            counts[arm], selected[arm] = accuracy["correct"], t["selected_epoch"]
            rate = results[arm]["runtime"]["timings"]["cuda_graph"]["tokens_per_second"]
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("Invalid runtime")
            rates[arm] = rate
            if results[arm]["tensor_bytes"] != t["tensor_bytes"] or t["tensor_bytes"] <= 0:
                raise ValueError("Training/evaluation footprint mismatch")
            sizes.add(t["tensor_bytes"])
        if len(sizes) != 1:
            raise ValueError("Deployed architectures differ between arms")
        return cls(study_id=result["study_id"], scientific_fingerprint=result["fingerprint"],
                   native_correct=results["native"]["accuracy"]["correct"], correct_by_arm=counts,
                   selected_epoch_by_arm=selected, graph_tokens_per_second_by_arm=rates,
                   tensor_bytes=sizes.pop())

    def to_dict(self):
        return asdict(self)
