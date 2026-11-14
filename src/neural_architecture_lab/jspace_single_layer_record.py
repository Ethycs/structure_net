"""Typed development-only evidence for one recovered Qwen residual block."""
from collections import Counter
from dataclasses import asdict, dataclass
import math

STUDY_ID = "JSPACE-QWEN-SINGLE-LAYER-RECOVERY-01"
SCOPE = ("One trained seed and the original64 development cases,16 per family; only block0 is compressed, "
         "blocks1..35, tied embedding/readout and final norm remain native. No confirmation, "
         "whole-stack compression, deployment speed or free-generation claim.")
FEATURE_KEYS = ("relative_mse", "mse", "first_relative_mse", "content_relative_mse", "answer_relative_mse",
                "irreducible_outside_relative_mse", "retained_relative_mse")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scores(records, native=None):
    require(len(records) == 64 and len({r["identity"] for r in records}) == 64, "Exactly64 unique development cases required")
    require(Counter(r["family"] for r in records) == {f:16 for f in ("arc", "boolq", "hellaswag", "winogrande")},
            "Four balanced development families required")
    correct = 0; families = {}; gains = losses = changed = 0
    for index, row in enumerate(records):
        logits = row["logits"]
        require(2 <= len(logits) <= 4 and all(math.isfinite(x) for x in logits)
                and type(row["answer"]) is int and 0 <= row["answer"] < len(logits)
                and type(row["prediction"]) is int and row["prediction"] == max(range(len(logits)), key=logits.__getitem__),
                "Finite choice logits, answer labels and exact argmax required")
        right = row["prediction"] == row["answer"]; correct += right
        family = families.setdefault(row["family"], dict(correct=0, count=0))
        family["correct"] += right; family["count"] += 1
        if native is not None:
            before = native[index]
            require(all(row[k] == before[k] for k in ("identity", "family", "answer")), "Unpaired identities or labels")
            original_right = before["prediction"] == before["answer"]
            gains += not original_right and right; losses += original_right and not right
            changed += before["prediction"] != row["prediction"]
    result = dict(correct=correct, count=64, accuracy=correct/64, by_family=families)
    if native is not None:
        result["paired_to_native"] = dict(gains=gains, losses=losses, changed_predictions=changed, net_correct=gains-losses)
    return result


@dataclass(frozen=True)
class JSpaceSingleLayerRecord:
    study_id: str
    classification: str
    initial: dict
    selected: dict
    native: dict
    projection_oracle: dict
    selected_epoch: int
    updates: int
    quality_pass: bool
    normalized_features: dict
    initial_normalized_features: dict
    normalization: dict
    storage: dict
    resource_failure_lineage: dict
    confirmed: bool = False
    independent_confirmations: int = 0
    scope: str = SCOPE

    @classmethod
    def from_evidence(cls, value):
        plan, train, fresh = value["plan"], value["training"], value["verification"]
        require(value["status"] == "completed" and value["confirmed"] is False, "Completed exploratory evidence required")
        require(plan["trained_layers"] == [0] and plan["native_suffix"] == list(range(1,36))
                and plan["rank"] == 1024 and plan["head_dim"] == 128 and plan["confirmation_used"] is False,
                "Only the declared half-rank/full-head block0 may change")
        require(plan["config"]["epochs"] == 12 and plan["config"]["microbatch"] == 4
                and plan["config"]["accumulation"] == 4 and train["updates"] == 384,
                "All12epochs/384updates are required")
        history = train["history"]
        require([r["epoch"] for r in history] == list(range(13)), "Complete ordered epoch history required")
        best = max(history, key=lambda r:(r["metrics"]["correct"], -r["metrics"]["relative_mse"]))
        require(train["status"] == "completed" and train["systems_only"] is False and train["strict_reload_exact"] is True
                and best["epoch"] == train["selected_epoch"] and best["metrics"] == train["selected"],
                "Completed scientific training and prospective checkpoint selection required")
        require(fresh["status"] == "completed" and fresh["passed"] is True and fresh["strict_state_load"] is True
                and fresh["selected_epoch"] == train["selected_epoch"]
                and fresh["campaign_fingerprint"] == plan["fingerprint"]
                and fresh["runtime_policy_fingerprint"] == value["runtime_policy"]["fingerprint"],
                "The exact selected checkpoint must pass independent verification")
        require(value["runtime_train"]["status"] == "completed" and value["runtime_train"]["stage"] == "train",
                "The allocator wrapper must complete too")
        native = scores(value["native_records"])
        initial = scores(value["initial_records"], value["native_records"])
        selected = scores(value["selected_records"], value["native_records"])
        selected["paired_to_initial"] = scores(value["selected_records"], value["initial_records"])["paired_to_native"]
        require(native["correct"] == train["native_correct"] == 50 and initial["correct"] == 46,
                "Matched50/64 native and46/64 initial controls required")
        require(selected["correct"] == train["selected"]["correct"] and train["selected"]["count"] == 64,
                "Selected metrics disagree with predictions")
        reload_scores = scores(fresh["records"], value["selected_records"])
        require(reload_scores["paired_to_native"]["changed_predictions"] == 0, "Fresh reload changed selected predictions")
        require(train["minimum_correct"] == 47 and train["quality_pass"] == (selected["correct"] >= 47)
                and fresh["training_quality_pass"] == train["quality_pass"], "Frozen47/64 gate changed")
        projection = scores(value["oracle"]["projected_update"]["records"], value["native_records"])
        cached = scores(value["oracle"]["native_boundary"]["records"], value["native_records"])
        require(projection["correct"] == 48 and cached["correct"] == 50, "Matched cached-native50 and projection48 controls required")
        projection.update(cached_native=cached,
            selected_prediction_agreement=sum(a["prediction"] == b["prediction"] for a,b in
                zip(value["selected_records"],value["oracle"]["projected_update"]["records"])),
            scope="Closest Euclidean teacher-update projection; not a task-accuracy upper bound")
        features = {key:train["selected"][key] for key in FEATURE_KEYS}
        initial_features = {key:history[0]["metrics"][key] for key in FEATURE_KEYS}
        require(all(math.isfinite(x) for x in [*features.values(),*initial_features.values()]), "Finite feature diagnostics required")
        norm = train["normalization"]
        require(norm["train_case_count"] == 512 and norm["floor"] > 0
                and math.isclose(norm["floor"], norm["median_teacher_update_energy"]*norm["floor_fraction"], rel_tol=1e-12),
                "A fixed training-only teacher-update floor is required")
        storage = value["storage"]
        require(storage["tied_embedding_readout_counted_once"] is True and storage["changed_layers"] == [0]
                and storage["selected_unpacked"]["tensor_bytes"] > storage["packed_block"]["tensor_bytes"] > 0
                and storage["whole_matched_fp32"]["selected_bytes"] == storage["native_matched_fp32_bytes"]
                    -storage["native_block_fp32_bytes"]+storage["packed_block"]["tensor_bytes"],
                "Honest full-model tensor accounting including geometry required")
        failure = value["v1_failure"]
        require(failure["status"] == "failed" and failure["error_type"] == "MemoryError"
                and failure["peak_reserved_bytes"] > sum(failure["allocation"].values())*1024**2,
                "Preserved v1 reservation-limit failure required")
        passed = selected["correct"] >= 47
        return cls(STUDY_ID, "single_layer_development_gate_passed" if passed else "single_layer_development_gate_failed",
            initial, selected, native, projection, train["selected_epoch"], train["updates"], passed,
            features, initial_features, norm, storage, failure)

    @property
    def primary_metric(self):
        return self.selected["accuracy"]

    def to_dict(self):
        return asdict(self)
