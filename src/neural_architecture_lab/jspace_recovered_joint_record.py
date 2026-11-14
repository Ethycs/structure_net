"""Typed development-only evidence for joint recovery of the complete assembly."""
from dataclasses import asdict, dataclass
import math

from .jspace_single_layer_record import require, scores

STUDY_ID = "JSPACE-QWEN-RECOVERED-JOINT-DISTILLATION-01"
SCOPE = ("Whole36-block selected compact assembly, jointly recovered on512 training cases and selected on the original64 "
         "development cases. One seed; no untouched confirmation, packing-parity, inference-speed or deployment claim.")
METRICS = ("loss","full_kl","choice_kl","choice_ce","weighted_full_kl","weighted_choice_kl","weighted_choice_ce",
           "teacher_accuracy","teacher_agreement","choice_margin","valid_label_probability","answer_full_vocab_ce")
CONFIG = dict(epochs=3,microbatch=1,accumulation=16,learning_rate=1e-5,gradient_clip=1.,seed=27865,
    optimizer_chunk_bytes=32*1024**2,max_cpu_state_bytes=24*1024**3,
    objective=dict(temperature=2.,full_kl_weight=.2,choice_kl_weight=.5,choice_ce_weight=.3))


def classify(correct):
    require(type(correct) is int and 0 <= correct <= 64,"Valid64-case correct count required")
    if correct <= 27:return "no_recovery_over_warm_start"
    if correct < 47:return "partial_recovery_below_gate"
    return "development_gate_passed"


@dataclass(frozen=True)
class JSpaceRecoveredJointRecord:
    study_id: str
    classification: str
    native: dict
    initial: dict
    selected: dict
    epochs: list
    selected_epoch: int
    initializer_selected: bool
    updates: int
    quality_pass: bool
    storage: dict
    confirmed: bool = False
    independent_confirmations: int = 0
    packing_validated: bool = False
    speed_measured: bool = False
    scope: str = SCOPE

    @classmethod
    def from_evidence(cls,value):
        plan,train,fresh,pilot = (value[k] for k in ("plan","training","verification","pilot"))
        require(value["status"] == train["status"] == fresh["status"] == "completed"
                and value["confirmed"] is False and train["confirmed"] is False and fresh["confirmed"] is False,
                "Completed exploratory training and independent verification required")
        require(plan["config"] == CONFIG and plan["initial_correct"] == 27 and plan["native_correct"] == 50
                and plan["minimum_correct"] == 47 and plan["confirmed"] is False
                and plan["partition"] == {"3060":list(range(24)),"2060_SUPER":list(range(24,36))},
                "Frozen objective,96-update schedule and complete model partition required")
        require(pilot["systems_only"] is True and pilot["updates"] == 1 and pilot["initial_control"]["passed"] is True
                and pilot["initial_control"]["correct"] == 27 and pilot["fixed_state_unchanged"] is True
                and pilot["parameter_changes"]["changed_blocks"] == list(range(36))
                and not pilot["parameter_changes"]["unchanged_blocks"],"All-block gradient and unchanged-geometry pilot required")
        teacher = value["teacher"]
        require(teacher["status"] == "completed" and teacher["passed"] is True and teacher["dtype"] == "float32"
                and teacher["shape"] == [576,2,151936] and teacher["slots"] == [1,3]
                and teacher["control"]["passed"] is True and teacher["control"]["predictions_exact"] is True
                and teacher["control"]["logits_pass"] is True,"Matched original FP32 midpoint/answer teacher required")
        native = scores(teacher["records"])
        require(native["correct"] == teacher["correct"] == 50 and teacher["count"] == 64,"Original native50/64 reference required")
        require(train["systems_only"] is False and train["updates"] == 96 and train["strict_reload_exact"] is True
                and [r["epoch"] for r in train["history"]] == list(range(4)),"Completed three scientific epochs required")
        best = max(train["history"],key=lambda r:(r["metrics"]["correct"],-r["metrics"]["choice_ce"]))
        require(train["selected_epoch"] == best["epoch"] and train["selected"] == best["metrics"]
                and train["selected_manifest_sha256"] == best["checkpoint_manifest_sha256"]
                and train["initializer_selected"] == (best["epoch"] == 0),"Accuracy/conditional-CE checkpoint selection changed")
        require(fresh["fingerprint"] == plan["fingerprint"] and fresh["selected_epoch"] == best["epoch"]
                and fresh["selected_manifest_sha256"] == train["selected_manifest_sha256"]
                and all(fresh[k] is True for k in ("passed","strict_state_load","predictions_exact","logits_pass","metrics_pass","fixed_state_unchanged")),
                "Selected state must pass independent fresh-process verification")
        require([r["epoch"] for r in value["epochs"]] == list(range(4)),"All epoch prediction records required")
        epochs = []
        for history,report in zip(train["history"],value["epochs"]):
            metrics = report["metrics"];result = scores(report["records"],teacher["records"])
            require(metrics == history["metrics"] and metrics["correct"] == result["correct"] and metrics["count"] == 64
                    and math.isclose(metrics["accuracy"],result["accuracy"],abs_tol=1e-12),"Task records disagree with metrics")
            require(all(math.isfinite(metrics[k]) for k in METRICS)
                    and math.isclose(metrics["teacher_accuracy"],50/64,abs_tol=1e-12),"Finite metrics and matched teacher decisions required")
            agreement = sum(a["prediction"] == b["prediction"] for a,b in zip(report["records"],teacher["records"]))/64
            require(math.isclose(metrics["teacher_agreement"],agreement,abs_tol=1e-12),"Teacher agreement changed")
            require(all(a["teacher_prediction"] == b["prediction"] for a,b in zip(report["records"],teacher["records"])),
                    "Per-case teacher predictions do not match the frozen FP32 cache")
            for key,weight,source in (("weighted_full_kl",.8,"full_kl"),("weighted_choice_kl",2.,"choice_kl"),("weighted_choice_ce",.3,"choice_ce")):
                require(math.isclose(metrics[key],weight*metrics[source],abs_tol=2e-6,rel_tol=2e-6),"Objective weighting changed")
            require(math.isclose(metrics["loss"],sum(metrics[k] for k in ("weighted_full_kl","weighted_choice_kl","weighted_choice_ce")),
                    abs_tol=2e-6,rel_tol=2e-6),"Loss terms do not sum")
            epochs.append(dict(epoch=report["epoch"],**result,metrics={k:metrics[k] for k in METRICS}))
        initial,selected = epochs[0],epochs[best["epoch"]]
        require(initial["correct"] == 27 and train["initial"] == train["history"][0]["metrics"],"Warm start must reproduce27/64")
        selected = dict(selected,paired_to_initial=scores(value["epochs"][best["epoch"]]["records"],value["epochs"][0]["records"])["paired_to_native"])
        reload = scores(fresh["records"],value["epochs"][best["epoch"]]["records"])
        require(reload["paired_to_native"]["changed_predictions"] == 0,"Fresh predictions differ")
        require(train["minimum_correct"] == 47 and train["native_correct"] == 50
                and train["quality_pass"] == (selected["correct"] >= 47)
                and fresh["training_quality_pass"] == train["quality_pass"],"Frozen final-model gate changed")
        require(value["storage"]["all_selected_fp32"] is True and value["storage"]["selected_blocks"] == 36
                and value["storage"]["tied_embedding_readout_counted_once"] is True,"Actual full selected FP32 inventory required")
        return cls(STUDY_ID,classify(selected["correct"]),native,initial,selected,epochs,best["epoch"],
            train["initializer_selected"],96,train["quality_pass"],value["storage"])

    @property
    def primary_metric(self):return self.selected["accuracy"]

    def to_dict(self):return asdict(self)
