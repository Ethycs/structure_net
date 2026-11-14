"""Development evidence for 34 independent Qwen layer replacements, not an assembly."""
from dataclasses import asdict, dataclass
import math

from .jspace_single_layer_record import FEATURE_KEYS, require, scores

STUDY_ID = "JSPACE-QWEN-PARALLEL-SINGLE-LAYER-RECOVERY-02"
CLASSIFICATION = "isolated_recovery_with_two_quality_misses"
SCOPE = ("34 independent replacements of layers2..35, each surrounded by the original native model, "
         "on the SAME64 development cases. The count of passing layers is not model accuracy. "
         "No assembled-model accuracy, confirmation, generation speed, or deployment claim.")
CONFIG = dict(epochs=12, microbatch=4, accumulation=4, learning_rate=1.2e-4,
              gradient_clip=1., seed=27864, floor_fraction=1e-4)


@dataclass(frozen=True)
class JSpaceParallelSingleLayerRecord:
    study_id: str
    classification: str
    native: dict
    layers: list
    summary: dict
    storage: dict
    confirmed: bool = False
    assembly_evaluated: bool = False
    assembly_status: str = "pending_separate_experiment"
    independent_confirmations: int = 0
    scope: str = SCOPE

    @classmethod
    def from_evidence(cls, value):
        result = value["coordinator_result"]
        require(value["status"] == result["status"] == "completed" and value["confirmed"] is False
                and result["confirmed"] is False and result["assembly_evaluated"] is False,
                "Completed isolated evidence without confirmation or assembly required")
        require(result["verified_layers"] == list(range(2,36)) and not result["failed_layers"]
                and not result["unattempted_layers"], "Every independent trial must complete and verify")
        native = scores(value["native_records"])
        require(native["correct"] == 50, "Matched native50/64 control required")
        require([row["plan"]["layer"] for row in value["trials"]] == list(range(2,36)), "Exactly layers2..35 required")
        layers = []
        for trial in value["trials"]:
            plan, train, fresh = trial["plan"], trial["training"], trial["verification"]
            layer = plan["layer"]
            require(plan["trained_layers"] == [layer] and plan["native_prefix"] == list(range(layer))
                    and plan["native_suffix"] == list(range(layer+1,36)) and plan["config"] == CONFIG
                    and plan["rank"] == 1024 and plan["head_dim"] == 128
                    and plan["intermediate_size"] == 11008 and plan["confirmation_used"] is False,
                    "Only one fixed-geometry full-head half-rank layer may change")
            require(train["status"] == "completed" and train["systems_only"] is False
                    and train["strict_reload_exact"] is True and train["updates"] == 384
                    and train["trained_layer"] == layer, "All12epochs/384updates and strict reload required")
            history = train["history"]
            require([r["epoch"] for r in history] == list(range(13)), "Complete ordered epoch history required")
            best = max(history, key=lambda r:(r["metrics"]["correct"],-r["metrics"]["relative_mse"]))
            require(best["epoch"] == train["selected_epoch"] and best["metrics"] == train["selected"]
                    and best["checkpoint_sha256"] == train["selected_checkpoint_sha256"]
                    and train["initializer_selected"] == (best["epoch"] == 0), "Prospective selection changed")
            require(fresh["status"] == "completed" and fresh["passed"] is True and fresh["strict_state_load"] is True
                    and fresh["fingerprint"] == fresh["campaign_fingerprint"] == plan["fingerprint"]
                    and fresh["checkpoint_sha256"] == train["selected_checkpoint_sha256"]
                    and fresh["selected_epoch"] == best["epoch"] and fresh["confirmed"] is False,
                    "Exact selected checkpoint needs completed independent verification")
            comparison = fresh["comparison"]
            require(all(comparison[k] is True for k in ("passed","feature_pass","logits_pass","predictions_exact")),
                    "Fresh feature/logit/prediction parity required")
            control = trial["pilot"]["native_control"]
            require(trial["pilot"]["systems_only"] is True and trial["pilot"]["updates"] == 1
                    and all(control[k] is True for k in ("passed","predictions_exact","logits_pass"))
                    and control["correct"] == 50 and control["count"] == 64, "Native substitution pilot required")
            initial = scores(trial["initial_records"], value["native_records"])
            selected = scores(trial["selected_records"], value["native_records"])
            selected["paired_to_initial"] = scores(trial["selected_records"],trial["initial_records"])["paired_to_native"]
            reloaded = scores(fresh["records"],trial["selected_records"])
            require(reloaded["paired_to_native"]["changed_predictions"] == 0
                    and selected["correct"] == train["selected"]["correct"]
                    and initial["correct"] == history[0]["metrics"]["correct"], "Predictions differ from recorded metrics")
            require(train["native_correct"] == 50 and train["minimum_correct"] == 47
                    and train["quality_pass"] == (selected["correct"] >= 47)
                    and fresh["training_quality_pass"] == train["quality_pass"], "Frozen isolated gate changed")
            norm = train["normalization"]
            require(norm["train_case_count"] == 512 and norm["floor_fraction"] == 1e-4
                    and norm["floor"] > 0 and math.isclose(norm["floor"],norm["median_teacher_update_energy"]*1e-4,rel_tol=1e-12),
                    "Training-only fixed teacher-update normalization required")
            features = {key:train["selected"][key] for key in FEATURE_KEYS}
            initial_features = {key:history[0]["metrics"][key] for key in FEATURE_KEYS}
            require(all(math.isfinite(x) for x in [*features.values(),*initial_features.values()]), "Nonfinite feature diagnostics")
            layers.append(dict(layer=layer,initial=initial,selected=selected,selected_epoch=best["epoch"],
                initializer_selected=train["initializer_selected"],quality_pass=train["quality_pass"],updates=384,
                selected_checkpoint_sha256=train["selected_checkpoint_sha256"],normalization=norm,
                initial_features=initial_features,selected_features=features,
                verification_logit_max_abs_delta=comparison["logit_max_abs_delta"]))
        misses = [r["layer"] for r in layers if not r["quality_pass"]]
        require(misses == result["isolated_quality_misses"] == [12,15], "Preserve both observed quality misses")
        storage = value["storage"]
        require(storage["same_fp32_tensor_signature"] is True and storage["tied_embedding_readout_counted_once"] is True
                and len(storage["per_layer"]) == 34 and storage["projection_only"] is True,
                "Actual equal-format inventories and explicitly projected assembly accounting required")
        per = storage["packed_block_tensor_bytes"]
        require(all(r["packed_tensor_bytes"] == per for r in storage["per_layer"])
                and storage["independent_packed_state_bank_bytes"] == 34*per
                and storage["all36_matched_fp32_projection"]["bytes"] == storage["native_matched_fp32_bytes"]
                    -36*storage["native_block_fp32_bytes"]+36*per, "Storage accounting differs from actual shapes")
        summary = dict(isolated_trials=34,isolated_layers_passing_gate=32,isolated_quality_misses=misses,
            initializer_selected_layers=[r["layer"] for r in layers if r["initializer_selected"]],
            layers_improved_correct_count=sum(r["selected"]["correct"] > r["initial"]["correct"] for r in layers),
            layers_unchanged_correct_count=sum(r["selected"]["correct"] == r["initial"]["correct"] for r in layers),
            initial_layers_passing_gate=sum(r["initial"]["correct"] >= 47 for r in layers),
            optimizer_updates=34*384,unique_development_cases=64,training_cases_per_layer=512,
            primary_metric_name="isolated_layers_passing_gate",primary_metric_unit="count")
        return cls(STUDY_ID,CLASSIFICATION,native,layers,summary,storage)

    @property
    def primary_metric(self):
        return self.summary["isolated_layers_passing_gate"]

    def to_dict(self):
        return asdict(self)
