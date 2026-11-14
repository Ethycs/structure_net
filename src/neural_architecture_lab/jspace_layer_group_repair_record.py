"""Typed actual all-compact task recovery after training a selected triple."""
from dataclasses import asdict, dataclass
import math

from .jspace_single_layer_record import require,scores
from .jspace_recovered_joint_record import CONFIG as ENDPOINT_CONFIG,METRICS as ENDPOINT_METRICS,classify

STUDY_ID = "JSPACE-QWEN-LAYER-GROUP-REPAIR-01"
SCOPE = ("Six-epoch repair of exactly three search-selected blocks in the complete36-block compact Qwen model. "
         "Original norm/readout, other33 blocks and all geometry fixed. Selected on reused64-case development data; "
         "no independent confirmation, packing parity, speed, or attribution solely to group selection.")
CONFIG = dict(ENDPOINT_CONFIG,epochs=6,seed=27866,feature_weight=1.)
METRICS = (*ENDPOINT_METRICS,"endpoint_loss","feature_loss","weighted_feature_loss")


def close(a,b): return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,abs_tol=2e-6,rel_tol=2e-6)


def epoch_score(report,native,layers):
    rows,metrics = report["records"],report["metrics"]; actual = scores(rows,native)
    require(metrics["correct"] == actual["correct"] and metrics["count"] == 64 and metrics["accuracy"] == actual["accuracy"],
        "Actual complete-model decisions must determine epoch accuracy")
    require(all(math.isfinite(metrics[k]) for k in METRICS),"Finite endpoint/feature metrics required")
    keys = {str(i) for i in layers}; layer_mean = {key:0. for key in keys}; agreement = 0
    totals = {key:0. for key in ("feature_loss","choice_ce","choice_margin","choice_kl","full_kl")}
    for row,teacher in zip(rows,native):
        require(row["teacher_prediction"] == teacher["prediction"] and len(row["logits"]) == len(teacher["logits"]),"Original frozen teacher choices changed")
        logits = row["logits"]; peak = max(logits); answer = row["answer"]
        ce = math.log(sum(math.exp(x-peak) for x in logits))+peak-logits[answer]
        margin = logits[answer]-max(x for i,x in enumerate(logits) if i != answer)
        require(math.isclose(row["choice_ce"],ce,abs_tol=1e-4,rel_tol=1e-5)
            and math.isclose(row["choice_margin"],margin,abs_tol=1e-5,rel_tol=1e-5),"Conditional decisions differ from saved logits")
        features = row["feature_loss_by_layer"]
        require(set(features) == keys and all(math.isfinite(x) and x >= 0 for x in features.values())
            and close(row["feature_loss"],sum(features.values())/3),"Per-case feature loss must average exactly the chosen three layers")
        for key in keys: layer_mean[key] += features[key]/64
        for key in totals:
            require(math.isfinite(row[key]),"Finite per-case metrics required"); totals[key] += row[key]/64
        agreement += row["prediction"] == teacher["prediction"]
    require(all(close(metrics[key],value) for key,value in totals.items()) and close(metrics["teacher_accuracy"],50/64)
        and close(metrics["teacher_agreement"],agreement/64),"Aggregate objective/teacher metrics disagree with actual records")
    for key,weight,source in (("weighted_full_kl",.8,"full_kl"),("weighted_choice_kl",2.,"choice_kl"),("weighted_choice_ce",.3,"choice_ce")):
        require(close(metrics[key],weight*metrics[source]),"Endpoint weighting changed")
    require(close(metrics["endpoint_loss"],sum(metrics[k] for k in ("weighted_full_kl","weighted_choice_kl","weighted_choice_ce")))
        and close(metrics["weighted_feature_loss"],metrics["feature_loss"])
        and close(metrics["loss"],metrics["endpoint_loss"]+metrics["weighted_feature_loss"]),"Feature and endpoint objectives do not sum")
    return dict(epoch=report["epoch"],**actual,metrics={k:metrics[k] for k in METRICS},feature_loss_by_layer=layer_mean)


@dataclass(frozen=True)
class JSpaceLayerGroupRepairRecord:
    study_id: str
    classification: str
    selected_layers: list
    native: dict
    initial: dict
    selected: dict
    epochs: list
    selected_epoch: int
    initializer_selected: bool
    quality_pass: bool
    search_context: dict
    storage: dict
    updates: int = 192
    confirmed: bool = False
    independent_confirmations: int = 0
    packing_validated: bool = False
    speed_measured: bool = False
    scope: str = SCOPE

    @classmethod
    def from_evidence(cls,value):
        plan,train,fresh,pilot,capture = (value[k] for k in ("plan","training","verification","pilot","capture"))
        layers = plan["selected_layers"]
        require(value["status"] == train["status"] == fresh["status"] == capture["status"] == "completed"
            and value["confirmed"] is False and all(v["confirmed"] is False for v in (plan,train,fresh)),"Completed unconfirmed repair and fresh verification required")
        require(plan["schema"] == "jspace.layer-group-repair.v1" and plan["config"] == CONFIG
            and type(layers) is list and len(layers) == 3 and layers == sorted(set(layers))
            and all(type(i) is int and 0 <= i < 36 for i in layers)
            and plan["initial_correct"] == 27 and plan["native_correct"] == 50 and plan["minimum_correct"] == 47
            and plan["partition"] == {"3060":list(range(24)),"2060_SUPER":list(range(24,36))},"Frozen triple, six-epoch objective and original architecture required")
        require(layers == plan["selection"]["selected_layers"] == value["selection_proof"]["selected_layers"]
            and plan["selection"] == value["selection_proof"],"Repair triple differs from authenticated frozen search selection")
        teacher = value["teacher"]; native = scores(teacher["records"])
        require(teacher["status"] == "completed" and teacher["passed"] is True and teacher["dtype"] == "float32"
            and teacher["shape"] == [576,2,151936] and teacher["slots"] == [1,3]
            and teacher["control"]["passed"] is True and native["correct"] == 50,"Original FP32 full-vocabulary teacher required")
        require(capture["passed"] is True and capture["fingerprint"] == plan["fingerprint"] and capture["dtype"] == "float32"
            and capture["selected_layers"] == layers and capture["shape"] == [plan["tokens"],2048]
            and capture["native_control"]["passed"] is True and capture["parity_atol"] == 1e-5 and capture["parity_rtol"] == 2e-4
            and set(capture["development_boundary_parity"]) == {str(i) for i in layers}
            and all(v["passed"] is True for v in capture["development_boundary_parity"].values()),"Matching original native next-boundary feature targets required")
        require(scores(capture["records"],teacher["records"])["paired_to_native"]["changed_predictions"] == 0,"Feature teacher decisions differ from endpoint teacher")
        require(pilot["systems_only"] is True and pilot["updates"] == 1 and pilot["initial_control"]["passed"] is True
            and pilot["initial_control"]["correct"] == 27 and pilot["changed_selected_layers"] == layers
            and pilot["frozen_state_unchanged"] is True and pilot["update"]["gradient_parameter_count"] == 21,
            "Group-only gradient/update pilot and unchanged native norm/other blocks required")
        require(train["systems_only"] is False and train["updates"] == 192 and train["strict_reload_exact"] is True
            and train["frozen_state_unchanged"] is True and train["selected_layers"] == layers
            and [r["epoch"] for r in train["history"]] == list(range(7)),"All six scientific epochs required")
        frozen = value["frozen_state"]
        require(frozen["passed"] is True and frozen["before"] == frozen["after"] == train["frozen_state"],"Other33 blocks, native norm, tiedE and geometry must remain fixed")
        winner = max(train["history"],key=lambda r:(r["metrics"]["correct"],-r["metrics"]["choice_ce"]))
        require(train["selected_epoch"] == winner["epoch"] and train["selected"] == winner["metrics"]
            and train["selected_manifest_sha256"] == winner["checkpoint_manifest_sha256"]
            and train["initializer_selected"] == (winner["epoch"] == 0),"Accuracy/conditional-CE selection changed")
        delta = value["delta_manifest"]
        require(delta["schema"] == plan["schema"] and delta["selected_layers"] == layers
            and delta["parent_assembly_fingerprint"] == plan["assembly_fingerprint"]
            and delta["search_fingerprint"] == plan["selection"]["search_fingerprint"]
            and set(delta["artifacts"]) == {f"block_{i:02d}.pt" for i in layers}
            and delta["metadata"] == dict(epoch=winner["epoch"],development=winner["metrics"]),"Selected delta must contain exactly three blocks and no norm update")
        require(fresh["fingerprint"] == plan["fingerprint"] and fresh["selected_layers"] == layers
            and fresh["selected_epoch"] == winner["epoch"] and fresh["selected_manifest_sha256"] == train["selected_manifest_sha256"]
            and all(fresh[k] is True for k in ("passed","strict_state_load","predictions_exact","logits_pass","metrics_pass","frozen_state_unchanged")),
            "Freshly reconstructed actual full model must verify")
        require([r["epoch"] for r in value["epochs"]] == list(range(7)),"All epoch decision/feature records required")
        epochs = []
        for history,report in zip(train["history"],value["epochs"]):
            require(report["metrics"] == history["metrics"],"Epoch metrics differ from authenticated history")
            epochs.append(epoch_score(report,teacher["records"],layers))
        initial,selected = epochs[0],epochs[winner["epoch"]]
        require(initial["correct"] == 27 and train["initial"] == train["history"][0]["metrics"],"Original all-compact27/64 warm start required")
        reference = value["epochs"][winner["epoch"]]["records"]
        epoch_score(dict(epoch=winner["epoch"],metrics=fresh["metrics"],records=fresh["records"]),teacher["records"],layers)
        require(scores(fresh["records"],reference)["paired_to_native"]["changed_predictions"] == 0
            and all(abs(x-y) <= 5e-4+1e-5*abs(y) for a,b in zip(fresh["records"],reference) for x,y in zip(a["logits"],b["logits"]))
            and all(math.isclose(a["feature_loss_by_layer"][str(i)],b["feature_loss_by_layer"][str(i)],abs_tol=1e-5,rel_tol=2e-4)
                for a,b in zip(fresh["records"],reference) for i in layers),"Fresh decision/feature records differ from selection")
        selected = dict(selected,paired_to_initial=scores(reference,value["epochs"][0]["records"])["paired_to_native"])
        passed = selected["correct"] >= 47
        require(train["minimum_correct"] == 47 and train["native_correct"] == 50 and train["quality_pass"] is passed
            and fresh["training_quality_pass"] is passed,"Actual all-compact repair must meet the cumulative task gate")
        context = value["search_context"]; restored = scores(context["records"],teacher["records"])
        require(context["selection"]["layers"] == layers and context["selection"]["group_size"] == 3
            and context["selection"]["restored_correct"] == restored["correct"]
            and context["selection"]["rescue_correct"] == restored["correct"]-27,"Search score must remain a separate native-restoration context")
        storage = value["storage"]
        require(storage["all_fp32"] is True and storage["trainable_layers"] == layers and storage["parameter_tensor_count"] == 21
            and storage["native_norm_frozen"] is True and storage["tied_embedding_readout_counted_once"] is True,"Actual three-block delta inventory required")
        return cls(STUDY_ID,classify(selected["correct"]),layers,native,initial,selected,epochs,winner["epoch"],
            train["initializer_selected"],passed,dict(selection=context["selection"],native_restored_score=restored),storage)

    @property
    def primary_metric(self): return self.selected["accuracy"]

    def to_dict(self): return asdict(self)
