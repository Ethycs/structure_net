"""Development-only evidence for extending the accepted Qwen suffix by one block."""
from collections import Counter
from dataclasses import asdict, dataclass
import math

from .jspace_single_layer_record import require, scores

STUDY_ID = "JSPACE-QWEN-INCREMENTAL-SUFFIX-01"
LAYERS = list(range(29, 36))
ATOL, RTOL = .03125, 1 / 128
CONFIG = dict(epochs=12, microbatch=4, accumulation=4, learning_rate=1.2e-4,
              temperature=2., kl_weight=.7, ce_weight=.3, gradient_clip=1.,
              checkpoint_blocks=True, dtype="bfloat16", seed=20861,
              optimizer_chunk_bytes=32*1024**2, max_cpu_state_bytes=32*1024**3)
SCOPE = ("One BF16 seven-block suffix, grown from accepted blocks30..35 by adding block29, "
         "jointly trained on512 original cases and selected on64 reused development cases. "
         "The native prefix0..28 is supplied by authenticated cached boundary29. "
         "No untouched confirmation, raw-token prefix recomputation, speed or general-generation claim.")


def quality_gates(correct, native_correct, parent_correct, count=64):
    require(count == 64 and all(type(v) is int and 0 <= v <= count
            for v in (correct, native_correct, parent_correct)), "Valid matched64-case counts required")
    native_minimum = math.ceil(native_correct - .05 * count)
    parent_minimum = math.ceil(parent_correct - .05 * count)
    return dict(native_minimum_correct=native_minimum, parent_minimum_correct=parent_minimum,
                minimum_correct=max(native_minimum, parent_minimum),
                native_pass=correct >= native_minimum, parent_pass=correct >= parent_minimum,
                quality_pass=correct >= max(native_minimum, parent_minimum), maximum_accuracy_drop=.05)


def parity(actual, reference, *, atol=ATOL, rtol=RTOL):
    require(atol == ATOL and rtol == RTOL, "Frozen BF16 parity tolerances changed")
    scores(reference)
    result = scores(actual, reference)
    require(all(len(a["logits"]) == len(b["logits"]) for a, b in zip(actual, reference)),
            "Parity requires matching complete choice vectors")
    exact = result["paired_to_native"]["changed_predictions"] == 0
    delta = max(abs(x-y) for a, b in zip(actual, reference) for x, y in zip(a["logits"], b["logits"]))
    close = all(abs(x-y) <= atol + rtol * abs(y)
                for a, b in zip(actual, reference) for x, y in zip(a["logits"], b["logits"]))
    return dict(passed=exact and close, predictions_exact=exact, logits_pass=close, max_abs_logit_delta=delta)


def summarize(report, native, parent):
    rows, metrics = report["records"], report["metrics"]
    actual = scores(rows, native)
    require(metrics["correct"] == actual["correct"] and metrics["count"] == 64
            and metrics["accuracy"] == actual["accuracy"], "Saved decisions must determine task accuracy")
    require(all(math.isfinite(metrics[k]) and metrics[k] >= -1e-5 for k in ("loss", "kl", "ce"))
            and math.isclose(metrics["loss"], 2.8*metrics["kl"]+.3*metrics["ce"], abs_tol=2e-5, rel_tol=2e-5),
            "Original full-vocabulary .7*T² KL + .3 CE objective required")
    actual.update(metrics={k: metrics[k] for k in ("loss", "kl", "ce")},
                  paired_to_parent=scores(rows, parent)["paired_to_native"],
                  native_agreement=sum(a["prediction"] == b["prediction"] for a,b in zip(rows,native))/64,
                  parent_agreement=sum(a["prediction"] == b["prediction"] for a,b in zip(rows,parent))/64,
                  best_constant_label_correct=max(Counter(r["answer"] for r in rows).values()))
    return actual


@dataclass(frozen=True)
class JSpaceIncrementalSuffixRecord:
    study_id: str
    classification: str
    native: dict
    parent: dict
    initial: dict
    selected: dict
    epochs: list
    selected_epoch: int
    initializer_selected: bool
    gates: dict
    storage: dict
    updates: int = 384
    compressed_layers: tuple = tuple(LAYERS)
    confirmed: bool = False
    independent_confirmations: int = 0
    packing_validated: bool = True
    speed_measured: bool = False
    scope: str = SCOPE

    @classmethod
    def from_evidence(cls, value):
        plan, train, fresh = (value[k] for k in ("plan", "training", "verification"))
        require(value["status"] == train["status"] == fresh["status"] == "completed"
                and value["confirmed"] is False, "Completed exploratory training and fresh verification required")
        require(plan["schema"] == "jspace.incremental-suffix.v1" and plan["cut"] == 29
                and plan["parent_cut"] == 30 and plan["layers"] == LAYERS and plan["config"] == CONFIG,
                "Only the declared one-block extension and original twelve-epoch recipe are supported")
        controls = value["controls"]
        native, parent = scores(controls["native"]["records"]), scores(controls["parent"]["records"])
        require(native["correct"] == controls["native"]["metrics"]["correct"] == 48
                and parent["correct"] == controls["parent"]["metrics"]["correct"] == 51,
                "Matched BF16 native48 and reproduced packed parent51 controls required")
        # Every control and epoch must contain the same identities, order and labels.
        scores(controls["parent"]["records"], controls["native"]["records"])
        for key in ("native", "parent", "seam"):
            control = controls[key]
            require(control["passed"] is True and parity(control["records"], control["reference_records"])["passed"],
                    "A native, parent or native29-to-parent control failed")
        require(parity(controls["seam"]["records"], controls["parent"]["records"])["passed"],
                "Introducing cached prefix29 changed the accepted parent decisions")
        pilot = value["pilot"]
        require(pilot["systems_only"] is True and pilot["updates"] == 1
                and pilot["master_changed_layers"] == LAYERS and pilot["norm_master_changed"] is True
                and pilot["gradient_layers"] == LAYERS and pilot["norm_gradient"] is True
                and pilot["fixed_state_unchanged"] is True and pilot["parent_inheritance_exact"] is True,
                "Real seven-block/norm FP32-master update, exact parent inheritance and fixed geometry required")
        require(train["systems_only"] is False and train["updates"] == 384
                and train["strict_reload_exact"] is True and train["fixed_state_unchanged"] is True
                and [r["epoch"] for r in train["history"]] == list(range(13)),
                "All twelve scientific epochs with an eligible initializer are required")
        history = train["history"]
        best = max(history, key=lambda r:(r["metrics"]["correct"], -r["metrics"]["ce"]))
        require(train["selected_epoch"] == best["epoch"] and train["selected"] == best["metrics"]
                and train["initial"] == history[0]["metrics"]
                and train["selected_checkpoint_sha256"] == best["checkpoint_sha256"],
                "Selection must use actual accuracy then original full-vocabulary CE, including epoch0")
        require([r["epoch"] for r in value["epochs"]] == list(range(13)), "Missing scientific epoch records")
        epochs = []
        for row, report in zip(history, value["epochs"]):
            require(report["metrics"] == row["metrics"], "Epoch metrics differ from authenticated history")
            epochs.append(dict(epoch=row["epoch"], **summarize(report, controls["native"]["records"], controls["parent"]["records"])))
        initial, selected = epochs[0], epochs[best["epoch"]]
        require(parity(value["epochs"][0]["records"],pilot["initial_records"])["passed"],
                "Scientific training must restart from the untouched pilot initializer")
        reference = value["epochs"][best["epoch"]]["records"]
        selected = dict(selected, paired_to_initial=scores(reference, value["epochs"][0]["records"])["paired_to_native"])
        require(fresh["fingerprint"] == plan["fingerprint"] and fresh["selected_epoch"] == best["epoch"]
                and fresh["selected_checkpoint_sha256"] == train["selected_checkpoint_sha256"]
                and all(fresh[k] is True for k in ("passed", "strict_state_load", "fixed_state_unchanged", "packed"))
                and parity(fresh["records"], reference)["passed"],
                "Fresh packed selected state must reproduce predictions and bounded BF16 logits")
        summarize(fresh, controls["native"]["records"], controls["parent"]["records"])
        require(all(math.isclose(fresh["metrics"][k], train["selected"][k], abs_tol=1e-7, rel_tol=1e-6)
                    for k in ("loss", "kl", "ce")), "Fresh full-vocabulary metrics changed")
        gates = quality_gates(selected["correct"], native["correct"], parent["correct"])
        require(train["gates"] == gates and fresh["quality_pass"] is gates["quality_pass"],
                "Both native and parent five-percentage-point gates must use matched counts")
        storage = value["storage"]
        require(storage["tied_embedding_readout_counted_once"] is True
                and storage["native_prefix_layers"] == list(range(29)) and storage["compact_layers"] == LAYERS
                and 0 < storage["packed_suffix_tensor_bytes"] <= storage["selected_suffix_tensor_bytes"]
                and storage["whole_model_tensor_bytes"] == storage["native_prefix_and_embedding_bytes"]+storage["packed_suffix_tensor_bytes"]+storage.get("rotary_bytes",0),
                "Actual packed suffix plus native prefix and tied E must be counted")
        classification = ("valid_expansion_within_budget" if gates["quality_pass"] else
                          "partial_recovery_below_budget" if selected["correct"] > initial["correct"] else
                          "no_recovery_below_budget")
        return cls(STUDY_ID, classification, native, parent, initial, selected, epochs,
                   best["epoch"], best["epoch"] == 0, gates, storage)

    @property
    def primary_metric(self): return self.selected["accuracy"]

    def to_dict(self):
        value = asdict(self); value["compressed_layers"] = list(self.compressed_layers)
        return value
