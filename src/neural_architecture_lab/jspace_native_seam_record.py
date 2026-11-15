"""Conservative records for one execution stop and its separate seam diagnosis."""
from dataclasses import asdict, dataclass
import math

from .jspace_single_layer_record import require, scores

NATIVE_ID = "JSPACE-QWEN3B-NATIVE-BUDGET-01"
SEAM_ID = "JSPACE-3B-SEAM-DIAGNOSTIC-01"
SCOPE = ("Reused64 development cases only. Eleven verified compressed blocks, not full36 completion; "
         "no independent confirmation, generation-quality or speed claim.")


def compare_rows(rows, reference):
    measured = scores(rows, reference)
    delta = max(abs(a-b) for x,y in zip(rows,reference) for a,b in zip(x["logits"],y["logits"]))
    require(all(len(x["logits"]) == len(y["logits"]) for x,y in zip(rows,reference)), "Choice widths changed")
    exact = measured["paired_to_native"]["changed_predictions"] == 0
    close = all(abs(a-b) <= .03125 + abs(b)/128 for x,y in zip(rows,reference)
                for a,b in zip(x["logits"],y["logits"]))
    return dict(passed=exact and close, predictions_exact=exact, logits_pass=close,
                max_abs_logit_delta=delta, correct=measured["correct"], count=64, accuracy=measured["accuracy"],
                atol=.03125, rtol=1/128)


def check_comparison(declared, rows, reference):
    actual = compare_rows(rows, reference)
    require(all(declared[k] == v for k,v in actual.items()), "Comparison disagrees with retained logits")
    return actual


def check_metrics(metrics, rows):
    actual = scores(rows)
    require(all(metrics[k] == actual[k] for k in ("correct", "count", "accuracy", "by_family")),
            "Metrics disagree with retained predictions")
    require(all(math.isfinite(metrics[k]) and metrics[k] >= 0 for k in ("loss", "kl", "ce", "original_kl")),
            "Finite nonnegative recovery metrics required")
    return actual


@dataclass(frozen=True)
class NativeBudgetFailureRecord:
    study_id: str
    classification: str
    accepted_cut: int
    accepted_compressed_layers: int
    initial: dict
    selected: dict
    selected_epoch: int
    quality: dict
    storage: dict
    stop: dict
    epochs: list
    confirmed: bool = False
    full36_recovered: bool = False
    independent_confirmations: int = 0
    speed_measured: bool = False
    scope: str = SCOPE

    @classmethod
    def from_evidence(cls, value):
        p,s = value["plan"],value["stage_plan"]
        phases=value["phases"]; t,f=phases["train"],phases["verify"]
        require(p["schema"] == s["schema"] == "jspace.qwen3b-native-budget.v1" and p["study_id"] == NATIVE_ID
                and p["confirmed"] is False and p["seed_cut"] == 26 and s["cut"] == 25
                and s["parent_cut"] == 26 and s["root_fingerprint"] == p["fingerprint"]
                and s["layers"] == list(range(25,36)), "Authenticated eleven-block continuation required")
        require(set(phases) == {"capture","fit","pilot","train","verify"}
                and all(x["status"] == "completed" for x in phases.values()), "All five accepted-stage phases required")
        require(t["systems_only"] is False and t["confirmed"] is False and t["updates"] == 384
                and t["completed_epochs"] == 12 and t["strict_reload_exact"] is True
                and t["fixed_state_unchanged"] is True and f["fixed_state_unchanged"] is True
                and f["passed"] is True and f["confirmed"] is False
                and f["selected_checkpoint_sha256"] == t["selected_checkpoint_sha256"],
                "Completed unchanged-geometry training and exact selected reload required")
        history=t["history"]; epochs=value["epochs"]
        require([h["epoch"] for h in history] == list(range(13)) and len(epochs)==13, "All13 development evaluations required")
        for h,e in zip(history,epochs):
            require(h["metrics"] == e["metrics"], "Epoch metrics changed")
            check_metrics(e["metrics"],e["records"])
        winner=max(history,key=lambda h:(h["metrics"]["correct"],-h["metrics"]["ce"]))
        require(winner["epoch"] == t["selected_epoch"] and winner["metrics"] == t["selected"]
                and t["initial"] == history[0]["metrics"], "Accuracy-first initializer-eligible selection changed")
        selected_rows=epochs[t["selected_epoch"]]["records"]
        selected=check_metrics(t["selected"],selected_rows)
        check_metrics(f["metrics"],f["records"])
        require(f["metrics"] == t["selected"] and f["records"] == selected_rows, "Fresh exact metrics/logits changed")
        require(check_comparison(f["comparison"],f["records"],selected_rows)["passed"], "Fresh prediction/logit gate failed")
        q=t["quality"]; native=s["native_correct"]; parent=s["parent_correct"]
        minimum=math.ceil(native-.05*64-1e-12)
        expected=dict(passed=selected["correct"]>=minimum,minimum_correct=minimum,count=64,maximum_drop=.05,
            native=dict(correct=native,minimum_correct=minimum,passed=selected["correct"]>=minimum),
            parent=dict(correct=parent,minimum_correct=math.ceil(parent-.05*64-1e-12),
                        passed=selected["correct"]>=math.ceil(parent-.05*64-1e-12)),
            policy="original_native_only",parent_gate_required=False)
        require(native == p["native_correct"] == 48 and q == f["quality"] == expected
                and q["passed"] is True and t["quality_pass"] is True, "Original-native budget changed")
        storage=f["storage"]
        require(storage["cut"]==25 and storage["compressed_layers"]==11 and f["compressed_blocks"]==11
                and storage["full_model_tensor_bytes"] == sum(storage[k] for k in
                    ("packed_suffix_and_norm_bytes","frozen_native_prefix_bytes","tied_embedding_readout_bytes","rotary_bytes"))
                and storage["native_full_tensor_bytes"]>storage["full_model_tensor_bytes"]>0
                and math.isclose(storage["full_model_reduction_fraction"],1-storage["full_model_tensor_bytes"]/storage["native_full_tensor_bytes"],abs_tol=1e-12),
                "Actual matched BF16 inventory required")
        failure=value["failure"]; failed=value["failed_stage_plan"]
        require(failed["cut"]==value["failed_cut"]==24 and failed["parent_cut"]==25
                and failed["root_fingerprint"]==p["fingerprint"]
                and failed["parent_checkpoint_sha256"]==t["selected_checkpoint_sha256"]
                and value["failed_phase"]=="capture" and value["failed_phase_completed"] is False
                and failure["status"]=="failed" and failure["failure_kind"]=="execution_failure",
                "The uncompleted next capture is an execution failure, not a scientific rejection")
        return cls(NATIVE_ID,"execution_failure_after_verified_partial_recovery",25,11,
            check_metrics(t["initial"],epochs[0]["records"]),selected,t["selected_epoch"],q,storage,
            dict(cut=24,phase="capture",failure_kind="execution_failure",scientific_quality_rejection=False),
            [dict(epoch=h["epoch"],**h["metrics"]) for h in history])

    @property
    def primary_metric(self): return float(self.accepted_compressed_layers)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class SeamDiagnosticRecord:
    study_id: str
    classification: str
    control: dict
    boundary: dict
    modes: dict
    frozen_state_sha256: str
    confirmed: bool = False
    full36_recovered: bool = False
    independent_confirmations: int = 0
    speed_measured: bool = False
    scope: str = SCOPE + " Read-only diagnostic; no training or parameter changes."

    @classmethod
    def from_evidence(cls,value):
        p,r,m=value["plan"],value["evaluation"],value["manifest"]
        require(p["schema"]=="jspace.3b-seam-diagnostic.v1" and p["cut"]==24 and p["parent_cut"]==25
                and p["confirmed"] is False and r["confirmed"] is False and r["status"]==m["status"]=="completed"
                and r["fingerprint"]==m["fingerprint"]==p["fingerprint"] and r["passed"] is True,
                "Completed frozen seam diagnostic required")
        require(r["unchanged_state"] is True and r["state_before"]==r["state_after"], "Model changed during read-only diagnosis")
        modes=r["modes"]; ref=modes["packed_inference"]["records"]
        require([x["identity"] for x in ref]==p["identities"], "Diagnostic development population changed")
        require(check_comparison(r["control"],ref,value["reference"]["records"])["passed"]
                and r["control"]["correct"]==p["parent_correct"]==46, "Cached selected-parent control failed")
        require(set(modes)=={"packed_inference","packed_torch","unpacked_eval","unpacked_grad", "native_batch4","native_singleton","native_singleton_padded"}, "All declared execution modes required")
        output={}
        for name,mode in modes.items():
            check_comparison(mode,mode["records"],ref)
            vals=mode["full_vocab_kl_t2_by_case"]
            require(len(vals)==64 and all(math.isfinite(x) and x>=-1e-7 for x in vals)
                    and math.isclose(sum(vals)/64,mode["full_vocab_kl_t2"],abs_tol=1e-12)
                    and math.isfinite(mode["full_vocab_max_abs"]) and mode["full_vocab_max_abs"]>=0
                    and all(math.isfinite(x) and x>=0 for x in mode["objective"].values()), "Finite full-vocabulary measurements required")
            output[name]={k:v for k,v in mode.items() if k not in ("records","full_vocab_kl_t2_by_case")}
        for name in ("packed_inference","packed_torch","native_singleton"):
            require(modes[name]["records"]==ref and modes[name]["full_vocab_max_abs"]==0
                    and modes[name]["full_vocab_kl_t2"]==0, "Singleton/fused-versus-Torch exactness not established")
        boundary=r["boundary"]; require(set(boundary)=={"batch4","singleton","singleton_padded"}, "All native layouts required")
        for b in boundary.values():
            rows=b["cases"]
            require(len(rows)==64 and b["exact"]==all(x["exact"] for x in rows)
                    and b["max_abs"]==max(x["max_abs"] for x in rows)
                    and math.isclose(b["case_mean_relative_rms"],sum(x["relative_rms"] for x in rows)/64,abs_tol=1e-12)
                    and all(math.isfinite(x[k]) and x[k]>=0 for x in rows for k in
                        ("max_abs","mismatch_fraction","relative_rms","answer_relative_rms")), "Boundary summaries changed")
        require(boundary["singleton"]["exact"] is True and boundary["singleton"]["max_abs"]==0
                and boundary["batch4"]["exact"] is False and modes["native_batch4"]["passed"] is False,
                "Batch/padding-sensitive failure and exact singleton control required")
        return cls(SEAM_ID,"singleton_seam_exact_packed_kernels_exact",r["control"],boundary,output,r["state_before"]["sha256"])

    @property
    def primary_metric(self): return sum(x["exact"] for x in self.boundary["singleton"]["cases"])/64
    def to_dict(self): return asdict(self)


def record_for(kind,value):
    require(kind in ("native","seam"),"Unknown closeout kind")
    return (NativeBudgetFailureRecord if kind=="native" else SeamDiagnosticRecord).from_evidence(value)
