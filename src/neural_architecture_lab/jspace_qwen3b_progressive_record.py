"""Conservative development evidence for one-block Qwen3B suffix expansion."""
from dataclasses import asdict, dataclass
import math

from .jspace_incremental_suffix_record import ATOL, RTOL, CONFIG, parity, quality_gates, summarize
from .jspace_single_layer_record import require, scores

STUDY_ID="JSPACE-QWEN3B-PROGRESSIVE-01"
PRIMARY_METRIC_NAME="verified_accepted_compressed_layer_count"
SCOPE=("Qwen2.5-3B BF16 progressive suffix recovery from the verified seven-block seed. "
       "One new preceding block is calibrated against each accepted parent; the complete suffix and final norm "
       "are jointly trained for12epochs on512 cases and selected on64 reused development cases. "
       "Native prefix boundaries are cached. No untouched confirmation, speed or general-generation evidence.")


def check_storage(value,cut):
    require(value["cut"]==cut and value["compressed_layers"]==36-cut,
            "Storage must describe the actual accepted suffix")
    keys=("packed_suffix_and_norm_bytes","frozen_native_prefix_bytes","tied_embedding_readout_bytes","rotary_bytes")
    require(all(type(value[k]) is int and value[k]>=0 for k in keys) and
        value["packed_suffix_and_norm_bytes"]>0 and value["tied_embedding_readout_bytes"]>0 and
        value["full_model_tensor_bytes"]==sum(value[k] for k in keys) and
        value["native_full_tensor_bytes"]>0 and
        math.isclose(value["full_model_reduction_fraction"],1-value["full_model_tensor_bytes"]/value["native_full_tensor_bytes"],abs_tol=1e-12),
        "Packed suffix, fixed geometry, native prefix and tied E arithmetic changed")
    require((value["frozen_native_prefix_bytes"]==0)==(cut==0),"Only cut0 omits the native prefix")


def stage_summary(value,native_rows,parent_rows,parent_correct):
    plan=value["plan"];cut=plan["cut"];layers=list(range(cut,36))
    require(plan["layers"]==layers and plan["parent_cut"]==cut+1 and
        plan["parent_correct"]==parent_correct and plan["native_correct"]==48 and
        plan["calibration_seed"]==20861+100000*(29-cut),"Progressive parent, layer range or probe seed changed")
    raw=value["raw_stages"];train,fresh,pilot,capture=(raw[k] for k in ("train","verify","pilot","capture"))
    require(all(raw[k]["status"]=="completed" for k in ("capture","fit","pilot","train","verify")),
            "Every phase must complete before a stage can be accepted")
    controls=capture["controls"]
    def checked_control(control,reference,correct):
        require(control["passed"] is True and control["predictions_exact"] is True and
            control["logits_pass"] is True and control["correct"]==correct and control["count"]==64 and
            control["accuracy"]==correct/64 and control["atol"]==ATOL and control["rtol"]==RTOL and
            math.isfinite(control["max_abs_logit_delta"]) and control["max_abs_logit_delta"]>=0,
            "Authenticated aggregate replay control failed")
        if "records" in control:
            require(parity(control["records"],reference)["passed"],"Saved control rows differ from reference")
    for key in ("parent","seam"):
        checked_control(controls[key],parent_rows,parent_correct)
    if cut==28:
        checked_control(controls["native"],native_rows,48)
    diagnostic=capture["teacher_cache_diagnostic"]
    require(diagnostic["predictions_exact"] is True and diagnostic["correct"]==parent_correct and
        capture["teacher_cache_logit_gate_applied"] is False,
        "Differentiable accepted-teacher cache must preserve parent decisions; logit rounding remains diagnostic")
    require(pilot["passed"] is True and pilot["systems_only"] is True and pilot["discarded"] is True and
        pilot["layer_indices"]==layers and pilot["fp32_master_changed_by_block"]==[True]*len(layers) and
        pilot["norm_fp32_master_changed"] is True and pilot["fixed_state_unchanged"] is True and
        pilot["checkpoint_strict_load"] is True and pilot["inheritance"]["passed"] is True and
        pilot["update"]["gradients_present_by_block"]==[True]*len(layers),
        "Discarded pilot must update every suffix block/norm while preserving geometry and parent initialization")
    require(train["systems_only"] is False and train["confirmed"] is False and train["config"]==CONFIG and
        train["completed_epochs"]==12 and train["updates"]==384 and train["cut"]==cut and
        train["layer_indices"]==layers and train["suffix_blocks"]==len(layers) and
        train["strict_reload_exact"] is True and train["fixed_state_unchanged"] is True and
        train["fixed_state"]==pilot["fixed_state"],"Complete unchanged12epoch joint recovery required")
    history=train["history"];reports=value["epochs"]
    require([r["epoch"] for r in history]==list(range(13)) and len(reports)==13,"All13 epoch records required")
    epochs=[]
    for row,report in zip(history,reports):
        require(row["metrics"]==report["metrics"],"Authenticated epoch/history mismatch")
        epochs.append(dict(epoch=row["epoch"],**summarize(report,native_rows,parent_rows)))
    best=max(history,key=lambda r:(r["metrics"]["correct"],-r["metrics"]["ce"]))
    require(train["selected_epoch"]==best["epoch"] and train["selected"]==best["metrics"] and
        train["selected_checkpoint_sha256"]==best["checkpoint_sha256"] and
        train["initial"]==history[0]["metrics"] and train["initializer_selected"]==(best["epoch"]==0),
        "Initializer-eligible accuracy-first/CE-tiebreak selection changed")
    require(parity(reports[0]["records"],pilot["initial_records"])["passed"],"Scientific initializer differs from pilot")
    reference=reports[best["epoch"]]["records"]
    require(fresh["passed"] is True and fresh["confirmed"] is False and fresh["fixed_state_unchanged"] is True and
        fresh["selected_checkpoint_sha256"]==train["selected_checkpoint_sha256"] and
        fresh["metrics"]==train["selected"] and parity(fresh["records"],reference)["passed"],
        "Fresh packed selected checkpoint must replay independently")
    actual=epochs[best["epoch"]];gates=quality_gates(actual["correct"],48,parent_correct)
    legacy=dict(passed=gates["quality_pass"],minimum_correct=gates["minimum_correct"],
        native=dict(correct=48,minimum_correct=gates["native_minimum_correct"],passed=gates["native_pass"]),
        parent=dict(correct=parent_correct,minimum_correct=gates["parent_minimum_correct"],passed=gates["parent_pass"]))
    require(train["quality_pass"] is gates["quality_pass"] and
        train["quality"]==dict(legacy,count=64,maximum_drop=.05) and fresh["quality"]==legacy,
        "Native and immediate-parent five-percentage-point budgets changed")
    require(value["update_audit"]==dict(count=384,all_blocks_and_norm=True),"Every update must cover all suffix blocks and norm")
    check_storage(fresh["storage"],cut)
    selected=dict(actual,paired_to_initial=scores(reference,reports[0]["records"])["paired_to_native"])
    return dict(cut=cut,compressed_layers=len(layers),parent_correct=parent_correct,initial=epochs[0],selected=selected,
        selected_epoch=best["epoch"],initializer_selected=best["epoch"]==0,quality=gates,storage=fresh["storage"],
        epochs=epochs,updates=384,selected_checkpoint=value["selected_checkpoint"],training_seconds=value["training_seconds"],
        control_rows_independently_recomputed=all("records" in controls[k] for k in ("parent","seam")+( ("native",) if cut==28 else ())))


@dataclass(frozen=True)
class JSpaceQwen3BProgressiveRecord:
    study_id:str
    classification:str
    terminal:bool
    accepted_cut:int
    accepted_compressed_layers:int
    accepted_correct:int
    accepted_storage:dict
    stages:list
    stop:dict
    confirmed:bool=False
    independent_confirmations:int=0
    speed_measured:bool=False
    scope:str=SCOPE

    @classmethod
    def from_evidence(cls,value):
        plan=value["plan"]
        require(value["confirmed"] is False and plan["confirmed"] is False and
            plan["schema"]=="jspace.qwen3b-progressive.v1" and plan["study_id"]==STUDY_ID and
            plan["config"]==CONFIG and plan["seed_cut"]==29 and plan["target_cut"]==0 and
            plan["seed_correct"]==51 and plan["native_correct"]==48,"Declared progressive3B campaign required")
        native=value["native_records"];parent=value["seed_records"]
        require(scores(native)["correct"]==48 and scores(parent,native)["correct"]==51,"Native48 and accepted seven-block51 seed required")
        check_storage(value["seed_storage"],29)
        accepted_cut,accepted_correct,storage=29,51,value["seed_storage"]
        stages=[];stop={};attempts=value["stages"]
        require([s["plan"]["cut"] for s in attempts]==list(range(28,28-len(attempts),-1)),"Attempted cuts must be contiguous from28")
        for index,stage in enumerate(attempts):
            require(stage["plan"]["parent_cut"]==accepted_cut and stage["plan"]["parent_correct"]==accepted_correct,
                    "An unaccepted model cannot become the next parent")
            if stage["status"]!="completed":
                require(index==len(attempts)-1 and stage["status"] in ("in_progress","resource_failure","execution_failure"),
                        "Only the last attempted stage may be unfinished")
                stop=dict(kind=stage["status"],cut=stage["plan"]["cut"],phase=stage.get("phase"),error=stage.get("error"))
                break
            summary=stage_summary(stage,native,parent,accepted_correct);stages.append(summary)
            if not summary["quality"]["quality_pass"]:
                require(index==len(attempts)-1,"Cannot advance after a completed quality miss")
                stop=dict(kind="quality_stop",cut=summary["cut"]);break
            accepted_cut=summary["cut"];accepted_correct=summary["selected"]["correct"];storage=summary["storage"]
            parent=stage["raw_stages"]["verify"]["records"]
        outcome=value.get("outcome")
        if outcome:
            require(outcome["confirmed"] is False and outcome["last_accepted_cut"]==accepted_cut and
                outcome["compressed_blocks"]==36-accepted_cut,"Campaign outcome contradicts accepted stage lineage")
            if outcome["status"]=="all_blocks_recovered":
                require(accepted_cut==0 and not stop,"Only verified passing cut0 means all36 blocks recovered")
                classification="all36_recovered_on_reused_development"
            elif outcome["status"]=="quality_stop":
                require(stop.get("kind")=="quality_stop" and outcome["failed_cut"]==stop["cut"],"Quality-stop evidence missing")
                classification="progressive_quality_stop"
            else:
                require(outcome["status"]=="requested_prefix_completed" and not stop and accepted_cut>0,"Unknown terminal outcome")
                classification="partial_requested_completion"
            terminal=True
        elif stop.get("kind") in ("resource_failure","execution_failure"):
            terminal=True;classification="progressive_"+stop["kind"]
        else:
            terminal=False;classification="progressive_in_progress"
        return cls(STUDY_ID,classification,terminal,accepted_cut,36-accepted_cut,accepted_correct,storage,stages,stop)

    @property
    def primary_metric(self):return self.accepted_compressed_layers

    def to_dict(self):return asdict(self)
