"""A teacher-guided trajectory diagnostic, explicitly separate from a model."""
from dataclasses import asdict,dataclass
import math

from .jspace_single_layer_record import require,scores

STUDY_ID="JSPACE-QWEN-STACKED-PROJECTION-ORACLE-01"
SCOPE=("Original teacher hidden states are supplied at every one of 36 layers on the original 64 development cases. "
       "This oracle is not a deployable model, learned recovery, or a task-accuracy upper bound. No confirmation or speed claim.")
RECURRENCE="h_next = h + ((native_next-h) @ U) @ U.T; frozen exact per-layer U; no learned branches"


@dataclass(frozen=True)
class JSpaceStackedOracleRecord:
    study_id:str
    classification:str
    native:dict
    teacher_guided:dict
    native_decision_agreement:float
    trajectory:list
    final_relative_rms:dict
    separate_student:dict
    native_control:dict
    confirmed:bool=False
    learned_model:bool=False
    deployable:bool=False
    task_accuracy_upper_bound:bool=False
    independent_confirmations:int=0
    scope:str=SCOPE

    @classmethod
    def from_evidence(cls,value):
        plan,result=value["plan"],value["evaluation"]
        require(value["status"]==result["status"]=="completed" and value["confirmed"] is False
                and plan["confirmed"] is False and result["confirmed"] is False,"Completed exploratory diagnostic required")
        require(plan["layers"]==result["layers"]==36 and plan["shape"]==result["shape"]
                and len(plan["basis_paths"])==len(set(plan["basis_paths"]))==36
                and plan["recurrence"]==RECURRENCE,"Exactly36 fixed projection subspaces and native-target recurrence required")
        require(result["fingerprint"]==plan["fingerprint"] and result["quality_reference_is_diagnostic_only"] is True,
                "Matching diagnostic result, not a learned-model gate, required")
        require(len(plan["cases"])==64 and all(c["split"]=="development" for c in plan["cases"]),"Original development-only population required")
        native=scores(result["native"]["records"])
        guided=scores(result["oracle"]["records"],result["native"]["records"])
        reference=scores(value["native_reference"]["records"],result["native"]["records"])
        control=result["native_control"]
        require(native["correct"]==result["native"]["correct"]==plan["native_correct"]==50
                and guided["correct"]==result["oracle"]["correct"] and result["oracle"]["count"]==64,
                "Prediction counts must agree with diagnostic summaries")
        require(all(control[k] is True for k in ("passed","predictions_exact","logits_pass"))
                and reference["paired_to_native"]["changed_predictions"]==0,"Authenticated native control must pass")
        for actual,case in zip(result["native"]["records"],plan["cases"]):
            require(all(actual[k]==case[k] for k in ("identity","family","answer")),"Cases differ from the frozen oracle population")
        agreement=1-guided["paired_to_native"]["changed_predictions"]/64
        paired=result["paired"]["overall"]
        require(paired["agreement"]==agreement and paired["wins"]==guided["paired_to_native"]["gains"]
                and paired["losses"]==guided["paired_to_native"]["losses"],"Paired decision agreement changed")
        trajectory=result["trajectory"]
        require([r["boundary"] for r in trajectory]==list(range(37)) and all(
            math.isfinite(r[k]) and r[k]>=0 for r in trajectory for k in ("valid_token_relative_rms","answer_relative_rms")),
            "All37 finite boundary diagnostics required")
        require(trajectory[0]["valid_token_relative_rms"]==trajectory[0]["answer_relative_rms"]==0.,"Native initial state required")
        student=scores(value["separate_student"]["records"],result["native"]["records"])
        require(student["correct"]==27 and value["separate_student"]["arm"]=="all_compact",
                "The27/64 learned assembly comparison must remain a separate authenticated artifact")
        student["scope"]="Separate previously evaluated real compact assembly before joint adaptation; not an oracle result"
        return cls(STUDY_ID,"teacher_guided_native_decisions_preserved" if agreement==1. else "teacher_guided_decision_mismatch",
            native,guided,agreement,trajectory,{k:trajectory[-1][k] for k in ("valid_token_relative_rms","answer_relative_rms")},student,control)

    @property
    def primary_metric(self):return self.native_decision_agreement

    def to_dict(self):return asdict(self)
