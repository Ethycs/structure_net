"""Typed, explicitly exploratory summary of the complete-model compact pass."""
from dataclasses import asdict,dataclass
import math


@dataclass(frozen=True)
class JSpaceFullCompactRecord:
    study_id:str
    scientific_fingerprint:str
    correct:int
    count:int
    original_correct:int
    tensor_bytes:int
    original_tensor_bytes:int
    graph_tokens_per_second:float
    original_graph_tokens_per_second:float
    trained_layers:int
    parameter_count:int
    confirmed:bool=False
    scope:str="exploratory_four_family_length_filtered_multiple_choice_reused_evaluation"

    @classmethod
    def from_result(cls,result):
        final=result["trajectory"]["36"]["evaluation"];original=result["controls"]["original"]
        record=cls(study_id=result["study_id"],scientific_fingerprint=result["fingerprint"],correct=final["correct"],
            count=final["count"],original_correct=original["evaluation"]["correct"],tensor_bytes=result["final_tensor_bytes"],
            original_tensor_bytes=original["tensor_bytes"],
            graph_tokens_per_second=result["adapted_runtime"]["timings"]["cuda_graph"]["tokens_per_second"],
            original_graph_tokens_per_second=original["runtime"]["timings"]["cuda_graph"]["tokens_per_second"],
            trained_layers=len(result["layers"]),parameter_count=result["final_parameters"])
        if result["status"]!="completed" or result["confirmed"] is not False:raise ValueError("Wrong result scope")
        if record.trained_layers!=36 or record.count!=256:raise ValueError("Incomplete all-layer result")
        if not 0<=record.correct<=record.count or not 0<=record.original_correct<=record.count:raise ValueError("Invalid counts")
        for value in (record.tensor_bytes,record.original_tensor_bytes,record.graph_tokens_per_second,record.original_graph_tokens_per_second):
            if not math.isfinite(value) or value<=0:raise ValueError("Invalid measurement")
        return record

    @property
    def accuracy(self):return self.correct/self.count

    @property
    def speedup(self):return self.graph_tokens_per_second/self.original_graph_tokens_per_second

    def to_dict(self):return asdict(self)
