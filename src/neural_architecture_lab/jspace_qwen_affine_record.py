"""Conservative typed record for the final-block Qwen affine experiment."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class JSpaceQwenAffineRecord:
    study_id: str
    scientific_fingerprint: str
    correct: int
    count: int
    original_correct: int
    tensor_bytes: int
    original_tensor_bytes: int
    graph_tokens_per_second: float
    native_graph_tokens_per_second: float
    native_repeat_graph_tokens_per_second: float
    confirmed: bool = False
    classification: str = "partial_quality_recovery_no_speedup"
    primary_arm: str = "r1536_h64_gb"
    scope: str = "exploratory_one_seed_final_block_four_family_length_filtered_multiple_choice"

    @classmethod
    def from_aggregate(cls, aggregate):
        result = aggregate["result"]
        if result["status"] != "completed" or result["confirmed"] is not False:
            raise ValueError("Wrong result status or scope")
        arms = ("r1536_h128", "r1536_h64", "r1024_h64", "r1536_h64_gb")
        for arm in arms:
            if aggregate["reload"][arm]["all_512_records_exact"] is not True:
                raise ValueError("Missing fresh-process reload verification")
            if result["results"][arm]["fresh_accuracy"]["count"] != 512:
                raise ValueError("Incomplete evaluation")
        native = result["results"]["native"]
        final = result["results"]["r1536_h64_gb"]
        record = cls(
            study_id=result["study_id"], scientific_fingerprint=result["fingerprint"],
            correct=final["fresh_accuracy"]["correct"], count=512,
            original_correct=native["fresh_accuracy"]["correct"],
            tensor_bytes=final["tensor_bytes"], original_tensor_bytes=native["tensor_bytes"],
            graph_tokens_per_second=final["runtime"]["timings"]["cuda_graph"]["tokens_per_second"],
            native_graph_tokens_per_second=native["runtime"]["timings"]["cuda_graph"]["tokens_per_second"],
            native_repeat_graph_tokens_per_second=result["results"]["native_repeat_runtime"]["timings"]["cuda_graph"]["tokens_per_second"],
        )
        if record.study_id != "JSPACE-QWEN-AFFINE-01" or aggregate["classification"] != record.classification:
            raise ValueError("Study/classification mismatch")
        if not 0 <= record.correct <= 512 or not 0 <= record.original_correct <= 512:
            raise ValueError("Invalid answer counts")
        for value in (record.tensor_bytes, record.original_tensor_bytes, record.graph_tokens_per_second,
                      record.native_graph_tokens_per_second, record.native_repeat_graph_tokens_per_second):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid measurement")
        return record

    @property
    def accuracy(self):
        return self.correct / self.count

    def to_dict(self):
        return asdict(self)
