"""Typed, conservative first-block residual/head ablation evidence."""
from collections import Counter
from dataclasses import asdict, dataclass
import math

STUDY_ID = "JSPACE-QWEN-RESIDUAL-DIAGNOSTICS-01"
CLASSIFICATION = "residual_rank_failure_dominates_first_block_head_reduction"
ARMS = ("native", "full_control", "half_head", "half_rank", "mean_rank")
VARIANTS = dict(full_control="full_rank_full_head", half_head="full_rank_half_head",
                half_rank="half_rank_full_head", mean_rank="mean_rank_full_head")
SCOPE = ("Exploratory matched64 development cases,16 per family; only block0 changed,35 native blocks retained. "
         "Native-weight initialization with no recovery training. No held-out quality, deployment speed, "
         "equivalence, or independent-confirmation claim; ongoing recovery is a separate study.")


def require(value, message):
    if not value:
        raise ValueError(message)


@dataclass(frozen=True)
class DiagnosticArm:
    name: str
    correct: int
    count: int
    gains: int
    losses: int
    changed_predictions: int
    max_abs_label_logit_delta: float
    fingerprint: str

    @property
    def accuracy(self):
        return self.correct/self.count


@dataclass(frozen=True)
class JSpaceResidualDiagnosticsRecord:
    study_id: str
    parent_study_id: str
    source_aggregate_sha256: str
    arms: tuple[DiagnosticArm, ...]
    classification: str = CLASSIFICATION
    confirmed: bool = False
    count: int = 64
    independent_confirmations: int = 0
    primary_arm: str = "half_rank"
    scope: str = SCOPE

    @classmethod
    def from_evidence(cls, aggregate, sources, source_sha256):
        require(aggregate["status"] == "completed" and aggregate["confirmed"] is False,
                "Completed exploratory evidence required")
        require(aggregate["study_id"] == "JSPACE-QWEN-ACCURACY-RECOVERY-01", "Wrong parent study")
        require(aggregate["classification"] == CLASSIFICATION, "Unexpected diagnostic classification")
        require(set(sources) == set(ARMS) == set(aggregate["arms"]), "All five matched arms required")
        require(len(source_sha256) == 64 and all(c in "0123456789abcdef" for c in source_sha256), "Invalid aggregate SHA")
        native = sources["native"]["records"]
        identities = [r["identity"] for r in native]
        require(len(native) == len(set(identities)) == 64, "Exactly64 unique matched cases required")
        require(Counter(r["family"] for r in native) == {f:16 for f in ("arc","boolq","hellaswag","winogrande")},
                "Four balanced development families required")
        arms = []
        for name in ARMS:
            source, claimed = sources[name], aggregate["arms"][name]
            rows = source["records"]
            require(source["status"] == "completed" and source["fingerprint"] == claimed["fingerprint"],
                    "Source status/fingerprint mismatch")
            require([r["identity"] for r in rows] == identities, "Unpaired or reordered cases")
            require(name == "native" or source["variant"] == VARIANTS[name], "Wrong ablation variant")
            gains = losses = changed = correct = 0
            delta = 0.
            family = {f:dict(correct=0,count=0) for f in ("arc","boolq","hellaswag","winogrande")}
            for baseline,row in zip(native, rows):
                require(row["answer"] == baseline["answer"] and row["family"] == baseline["family"], "Labels/families changed")
                logits = row["logits"]
                require(len(logits) == len(baseline["logits"]) and 2 <= len(logits) <= 4
                        and all(math.isfinite(v) for v in logits), "Invalid choice logits")
                require(type(row["answer"]) is int and 0 <= row["answer"] < len(logits)
                        and type(row["prediction"]) is int
                        and row["prediction"] == max(range(len(logits)), key=lambda i:logits[i]), "Wrong choice prediction")
                ac,bc = baseline["prediction"] == baseline["answer"], row["prediction"] == row["answer"]
                gains += not ac and bc
                losses += ac and not bc
                changed += row["prediction"] != baseline["prediction"]
                correct += bc
                delta = max(delta, *(abs(a-b) for a,b in zip(logits,baseline["logits"])))
                family[row["family"]]["count"] += 1
                family[row["family"]]["correct"] += bc
            require(source["count"] == claimed["count"] == 64 and source["correct"] == claimed["correct"] == correct,
                    "Answer-count mismatch")
            require(source["accuracy"] == correct/64, "Accuracy/count mismatch")
            require((claimed["wins_against_native"],claimed["losses_against_native"],claimed["changed_predictions"])
                    == (gains,losses,changed), "Paired-count mismatch")
            require(math.isclose(delta, claimed["max_abs_label_logit_delta"], abs_tol=1e-12, rel_tol=1e-12), "Logit-delta mismatch")
            require(claimed["by_family"] is None or claimed["by_family"] == family, "Aggregate family-count mismatch")
            require("by_family" not in source or source["by_family"] == family, "Source family-count mismatch")
            arms.append(DiagnosticArm(name,correct,64,gains,losses,changed,delta,source["fingerprint"]))
        values = {a.name:a for a in arms}
        require(values["full_control"].changed_predictions == 0, "Full-rank control did not preserve choices")
        require(values["half_rank"].correct < values["half_head"].correct < values["native"].correct
                and values["mean_rank"].correct < values["half_head"].correct,
                "Observed ordering does not support the descriptive classification")
        return cls(STUDY_ID,aggregate["study_id"],source_sha256,tuple(arms))

    @property
    def primary_metric(self):
        return next(a.accuracy for a in self.arms if a.name == self.primary_arm)

    def to_dict(self):
        return asdict(self)
