"""Typed four-arm development evidence for actual recovered-block assembly."""
from dataclasses import asdict, dataclass
import math

from .jspace_single_layer_record import require, scores

STUDY_ID = "JSPACE-QWEN-RECOVERED-ASSEMBLY-01"
ARMS = ("native", "accepted_two", "all_compact", "hybrid")
LAYERS = dict(native=[], accepted_two=[0, 1], all_compact=list(range(36)),
              hybrid=[i for i in range(36) if i not in (12, 15)])
SCOPE = ("Actual full36-block assembly on the same64 development cases, with hybrid as the fixed primary arm. "
         "MatchedFP32 tensor storage only; no confirmation, deployment speed, or general-generation claim.")


@dataclass(frozen=True)
class JSpaceRecoveredAssemblyRecord:
    study_id: str
    classification: str
    arms: dict
    primary_quality_pass: bool
    storage: dict
    model_parameters: dict
    primary_arm: str = "hybrid"
    confirmed: bool = False
    independent_confirmations: int = 0
    speed_measured: bool = False
    assembly_evaluated: bool = True
    scope: str = SCOPE

    @classmethod
    def from_evidence(cls, value):
        plan, aggregate, runs = value["plan"], value["aggregate"], value["arms"]
        require(value["status"] == aggregate["status"] == "completed" and value["confirmed"] is False,
                "Completed exploratory assembly evidence required")
        require(plan["schema"] == "jspace.recovered-assembly.v1" and plan["arms"] == LAYERS
                and plan["primary_arm"] == aggregate["primary_arm"] == "hybrid"
                and plan["count"] == 64 and plan["minimum_correct"] == aggregate["minimum_correct"] == 47
                and plan["maximum_accuracy_drop"] == aggregate["maximum_accuracy_drop"] == .05,
                "Frozen four-arm composition and cumulative47/64 gate required")
        require(aggregate["fingerprint"] == plan["fingerprint"] and aggregate["confirmed"] is False
                and aggregate["confirmation_used"] is False and aggregate["speed_measured"] is False
                and aggregate["original_native_bf16_storage_measured"] is False
                and aggregate["original_native_bf16_tensor_bytes"] is None,
                "No confirmation, speed or nativeBF16 storage claim is supported")
        require(set(runs) == set(ARMS), "All four completed arms must be retained")
        native = runs["native"]["records"]; arms, storage, parameters = {}, {}, {}
        for arm in ARMS:
            run = runs[arm]
            require(run["status"] == "completed" and run["fingerprint"] == plan["fingerprint"]
                    and run["arm"] == arm and run["confirmed"] is False and run["compact_layers"] == LAYERS[arm],
                    "Arm status, fingerprint or actual composition differs")
            score = scores(run["records"], native)
            for claimed in (run["accuracy"], aggregate["accuracy"][arm]):
                require(claimed["count"] == 64 and claimed["correct"] == score["correct"]
                        and claimed["accuracy"] == score["accuracy"]
                        and all(claimed["by_family"][f]["correct"] == row["correct"]
                            and claimed["by_family"][f]["count"] == 16 for f,row in score["by_family"].items()),
                        "Reported accuracy differs from actual paired records")
            arms[arm] = score
            state = run["resident_tensor_state"]
            require(state["dtype"] == "float32" and [r["layer"] for r in state["blocks"]] == list(range(36))
                    and all(r["compact"] == (r["layer"] in LAYERS[arm]) for r in state["blocks"]),
                    "Complete matchedFP32 layer inventory required")
            for key in ("storage_bytes", "logical_tensor_bytes"):
                components = [state["endpoints"][key], *[r[key] for r in state["blocks"]]]
                require(all(type(n) is int and n > 0 for n in components)
                        and state[key] == sum(components), "Tensor inventory does not sum to reported size")
            storage[arm] = dict(bytes=state["storage_bytes"], logical_tensor_bytes=state["logical_tensor_bytes"])
            layers = run["metrics"]["layers"]
            require([r["layer"] for r in layers] == list(range(36))
                    and all(type(r["parameter_bytes"]) is int and r["parameter_bytes"] > 0
                        and r["parameter_bytes"] % 4 == 0 for r in layers), "ActualFP32 parameter inventory required")
            parameters[arm] = value["endpoint_parameters"]+sum(r["parameter_bytes"]//4 for r in layers)
        require(arms["native"]["correct"] == plan["expected_native_correct"] == 50
                and arms["accepted_two"]["correct"] == plan["expected_accepted_two_correct"] == 49,
                "Both accepted accuracy controls must reproduce")
        for arm in ("native", "accepted_two"):
            reference = value["references"][arm]["records"]
            scores(reference, runs[arm]["records"])
            unchanged = all(a["prediction"] == b["prediction"] for a,b in zip(reference, runs[arm]["records"]))
            logit_ok = all(abs(observed-expected) <= 5e-4+1e-5*abs(expected)
                for ref, row in zip(reference, runs[arm]["records"]) for expected, observed in zip(ref["logits"], row["logits"]))
            control = runs[arm]["control"]
            require(unchanged and logit_ok and all(control[k] is True for k in ("passed", "predictions_exact", "logits_pass"))
                    and control["logit_atol"] == 5e-4 and control["logit_rtol"] == 1e-5,
                    "Actual control predictions and tolerant logits must match authenticated references")
        native_bytes = storage["native"]["bytes"]
        require(aggregate["native_tensor_storage_bytes"] == native_bytes, "NativeFP32 size changed")
        for arm in ARMS[1:]:
            comparison = aggregate["comparisons"][arm]; paired = comparison["paired"]["overall"]
            score = arms[arm]; delta = (50-score["correct"])/64
            require(paired["count"] == 64 and paired["native_correct"] == 50 and paired["compact_correct"] == score["correct"]
                    and paired["accuracy_drop"] == delta
                    and paired["wins"] == score["paired_to_native"]["gains"]
                    and paired["losses"] == score["paired_to_native"]["losses"], "Paired aggregate differs from raw predictions")
            reduction = 1-storage[arm]["bytes"]/native_bytes
            require(comparison["tensor_storage_bytes"] == storage[arm]["bytes"]
                    and math.isclose(comparison["tensor_storage_reduction"], reduction, rel_tol=1e-12, abs_tol=1e-12),
                    "MatchedFP32 size reduction differs from measured inventories")
            storage[arm]["matched_fp32_reduction"] = reduction
        passed = arms["hybrid"]["correct"] >= 47
        require(aggregate["primary_quality_pass"] is passed, "Primary hybrid outcome differs from frozen gate")
        classification = "hybrid_assembly_development_gate_passed" if passed else "hybrid_assembly_development_gate_failed"
        storage["scope"] = "MatchedFP32 tensors including fixed geometry and tiedE once; originalBF16 storage unmeasured; excludes transient execution memory"
        return cls(STUDY_ID, classification, arms, passed, storage, parameters)

    @property
    def primary_metric(self): return self.arms["hybrid"]["accuracy"]

    def to_dict(self): return asdict(self)
