"""Conservative typed index of the complete six-stage Qwen compression study."""
from dataclasses import asdict, dataclass
import math


STUDY_ID = "JSPACE-QWEN-ALLBLOCKS-01"
PASS = "all_targets_met_on_restricted_benchmark"
FAIL = "all_block_targets_not_jointly_met"
THRESHOLDS = dict(minimum_size_reduction=.38, minimum_speedup=1.4,
                  maximum_accuracy_drop=.01, maximum_family_drop=4/128)
V2_THRESHOLDS = {**THRESHOLDS, "maximum_accuracy_drop": .05, "family_quality_is_gate": False}
SCOPE = ("One exploratory Qwen2.5-3B checkpoint; all 36 blocks; six sequential teacher/student recovery groups, "
         "not six independent replications. Fresh 512-item four-task accuracy; batch-one whole-model throughput "
         "on eight development prompts and 32 generated tokens. Fixed position capacity 416. "
         "No general-language, long-context, noninferiority, or independent-confirmation claim.")


def required_stages():
    result = ["evaluation"]
    for group in range(6):
        prefix = f"groups/group_{group:02d}"
        result.extend(prefix + "/" + name for name in ("capture", "basis", "recovery"))
        result.extend(prefix + f"/block_{layer:02d}/fit_result" for layer in range(30 - group*6, 36 - group*6))
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


@dataclass(frozen=True)
class JSpaceAllBlocksRecord:
    study_id: str
    scientific_fingerprint: str
    classification: str
    targets_met: bool
    native_correct: int
    compact_correct: int
    native_tensor_bytes: int
    compact_tensor_bytes: int
    size_reduction: float
    native_graph_rates: tuple
    compact_graph_tokens_per_second: float
    whole_model_speedup: float
    gates: dict
    selected_epoch_by_group: dict
    development_ce_by_group: dict
    protocol_schema: str
    quality_thresholds: dict
    quality_diagnostics: dict
    count: int = 512
    compressed_blocks: int = 36
    completed_groups: int = 6
    independent_confirmations: int = 0
    confirmed: bool = False
    scope: str = SCOPE

    @classmethod
    def from_evidence(cls, evidence):
        plan, result = evidence["plan"], evidence["result"]
        config = plan["config"]
        schema = plan["schema"]
        require(schema in {"jspace.qwen-parallel.v1", "jspace.qwen-parallel.v2"}, "Unknown protocol version")
        amended = schema.endswith(".v2")
        thresholds = V2_THRESHOLDS if amended else THRESHOLDS
        require(plan["study_id"] == result["study_id"] == STUDY_ID and not plan["systems_only"], "Wrong study or systems-only evidence")
        require(result["status"] == "completed" and result["confirmed"] is False, "Completed exploratory evidence required")
        require(result["fingerprint"] == plan["fingerprint"], "Scientific fingerprint mismatch")
        require(all(config[k] == v for k, v in thresholds.items()), "User targets changed")
        if amended:
            require(config["rank_gate"] == "actual_sketch_svd_shape_fp32", "Changed numerical-rank amendment")
        require(config["layers"] == 36 and config["group_size"] == 6 and config["epochs"] == 12, "Incomplete all-block protocol")
        require(result["evaluation_count"] == 512, "Expected 512 fresh cases")
        for name in required_stages():
            require(name in evidence["manifests"], "Missing completed stage: " + name)
            manifest = evidence["manifests"][name]
            require(manifest["status"] == "completed" and manifest["fingerprint"] == plan["fingerprint"], "Incomplete/mismatched stage: " + name)
        require(set(evidence["training"]) == {str(i) for i in range(6)}, "All six recovery groups required")
        selected, trajectories = {}, {}
        expected_updates = 12 * math.ceil(512 / (config["microbatch"] * config["accumulation"]))
        for group, training in evidence["training"].items():
            require(training["status"] == "completed" and training["completed_epochs"] == 12 and
                    training["updates"] == expected_updates, "Incomplete recovery budget: " + group)
            require(training["selection_metric"] == "packed_development_ce", "Changed checkpoint selection")
            for key in ("microbatch", "accumulation", "epochs", "learning_rate", "temperature", "kl_weight", "ce_weight", "gradient_clip"):
                require(training["config"][key] == config[key], "Training config differs from frozen plan: " + key)
            history = training["history"]
            require([r["epoch"] for r in history] == list(range(13)), "Missing group development trajectory")
            values = [r["development"]["ce"] for r in history]
            require(all(math.isfinite(v) for v in values), "Nonfinite development CE")
            best = min(range(13), key=lambda i: values[i])
            require(training["selected_epoch"] == best and training["selected"] == history[best]["development"], "Wrong selected checkpoint")
            require(all(r["development"]["count"] == 64 for r in history), "Development population changed")
            selected[group], trajectories[group] = best, values
        quality = result["quality"]
        families = quality["by_family"]
        require(set(families) == {"arc", "boolq", "hellaswag", "winogrande"}, "Wrong task families")
        for q in (quality["overall"], *families.values()):
            n = q["count"]
            require(n == (512 if q is quality["overall"] else 128), "Wrong accuracy population")
            require(all(isinstance(q[k], int) and 0 <= q[k] <= n for k in ("native_correct", "compact_correct", "wins", "losses")), "Invalid correctness counts")
            require(q["wins"] - q["losses"] == q["compact_correct"] - q["native_correct"], "Paired counts disagree")
            require(close(q["native_accuracy"], q["native_correct"]/n) and close(q["compact_accuracy"], q["compact_correct"]/n) and
                    close(q["accuracy_drop"], (q["native_correct"]-q["compact_correct"])/n), "Accuracy count/rate mismatch")
        for key in ("native_correct", "compact_correct", "wins", "losses"):
            require(sum(q[key] for q in families.values()) == quality["overall"][key], "Family counts do not add up")
        measurements = result["results"]
        require(set(measurements) == {"native_before", "compact", "native_after"}, "Native-before/compact/native-after controls required")
        sizes, rates = {}, {}
        for arm, value in measurements.items():
            sizes[arm] = value["tensor_bytes"]
            rates[arm] = value["runtime"]["timings"]["cuda_graph"]["tokens_per_second"]
            require(isinstance(sizes[arm], int) and sizes[arm] > 0 and math.isfinite(rates[arm]) and rates[arm] > 0, "Invalid deployed size or whole-model throughput")
        require(sizes["native_before"] == sizes["native_after"] == config["native_bytes"], "Native reference size changed")
        baseline = (rates["native_before"] + rates["native_after"]) / 2
        reduction, speedup = 1 - sizes["compact"] / sizes["native_before"], rates["compact"] / baseline
        gates = dict(overall_quality=quality["overall"]["accuracy_drop"] <= thresholds["maximum_accuracy_drop"] + 1e-12,
            every_family_quality=all(q["accuracy_drop"] <= 4/128 + 1e-12 for q in families.values()),
            deployed_size=reduction + 1e-12 >= .38, whole_model_speed=speedup + 1e-12 >= 1.4,
            packed_reload_exact=result["packed_reload_exact"] is True,
            graph_static_tokens_exact=all(v["runtime"]["graph_static_exact"] is True for v in measurements.values()))
        acceptance = result["acceptance"]
        diagnostics = {}
        if amended:
            diagnostics = {"every_family_quality": gates.pop("every_family_quality"),
                "original_one_percentage_point_quality": quality["overall"]["accuracy_drop"] <= .01 + 1e-12}
            require(acceptance.get("diagnostics") == diagnostics and acceptance.get("primary_quality_scope") == "aggregate_accuracy",
                    "Amended quality diagnostics or aggregate scope changed")
        require(acceptance["thresholds"] == thresholds and acceptance["gates"] == gates, "Reported gates differ from measured joint gates")
        require(close(acceptance["measured_size_reduction"], reduction) and close(acceptance["whole_model_speedup"], speedup) and
                close(acceptance["native_baseline_tokens_per_second"], baseline), "Wrong size/speed aggregation")
        passed = all(gates.values()); classification = PASS if passed else FAIL
        require(result["targets_met"] is passed and acceptance["passed"] is passed and result["classification"] == classification,
                "Partial successes cannot pass the joint target")
        return cls(STUDY_ID, plan["fingerprint"], classification, passed,
            quality["overall"]["native_correct"], quality["overall"]["compact_correct"],
            sizes["native_before"], sizes["compact"], reduction, (rates["native_before"], rates["native_after"]),
            rates["compact"], speedup, gates, selected, trajectories, schema, dict(thresholds), diagnostics)

    def to_dict(self):
        value = asdict(self)
        value["native_graph_rates"] = list(self.native_graph_rates)
        return value
