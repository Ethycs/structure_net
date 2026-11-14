"""Conservative evidence gates for the sequential Qwen quadratic experiment."""
from dataclasses import asdict, dataclass
import math
import statistics

STUDY_ID = "JSPACE-QWEN-QUADRATIC-01"
ARMS = ("softmax", "quadratic")
PASS = "exploratory_quadratic_targets_met"
FAIL = "exploratory_quadratic_targets_not_met"
SCOPE = ("One exploratory Qwen checkpoint and a shared 512-case four-task holdout; 36 sequential stages per arm, "
    "not independent replications. Accuracy uses FP32 layer-streamed GPU inference; speed uses full-model "
    "FP32 one-thread compiled CPU prefill at 128/256 tokens. Timed development-prefix parity is not "
    "512-case compiled-CPU accuracy. No decoding-throughput, long-context, or summarization-quality claim.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def required_stages():
    return ["basis", "inputs", "pilot"] + [f"arms/{arm}/layer_{layer:02d}" for arm in ARMS for layer in range(36)] + [
        f"evaluation/{arm}" for arm in ("native", *ARMS)] + [f"{phase}/{arm}" for phase in ("export", "cpu_benchmark") for arm in ARMS]


def paired_quality(native, candidate):
    require(len(native) == len(candidate) == 512, "Expected 512 paired evaluation cases")
    require(len({row["identity"] for row in native}) == 512, "Duplicate evaluation identity")
    require(all(all(a[k] == b[k] for k in ("identity", "family", "answer")) for a, b in zip(native, candidate)), "Unpaired evaluation rows")
    families = {row["family"] for row in native}
    require(families == {"arc", "boolq", "hellaswag", "winogrande"}, "Changed task families")
    def summary(pairs):
        n = len(pairs)
        ac = sum(a["prediction"] == a["answer"] for a, _ in pairs)
        bc = sum(b["prediction"] == b["answer"] for _, b in pairs)
        wins = sum(a["prediction"] != a["answer"] and b["prediction"] == b["answer"] for a, b in pairs)
        losses = sum(a["prediction"] == a["answer"] and b["prediction"] != b["answer"] for a, b in pairs)
        discordant = wins + losses
        p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant)
        return dict(count=n, native_correct=ac, compact_correct=bc, native_accuracy=ac/n, compact_accuracy=bc/n,
            accuracy_drop=(ac-bc)/n, net_lost_correct=ac-bc, wins=wins, losses=losses,
            agreement=sum(a["prediction"] == b["prediction"] for a, b in pairs)/n, exact_binomial_p=p)
    pairs = list(zip(native, candidate))
    by_family = {name: summary([(a, b) for a, b in pairs if a["family"] == name]) for name in sorted(families)}
    require(all(v["count"] == 128 for v in by_family.values()), "Unbalanced held-out families")
    return {"overall": summary(pairs), "by_family": by_family}


def benchmark_measurements(timing):
    require(timing["threads"] == 1 and timing["repeats"] == 5 and timing["lengths"] == [128, 256], "Changed CPU timing budget")
    compilation = timing["compilation"]
    require(compilation["backend"] == "inductor" and compilation["fullgraph"] is True and compilation["dynamic"] is False
        and compilation["eager_graph_break_fallback"] is False and compilation["options"]["freezing"] is True, "Matched full-graph compilation required")
    require(all(compilation["fx_graphs"][arm] for arm in ("native", "compact")), "Missing compiled graph evidence")
    for arm in ("native", "compact"):
        for mode in ("eager", "compiled"):
            for length in (128, 256):
                value = timing["latencies"][arm][mode][str(length)]
                rows = value["samples"]
                require(len(rows) == 10 and {(r["repeat"], r["prompt"]) for r in rows} == {(r, p) for r in range(5) for p in range(2)}, "Missing or duplicate timing samples")
                require(all(math.isfinite(r["seconds"]) and r["seconds"] > 0 for r in rows), "Invalid CPU wall latency")
                require(close(value["median_seconds"], statistics.median(r["seconds"] for r in rows)), "Reported CPU median differs from samples")
    ratios = {str(length): timing["latencies"]["native"]["compiled"][str(length)]["median_seconds"] /
              timing["latencies"]["compact"]["compiled"][str(length)]["median_seconds"] for length in (128, 256)}
    require(set(timing["compiled_speedup"]) == {"128", "256"} and all(close(ratios[k], timing["compiled_speedup"][k]) for k in ratios), "Speedups do not compare matched compiled models")
    parity = timing["parity"]
    require(len(parity) == 8 and {(p["model"], p["length"], p["prompt"]) for p in parity} ==
            {(a, length, p) for a in ("native", "compact") for length in (128, 256) for p in range(2)}, "Incomplete compilation parity coverage")
    exact = all(p["greedy_equal"] is True for p in parity)
    require(timing["all_compiled_greedy_equal"] is exact, "Wrong compiler parity summary")
    sizes = timing["resident_tensor_bytes"]
    require(set(sizes) == {"native", "compact"} and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in sizes.values()), "Invalid measured resident state size")
    return ratios, 1 - sizes["compact"] / sizes["native"], exact


@dataclass(frozen=True)
class JSpaceQuadraticRecord:
    study_id: str
    scientific_fingerprint: str
    classification: str
    targets_met: bool
    native_correct: int
    softmax_correct: int
    quadratic_correct: int
    cpu_compiled_prefill_speedups: dict
    tensor_reductions: dict
    tensor_bytes: dict
    gates: dict
    count: int = 512
    layers_per_arm: int = 36
    completed_training_stages: int = 72
    confirmed: bool = False
    independent_confirmations: int = 0
    scope: str = SCOPE

    @classmethod
    def from_evidence(cls, evidence):
        plan, result = evidence["plan"], evidence["result"]
        config = plan["config"]
        require(plan["schema"] == result["schema"] == "jspace.quadratic-recipe.v1" and
            plan["study_id"] == result["study_id"] == STUDY_ID, "Wrong experiment identity")
        require(result["status"] == "completed" and result["confirmed"] is False and result["fingerprint"] == plan["fingerprint"], "Completed exploratory evidence required")
        fixed = dict(layers=36, rank=1024, head_dim=64, native_mlp_width=11008, epochs=12, temperature=2.,
                     weight_decay=0., hyperdeterminant=False, dtype="float32", minimum_size_reduction=.38,
                     minimum_speedup=1.4, maximum_accuracy_drop=.05, cpu_threads=1)
        require(all(config[k] == v for k, v in fixed.items()), "Changed scientific/quality protocol")
        for name in required_stages():
            manifest = evidence["manifests"].get(name, {})
            require(manifest.get("status") == "completed" and manifest.get("fingerprint") == plan["fingerprint"], "Incomplete stage: " + name)
        expected = {f"{arm}/{layer}" for arm in ARMS for layer in range(36)}
        require(set(evidence["training"]) == expected, "Both complete sequential arms required")
        for key, training in evidence["training"].items():
            arm, layer = key.split("/"); layer = int(layer)
            require(training["status"] == "completed" and training["mode"] == arm and training["layer"] == layer and
                training["completed_epochs"] == 12 and training["updates"] == 12 * math.ceil(512 / (config["microbatch"] * config["accumulation"])), "Incomplete or mismatched training budget")
            for name in ("epochs", "microbatch", "accumulation", "learning_rate", "temperature", "gradient_clip", "seed"):
                require(training["config"][name] == config[name], "Training config differs: " + name)
            expected_loss = "normalized_feature + T^2 KL" if layer == 35 else "normalized_feature"
            require(training["objective"] == expected_loss and training["final_layer"] is (layer == 35) and training["disk_reload_exact"] is True, "Changed layer objective or missing selected-checkpoint reload")
            history = training["history"]
            require([r["epoch"] for r in history] == list(range(13)) and all(r["development"]["count"] == 64 for r in history), "Incomplete development history")
            require(all(math.isfinite(r["development"]["loss"]) for r in history), "Nonfinite checkpoint selection loss")
            chosen = min(history, key=lambda r: r["development"]["loss"])
            require(training["selected_epoch"] == chosen["epoch"] and training["selected"] == chosen["development"], "Wrong selected checkpoint")
            require(all(v["gradient_updates"] == v["optimizer_updates"] == training["updates"] for v in training["trainable_parameters"].values()) and training["trainable_parameters"], "Missing actual gradient/optimizer participation")
        rows = evidence["records"]
        qualities = {arm: paired_quality(rows["native"], rows[arm]) for arm in ARMS}
        require(result["quadratic_vs_softmax"] == paired_quality(rows["softmax"], rows["quadratic"]), "Wrong paired architecture control")
        ratios, reductions, sizes, gates = {}, {}, {}, {}
        for arm in ARMS:
            ratios[arm], reductions[arm], parity = benchmark_measurements(evidence["timing"][arm])
            sizes[arm] = evidence["timing"][arm]["resident_tensor_bytes"]
            require(evidence["exports"][arm]["tensor_bytes"] == sizes[arm]["compact"] and evidence["exports"][arm]["model_parameters"] > 0, "Export/runtime state disagreement")
            gates[arm] = dict(aggregate_accuracy=qualities[arm]["overall"]["accuracy_drop"] <= .05 + 1e-12,
                tensor_reduction=reductions[arm] >= .38, cpu_prefill_speed=min(ratios[arm].values()) >= 1.4,
                compiled_greedy_parity=parity)
            given = result["arms"][arm]
            require(given["quality"] == qualities[arm] and given["gates"] == gates[arm] and given["targets_met"] is all(gates[arm].values()), "Quality or joint gates differ from raw evidence")
            require(close(given["tensor_reduction"], reductions[arm]) and given["tensor_bytes"] == sizes[arm] and given["cpu_compiled_prefill_speedups"] == ratios[arm], "Reported size/speed differs from measurements")
        require(sizes["softmax"]["native"] == sizes["quadratic"]["native"], "Native FP32 reference state changed between benchmarks")
        passed = all(gates["quadratic"].values())
        require(result["targets_met"] is passed, "Partial success cannot pass the primary target")
        return cls(STUDY_ID, plan["fingerprint"], PASS if passed else FAIL, passed,
            qualities["quadratic"]["overall"]["native_correct"], qualities["softmax"]["overall"]["compact_correct"],
            qualities["quadratic"]["overall"]["compact_correct"], ratios, reductions, sizes, gates)

    def to_dict(self):
        return asdict(self)
