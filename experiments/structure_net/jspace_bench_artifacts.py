"""CPU-only storage accounting and presentation for the frozen J-space pilot."""
import argparse
import json
from pathlib import Path

import torch

from . import jspace_bench_tensor as base
from .jspace_bench_tensor_v2 import DEFAULT_ROOT as PARENT
from .jspace_bench_covered import DEFAULT_ROOT as COVERED
from .jspace_bench_data import sha, write


def read(path):
    return json.loads(path.read_text())


def storage_audit(root, stage, arm, baseline_key="baseline", outside_buffer_bytes=0):
    plan = base.verify(root)
    manifest = base.checked(root, stage)
    evaluation = read(root / "evaluation.json")
    config_path = Path(plan["model"]["path"]) / "config.json"
    if sha(config_path) != plan["model"]["files"]["config.json"]:
        raise ValueError("Pinned model configuration changed")
    config = read(config_path)
    hidden, intermediate = config["hidden_size"], config["intermediate_size"]
    modules = torch.load(root / "compressed_modules.pt", map_location="cpu", weights_only=True)
    weight_names = {"quadratic_weight", "dropped_weight", "coefficient", "output_bias", "gate_up_weight", "down_weight"}
    normalization_names = {"anchor_mean", "anchor_scale", "dropped_mean", "dropped_scale"}
    index_names = {"anchor_positions", "pair_i", "pair_j", "feature_i", "feature_j"}
    layers = {}
    totals = {"weight_values": 0, "weight_bytes": 0, "normalization_bytes": 0, "index_bytes": 0}
    for layer, state in modules.items():
        shapes, counts = {}, dict.fromkeys(totals, 0)
        for name, tensor in state.items():
            size = tensor.numel() * tensor.element_size()
            shapes[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "bytes": size}
            if name.endswith(".weight") or name in weight_names:
                counts["weight_values"] += tensor.numel()
                counts["weight_bytes"] += size
            elif name in normalization_names:
                counts["normalization_bytes"] += size
            elif name in index_names:
                counts["index_bytes"] += size
            else:
                raise ValueError(f"Unclassified replacement state: {name}")
        if "gate_up_weight" in state:
            removed = evaluation["removed_counts"][layer]
            actual_deleted_rows = intermediate-state["gate_up_weight"].shape[0]//2
            assert 0 <= removed-actual_deleted_rows < 16
            assert state["down_weight"].shape == (hidden, intermediate)
        else:
            removed = intermediate - state["gate_proj.weight"].shape[0]
            actual_deleted_rows = removed
            assert state["up_proj.weight"].shape == state["gate_proj.weight"].shape
            assert state["down_proj.weight"].shape == (hidden, intermediate - removed)
        layers[layer] = {"removed_gate_up_channels": removed, "state": shapes, **counts,
                         "net_gate_up_rows_saved_after_alignment_padding": actual_deleted_rows,
                         "eliminated_down_columns_retained_as_output_factor": "dropped_weight" in state}
        for key in totals:
            totals[key] += counts[key]
    original_selected_values = len(modules) * 3 * hidden * intermediate
    original_bytes = evaluation["timings"][baseline_key]["resident_model_tensor_bytes"]
    # The selected original Qwen MLP projections are BF16 and have no biases.
    replacement_bytes = sum(totals[k] for k in ("weight_bytes", "normalization_bytes", "index_bytes"))
    combined_bytes = original_bytes - original_selected_values * 2 + replacement_bytes + outside_buffer_bytes
    measured = evaluation["timings"][arm]["resident_model_tensor_bytes"]
    if combined_bytes != measured:
        raise ValueError("Checkpoint accounting differs from measured resident tensor bytes")
    result = {
        "scientific_fingerprint": plan["scientific_fingerprint"],
        "original_model_parameters": manifest["model_parameters"],
        "original_selected_mlp_weight_values": original_selected_values,
        "replacement": totals,
        "combined_learned_weight_values_including_tensor_buffers":
            manifest["model_parameters"] - original_selected_values + totals["weight_values"],
        "weight_count_scope": "Includes fitted/derived polynomial and output-bias buffers; excludes normalization and indices.",
        "original_resident_model_tensor_bytes": original_bytes,
        "combined_resident_model_tensor_bytes": combined_bytes,
        "additional_buffer_bytes_outside_replaced_layers": outside_buffer_bytes,
        "resident_tensor_bytes_saved": original_bytes - combined_bytes,
        "replacement_checkpoint_file_bytes": (root / "compressed_modules.pt").stat().st_size,
        "replacement_checkpoint_sha256": sha(root / "compressed_modules.pt"),
        "checkpoint_scope": "Replacement-module delta; requires pinned original Qwen and fit/coverage metadata.",
        "layers": layers,
    }
    write(root / "storage_audit.json", result)
    return result


def redundancy():
    fit = read(PARENT / "fit.json")
    layers = {}
    histogram = [0] * 4
    for layer, item in fit.items():
        r = item["redundancy"]
        pairwise = r["pairwise_shared_counts"]
        pair_count = sum(pairwise[i][j] for i in range(4) for j in range(i + 1, 4))
        all_four, at_least_two = r["shared_by_all_tasks"], r["shared_by_at_least_two_tasks"]
        # Invert the subset-count identities to recover the membership histogram.
        three_numerator = pair_count - at_least_two - 5 * all_four
        assert three_numerator % 2 == 0
        three = three_numerator // 2
        two = at_least_two - three - all_four
        one = r["distinct_shared_elements"] - at_least_two
        counts = [one, two, three, all_four]
        assert min(counts) >= 0
        assert sum((k + 1) * n for k, n in enumerate(counts)) == r["total_task_occurrences"]
        assert sum(counts) == r["distinct_shared_elements"]
        layers[layer] = {**r, "elements_used_by_exactly_k_tasks": counts}
        histogram = [a + b for a, b in zip(histogram, counts)]
    totals = {key: sum(item[key] for item in layers.values()) for key in (
        "total_task_occurrences", "distinct_shared_elements", "redundant_occurrences",
        "shared_by_at_least_two_tasks", "shared_by_all_tasks")}
    result = {"layers": layers, "totals": totals,
              "elements_used_by_exactly_k_tasks": histogram,
              "redundant_fraction": totals["redundant_occurrences"] / totals["total_task_occurrences"],
              "scope": "Common graph generators counted within each layer; no cross-layer identification. "
                       "Task support means development NRMSE <=0.5, not exact neural identities."}
    write(PARENT / "basis_redundancy_summary.json", result)
    return result


def plot(reuse):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#d6dde4", "#89b5d4", "#377aa8", "#183f60"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    keys = sorted(reuse["layers"], key=int)
    bottoms = [0] * len(keys)
    for k, color in enumerate(colors):
        counts = [reuse["layers"][key]["elements_used_by_exactly_k_tasks"][k] for key in keys]
        axes[0].bar(keys, counts, bottom=bottoms, color=color, label=f"{k+1} task{'s' if k else ''}")
        bottoms = [a + b for a, b in zip(bottoms, counts)]
    axes[0].set(xlabel="MLP block (zero based)", ylabel="Distinct qualifying generators",
                title="How broadly is each common relation reusable?")
    axes[0].legend(frameon=False, ncol=2)
    totals = reuse["totals"]
    axes[1].bar(["All task uses", "Distinct generators", "Repeated uses"],
                [totals["total_task_occurrences"], totals["distinct_shared_elements"], totals["redundant_occurrences"]],
                color=["#89b5d4", "#377aa8", "#183f60"])
    for i, value in enumerate([totals["total_task_occurrences"], totals["distinct_shared_elements"], totals["redundant_occurrences"]]):
        axes[1].text(i, value + 60, f"{value:,}", ha="center")
    axes[1].set(title="18.8% of task uses reuse an element", ylim=(0, 4200), ylabel="Count")
    axes[1].tick_params(axis="x", labelsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Shared task-Jacobian chart → polynomial graph basis", fontsize=14)
    fig.text(.5, .015, "4 tasks · 3 MLP blocks · 73 generators qualify on all 4 tasks · approximate NRMSE ≤ 0.5", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .93))
    for extension in ("png", "svg"):
        fig.savefig(PARENT / f"basis_redundancy.{extension}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stacked-only", action="store_true")
    args = parser.parse_args()
    base.cpu_only()
    if args.stacked_only:
        from .jspace_bench_stacked import DEFAULT_ROOT as STACKED
        # Each of the other 33 generic stacked modules owns 44 bytes of dummy
        # index/normalization buffers; its weight count is unchanged.
        result = storage_audit(STACKED, "stacked", "stacked_polynomial_core", "original_before", 33*44)
        print(json.dumps({"stacked_saved_bytes": result["resident_tensor_bytes_saved"]}, allow_nan=False))
        return
    parent = storage_audit(PARENT, "evaluation", "compressed")
    covered = storage_audit(COVERED, "covered", "factorized")
    reuse = redundancy()
    plot(reuse)
    print(json.dumps({"parent_saved_bytes": parent["resident_tensor_bytes_saved"],
                      "covered_saved_bytes": covered["resident_tensor_bytes_saved"],
                      "reuse": reuse["totals"]}, allow_nan=False))


if __name__ == "__main__":
    main()
