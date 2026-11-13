"""Descriptive plots and paired examples from the frozen attention transport pilot."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .jspace_attention_study import DEFAULT_ROOT, verify, checked
from .jspace_causal_components import cpu_only
from .jspace_relation_study import write_json, file_hash
from .jspace_tasks import wilson_interval


def main(root=DEFAULT_ROOT):
    cpu_only()
    verify(root)
    checked(root, "evaluation")
    result = json.loads((root/"result.json").read_text())
    trace = json.loads((root/"trace.json").read_text())
    records = json.loads((root/"evaluation/records.json").read_text())
    pairs = json.loads((root/"tasks.json").read_text())["pairs"]["evaluation"]
    method = result["selected_origin"]["method"]
    eligible = [r for r in records if r["source"]["original_correct"] and r["donor"]["original_correct"]]
    intervals = {arm: wilson_interval(value["target_correct"], value["eligible_total"]) for arm,value in result["arms"].items()}
    split_counts = {}
    for group in ("seen", "heldout"):
        rows = [r for r,p in zip(records,pairs) if p["source"]["composition_heldout"] == (group == "heldout")]
        clean = [r for r in rows if r["source"]["original_correct"] and r["donor"]["original_correct"]]
        split_counts[group] = {"total": len(rows), "teacher_correct": sum(r["source"]["original_correct"] for r in rows),
                               "eligible": len(clean), "target_counts": {a:sum(r["edits"][a]["target_correct"] for r in clean) for a in result["arms"]}}
    examples = []
    for outcome in (True, False):
        rows = [(r,p) for r,p in zip(records,pairs) if r["source"]["original_correct"] and r["donor"]["original_correct"] and r["edits"][method]["target_correct"] == outcome]
        for row,pair in rows[:3]:
            examples.append({"case_id": row["source"]["case_id"], "country": pair["source"]["intermediate"], "donor_country": pair["source"]["swap_to"],
                             "attribute": pair["source"]["attribute"], "source_answer": row["source"]["answer"], "donor_answer": row["donor"]["answer"],
                             "expected_target": pair["source"]["swap_answer"], "edited_answers": {a:r["answer"] for a,r in row["edits"].items()}, "J_target_correct": outcome})
    diag = {"result_sha256": file_hash(root/"result.json"), "trace_sha256": file_hash(root/"trace.json"),
            "teacher_interval": wilson_interval(result["teacher"]["correct"], result["teacher"]["total"]),
            "intervention_intervals": intervals, "groups": split_counts, "examples": examples,
            "example_selection": "first three eligible J successes and failures in frozen shuffled evaluation order",
            "J_patch_medians": {key: float(np.median([r["edits"][method]["patch"][key] for r in records])) for key in
                                ("requested_l2", "actual_l2", "relative_l2", "rounding_relative_error", "actual_outside_J_relative_norm")}}
    write_json(root/"diagnostics.json", diag)
    curve = trace["curve"]
    positions = np.arange(len(curve))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.7), layout="constrained")
    arms = list(result["arms"])
    rates = [100*result["arms"][a]["target_success"] if eligible else 0 for a in arms]
    ax[0].bar(np.arange(len(arms)), rates, color=["#777777", "#007f8b", "#bbbbbb", "#ab7031", "#6b559b"])
    ax[0].set_xticks(np.arange(len(arms)), [a.replace("_", "\n") for a in arms], fontsize=8)
    ax[0].set_ylim(0, 105)
    ax[0].set_ylabel("Target answers (%)")
    ax[0].set_title(f"Attention intervention ({len(eligible)} eligible worlds)")
    for i,a in enumerate(arms):
        ax[0].text(i, rates[i]+2, str(result["arms"][a]["target_correct"]), ha="center", fontsize=9)
    ax[1].plot(positions, [100*r["country_correct"]/r["total"] for r in curve], label="Actual clean country", color="#007f8b")
    ax[1].plot(positions, [100*r["patched_country_target_correct"]/r["total"] for r in curve], label="Actual edited target", color="#c45a28")
    origin = result["origin_probe"]
    ax[1].scatter([-1], [100*origin["country_correct"]/origin["total"]], color="#007f8b", marker="x")
    ax[1].scatter([-1], [100*origin["patched_target_correct"]/origin["total"]], color="#c45a28", marker="x")
    ax[1].axhline(25, color="gray", linestyle=":", label="Uniform chance")
    ax[1].set_ylim(0, 105)
    ax[1].set_ylabel("Country probe accuracy (%)")
    ax[1].set_title("Readability through the intact model")
    ax[1].legend(fontsize=8)
    for kind,color in (("linear", "#007f8b"), ("quadratic", "#c45a28")):
        values = [r["methods"][kind]["intervention_delta"]["nrmse"] for r in curve]
        ax[2].plot(positions, [v if v is not None else np.nan for v in values], label=kind, color=color)
    ax[2].plot(positions, [r["carry_delta"]["nrmse"] for r in curve], label="Carry initial change", color="gray", linestyle="--")
    ax[2].axhline(1, color="gray", linestyle=":")
    ax[2].axvline(7, color="black", linestyle=":", linewidth=1)
    ax[2].set_yscale("symlog", linthresh=1)
    ax[2].set_ylabel("Normalized intervention-change error")
    ax[2].set_title("Composed predictions; dotted = primary endpoint")
    ax[2].legend(fontsize=8)
    ticks = sorted(set([0, 7, len(curve)-1]+list(range(1,len(curve),6))))
    for a in ax[1:]:
        a.set_xticks(ticks, [curve[i]["node"].replace("residual", "post-A").replace("block", "post-MLP") for i in ticks], rotation=45, ha="right", fontsize=8)
        a.set_xlabel("Zero-based layer and residual boundary")
    ax[1].set_xticks([-1]+ticks, [origin["node"]]+[curve[i]["node"].replace("residual", "post-A").replace("block", "post-MLP") for i in ticks], rotation=45, ha="right", fontsize=8)
    fig.suptitle(f"J-space attention transport: Qwen2.5-3B, layer {result['selected_origin']['layer']}, {method}", fontsize=13)
    for extension in ("png", "svg"):
        fig.savefig(root/f"attention_transport.{extension}", dpi=160)
    plt.close(fig)
    write_json(root/"presentation_manifest.json", {"scope": "post-evaluation descriptive presentation, no selection changes",
               "source_sha256": file_hash(__file__), "artifacts": {name:file_hash(root/name) for name in
                ("result.json", "trace.json", "diagnostics.json", "attention_transport.png", "attention_transport.svg")}})
    return diag


if __name__ == "__main__":
    print(json.dumps(main(), allow_nan=False))
