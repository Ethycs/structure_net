#!/usr/bin/env python3
"""Validate MoBA raw evidence, render its measured report, and register read-back."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from experiments.structure_net import tinyllm_moba_distillation as m
from neural_architecture_lab.moba_distillation_hypothesis import MoBADistillationEvidence, store_evidence

DEFAULT_REPORT = Path("docs/08 - Analysis/2026-09-09_tinyllm-moba-distillation.md")


def render_figure(campaign, results, report):
    """Export a static research figure; seed spread is descriptive, not a CI."""
    if campaign["summary"]["completed"] != 40:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    destination = report.parent / "figures" / "2026-09-09_tinyllm-moba-distillation.png"
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    arms = ("LM", "KD", "route_mass", "route_contribution", "combined_mass", "combined_contribution")
    colors = dict(zip(arms, ("#777777", "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")))
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for arm in arms:
        selected = [r for r in results if r["arm"] == arm]
        curve = np.array([[h["metrics"]["logits_kl"] for h in r["history"]] for r in selected])
        axs[0, 0].plot(np.arange(5) * 8192, curve.mean(0), marker="o", ms=3, label=arm, color=colors[arm])
        axs[0, 0].fill_between(np.arange(5) * 8192, curve.min(0), curve.max(0), color=colors[arm], alpha=0.07)
        axs[0, 1].plot([1, 2, 4, 8], [statistics.mean(r["curve"][str(k)]["logits_kl"] for r in selected) for k in (1, 2, 4, 8)], marker="o", ms=3, color=colors[arm])
    axs[0, 0].axvline(24576, color="black", lw=0.8, ls="--")
    axs[0, 0].set(xlabel="Adaptation target tokens", ylabel="Teacher logits KL (monitor)", title="Adaptation; dashed line starts teacher-free finish")
    axs[0, 0].legend(fontsize=7, ncol=2)
    axs[0, 1].set(xlabel="Selected blocks, including local", ylabel="Teacher logits KL (test)", title="Final conversion curves", xticks=[1, 2, 4, 8])
    means = campaign["arm_means"]
    for i, arm in enumerate(arms):
        selected = [r for r in results if r["arm"] == arm]
        values = np.array([r["metrics"]["mass_recall"] for r in selected])
        axs[1, 0].bar(i, values.mean(), color=colors[arm], alpha=0.85)
        axs[1, 0].scatter(np.full(5, i), values, s=9, color="black", zorder=3)
    axs[1, 0].axhline(means["M0"]["mass_recall"], color="black", ls="--", lw=0.8, label="Zero-shot mean")
    axs[1, 0].set(xticks=range(len(arms)), xticklabels=arms, ylabel="Teacher mass recall, excluding local", title="Final k=2 routing recall; dots are adaptation seeds")
    axs[1, 0].tick_params(axis="x", rotation=25, labelsize=8)
    axs[1, 0].legend(fontsize=8)
    for seed in campaign["configuration"]["seeds"]:
        r = {r["arm"]: r["metrics"] for r in results if r["seed"] == seed}
        ratio = r["combined_mass"]["logits_kl"] / r["LM"]["logits_kl"]
        gain = r["combined_mass"]["mass_recall"] - r["M0"]["mass_recall"]
        axs[1, 1].scatter(ratio, gain, color=colors["combined_mass"], s=35)
        axs[1, 1].annotate(str(seed), (ratio, gain), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axs[1, 1].axvline(0.8, color="black", ls="--", lw=0.8)
    axs[1, 1].axhline(0.05, color="black", ls="--", lw=0.8)
    axs[1, 1].set(xlabel="Combined-mass KL / LM KL (pass ≤0.80)", ylabel="Mass recall gain over M0 (pass ≥0.05)",
                  title=f"Primary joint gate: {campaign['summary']['joint_passes']}/5; NLL also gated")
    for ax in axs.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("TinyLLM MoBA distillation — one pretrained checkpoint, five adaptation streams", fontsize=13)
    fig.savefig(destination, dpi=180)
    fig.savefig(destination.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def validate_campaign(path, expected_sha):
    if m.sha256(path) != expected_sha:
        raise RuntimeError("campaign hash mismatch")
    campaign = json.loads(path.read_text())
    if campaign.get("status") != "completed" or campaign.get("hypothesis_id") != m.HYPOTHESIS_ID or campaign.get("profile") != "PREREGISTERED":
        raise RuntimeError("requires completed preregistered evidence")
    config_data = dict(campaign["configuration"])
    for key in ("seeds", "betas", "evaluation_ks", "stage_ks"):
        config_data[key] = tuple(config_data[key])
    config = m.StudyConfig(**config_data)
    producer = campaign["producer"]
    identity = {key: producer[key] for key in ("source", "data", "configuration", "environment")}
    if m.digest(identity) != producer["fingerprint"]:
        raise RuntimeError("producer fingerprint mismatch")
    for relative, sha in producer["source"].items():
        if m.sha256(path.parent / "producer_source" / relative) != sha:
            raise RuntimeError("frozen source mismatch")
    results = []
    for artifact in campaign["results"]:
        result_path = Path(artifact["path"])
        if m.sha256(result_path) != artifact["sha256"]:
            raise RuntimeError("per-run result hash mismatch")
        r = json.loads(result_path.read_text())
        fingerprint = m.digest({"producer": producer["fingerprint"], "arm": r["arm"], "seed": r["seed"],
                                "schedule": r["artifacts"]["schedule"]["sha256"],
                                "schema": m.SCHEMA_VERSION, "hypothesis": m.HYPOTHESIS_ID})
        m.reuse_result(result_path, fingerprint)
        expected_id = f"{m.HYPOTHESIS_ID}/{r['arm']}/seed_{r['seed']}"
        if (r["configuration"] != campaign["configuration"] or r["experiment_id"] != artifact["experiment_id"]
                or r["experiment_id"] != expected_id or r["hypothesis_id"] != m.HYPOTHESIS_ID):
            raise RuntimeError("per-run configuration or identity mismatch")
        if r["checkpoint_reload_max_error"] != 0:
            raise RuntimeError("checkpoint lifecycle failed")
        expected_updates = 0 if r["arm"] in ("F", "M0") else 128
        if r["training_updates"] != expected_updates or r["training_tokens"] != expected_updates * 256:
            raise RuntimeError("fixed adaptation budget mismatch")
        results.append(r)
    ids = [r["experiment_id"] for r in results]
    if len(set(ids)) != len(ids) or len(ids) != campaign["summary"]["completed"]:
        raise RuntimeError("duplicate or missing direct evidence")
    if not campaign["teacher_unchanged"]:
        raise RuntimeError("teacher not frozen")
    if not all(p["pass"] and p["max_absolute_logit_error"] <= config.parity_tolerance for p in campaign["validity"]):
        classification = "invalid_full_budget_conversion"
    else:
        zeros = [r for r in results if r["arm"] == "M0"]
        if len(zeros) != len(config.seeds):
            raise RuntimeError("missing zero-shot control")
        if sum(r["metrics"]["logits_kl"] >= config.material_kl for r in zeros) < config.required_seeds:
            classification = "no_material_conversion_gap"
        else:
            expected_cells = {(arm, seed) for arm in m.ARMS for seed in config.seeds}
            if {(r["arm"], r["seed"]) for r in results} != expected_cells:
                raise RuntimeError("missing scheduled cells")
            gates = []
            for seed in config.seeds:
                arms = {r["arm"]: r for r in results if r["seed"] == seed}
                if len({r["artifacts"]["schedule"]["sha256"] for r in arms.values()}) != 1:
                    raise RuntimeError("unpaired schedules")
                gates.append({"seed": seed, **m.seed_gate(*(arms[a]["metrics"] for a in ("F", "M0", "LM", "combined_mass")), config)})
            if gates != campaign["per_seed_gates"]:
                raise RuntimeError("joint seed gates do not replay")
            classification = m.classify(gates, config)
    if classification != campaign["classification"] or campaign["confirmed"] != (classification == "routing_distillation_supported"):
        raise RuntimeError("classification does not replay")
    for arm, expected in campaign["arm_means"].items():
        selected = [r for r in results if r["arm"] == arm]
        for key, value in expected.items():
            actual = statistics.mean(r["blimp"]["accuracy"] if key == "blimp_accuracy" else r["metrics"][key] for r in selected)
            if abs(actual - value) > 1e-10:
                raise RuntimeError("aggregate mean does not replay")
    return campaign, results


def render_report(campaign, results, campaign_path, campaign_sha):
    c, summary = campaign["classification"], campaign["summary"]
    passed = summary["joint_passes"]
    lines = ["# TinyLLM MoBA teacher distillation", "",
             f"Status: completed. Date: 2026-09-09. Profile: PREREGISTERED.",
             f"Hypothesis: `{m.HYPOTHESIS_ID}`. Classification: **`{c}`**.",
             f"Primary joint gate: **{passed}/5** adaptation seeds; confirmation: **{str(campaign['confirmed']).lower()}**.", "",
             f"Artifact: `{campaign_path}`. SHA-256: `{campaign_sha}`.", "",
             "[Preregistration](../07%20-%20Status%20Reports/2026-09-09_tinyllm-moba-distillation-preregistration.md)", "",
             "## Measured comparison", "",
             "All trained arms started from the same native-256 d8 checkpoint and used 128 updates / 32768 target tokens per arm. The table reports seed means at k=2 (local block plus one routed completed past block). F uses full attention.", "",
             "| Arm | NLL | Perplexity | Teacher logits KL | Mass recall | Contribution recall | Attention MSE | Hidden MSE | BLiMP accuracy |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, a in campaign["arm_means"].items():
        lines.append(f"| {arm} | {a['nll']:.5f} | {a['perplexity']:.3f} | {a['logits_kl']:.6f} | {a['mass_recall']:.4f} | {a['contribution_recall']:.4f} | {a['attention_mse']:.6f} | {a['hidden_mse']:.5f} | {a.get('blimp_accuracy', 0):.2%} |")
    if summary["completed"] == 40:
        lines += ["", "![Measured adaptation, sparsity, routing, and joint-gate curves](figures/2026-09-09_tinyllm-moba-distillation.png)", ""]
        means = campaign["arm_means"]
        kl_change = 100 * (1 - means["combined_mass"]["logits_kl"] / means["LM"]["logits_kl"])
        kl_description = (f"reduced mean teacher-logit KL by {kl_change:.2f}%" if kl_change >= 0
                          else f"increased mean teacher-logit KL by {-kl_change:.2f}%")
        recall_gain = means["combined_mass"]["mass_recall"] - means["M0"]["mass_recall"]
        nll_change = 100 * (means["combined_mass"]["nll"] / means["F"]["nll"] - 1)
        lines += ["", f"Combined mass distillation {kl_description} relative to LM-only, changed routing recall by {100 * recall_gain:.2f} percentage points relative to zero-shot conversion, and changed NLL by {nll_change:.2f}% relative to F. These are descriptive ratios of means; the decision uses the paired seed gates below.", "",
                  "| Seed | KL ratio vs LM (≤0.80) | Recall gain vs M0 (≥0.05) | NLL ratio vs F (≤1.05) | KL | Recall | NLL | Joint |",
                  "| --- | ---: | ---: | ---: | --- | --- | --- | --- |"]
        for g in campaign["per_seed_gates"]:
            arms = {r["arm"]: r["metrics"] for r in results if r["seed"] == g["seed"]}
            a = arms["combined_mass"]
            flags = " | ".join("pass" if g[k] else "fail" for k in ("kl", "recall", "nll", "joint"))
            lines.append(f"| {g['seed']} | {a['logits_kl'] / arms['LM']['logits_kl']:.4f} | {a['mass_recall'] - arms['M0']['mass_recall']:.4f} | {a['nll'] / arms['F']['nll']:.4f} | {flags} |")
        lines += ["", "## Adaptation trajectory and sparsity", "",
                  "Monitoring windows are disjoint from final evaluation; no checkpoint was selected from these values. The last 32 updates use LM-only with the teacher removed.", "",
                  "| Arm | KL at update 0 | 32 | 64 | 96 | 128 | Final parameter deformation |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for arm in m.ARMS[2:]:
            selected = [r for r in results if r["arm"] == arm]
            values = [statistics.mean(r["history"][i]["metrics"]["logits_kl"] for r in selected) for i in range(5)]
            lines.append("| " + arm + " | " + " | ".join(f"{v:.5f}" for v in values) + f" | {statistics.mean(r['relative_parameter_deformation'] for r in selected):.5f} |")
        lines += ["", "| Arm | Teacher KL, k=1 | k=2 | k=4 | k=8 |", "| --- | ---: | ---: | ---: | ---: |"]
        for arm in m.ARMS[1:]:
            selected = [r for r in results if r["arm"] == arm]
            lines.append("| " + arm + " | " + " | ".join(f"{statistics.mean(r['curve'][str(k)]['logits_kl'] for r in selected):.5f}" for k in (1, 2, 4, 8)) + " |")
        lines += ["", "## Downstream control and mechanistic boundaries", "",
                  "BLiMP accuracy uses 128 fixed minimal pairs across four agreement suites and summed sentence log probability. These examples are shared across adaptation seeds; five evaluations are not five independent BLiMP datasets.", "",
                  "| Arm | Subject–verb 1 | Subject–verb 2 | Anaphor number | Determiner–noun | Sparse-active examples |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for arm in m.ARMS:
            selected = [r for r in results if r["arm"] == arm]
            per_suite = [statistics.mean(r["blimp"]["per_suite"][suite] for r in selected) for suite in m.BLIMP_HASHES]
            active = statistics.mean(r["blimp"]["sparsity_active_fraction"] for r in selected)
            lines.append("| " + arm + " | " + " | ".join(f"{v:.2%}" for v in per_suite + [active]) + " |")
        lines += ["", f"At k=2 the combined-mass arm used {means['combined_mass']['edge_fraction']:.2%} of causal attention edges and captured {means['combined_mass']['teacher_mass_captured']:.2%} of teacher attention mass (averaged over layers, heads, and queries). These are mask statistics; the reference implementation still computes dense tensors."]
        lines += ["", "| Arm | Mass routing KL | Contribution routing KL | Next-token accuracy | Teacher mass captured | Causal edges selected |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for arm, a in means.items():
            lines.append(f"| {arm} | {a['route_mass_kl']:.5f} | {a['route_contribution_kl']:.5f} | {a['next_token_accuracy']:.2%} | {a['teacher_mass_captured']:.2%} | {a['edge_fraction']:.2%} |")
    lines += ["", "## Campaign integrity and validity", "",
              f"Requested {summary['requested']} cells; completed {summary['completed']}; failed {summary['failed']}; retried {summary['retried']}; excluded {summary['excluded']}; reused {summary['reused']}; unexecuted by gate {summary['unexecuted_by_gate']}.",
              f"Full-budget parity passed {sum(p['pass'] for p in campaign['validity'])}/{len(campaign['validity'])}; maximum logit error {max(p['max_absolute_logit_error'] for p in campaign['validity']):.8g}, tolerance 1e-4. The teacher state hash remained unchanged. Saved checkpoints reloaded with exactly matching tensors and logits.", "",
              f"Producer fingerprint: `{campaign['producer']['fingerprint']}`. Producing Git base: `{campaign['producer']['git_head']}`. Uncommitted dependencies were hashed and copied under `producer_source/`; existing user work was preserved.",
              f"Environment: {campaign['producer']['environment']}. One sequential worker, physical GPU 2. Shared-device times are operational diagnostics, not benchmarks.", "",
              "Validation: 13 CPU numerical/experiment contract tests and four registry contract tests; eight-arm tiny CPU lifecycle and completed-result reuse; eight-arm representative CUDA lifecycle at the intended one-worker concurrency (peak allocated memory 1.64 GB). The initial CPU shakedown exposed a PyTorch argsort keyword incompatibility; its root was retained, the implementation was corrected before primary execution, and the full test/lifecycle ladder passed. No primary evidence was reused from that failed systems attempt.", "",
              "All direct result, checkpoint, schedule and preserved-source hashes were checked during finalization. The finalizer recomputed primary seed gates, classification and aggregate means from direct records.", "",
              "## Interpretation", ""]
    if campaign["confirmed"]:
        lines.append("The fixed experiment supports the operational conversion hypothesis for this checkpoint and adaptation budget. It does not establish an asymptotic or universal property of pretrained routing geometry.")
    else:
        lines.append("The full preregistered hypothesis is not confirmed. Secondary improvements and contribution-target comparisons remain informative but cannot change the failed primary gate.")
    if summary["completed"] == 40:
        means = campaign["arm_means"]
        route_drop = 100 * (1 - means["combined_mass"]["route_mass_kl"] / means["M0"]["route_mass_kl"])
        recall_change = means["combined_mass"]["mass_recall"] - means["M0"]["mass_recall"]
        lines += ["", f"The zero-shot router already recalled {means['M0']['mass_recall']:.2%} of teacher-important past blocks under the declared recall estimator. Combined mass distillation reduced soft-routing KL by {route_drop:.2f}% relative to zero-shot, while hard-block recall changed by {recall_change * 100:.2f} percentage points. The distinction between matching the soft routing distribution and improving discrete block selection is a direct measured finding; it does not establish the cause of the fidelity tradeoff."]
        if all(r["blimp"]["sparsity_active_fraction"] == 0 for r in results):
            lines += ["", "No BLiMP example activated sparsity at the tested budget: each sentence fit within the available two-block coverage. Thus BLiMP differences measure consequences of weight adaptation on short sentences, and provide no evidence for long-range sparse routing quality."]
    lines += ["", "The combined objectives and common unfreezing/curriculum schedule test a conversion package. Route-only versus LM isolates adding a routing loss within that schedule. Projected value norms are contribution proxies, not causal ablation measurements. Top-k uses teacher-important completed past blocks and excludes the forced local block from recall.", "",
              "This experiment covers one pretrained model, native length 256 and one small token budget. It does not establish orders-of-magnitude data savings, long-context quality, or accelerated inference. Short BLiMP examples primarily test weight adaptation rather than long-range routing. The parent validation suffix was reused with disjoint adaptation, monitoring and final-evaluation partitions; it is not historically unseen model-selection data.", "",
              "## Reproduction and evidence", "", "```bash",
              "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 pixi run python -m \\",
              "  experiments.structure_net.tinyllm_moba_distillation --device cuda:0 \\",
              "  --output data/experiments/tinyllm_moba_distillation/20260909_registered",
              "```", "",
              "This command hash-verifies and reuses completed evidence. A changed producer or protocol requires a new dated root. The parent checkpoint and token-stream SHA-256 values are fixed in the preregistration. Per-arm result files include k-curves, stage losses, BLiMP item scores, complete configuration, and checkpoint/schedule hashes.", "",
              "Sources: [MoBA](https://github.com/MoonshotAI/MoBA), [MoBA paper](https://arxiv.org/abs/2502.13189), [BLiMP](https://github.com/alexwarstadt/blimp).", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args(argv)
    campaign, results = validate_campaign(args.campaign, args.expected_sha256)
    render_figure(campaign, results, args.report)
    report = render_report(campaign, results, args.campaign, args.expected_sha256)
    if args.report.exists() and args.report.read_text() != report:
        raise RuntimeError("report already exists with different content; preserve and amend")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if not args.report.exists():
        args.report.write_text(report)
    evidence = MoBADistillationEvidence(campaign["classification"], campaign["summary"]["completed"],
                                        campaign["summary"]["joint_passes"], str(args.campaign), args.expected_sha256,
                                        str(args.report), m.sha256(args.report), tuple(r["experiment_id"] for r in results))
    if args.register:
        # The repository's Chroma 0.4 index still references this removed
        # NumPy 2 alias. Keep compatibility local to the registry process;
        # no training or measurement producer is changed.
        import numpy as np
        if "NaN" not in np.__dict__:
            np.NaN = np.nan
        from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
        logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",
            queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected",
            enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
        registry = store_evidence(evidence, logger, Path("data/hypothesis_registry/tinyllm-moba-distillation-v1.json"))
        print(json.dumps(registry["storage"]))
    print(json.dumps({"classification": evidence.classification, "confirmed": evidence.confirmed,
                      "completed_cells": evidence.completed_cells, "report": str(args.report)}))


if __name__ == "__main__":
    main()
