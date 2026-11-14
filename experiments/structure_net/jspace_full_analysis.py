"""Postprocess the frozen full-model pass without selecting another checkpoint."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from . import jspace_full_capture as base
from .jspace_full_affine_v2 import DEFAULT_ROOT
from .jspace_full_attribution import DEFAULT_ROOT as ATTRIBUTION_ROOT,ARMS
from .jspace_bench_data import DATASETS,sha,write

EXPORT_ROOT=base.ROOT/"data/experiments/jspace_full_compact/20261113_affine_export_v2"
V1_ROOT=base.ROOT/"data/experiments/jspace_full_compact/20261113_v1"
REFINEMENT_ROOT=base.ROOT/"data/experiments/jspace_full_compact/20261113_refinement_v1"
DIAGNOSTIC_ROOT=base.ROOT/"data/experiments/jspace_full_compact/20261113_projection_diagnostic_v1"


def analyze(root):
    base.cpu_only();plan=base.verify(root)
    result=json.loads((root/"result.json").read_text())
    fit=json.loads((Path(plan["fit_root"])/"fit.json").read_text())
    original=json.loads((root/"controls/original_evaluation.json").read_text())
    final=json.loads((root/"train/adapted_36_evaluation.json").read_text())
    if [v["identity"] for v in original]!=[v["identity"] for v in final]:raise ValueError("Unpaired answers")
    a=np.array([v["prediction"]==v["answer"] for v in original]);b=np.array([v["prediction"]==v["answer"] for v in final])
    if int(b.sum())!=result["trajectory"]["36"]["evaluation"]["correct"]:raise ValueError("Accuracy mismatch")
    from scipy.stats import binomtest
    wins=int((~a&b).sum());losses=int((a&~b).sum());discordant=wins+losses
    reused=[v["redundancy"] for v in fit["layers"].values()]
    uses=sum(v["task_uses"] for v in reused);repeats=sum(v["redundant_occurrences"] for v in reused)
    basis={"core_rank":1024,"retained_residual_rank":1024,"global_jacobian_energy_fraction":fit["global_core_energy_fraction"],
        "jacobian_target":"correct-option logit minus mean of other option logits",
        "sampled_calibration_token_positions_per_case":16,
        "per_layer_jacobian_energy_fraction":fit["jacobian_energy_fractions"],"task_uses":uses,
        "unique_elements":sum(v["unique_elements"] for v in reused),"redundant_occurrences":repeats,"redundant_fraction":repeats/uses,
        "shared_all_four":sum(v["shared_all_four"] for v in reused),"removed_nonlinear_channels":36*1024,
        "normal_form_features_per_layer":153,"scope":"Reuse counted across four tasks separately within each layer; threshold NRMSE <=0.5"}
    paired={"gains":wins,"losses":losses,"unchanged_correct":int((a&b).sum()),"unchanged_wrong":int((~a&~b).sum()),
        "net_correct_change":int(b.sum()-a.sum()),"exact_mcnemar_two_sided_p":float(binomtest(wins,discordant,.5).pvalue) if discordant else 1.,
        "scope":"descriptive paired comparison on a reused exploratory evaluation pool"}
    mlp_rows=json.loads((REFINEMENT_ROOT/"mlp_runtime/mlp_runtime_evaluation.json").read_text())
    if [v["identity"] for v in original]!=[v["identity"] for v in mlp_rows]:raise ValueError("Unpaired MLP answers")
    m=np.array([v["prediction"]==v["answer"] for v in mlp_rows])
    mlp_paired={"gains":int((~a&m).sum()),"losses":int((a&~m).sum()),"net_correct_change":int(m.sum()-a.sum())}
    reload=json.loads((EXPORT_ROOT/"reload/verification.json").read_text())
    if reload["all_256_records_exact"] is not True:raise ValueError("Reload failed")
    attribution={arm:json.loads((ATTRIBUTION_ROOT/arm/"result.json").read_text()) for arm in ARMS}
    losses=[]
    for layer in range(36):losses+=json.loads((root/f"train/losses_{layer:02d}.json").read_text())
    local=np.array([.1*v["local_relative_mse"] for v in losses]);functional=np.array([2.8*v["kl"]+.3*v["ce"] for v in losses])
    training={"steps":len(losses),"unique_questions":len({v["identity"] for v in losses}),
        "selected_token_targets":len(losses)*4,"local_term_dominates_fraction":float((local>functional).mean()),
        "median_weighted_local_loss":float(np.median(local)),"median_weighted_functional_loss":float(np.median(functional)),
        "maximum_local_relative_mse":max(v["local_relative_mse"] for v in losses)}
    aggregate={"study_id":base.STUDY_ID,"status":"completed","confirmed":False,"created_at":datetime.now(timezone.utc).isoformat(),
        "result_sha256":sha(root/"result.json"),"result":result,"basis":basis,"paired_comparison":paired,"checkpoint_reload":reload,
        "export_root":str(EXPORT_ROOT),"export":json.loads((EXPORT_ROOT/"export.json").read_text()),
        "attribution_root":str(ATTRIBUTION_ROOT),"attribution":attribution,"training_diagnostics":training,
        "attribution_scope":"v1 before value-mean correction; development-only decomposition",
        "v1":{"root":str(V1_ROOT),"result_sha256":sha(V1_ROOT/"result.json"),"result":json.loads((V1_ROOT/"result.json").read_text())},
        "refinement":{"root":str(REFINEMENT_ROOT),"mlp_paired_comparison":mlp_paired,**{a:json.loads((REFINEMENT_ROOT/a/"result.json").read_text()) for a in ("attention_identity","mlp_runtime")}},
        "projection_diagnostic":{"root":str(DIAGNOSTIC_ROOT),"result_sha256":sha(DIAGNOSTIC_ROOT/"result.json"),"result":json.loads((DIAGNOSTIC_ROOT/"result.json").read_text())},
        "source_sha256":{str(Path(__file__)):sha(__file__)},"scope":"one-seed all-layer engineering experiment, not independent confirmation"}
    write(root/"campaign_results.json",aggregate)
    plot(root,aggregate)
    report(root,aggregate)
    return {"accuracy":result["trajectory"]["36"]["evaluation"],"matched_graph_speedup":result["matched_graph_speedup"],
            "tensor_reduction":result["tensor_byte_reduction_fraction"],"basis":basis,"paired":paired}


def plot(root,aggregate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r=aggregate["result"];controls=r["controls"];mlp=aggregate["refinement"]["mlp_runtime"]
    names=["Original","Rotated","Compact\ninit.","Adapted","MLP only"]
    values=[controls[k]["evaluation"]["correct"] for k in ("original","rotated","initialized_compact")]+[r["trajectory"]["36"]["evaluation"]["correct"],mlp["evaluation"]["correct"]]
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout="constrained")
    colors=["#657b83","#80a4a8","#c7974c","#267d77","#8a78a6"]
    bars=axes[0,0].bar(names,np.array(values)/256*100,color=colors)
    axes[0,0].bar_label(bars,labels=[f"{v}/256" for v in values],padding=3)
    axes[0,0].set_ylim(0,100);axes[0,0].set_ylabel("Accuracy (%)");axes[0,0].set_title("Four-task answers; reused evaluation")
    original=controls["original"]["runtime"]["timings"];adapted=r["adapted_runtime"]["timings"]
    x=np.arange(2)
    for offset,source,label,color in ((-.26,original,"Original",colors[0]),(0,adapted,"Adapted",colors[3]),(.26,mlp["runtime"]["timings"],"MLP only",colors[4])):
        bars=axes[0,1].bar(x+offset,[source[a]["tokens_per_second"] for a in ("hf","cuda_graph")],width=.25,label=label,color=color)
        axes[0,1].bar_label(bars,fmt="%.1f",padding=3)
    axes[0,1].set_xticks(x,["HF eager","Matched CUDA graph"]);axes[0,1].set_ylabel("Generated tokens/s")
    axes[0,1].set_title("Batch one, 32 tokens, prefill included");axes[0,1].legend()
    sizes=[controls[k]["tensor_bytes"]/1e9 for k in ("original","rotated","initialized_compact")]+[r["final_tensor_bytes"]/1e9,mlp["tensor_bytes"]/1e9]
    bars=axes[1,0].bar(names,sizes,color=colors);axes[1,0].bar_label(bars,fmt="%.3f",padding=3)
    axes[1,0].set_ylabel("Resident tensor bytes (decimal GB)");axes[1,0].set_ylim(0,max(sizes)*1.15)
    axes[1,0].set_title("All learned weights and buffers")
    axes[1,1].plot(np.arange(36),np.array(aggregate["basis"]["per_layer_jacobian_energy_fraction"])*100,color=colors[-1])
    axes[1,1].set_ylim(0,100);axes[1,1].set_xlabel("Decoder layer (zero indexed)");axes[1,1].set_ylabel("Task-Jacobian energy retained (%)")
    axes[1,1].set_title("1,024 core + 1,024 retained residual coordinates")
    fig.suptitle("JSPACE-FULL-COMPACT-01 · Qwen2.5-3B · RTX 3060")
    fig.savefig(root/"full_compact_results.png",dpi=180);fig.savefig(root/"full_compact_results.svg");plt.close(fig)


def report(root,a):
    r=a["result"];c=r["controls"];old=a["v1"]["result"];mlp=a["refinement"]["mlp_runtime"]
    def score(x):return f"{x['correct']}/{x['count']} ({100*x['overall']:.2f}%)"
    def speed(x):return x["timings"]["cuda_graph"]["tokens_per_second"]
    rows=[("Native",c["original"]["evaluation"],c["original"]["tensor_bytes"],speed(c["original"]["runtime"])),
          ("Full-coordinate rotation",c["rotated"]["evaluation"],c["rotated"]["tensor_bytes"],None),
          ("v1 initialization; missing value mean",old["controls"]["initialized_compact"]["evaluation"],old["controls"]["initialized_compact"]["tensor_bytes"],None),
          ("v1 adapted",old["trajectory"]["36"]["evaluation"],old["final_tensor_bytes"],speed(old["adapted_runtime"])),
          ("Corrected v2 initialization",c["initialized_compact"]["evaluation"],c["initialized_compact"]["tensor_bytes"],None),
          ("Corrected v2 adapted",r["trajectory"]["36"]["evaluation"],r["final_tensor_bytes"],speed(r["adapted_runtime"])),
          ("All 36 MLPs compressed; native attention",mlp["evaluation"],mlp["tensor_bytes"],speed(mlp["runtime"]))]
    table="\n".join(f"| {name} | {score(metric)} | {size/1e9:.4f} | {rate:.2f} |" if rate is not None else
                    f"| {name} | {score(metric)} | {size/1e9:.4f} | — |" for name,metric,size,rate in rows)
    b=a["basis"];d=a["projection_diagnostic"]["result"]["rows"]
    trajectory="; ".join(f"{layer} layers: {score(v['evaluation'])}" for layer,v in r["trajectory"].items())
    component="\n".join(f"| {arm} | {score(v['accuracy'])} |" for arm,v in a["attribution"].items())
    family="\n".join(f"| {name} | {c['original']['evaluation']['by_task'][name]['correct']}/64 | {r['trajectory']['36']['evaluation']['by_task'][name]['correct']}/64 | {mlp['evaluation']['by_task'][name]['correct']}/64 |" for name in DATASETS)
    final=r["trajectory"]["36"]["evaluation"]
    conclusion=("The corrected full model preserves the measured native correct count." if final["correct"]>=c["original"]["evaluation"]["correct"] else
                "The corrected full model does not preserve native task accuracy at this budget.")
    text=f"""# Full-model functional-coordinate and polynomial-core experiment

Status: completed exploratory campaign. Study: `JSPACE-FULL-COMPACT-01`.
`confirmed=false`. Latest root: `{root.relative_to(base.ROOT)}`.
Aggregate SHA-256: `{sha(root/'campaign_results.json')}`.

{conclusion} The full 36-layer pass was executed and adapted, including a
correction to an affine-value omission in the first implementation. All versions
remain recorded. The MLP-only branch is a separate, untrained follow-up selected
for measurement from development diagnostics; it is not the fully compressed
attention model and is not an independent confirmation.

## Measured results

| Model | Evaluation accuracy | Resident tensor GB | Graph tok/s |
| --- | ---: | ---: | ---: |
{table}

GB is decimal. Throughput uses the same eight pilot prompts, batch one and 32
greedy generated tokens, including prefill. Both native and compressed models use
the same CUDA-graph runtime. Corrected v2's matched speed ratio is
**{r['matched_graph_speedup']:.4f}×**, and its tensor-byte reduction is
**{100*r['tensor_byte_reduction_fraction']:.2f}%**. The MLP-only speed ratio is
{mlp['matched_graph_speedup']:.4f}× against its contemporaneous native control;
that small difference is not a demonstrated speed improvement.
The MLP-only model gains {a['refinement']['mlp_paired_comparison']['gains']} answers
and loses {a['refinement']['mlp_paired_comparison']['losses']}; the five-answer net
drop does not mean only five predictions changed.

Corrected v2's ordinary HF generation measured
{r['adapted_runtime']['timings']['hf']['tokens_per_second']:.2f} tok/s versus
{c['original']['runtime']['timings']['hf']['tokens_per_second']:.2f} native.
Graph capture takes {r['adapted_runtime']['graph_capture_seconds']:.2f} seconds and
is amortized. The fast-path prefill scores
{score(r['adapted_runtime']['graph_prefill_accuracy'])}. Fixed-length timing
does not establish free-generation or summarization quality.

| Task | Native | Corrected v2 adapted | MLP-only |
| --- | ---: | ---: | ---: |
{family}

Corrected progressive checkpoints: {trajectory}. These measurements did not
select a checkpoint or change the training schedule. The paired final comparison
has {a['paired_comparison']['gains']} gained answers and
{a['paired_comparison']['losses']} lost answers; descriptive exact McNemar
two-sided p={a['paired_comparison']['exact_mcnemar_two_sided_p']:.4g} on this
reused evaluation pool.

## What was applied

The pinned Qwen2.5-3B-Instruct checkpoint has 36 layers and 3,085,938,688 native
parameters. A common orthogonal task-Jacobian/Ritz basis carries 1,024 functional
and 1,024 retained residual coordinates throughout the body. The Gramian uses
outer products of the correct-option-minus-mean-other-option margin gradient at
16 token positions per calibration case. This scalar task contrast is narrower
than the supplied recipe's full downstream-logit Jacobian; it does not measure
full-vocabulary generation sensitivity. RMS gains and
coordinate maps are folded into linear weights; the RMS denominator retains the
original 2,048-wide residual energy. Entry/exit rotations preserve the tied
vocabulary matrix. The complete-coordinate control is algebraically exact in
real arithmetic; its measured BF16 rounding effect is shown above.

Each attention block keeps one native GQA group and compiles the other eight
queries plus their shared K/V head. That group reads 1,024 dynamic input
coordinates with a mean anchor for the other coordinates, uses 48 RoPE frequency
pairs (96 coordinates), and retains the original 1/sqrt(128) score scale. All
causal token positions remain. Cache padding means this version saves no KV-cache
bytes. Corrected v2 restores the omitted value-mean output and packs both groups
into one attention call and one output projection. That fused call pads the
compact Q/K/V vectors back to 128 lanes, so it also forgoes the head-dimension
saving inside the attention matmuls; the incoming and outgoing projections still
use the compiled coefficients. This is an explicit runtime tradeoff, not a
demonstration of a fully 96-wide fused attention kernel.
Keeping the other native GQA group does not reconstruct the variable directions
discarded from the compiled group; those directions are mean-anchored here.

Each MLP removes 1,024 nonanchor nonlinear channels and compiles their contribution
through 153 degree-two normal-form features of 16 Jacobian-selected anchors.
The remaining 9,984 native nonlinear channels are retained. The monic graph
generators `y_i-p_i(z)` admit the relatively-prime-leading-term Gröbner criterion;
representative exact CAS checks accompany every layer. These certificates cover
the initialized rational surrogate. Relation defects in the neural activations
are measured approximations; adaptation learns the output coefficients. The
fixed 1,024-channel removal budget is separate from the cross-task eligibility
threshold: this run does not establish reliable coverage of every removed
channel. Retaining native channels is also not exact correction of every
discarded function.

Every layer receives 96 batch-one AdamW updates at 1e-4, for 3,456 updates per
variant. Only the current compact coefficients are trained. Its student prefix
is streamed from disk; future native layers remain frozen for continuation. Loss
is `0.7*T²*KL + 0.3*CE + 0.1*local_relative_MSE`, T=2, at three prompt positions
and the final option letter. This is a small adaptation experiment, not a large
language-model retraining campaign. The weighted local term dominates the
functional term on {100*a['training_diagnostics']['local_term_dominates_fraction']:.1f}%
of corrected-v2 updates; this is a training concern, not proof of its causal role.

## Data, reuse and interpretation

Use 512 training, 64 development and 256 evaluation questions across pinned ARC,
BoolQ, HellaSwag and WinoGrande files, limited to 384 prompt tokens. Splits are
disjoint. The evaluation pool is reused from the earlier pilot. Calibration uses
32 training examples per task and 16 token positions each. Teacher capture has
78,115 prompt tokens across train and development and stores all 37 boundaries
as exact BF16 bits. No external teacher calls were used.

The common 1,024-coordinate core retains {100*b['global_jacobian_energy_fraction']:.2f}%
of pooled Jacobian trace energy. Reuse at per-task reconstruction NRMSE <=0.5 is
{b['task_uses']:,} task uses, {b['unique_elements']:,} unique layer-specific elements,
and {b['redundant_occurrences']:,} repeated uses ({100*b['redundant_fraction']:.2f}%).
{b['shared_all_four']:,} elements qualify on all four tasks. Anchor definitions are
excluded. These are common-fit eligibility counts, not identical independently
fitted coefficients or identified semantic CAVs. Layer one contributes much of
the reuse; the aggregate is not directly comparable to the earlier three-layer
pilot's percentage.

A CPU diagnostic on 128 calibration final-token states finds that the compact
input branch replaces {100*min(v['input_remainder_energy_fraction'] for v in d):.1f}%–
{100*max(v['input_remainder_energy_fraction'] for v in d):.1f}% of squared deviation
from its anchor by the mean. High Jacobian trace retention therefore accompanies
a large finite input change. The diagnostic's pooled linearized quadratic-form
ratio is not a bound on the nonlinear model's finite error. The algebraic folding
control and the MLP-only result support continuing the retained-residual approach;
this global attention cut and short adaptation budget do not establish that the
same approximation remains valid through the entire stack.

## Failure localization and corrections

These development-only diagnostics concern v1, before the value-mean correction:

| Initialized component | Development accuracy |
| --- | ---: |
| Native | 48/64 |
{component}

The full-rank, full-head split-attention control scores
{score(a['refinement']['attention_identity']['evaluation'])}. Thus the split
implementation works at full width; the aggressive approximations are materially
different interventions. The value PCA in v1 omitted its affine mean correction,
dropping 70.87% and 81.89% of selected value-mean energy in its first two layers.
This was an implementation omission, not evidence against the supplied technique.
Corrected v2 restores `sum_h O_h (I-SS^T) mean(v)` and repeats the complete run.
Q/K frequency and input-coordinate omissions still require empirical validation.

The first fit also stopped on a transient reservation-readiness discrepancy.
Its retry changes only bounded readiness waiting and performs no GPU work while
waiting. The original root and failure record are preserved.

## Artifacts and reproduction

Seventeen CPU tests pass. The real GPU shakedowns include backward gradients,
frozen native weights, parameter reload and graph/uncaptured token agreement.
The corrected materialized checkpoint was loaded in a fresh reserved process;
all 256 evaluation records, including logits, match exactly.

The initial serializer copied the tied vocabulary weights twice. The separate
export restores that storage alias with every tensor unchanged. The corrected
export is {a['export']['export_file_bytes']:,} bytes, versus
{a['export']['source_file_bytes']:,} bytes for its preserved raw training file.
Checkpoint SHA-256: `{a['export']['export_sha256']}`.

- Corrected checkpoint: `{a['checkpoint_reload']['checkpoint']}`.
- MLP-only checkpoint: `{REFINEMENT_ROOT/'mlp_runtime/model.pt'}`.
- Protocol and amendments: `docs/07 - Status Reports/jspace/2026-11-13_jspace-full-compact-plan.md`,
  `2026-11-13_jspace-full-compact-reservation-amendment.md`, and `2026-11-13_jspace-affine-correction.md`.
- Producer snapshots, source/data hashes, loss histories and per-case results are
  retained in every dated root. These are custom PyTorch research checkpoints,
  loaded by the corresponding experiment driver.

All GPU stages use root locked `ncm` Pixi under inference-ctl leases. To create
and verify a fresh identical export without retraining, choose an unused root:

```bash
CUDA_VISIBLE_DEVICES='' pixi run --locked python -m experiments.structure_net.jspace_full_export \\
  prepare --parent data/experiments/jspace_full_compact/20261113_affine_v2 \\
  --root data/experiments/jspace_full_compact/20261113_affine_recheck
CUDA_VISIBLE_DEVICES='' pixi run --locked python -m experiments.structure_net.jspace_full_export \\
  export --root data/experiments/jspace_full_compact/20261113_affine_recheck
inference-ctl lease run --allocation 3060=11264 --minutes 60 \\
  --reason "JSPACE corrected export verification" -- \\
  pixi run --locked -e ncm python -m experiments.structure_net.jspace_full_affine_v2 \\
  export_reload --root data/experiments/jspace_full_compact/20261113_affine_recheck
```

Stages are append-only: use a new prepared root to repeat a completed stage.
The combined plot is `full_compact_results.png`/`.svg` in the latest root.
"""
    target=base.ROOT/"docs/08 - Analysis/jspace/2026-11-13_jspace-full-compact.md"
    if target.exists():raise FileExistsError(target)
    target.write_text(text)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    a=p.parse_args();print(json.dumps(analyze(a.root),allow_nan=False))
