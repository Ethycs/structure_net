"""Separate actual summary accuracy from code learning and implementation parity."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import PARENT, ROOT, verify
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT / "20261022_generalization_diagnosis_v1"
REPORT = Path("docs/08 - Analysis/2026-10-22_ncm-generalization-diagnosis.md")
RUNS = ("20261022_current_ncm_grade_secondary_v1", "20261022_current_parent_grade_secondary_v1",
        "20261022_body_ablation_measure_v1", "20261022_released_xsum_control_v2")


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh append-only diagnosis required")
    ncm, parent, ablation, released = [q.read(ROOT / name / "result.json") for name in RUNS]
    if any(r["status"] != "completed" or r["valid"] != 64 or r["invalid"] for r in (ncm, parent)):
        raise ValueError("complete matched secondary grading required")
    if ncm["passed"] != 0 or parent["passed"] != 50:
        raise ValueError("recorded task outcomes differ")
    if (ablation["status"] != "completed_training_channel_diagnostic" or ablation["cases"] != 128
            or not ablation["anchor_exact"] or not ablation["reload_exact"]):
        raise ValueError("complete matched channel diagnostic required")
    if released["status"] != "completed_inference_path_control" or not released["author_local_exact"] or released["cases"] != 32:
        raise ValueError("complete released evaluator parity required")
    paths = [Path(__file__), Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for name in RUNS:
        root = ROOT / name
        verify(q.read(root / "producer.json"))
        paths.extend(p for p in root.glob("*") if p.is_file())
    teacher = ROOT / "20261022_teacher_v1/train.json"
    selection = PARENT / "selection.json"
    original = {r["id"]: r for r in q.read(selection)["train"]}
    rows = q.read(teacher)
    alignment = {k: sum(r[k] == original[r["id"]][k] for r in rows)
                 for k in ("source", "source_sha256", "parent_ids")}
    if len(rows) != 1999 or any(n != 1999 for n in alignment.values()):
        raise ValueError("original cached-feature/source correspondence failed")
    paths.extend([teacher, selection, ROOT / "20261022_combined_primary_pilot_v1/result.json"])
    failed = ROOT / "20261022_released_xsum_pilot_v1"
    paths.extend(path for path in failed.glob("*") if path.is_file())
    DEST.mkdir()
    aggregate = {"schema": "nal.ncm-generalization-diagnosis.v1", "secondary_ncm": ncm,
        "secondary_parent": parent, "training_channel_ablation": ablation, "released_path_control": released,
        "training_feature_source_alignment": alignment, "primary_quality_pending": True,
        "joint_judge_task_accuracy": None, "final_test_opened": False, "speed_tested": False,
        "underpowered": True, "confirmed": False, "artifacts": {str(p): file_hash(p) for p in paths}}
    write_json(DEST / "campaign_results.json", aggregate)
    table = "\n".join(f"|{arm}|{m['coarse_correct']}/512={100*m['coarse_accuracy']:.4f}%|"
                      f"{m['nonpad_correct']}/4,477={100*m['nonpad_reconstruction']:.4f}%|{m['exact_target_texts']}/128|"
                      for arm, m in ablation["metrics"].items())
    REPORT.write_text(f'''# Why the current frozen-body NCM summarizer is failing

Date: 2026-10-22. The final5,120-update NCM continuation still fails actual
complete-summary grading. Training reconstruction and code accuracy are not
summary accuracy. **The95% task goal and any speedup claim remain unestablished.**

| Model, same64 source-disjoint development passages | Complete summaries passing Terra | Wilson95% interval |
| --- | --- | --- |
| Final NCM continuation |0/64=0%|0–5.6624%|
| Original full Llama-3.2-3B-Instruct |50/64=78.125%|66.5672–86.4977%|

All128 judgments were valid, with zero retries. Both use the unchanged rubric:
every material claim supported, salient main information, readable, nonempty,
and at most128cl100ktokens. Neither uses exact-reference matching. NCM failed
all three semantic flags on all64 cases. Parent failures include13faithfulness,
one salience and one readability flag; these overlap. Examples include changing
an optional advertising suggestion into a definite plan, generalizing one
item's properties to every item in a list, and reversing story chronology.
This is model-based evaluation, not independently collected human gold.

Parent generation used the same source prompts and all28frozen layers, with
its original output head and96-Llama-token cap. NCM has a48-GPT2-token canvas.
Output budgets therefore differ; this is a task-quality reference, not a
matched speed benchmark. Parent performance is not a logical upper bound on a
student taught by a stronger teacher, but preserving this reference alone would
not satisfy95%. The original parent body and its fully connected layers have
not been retrained.

The intended primary Sonnet quality runs are **still pending**. A fresh combined
four-judge/four-teacher systems pilot returned three valid judgments and one
invalid after its schema/consistency retry cap; it did not pass. Primary quality
therefore waits for the existing broader-label job to finish. No joint-judge
accuracy or final-test result is reported here. The final256-case test is sealed.

## The model uses both source channels, but does not generalize

The completed follow-up isolates each input on128hash-selected training cases.
All conditions use the exact final checkpoint, frozen polished codec, BF16,
batch4, coarse AR and the same discrete generation procedure. Changed native
inputs carry their own padding masks; cyclic shuffles preserve channel marginals.
Zero and disabled-body conditions are out-of-distribution stress tests.

| Input condition | Correct coarse codes | Correct target text tokens | Exact training texts |
| --- | --- | --- | --- |
{table}

The matched128-case sample reconstructs98.1237% of training tokens while every
held-out complete summary fails. Shifting Llama features reduces reconstruction
to20.7505%, and shifting native text reduces it to6.0308%. This establishes
dependence on both channels for training predictions. It does not establish
semantic transfer from Llama or prove the one-vector interface is sufficient.
Strong memorization/task-transfer failure is supported; ignored Llama input
does not explain these results. More data alone has not been proved a cure.

The four-case lifecycle pilot and full run reproduced original training codes
and text tokens exactly and reloaded exactly. All codec, generator and adapter
weights remained fixed. All1,999training sources, source hashes and parent
prompt IDs match their original cached-feature records. The development feature
path uses the same parent input construction before right-packing its valid
hidden states; final-valid-state extraction handles both padding directions.

## The released evaluator is consistent with the local inference path

The untouched original XSum model and the untouched author's evaluator produce
the same32/32 output texts as our local FP32 path on the author's first32 test
inputs. All outputs are nonempty; exact reference matches are
{released['exact_references']}/32. Data are pinned to official revision
`6580ba41a4b0a976dde876fbe84d25327617b0e6` with published LFS SHA256 verification.
This control explicitly uses the author's first512tokens, with no task-transfer
finetuning or Llama bridge. Original models remain unchanged.

The first four inspected outputs still show factual and wording defects, such
as confusing which football club appointed a director and producing a malformed
description of a Chicago kidnapping case. These examples are qualitative,
not a measured XSum task-accuracy rate. **Exact evaluator parity is an
implementation control, not evidence of95% faithful summarization.** The
author's released evaluation reports ROUGE/perplexity rather than our full
faithfulness/readability/salience pass criterion.
[Official release and evaluation instructions](https://github.com/HazyResearch/ncm).

An initial author-subprocess pilot stopped before model loading because the
existing NCM dependency path was not forwarded. Its log/producer remain intact.
Fresh v2 pilot/control forward that existing path without modifying vendor
code, packages, checkpoints or data. Both passed their declared parity checks.

The recent codec polish reaches99.7603% training-token reconstruction with
known correct codes, and the residual-capacity oracle can repair over92% of
token losses after nearby coarse-code changes when it knows the target latent.
Neither gives the generator the right facts on a new passage. The next training
experiments need broader audited teacher supervision and explicit tests of
error-conditioned finer-code correction, with real development accuracy as
the deciding endpoint. These are proposed remedies, not established outcomes.

Aggregate: `{DEST / 'campaign_results.json'}`; SHA256
`{file_hash(DEST / 'campaign_results.json')}`. Raw predictions, first-valid
judgments, producer/source copies, checkpoint seals, public-data receipts,
failed pilot and exact reload records remain in their immutable run roots.
Registry records for the source-channel diagnostic and released-path control
remain confirmed=false and underpowered=true; the two-judge task campaigns
remain unfinished. The prototype's512source/48target-token range still does
not cover the full application limits.
''')
    print(json.dumps({"status": "diagnosis_recorded", "report": str(REPORT)}), flush=True)


def register():
    register_records(DEST / "campaign_results.json", REPORT, DEST, {
        "ncm-body-contribution-v1": {"name": "Frozen NCM source-channel contribution",
            "description": "The final NCM uses both native text and frozen Llama features on128training cases; training token reconstruction98.12percent coexists with0of64held-out complete summaries passing the secondary judge.",
            "confirmation_status": "training_channel_dependence_only", "task_accuracy_measured": False},
        "ncm-released-xsum-control-v1": {"name": "Released NCM author inference parity",
            "description": "On32pinned released XSum inputs, the unchanged author's evaluator and local FP32 path agree exactly. This checks implementation parity, not faithful summary accuracy.",
            "confirmation_status": "inference_path_control_only", "task_accuracy_measured": False},
    })
    print(json.dumps({"status": "registered_and_read_back", "studies": 2}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
