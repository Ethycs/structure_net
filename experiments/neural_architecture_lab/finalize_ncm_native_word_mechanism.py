"""Archive the crossed diagnostic without treating oracle recovery as accuracy."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_native_word_mechanism as m
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT / "20261023_native_word_mechanism_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-word-mechanism-and-upstream-actions.md")


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh finalization required")
    paths = [Path(__file__), Path(m.__file__), m.PLAN, Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in (m.PILOT, m.DEST):
        if m.run.q.read(root / "worker_terminal_receipt.json")["exit_code"] != 0:
            raise ValueError("both diagnostic workers must be observed terminal")
        verify(m.run.q.read(root / "producer.json"))
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    result = m.run.q.read(m.DEST / "result.json")
    rows = m.run.q.read(m.DEST / "outputs.json")
    if (len(rows) != 8 or result["metrics"] != m.aggregate(rows)
            or file_hash(m.DEST / "outputs.json") != result["outputs_sha256"]
            or not result["fixed_A_replay_exact"] or not result["model_and_selected_cache_unchanged"]):
        raise ValueError("fixed diagnostic replay differs")
    classification = "parallel_word_recovery_failed_even_with_oracle_plan"
    aggregate = {"schema": "nal.ncm-native-word-mechanism-results.v1", "classification": classification,
        "result": result, "task_accuracy_measured": False, "NCM_success_established": False,
        "final_test_opened": False, "speed_tested": False, "confirmed": False,
        "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(paths)}}
    DEST.mkdir(); campaign = DEST / "campaign_results.json"; write_json(campaign, aggregate)
    table = []
    for cohort, arms in result["metrics"].items():
        for arm, value in arms.items():
            table.append(f"| {cohort} | {arm} | {'Target-derived oracle' if value['oracle_plan'] else 'Source-predicted'} | {'True previous words' if value['teacher_forced'] else 'Four parallel refinements'} | {value['correct']}/{value['total']} ({100*value['token_recovery']:.2f}%) |")
    REPORT.write_text('''# The word stage fails even when the coarse plan is supplied

Date: 2026-10-23. Exploratory hypothesis `ncm-native-word-mechanism-v1`.
Classification `parallel_word_recovery_failed_even_with_oracle_plan`.
The actual source-disjoint summary endpoint remains **0/64 whole-summary
passes**, with64valid Terra judgments and0readable summaries.

The final failed checkpoint was tested on the first four fixed summary
TRAIN cases and first four raw-language diagnostic cases. The table measures
valid-position token recovery, **not summary accuracy**. Oracle plans and
true previous words reveal target information and are unavailable at inference.

| Cohort | Arm | Plan | Word input | Token recovery, not task accuracy |
| --- | --- | --- | --- | --- |
''' + '\n'.join(table) + f'''

Correct coarse plans barely help the four-pass decoder. Stronger recovery
under true previous words shows retained teacher-forced language ability,
especially on training summaries. It does not demonstrate free-running fluency
or generalization. The contrast locates a major failure in the noisy parallel
word interface; predicting better coarse codes alone cannot repair these cases.
It does not prove the coarse predictor is good, isolate one training defect,
or establish a failure of the NCM paper itself.

The two-case pilot passed before the fixed eight-case study. Arm A reproduced
the stored training reload probe exactly, including all raw word IDs and
coarse codes. Body, books, new weights and selected CPU caches remained
unchanged; all logits were finite. Both workers terminated successfully.
No new teacher calls, training, development targets, final test, fallback,
external verifier or speed benchmark were used.

The most relevant published training implementation is
[JacobiForcing](https://github.com/hao-ai-lab/JacobiForcing/tree/028dbd6283fdc6fa356f0b109ce04c5ca538e49f).
Its actual multiblock trainer combines clean AR CE×10 and noisy-token soft
consistency with a detached current clean branch in the same objective.
Our failed model randomly alternated clean/noisy word updates, used offline
original-teacher top32targets, and forced predicted coarse histories on every
word update. It never qualified correct-plan parallel decoding before scaling.
The released training/inference recipe is materially different; its published
coding/math results have not been reproduced locally.

| Reference | What we can copy | Boundary |
| --- | --- | --- |
| [Official NCM](https://github.com/HazyResearch/ncm/tree/c70274273f489abc73832b2b98584a8483b6b203) | Released architecture, codec/checkpoints, inference tests | Inspected release contains no trainer; local32/32decoded-text inference agreement is not training replication. |
| [JacobiForcing trainer](https://github.com/hao-ai-lab/JacobiForcing/blob/028dbd6283fdc6fa356f0b109ce04c5ca538e49f/JacobiForcing/train/soft_flexattn_cllm_trainer_multiblock.py) | Paired clean/noisy losses, detached clean targets, explicit trajectory/mask handling | Its full-model examples and iterative block decoding differ from our frozen3B/nine-pass system. |
| [SpecForge](https://github.com/sgl-project/SpecForge/tree/f586b1ee40b5e4bf3ef03954849024b57a5edb7a) | Native vocabulary/features, concrete DFlash training and Qwen3-8B configuration | Full-parent verification protects speculative outputs; it is not standalone NCM accuracy. |

Next, test paired clean/noisy word updates and a mix of correct versus predicted
coarse plans in a small, explicitly target-conditioned decoding control.
Require that control to work before another large end-to-end run. Keep original
MLP weights frozen, retain source-only generation for all task evaluation,
and judge complete unseen summaries before any acceleration claim. A passing
training control will admit more investigation, never establish95%quality.

Aggregate `{campaign}`, SHA256 `{file_hash(campaign)}`. Confirmed=false.
''')
    print(json.dumps({"status": "finalized", "classification": classification}), flush=True)


def register():
    register_records(DEST / "campaign_results.json", REPORT, DEST, {"ncm-native-word-mechanism-v1": {
        "name": "Native NCM parallel word failure persists under oracle coarse plans",
        "description": "Crossed8case mechanism probe:4training summaries and4raw diagnostics. Predicted/oracle plans yield9.25/10.69percent training word recovery withfourparallelpasses, versus95.66/95.95percent withtruepreviouswords. This is targetconditionedmechanism evidence, not taskaccuracy; actual student0/64 remainsfailed.",
        "confirmation_status": "parallel_word_recovery_failed_even_with_oracle_plan",
        "task_accuracy_measured": False, "NCM_success_established": False}})
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
