"""Archive target-free convergence as a diagnosis, not a speed/quality win."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_native_word_convergence as study
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT / "20261023_native_word_convergence_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-word-convergence-control.md")


def prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh finalization required")
    if study.m.run.q.read(study.DEST / "worker_terminal_receipt.json")["exit_code"] != 0:
        raise ValueError("observed terminal convergence worker required")
    verify(study.m.run.q.read(study.DEST / "producer.json"))
    result = study.m.run.q.read(study.DEST / "result.json")
    rows = study.m.run.q.read(study.DEST / "outputs.json")
    if (result["cases"] != 4 or not result["all_final_fixed_points_verified"]
            or not result["four_pass_replay_exact"] or not result["model_books_selected_cache_unchanged"]
            or file_hash(study.DEST / "outputs.json") != result["outputs_sha256"]):
        raise ValueError("complete valid convergence diagnosis required")
    originals = {r["id"]: r for r in study.m.run.q.read(study.m.run.TRAIN / "reload_probe.json")["outputs"]}
    targets = {r["id"]: r["target_ids"] for r in study.m.run.q.read(study.m.run.data.DEST / "cases.json")["summary_train"]}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(study.m.run.MODEL_PATH, local_files_only=True)
    generation = study.m.run.q.read(study.m.run.MODEL_PATH / "generation_config.json")
    configured = generation.get("eos_token_id", [])
    stops = {tokenizer.eos_token_id, *([configured] if isinstance(configured, int) else configured)}
    stabilized = []
    for row in rows:
        states = row["all_word_states"]
        if (len(states) != 128 or any(len(s) != 128 for s in states)
                or states[3] != originals[row["id"]]["raw_parallel_ids"]
                or not row["source_only"] or row["target_used_in_generation"] or not row["fixed_point_verified"]):
            raise ValueError("target-free fixed original cohort changed")
        previous = row["initial_ids"]
        expected_prefixes = []
        stable_text_at = None
        gold = targets[row["id"]]; padded = gold + [-100] * (128 - len(gold))
        for i, state in enumerate(states):
            expected_prefixes.append(prefix(previous, state))
            active_end = next((j + 1 for j, token in enumerate(state) if token in stops), None)
            if stable_text_at is None and active_end is not None and expected_prefixes[-1] >= active_end:
                stable_text_at = i + 1
            previous = state
        if row["consecutive_matching_prefixes"] != expected_prefixes:
            raise ValueError("consecutive prefix trace differs")
        for n in study.SNAPSHOTS:
            expected = study.m.score(states[n-1], padded, stops, tokenizer)
            if row["snapshots"][str(n)] != expected:
                raise ValueError("snapshot raw score differs")
        stabilized.append({"id": row["id"], "first_stable_complete_text_iteration": stable_text_at})
    for n in study.SNAPSHOTS:
        selected = [r["snapshots"][str(n)] for r in rows]
        correct, total = sum(r["correct"] for r in selected), sum(r["total"] for r in selected)
        if result["metrics"][str(n)] != {"correct": correct, "total": total, "training_token_recovery": correct / total, "task_accuracy": None}:
            raise ValueError("convergence token aggregate differs")
    paths = [Path(__file__), Path(study.__file__), study.PLAN,
        Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    paths += [p for p in study.DEST.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    classification = "causal_fixed_point_reached_quality_unmeasured"
    aggregate = {"schema": "nal.ncm-word-convergence-results.v1", "classification": classification,
        "result": result, "stability": stabilized, "raw_state_scores_and_prefixes_replayed": True,
        "task_accuracy_measured": False, "NCM_success_established": False, "confirmed": False,
        "final_test_opened": False, "speed_tested": False,
        "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(paths)}}
    DEST.mkdir(); campaign = DEST / "campaign_results.json"; write_json(campaign, aggregate)
    table = []
    for n in study.SNAPSHOTS:
        score = result["metrics"][str(n)]
        table.append(f"| {n} | {score['correct']}/{score['total']} ({100*score['training_token_recovery']:.2f}%) |")
    first = rows[0]
    REPORT.write_text(f'''# Word convergence diagnosis: four passes were not qualified

Date: 2026-10-23. Hypothesis `ncm-word-convergence-control-v1`.
Classification `{classification}`. Aggregate `{campaign}`,
SHA256 `{file_hash(campaign)}`.

The original failed 2,048-update model was restored unchanged. On the first
four fixed training summaries, source-only predicted coarse plans remained
fixed while the word stage ran 128 parallel updates. A 129th pass verified
exact full-frame fixed points on every case. No oracle plan or previous gold
words entered generation. This is a decoding diagnosis, not a new summary
accuracy or throughput result. The original nine-pass model remains 0/64 on
source-disjoint development summaries.

| Word updates | Training token recovery, not summary accuracy |
| --- | --- |
''' + '\n'.join(table) + f'''

On the first fixed case, four updates produced:

> {first['snapshots']['4']['text']}

At 128 updates, the same unchanged model produced:

> {first['snapshots']['128']['text']}

These texts have no new semantic judgments. Convergence and training-target
matches do not establish faithful, salient summaries or generalization.
Increasing the iteration count can remove a parallel compute advantage; it
does not demonstrate acceleration. Raw states, configured stop IDs, exact
four-pass replay, token matches and prefix traces were retained and replayed.
Original body, books, new weights and selected cache remained unchanged.

The reference JacobiForcing implementation uses iterative block decoding and
trains clean/noisy trajectories. Our fixed whole-frame four-update shortcut
was never qualified to converge. The neighboring paired-loss control improved
training recovery but failed its 90% oracle capacity gate; it was not scaled.
Before more large training, distinguish decoder convergence from the quality
of the final fixed point and from a learned codec's ability to skip sequential
dependencies. Any adaptive/block decoding would be a separately registered
hybrid study, with its full compute costs and complete unseen-summary grading.
It cannot retroactively rescue the failed fixed-nine-pass NCM endpoint.
''')
    print(json.dumps({"status": "finalized", "classification": classification, "metrics": result["metrics"]}), flush=True)


def register():
    register_records(DEST / "campaign_results.json", REPORT, DEST, {"ncm-word-convergence-control-v1": {
        "name": "Source-only convergence of original failed NCM word decoder",
        "description": "Four fixed training summaries, unchanged original2048-update model, source-only coarse plans,128wordupdates plus129th fixedpointcheck. Every rawstate and prefix retained; originalfourpass outputs replayexact. Convergence is mechanism evidence only; no taskaccuracy or speed, studentdevelopment remains0/64.",
        "confirmation_status": "causal_fixed_point_reached_quality_unmeasured",
        "task_accuracy_measured": False, "NCM_success_established": False}})
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
