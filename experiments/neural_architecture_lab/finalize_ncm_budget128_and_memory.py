"""Archive judge-budget qualification and a no-op interface lifecycle separately."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_budget128_profile as profile
from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.structure_net.ncm_source_readout_grade import audit_row
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT / "20261023_budget128_and_memory_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-budget128-and-intact-memory.md")


def finalize():
    if DEST.exists() or REPORT.exists(): raise ValueError("fresh result root required")
    roots = [ROOT / ("20261023_budget128_judge_"+role+"_v1") for role in ("secondary", "primary")]
    roots += [ROOT / "20261023_intact_memory_pilot_v2", ROOT / "20261023_native_labels_diagnostic_v1"]
    for root in roots:
        verify(q.read(root / "producer.json"))
        if q.read(root / "worker_terminal_receipt.json")["exit_code"] != 0:
            raise ValueError("all workers must be observed terminal")
    profile.install()
    judges = {}
    for role, root in zip(("secondary", "primary"), roots):
        expected = q.read(root / "expected.json")
        scores = q.read(root / "scores.json")
        correct = controls = 0
        for packet in q.read(root / "packets.json"):
            row = q.read(root / "rows" / (packet["case_id"]+".json"))
            audit_row(packet, role, row)
            if row["status"] != "valid": raise ValueError("invalid calibration request")
            values = {e["candidate_id"]: e for e in row["evaluations"]}
            indexed = {e["candidate_id"]: e for e in scores}
            for candidate in packet["candidates"]:
                name = candidate["candidate_id"]
                actual = q.passes(values[name], candidate["text"], indexed[name]["tokens"])
                if actual != indexed[name]["actual"] or values[name] != indexed[name]["evaluation"]:
                    raise ValueError("calibration decision changed")
                controls += 1
                correct += actual == expected[name]["expected_pass"]
        result = q.read(root / "result.json")
        if controls != 50 or correct != 50 or result["status"] != "passed" or result["correct"] != correct:
            raise ValueError("both judges must meet fixed50/50 gate")
        judges[role] = result
    memory = q.read(roots[2] / "result.json")
    for flag in ("initial_logits_and_greedy_exact", "native_weights_frozen", "features_immutable", "reload_exact"):
        if memory[flag] is not True: raise ValueError("memory lifecycle check failed")
    if memory["status"] != "passed" or memory["updates"] != 2 or memory["NCM_model"]:
        raise ValueError("wrong systems result")
    failure = ROOT / "20261023_intact_memory_pilot_v1"
    if q.read(failure / "worker_terminal_receipt.json")["exit_code"] != 1:
        raise ValueError("original failure must remain recorded")
    diagnostic = q.read(roots[3] / "result.json")
    if diagnostic["cases"] != 10 or diagnostic["passed"] != 8 or diagnostic["valid"] != 10:
        raise ValueError("unchanged old-profile diagnostic required")
    paths = [Path(__file__), Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in [*roots, failure]:
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    DEST.mkdir()
    aggregate = {"schema": "nal.ncm-budget128-and-memory.v1", "judges": judges, "memory_lifecycle": memory,
        "old_profile_label_diagnostic": diagnostic, "original_memory_pilot_failed": True,
        "classification": "budget128_judges_qualified_and_memory_lifecycle_passed_only",
        "NCM_success_established": False, "confirmed": False, "task_accuracy": None,
        "final_test_opened": False, "speed_tested": False,
        "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(paths)}}
    campaign = DEST / "campaign_results.json"
    write_json(campaign, aggregate)
    REPORT.write_text(f'''# Corrected budget qualified; intact-memory lifecycle passes

Date: 2026-10-23. Exploratory, systems/evaluation evidence only. Classification
`{aggregate['classification']}`. The95%summary-task objective remains unmet.

The inherited judge rubric targeted approximately32words, while the application's
stored summary limit is128tokens. One of10teacher-pilot judgments explicitly
penalized a supported summary for length inside that storage cap. The new profile
changes only this budget instruction; source-grounded faithfulness, main-point
coverage, readability and the code-enforced128-token cap remain. Every old score
and first negative judgment remains unchanged. The old pilot scored8/10; one
other failure was a real recipe-order contradiction, not a budget artifact.

Both Terra and Sonnet passed **50/50** authored/source-reviewed calibration
controls, with13/13valid requests per judge. Each set contains the48original
controls plus a long supported coffee summary and a terminal-number contradiction.
All raw responses and decisions were replayed. This is judge qualification,
not model accuracy, independent human gold or an NCM result. Fixed model outputs
have not yet been re-evaluated under this new profile; old50/64,49/64and0/64
counts must not be relabeled as new-profile measurements.

The original zero-output memory correction passed CPU tests but failed its GPU
initial exact-identity check before any updates. The v2 inference branch returns
the exact original hidden tensor while its output matrix is zero and autograd is
disabled. This retains storage/strides; training still runs correction arithmetic
so its initial output projection receives gradients. The original numerical
cause was not isolated. Original producer and failure stay append-only.

The v2 GPU lifecycle passed initial exact native logits and32-token greedy IDs,
two real updates, no native gradients, frozen full-model hash, immutable source
features, correction bound and exact new-module reload. Native16-block1B weights
were FP16, the new7,340,032parameters FP32, batch2, physicalGPU2. Peak allocation
was3,226,645,504bytes. Four accepted training cases and target-independent
synthetic concept vectors were systems probes only. This is not an NCM model;
no new held-out summary quality or speed was measured. Its no-op initialization
does not guarantee quality after training.

Aggregate `{campaign}`, SHA256 `{file_hash(campaign)}`. Registered records retain
confirmed=false. Follow-up label production uses a separately fixed draft /
critique / one-revision algorithm and a prospective128-case audit; neither these
systems passes nor old training memorization admit an NCM success claim.
''')
    print(json.dumps({"status": aggregate["classification"], "judge_controls_correct": {r: j["correct"] for r,j in judges.items()},
                      "memory_lifecycle": memory["status"], "task_accuracy": None}), flush=True)


def register():
    specs = {
        "ncm-budget128-judge-profile-v1": {"name": "Application128-token judge qualification",
            "description": "Corrected128-token profile: Terra50/50 and Sonnet50/50 source-reviewed controls. Old32-word judgments preserved; no model-quality result.",
            "confirmation_status": "both_budget128_judges_qualified"},
        "ncm-intact-memory-lifecycle-v1": {"name": "Identity-preserving intact decoder memory lifecycle",
            "description": "Original GPU zero-identity check failed; fresh v2 passed exact initial logits/greedy, two updates with frozen16-block1B, reload and memory checks. Synthetic concepts only, no NCM/task-quality claim.",
            "confirmation_status": "v2_lifecycle_passed_original_failure_preserved"}}
    for spec in specs.values(): spec.update(task_accuracy_measured=False, NCM_model=False, NCM_success_established=False)
    register_records(DEST / "campaign_results.json", REPORT, DEST, specs)
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
