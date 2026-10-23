"""Target-free convergence control for the original failed NCM word decoder."""
import hashlib
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_native_word_mechanism as m
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_native_query_distillation import pack_prefix
from experiments.structure_net.ncm_native_readout import frozen_hash, restore_trainable, trainable_state
from experiments.structure_net.ncm_native_word_neighborhood import initial_state, native_word_logits, predicted_concepts, special_policy
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-word-convergence-plan.md")
DEST = ROOT / "20261023_native_word_convergence_v1"
SNAPSHOTS = (4, 8, 16, 32, 64, 128)


def matching_prefix(previous, current):
    return previous.eq(current).long().cumprod(-1).sum(-1)


def iterate(step, initial, count):
    if count < 1:
        raise ValueError("positive fixed iteration budget required")
    state = initial
    states, prefixes = [], []
    for _ in range(count):
        updated = step(state)
        prefixes.append(matching_prefix(state, updated).cpu().tolist())
        states.append(updated.cpu().tolist())
        state = updated
    verified = torch.equal(step(state), state)
    return state, states, prefixes, verified


@torch.no_grad()
def main():
    m.run.require_gpu()
    if (m.run.q.read(m.PILOT / "worker_terminal_receipt.json")["exit_code"] != 0
            or m.run.q.read(m.PILOT / "result.json")["status"] != "systems_lifecycle_passed_not_quality_evidence"):
        raise ValueError("native-word lifecycle pilot required")
    verify(m.run.q.read(m.PILOT / "producer.json"))
    cache = m.run.cache_preparation.CACHE
    manifest = m.run.q.read(cache / "manifest.json")
    ids = manifest["cohorts"]["summary_train"][:4]
    entries = [manifest["cases"][i] for i in ids]
    for entry in entries:
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("selected cache changed")
    if file_hash(m.run.TRAIN / "model.pt") != m.run.q.read(m.run.TRAIN / "result.json")["checkpoint_sha256"]:
        raise ValueError("original failed checkpoint changed")
    paths = [Path(__file__), PLAN, Path(m.__file__), Path(m.run.__file__)]
    paths += [Path("experiments/structure_net") / (name + ".py") for name in
        ("ncm_native_word_neighborhood", "ncm_native_query_concepts", "ncm_native_query_cache", "ncm_native_query_distillation", "ncm_native_readout")]
    inputs = [m.run.TRAIN / n for n in ("model.pt", "producer.json", "result.json", "reload_probe.json")]
    inputs += [cache / "manifest.json", m.run.cache_preparation.BOOKS / "concepts.pt"]
    inputs += [Path(e["path"]) for e in entries]
    producer = freeze(DEST, "source_only_word_convergence_diagnosis", inputs + paths,
        {"hypothesis_id": "ncm-word-convergence-control-v1", "selected_training_ids": ids,
         "fixed_word_updates": 128, "verification_passes": 1, "snapshots": SNAPSHOTS,
         "batch": 2, "physical_gpu": 0, "precision": "BF16native/FP32new", "task_accuracy": None})
    for path in paths:
        producer["source_hashes"][str(path)] = file_hash(path)
        (DEST / "producer_source" / ("convergence_" + path.name)).write_bytes(path.read_bytes())
    producer["fingerprint"] = m.run.q.digest({k: v for k, v in producer.items() if k != "fingerprint"})
    write_json(DEST / "producer.json", producer)
    net, tokenizer = m.run.model_and_tokenizer()
    restore_trainable(net, torch.load(m.run.TRAIN / "model.pt", map_location="cpu", weights_only=True)["trainable"])
    frozen = frozen_hash(net)
    books = m.tensor_digest({str(i): b for i, b in enumerate(net.books)})
    weights = m.tensor_digest(trainable_state(net))
    originals = {r["id"]: r for r in m.run.q.read(m.run.TRAIN / "reload_probe.json")["outputs"]}
    stops, _ = special_policy(net, tokenizer)
    rows = []
    for start in range(0, 4, 2):
        examples = [torch.load(manifest["cases"][i]["path"], map_location="cpu", weights_only=True) for i in ids[start:start+2]]
        memory = pack_prefix([e["prefix"] for e in examples], "cuda")
        codes, prior = predicted_concepts(net, memory)
        initial = initial_state(net, prior, tokenizer, 2048)
        def step(state):
            logits = native_word_logits(net, memory, prior, state)
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite convergence logits")
            return logits.argmax(-1)
        _, states, prefixes, verified = iterate(step, initial, 128)
        for j, example in enumerate(examples):
            if states[3][j] != originals[example["id"]]["raw_parallel_ids"] or [c[j].tolist() for c in codes] != originals[example["id"]]["codes"]:
                raise ValueError("original four-pass replay differs")
            targets = example["targets"].tolist()
            rows.append({"id": example["id"], "source_only": True, "target_used_in_generation": False,
                "codes": [c[j].tolist() for c in codes], "initial_ids": initial[j].tolist(),
                "all_word_states": [state[j] for state in states],
                "consecutive_matching_prefixes": [p[j] for p in prefixes],
                "snapshots": {str(n): m.score(states[n-1][j], targets, stops, tokenizer) for n in SNAPSHOTS},
                "fixed_point_verified": verified})
        write_json(DEST / "outputs.json", rows)
        if not verified:
            raise ValueError("128-position causal update did not reach fixed point")
        print(json.dumps({"completed": len(rows), "fixed_point_verified": verified}), flush=True)
        del memory
    if frozen_hash(net) != frozen or m.tensor_digest(trainable_state(net)) != weights or m.tensor_digest({str(i): b for i, b in enumerate(net.books)}) != books:
        raise ValueError("model mutated")
    verify(producer)
    metrics = {}
    for n in SNAPSHOTS:
        selected = [r["snapshots"][str(n)] for r in rows]
        correct = sum(s["correct"] for s in selected); total = sum(s["total"] for s in selected)
        metrics[str(n)] = {"correct": correct, "total": total, "training_token_recovery": correct / total, "task_accuracy": None}
    result = {"status": "source_only_convergence_diagnosis_completed", "metrics": metrics,
        "cases": 4, "four_pass_replay_exact": True, "all_final_fixed_points_verified": True,
        "model_books_selected_cache_unchanged": True, "task_accuracy": None,
        "NCM_success_established": False, "speed_tested": False, "final_test_opened": False,
        "producer_fingerprint": producer["fingerprint"], "outputs_sha256": file_hash(DEST / "outputs.json")}
    write_json(DEST / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
