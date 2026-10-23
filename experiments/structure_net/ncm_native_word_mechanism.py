"""Cross coarse-plan quality with free versus teacher-forced word inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.structure_net import train_ncm_native_query_neighborhood as run
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_native_concepts import ConceptConfig, encode
from experiments.structure_net.ncm_native_query_distillation import pack_prefix
from experiments.structure_net.ncm_native_readout import frozen_hash, restore_trainable, trainable_state
from experiments.structure_net.ncm_native_word_neighborhood import (
    initial_state, native_word_logits, predicted_concepts, special_policy,
)
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-native-word-mechanism-plan.md")
PILOT = ROOT / "20261023_native_word_mechanism_pilot_v1"
DEST = ROOT / "20261023_native_word_mechanism_v1"
ARMS = {"A": (False, False), "B": (True, False), "C": (False, True), "D": (True, True)}


def tensor_digest(values):
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        h.update(name.encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def score(ids, targets, stops, tokenizer):
    if len(ids) != len(targets) or not any(v != -100 for v in targets):
        raise ValueError("matching nonempty native frames required")
    end = next((i + 1 for i, value in enumerate(ids) if value in stops), len(ids))
    active = ids[:end]
    valid = [(i, v) for i, v in enumerate(targets) if v != -100]
    return {"raw_parallel_ids": ids, "active_ids": active,
            "text": tokenizer.decode(active, skip_special_tokens=True).strip(),
            "correct": sum(ids[i] == v for i, v in valid), "total": len(valid),
            "active_correct": sum(i < end and ids[i] == v for i, v in valid)}


def aggregate(rows):
    result = {}
    for cohort in sorted({r["cohort"] for r in rows}):
        result[cohort] = {}
        for arm in ARMS:
            selected = [r["arms"][arm] for r in rows if r["cohort"] == cohort]
            correct = sum(r["correct"] for r in selected)
            total = sum(r["total"] for r in selected)
            result[cohort][arm] = {"cases": len(selected), "correct": correct, "total": total,
                "token_recovery": correct / total,
                "valid_position_CE": sum(r["valid_position_CE"] * r["total"] for r in selected) / total,
                "oracle_plan": ARMS[arm][0], "teacher_forced": ARMS[arm][1], "task_accuracy": None}
    return result


@torch.no_grad()
def main(pilot=False):
    run.require_gpu()
    train_result = run.q.read(run.TRAIN / "result.json")
    if run.q.read(run.TRAIN / "worker_terminal_receipt.json")["exit_code"] != 0:
        raise ValueError("training must be observed terminal")
    if file_hash(run.TRAIN / "model.pt") != train_result["checkpoint_sha256"]:
        raise ValueError("final checkpoint differs")
    cache = run.cache_preparation.CACHE
    if file_hash(cache / "manifest.json") != run.q.read(cache / "result.json")["manifest_sha256"]:
        raise ValueError("cache manifest differs")
    manifest = run.q.read(cache / "manifest.json")
    selected = {"summary_train": manifest["cohorts"]["summary_train"][:2 if pilot else 4]}
    if not pilot:
        if (run.q.read(PILOT / "worker_terminal_receipt.json")["exit_code"] != 0
                or run.q.read(PILOT / "result.json")["status"] != "systems_lifecycle_passed_not_quality_evidence"):
            raise ValueError("terminal successful four-arm pilot required")
        verify(run.q.read(PILOT / "producer.json"))
        selected["language_diagnostic"] = manifest["cohorts"]["language_diagnostic"][:4]
    entries = [manifest["cases"][i] for ids in selected.values() for i in ids]
    for entry in entries:
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("selected immutable cache differs")
    paths = [Path(__file__), PLAN, Path(run.__file__),
        Path("experiments/structure_net/ncm_native_word_neighborhood.py"),
        Path("experiments/structure_net/ncm_native_query_distillation.py"),
        Path("experiments/structure_net/ncm_native_query_concepts.py"),
        Path("experiments/structure_net/ncm_native_query_cache.py"),
        Path("experiments/structure_net/ncm_native_concepts.py"),
        Path("experiments/structure_net/ncm_native_readout.py")]
    inputs = [run.TRAIN / n for n in ("producer.json", "result.json", "model.pt", "reload_probe.json")]
    inputs += [cache / "manifest.json", run.cache_preparation.BOOKS / "concepts.pt"]
    inputs += [Path(e["path"]) for e in entries]
    if not pilot:
        inputs += [PILOT / n for n in ("producer.json", "result.json", "worker_terminal_receipt.json")]
    root = PILOT if pilot else DEST
    producer = freeze(root, "systems_mechanism_pilot" if pilot else "exploratory_word_mechanism", inputs + paths,
        {"hypothesis_id": "ncm-native-word-mechanism-v1", "selected": selected,
         "arms": ARMS, "word_passes": 4, "batch": 2, "physical_gpu": 0,
         "precision": "BF16native/FP32new", "allow_tf32": False,
         "training": False, "task_accuracy": None, "sealed_cases_opened": False})
    for path in paths:
        producer["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("mechanism_" + path.name)).write_bytes(path.read_bytes())
    producer["fingerprint"] = run.q.digest({k: v for k, v in producer.items() if k != "fingerprint"})
    write_json(root / "producer.json", producer)
    net, tokenizer = run.model_and_tokenizer()
    restore_trainable(net, torch.load(run.TRAIN / "model.pt", map_location="cpu", weights_only=True)["trainable"])
    frozen = frozen_hash(net)
    if frozen != train_result["original_frozen_hash_unchanged"]:
        raise ValueError("original frozen weights differ")
    books = tensor_digest({str(i): b for i, b in enumerate(net.books)})
    weights = tensor_digest(trainable_state(net))
    probe = {r["id"]: r for r in run.q.read(run.TRAIN / "reload_probe.json")["outputs"]}
    cfg = ConceptConfig(canvas=net.canvas, levels=net.levels)
    stops, _ = special_policy(net, tokenizer)
    rows = []
    for cohort, ids in selected.items():
        for start in range(0, len(ids), 2):
            examples = [torch.load(manifest["cases"][i]["path"], map_location="cpu", weights_only=True) for i in ids[start:start+2]]
            memory = pack_prefix([e["prefix"] for e in examples], "cuda")
            targets = torch.stack([e["targets"] for e in examples]).cuda()
            gold = targets.masked_fill(targets.eq(-100), tokenizer.eos_token_id)
            actual_codes, actual_prior = predicted_concepts(net, memory)
            oracle_codes, oracle_prior, _ = encode(net.backbone.language.model.embed_tokens(gold).float(), net.books, cfg)
            chunk = [{"id": e["id"], "cohort": cohort, "target_ids": targets[j].tolist(), "arms": {}} for j, e in enumerate(examples)]
            for arm, (oracle, teacher_forced) in ARMS.items():
                codes, prior = (oracle_codes, oracle_prior) if oracle else (actual_codes, actual_prior)
                trajectory = []
                if teacher_forced:
                    logits = native_word_logits(net, memory, prior, gold)
                    state = logits.argmax(-1)
                else:
                    state = initial_state(net, prior, tokenizer, 2048)
                    trajectory.append(state.cpu().tolist())
                    for _ in range(4):
                        logits = native_word_logits(net, memory, prior, state)
                        if not torch.isfinite(logits).all():
                            raise ValueError("nonfinite free-state logits")
                        state = logits.argmax(-1); trajectory.append(state.cpu().tolist())
                if not torch.isfinite(logits).all():
                    raise ValueError("nonfinite word logits")
                losses = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="none").reshape_as(targets)
                for j, e in enumerate(examples):
                    value = score(state[j].tolist(), targets[j].tolist(), stops, tokenizer)
                    value.update({"codes": [c[j].tolist() for c in codes], "oracle_plan": oracle,
                        "teacher_forced": teacher_forced,
                        "valid_position_CE": float(losses[j].sum() / targets[j].ne(-100).sum()),
                        "trajectory_ids": [s[j] for s in trajectory]})
                    if arm == "A" and cohort == "summary_train":
                        if value["raw_parallel_ids"] != probe[e["id"]]["raw_parallel_ids"] or value["codes"] != probe[e["id"]]["codes"]:
                            raise ValueError("A-arm fixed checkpoint replay differs")
                    chunk[j]["arms"][arm] = value
                del logits
            rows.extend(chunk)
            write_json(root / "outputs.json", rows)
            print(json.dumps({"completed": len(rows), "cohort": cohort}), flush=True)
            del memory
    if frozen_hash(net) != frozen or tensor_digest(trainable_state(net)) != weights or tensor_digest({str(i): b for i, b in enumerate(net.books)}) != books:
        raise ValueError("model mutated during diagnostic")
    verify(producer)
    result = {"status": "systems_lifecycle_passed_not_quality_evidence" if pilot else "exploratory_mechanism_measured",
        "metrics": aggregate(rows), "cases": len(rows), "task_accuracy": None,
        "fixed_A_replay_exact": True, "model_and_selected_cache_unchanged": True,
        "sealed_cases_opened": False, "speed_tested": False, "NCM_success_established": False,
        "producer_fingerprint": producer["fingerprint"], "outputs_sha256": file_hash(root / "outputs.json")}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pilot", action="store_true")
    main(parser.parse_args().pilot)
