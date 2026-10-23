"""Small paired clean/noisy word capacity control, never task accuracy."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.structure_net import train_ncm_native_query_neighborhood as run
from experiments.structure_net import ncm_native_word_mechanism as mechanism
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_native_concepts import ConceptConfig, encode
from experiments.structure_net.ncm_native_query_distillation import pack_prefix
from experiments.structure_net.ncm_native_readout import frozen_hash, restore_trainable, trainable_state
from experiments.structure_net.ncm_native_word_neighborhood import initial_state, native_word_logits, predicted_concepts, special_policy
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-paired-word-control-plan.md")
PILOT = ROOT / "20261023_paired_word_control_pilot_v1"
TRAIN = ROOT / "20261023_paired_word_control_train_v1"


@dataclass(frozen=True)
class Config:
    updates: int = 128
    batch: int = 2
    lr: float = 1e-4
    weight_decay: float = .01
    clip: float = 1.
    clean_loss_weight: float = 10.
    oracle_warmup: int = 16
    word_refinements: int = 4


def noisy_mask(state, gold, targets):
    if state.shape != gold.shape or state.shape != targets.shape or state.ndim != 2:
        raise ValueError("matching native frames required")
    # Query j consumes words <j. A wrong word j first changes query j+1.
    equal_prefix = state[:, :-1].eq(gold[:, :-1]).long().cumprod(-1).bool()
    different_context = torch.cat((torch.zeros_like(state[:, :1], dtype=torch.bool), ~equal_prefix), -1)
    return targets.ne(-100) & different_context


def soft_consistency(logits, teacher_probabilities, mask, chunk=16):
    if logits.shape != teacher_probabilities.shape or logits.shape[:-1] != mask.shape or chunk < 1:
        raise ValueError("matching probability/logit/mask frames required")
    indices = mask.flatten().nonzero().flatten()
    if not len(indices):
        return logits.sum() * 0.
    flat = logits.flatten(0, 1)
    teacher = teacher_probabilities.detach().flatten(0, 1)
    loss = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, len(indices), chunk):
        chosen = indices[start:start+chunk]
        values = flat[chosen].float()
        probabilities = teacher[chosen].float()
        loss = loss + (values.logsumexp(-1) - (probabilities * values).sum(-1)).sum()
    return loss / len(indices)


@torch.no_grad()
def plans(net, memory, gold, oracle):
    if oracle:
        cfg = ConceptConfig(canvas=net.canvas, levels=net.levels)
        codes, prior, _ = encode(net.backbone.language.model.embed_tokens(gold).float(), net.books, cfg)
        return codes, prior
    return predicted_concepts(net, memory)


def paired_backward(net, memory, targets, tokenizer, oracle, passes, clean_weight):
    gold = targets.masked_fill(targets.eq(-100), tokenizer.eos_token_id)
    with torch.no_grad():
        _, prior = plans(net, memory, gold, oracle)
        state = initial_state(net, prior, tokenizer, 2048)
        for _ in range(passes):
            state = native_word_logits(net, memory, prior, state).argmax(-1)
        mask = noisy_mask(state, gold, targets)
    clean_logits = native_word_logits(net, memory, prior, gold)
    if not torch.isfinite(clean_logits).all():
        raise ValueError("nonfinite clean logits")
    probabilities = clean_logits.detach().float().softmax(-1)
    clean = F.cross_entropy(clean_logits.float().flatten(0, 1), targets.flatten())
    (clean * clean_weight).backward()
    clean_value = float(clean.detach())
    del clean, clean_logits
    logits = native_word_logits(net, memory, prior, state)
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite noisy logits")
    noisy = soft_consistency(logits, probabilities, mask)
    noisy.backward()
    return {"clean_CE": clean_value, "noisy_soft_CE": float(noisy.detach()),
        "oracle_plan": oracle, "trajectory_passes": passes, "noisy_positions": int(mask.sum()),
        "valid_positions": int(targets.ne(-100).sum())}


@torch.no_grad()
def probe(net, examples, tokenizer):
    stops, _ = special_policy(net, tokenizer)
    rows = []
    for start in range(0, len(examples), 2):
        chunk = examples[start:start+2]
        memory = pack_prefix([e["prefix"] for e in chunk], "cuda")
        targets = torch.stack([e["targets"] for e in chunk]).cuda()
        gold = targets.masked_fill(targets.eq(-100), tokenizer.eos_token_id)
        output = [{"id": e["id"], "arms": {}} for e in chunk]
        for arm, oracle in (("A", False), ("B", True)):
            codes, prior = plans(net, memory, gold, oracle)
            state = initial_state(net, prior, tokenizer, 2048)
            for _ in range(4):
                state = native_word_logits(net, memory, prior, state).argmax(-1)
            for j, row in enumerate(output):
                row["arms"][arm] = {**mechanism.score(state[j].tolist(), targets[j].tolist(), stops, tokenizer),
                    "oracle_plan": oracle, "codes": [c[j].tolist() for c in codes]}
        rows.extend(output)
    return rows


def metrics(rows):
    result = {}
    for arm in ("A", "B"):
        correct = sum(r["arms"][arm]["correct"] for r in rows)
        total = sum(r["arms"][arm]["total"] for r in rows)
        result[arm] = {"correct": correct, "total": total, "token_recovery": correct / total,
            "oracle_plan": arm == "B", "task_accuracy": None}
    return result


def main(pilot=False):
    run.require_gpu(); config = Config()
    if not pilot:
        if (run.q.read(PILOT / "worker_terminal_receipt.json")["exit_code"] != 0
                or run.q.read(PILOT / "result.json")["status"] != "systems_lifecycle_passed_not_quality_evidence"):
            raise ValueError("terminal successful paired-loss CUDA pilot required")
        verify(run.q.read(PILOT / "producer.json"))
    cache = run.cache_preparation.CACHE
    manifest = run.q.read(cache / "manifest.json")
    if file_hash(cache / "manifest.json") != run.q.read(cache / "result.json")["manifest_sha256"]:
        raise ValueError("cache manifest changed")
    selected = {"summary_train": manifest["cohorts"]["summary_train"][:2 if pilot else 4]}
    if pilot:
        selected["language_train"] = manifest["cohorts"]["language_train"][:2]
    entries = [manifest["cases"][i] for ids in selected.values() for i in ids]
    for entry in entries:
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("selected cache changed")
    if file_hash(run.TRAIN / "model.pt") != run.q.read(run.TRAIN / "result.json")["checkpoint_sha256"]:
        raise ValueError("initial failed checkpoint changed")
    root = PILOT if pilot else TRAIN
    paths = [Path(__file__), PLAN, Path(mechanism.__file__)]
    for name in ("ncm_native_word_neighborhood", "ncm_native_query_distillation", "ncm_native_query_concepts",
                 "ncm_native_query_cache", "ncm_native_concepts", "ncm_native_readout", "train_ncm_native_query_neighborhood"):
        paths.append(Path("experiments/structure_net") / (name + ".py"))
    inputs = [run.TRAIN / n for n in ("model.pt", "result.json", "producer.json", "worker_terminal_receipt.json")]
    inputs += [cache / "manifest.json", run.cache_preparation.BOOKS / "concepts.pt"] + [Path(e["path"]) for e in entries]
    if not pilot:
        inputs += [PILOT / n for n in ("producer.json", "result.json", "worker_terminal_receipt.json")]
    updates = 4 if pilot else config.updates
    producer = freeze(root, "paired_word_systems_pilot" if pilot else "paired_word_training_capacity_control",
        inputs + paths, {**asdict(config), "pilot": pilot, "actual_updates": updates, "selected": selected,
        "sampler_seed": 7042, "trajectory_seed": 8042, "task_accuracy": None,
        "original_weights_frozen": True, "physical_gpu": 0, "precision": "BF16native/FP32new"})
    for path in paths:
        producer["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("paired_" + path.name)).write_bytes(path.read_bytes())
    producer["fingerprint"] = run.q.digest({k: v for k, v in producer.items() if k != "fingerprint"})
    write_json(root / "producer.json", producer)
    net, tokenizer = run.model_and_tokenizer()
    restore_trainable(net, torch.load(run.TRAIN / "model.pt", map_location="cpu", weights_only=True)["trainable"])
    net.eval()
    frozen = frozen_hash(net)
    if frozen != run.q.read(run.TRAIN / "result.json")["original_frozen_hash_unchanged"]:
        raise ValueError("original frozen model differs from initial checkpoint")
    books = mechanism.tensor_digest({str(i): b for i, b in enumerate(net.books)})
    cohorts = {k: [torch.load(manifest["cases"][i]["path"], map_location="cpu", weights_only=True) for i in ids] for k, ids in selected.items()}
    examples = cohorts["summary_train"]
    before = probe(net, examples, tokenizer); write_json(root / "before_outputs.json", before)
    parameters = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    sampler = torch.Generator().manual_seed(7042); trajectory = torch.Generator().manual_seed(8042)
    torch.cuda.reset_peak_memory_stats()
    for step in range(updates):
        task = ("summary_train", "language_train", "summary_train", "language_train")[step] if pilot else "summary_train"
        available = cohorts[task]
        chosen = torch.randint(len(available), (2,), generator=sampler).tolist()
        batch = [available[i] for i in chosen]
        memory = pack_prefix([e["prefix"] for e in batch], "cuda")
        targets = torch.stack([e["targets"] for e in batch]).cuda()
        oracle = (step in (0, 3)) if pilot else (step < config.oracle_warmup or step % 2 == 0)
        passes = int(torch.randint(config.word_refinements, (1,), generator=trajectory))
        optimizer.zero_grad(set_to_none=True)
        row = paired_backward(net, memory, targets, tokenizer, oracle, passes, config.clean_loss_weight)
        if any(p.grad is not None for p in net.parameters() if not p.requires_grad):
            raise ValueError("original weights received gradients")
        norm = torch.nn.utils.clip_grad_norm_(parameters, config.clip, error_if_nonfinite=True)
        if not float(norm) > 0:
            raise ValueError("no new-parameter gradient")
        optimizer.step()
        row.update({"step": step + 1, "task": task, "ids": [e["id"] for e in batch], "gradient_norm": float(norm)})
        with (root / "trace.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        if pilot or (step + 1) % 8 == 0:
            print(json.dumps(row), flush=True)
        del memory
    after = probe(net, examples, tokenizer); write_json(root / "after_outputs.json", after)
    checkpoint = {"trainable": trainable_state(net), "optimizer": optimizer.state_dict(), "step": updates,
        "sampler_rng": sampler.get_state(), "trajectory_rng": trajectory.get_state(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "producer_fingerprint": producer["fingerprint"]}
    torch.save(checkpoint, root / "model.pt")
    with torch.no_grad():
        for parameter in parameters:
            parameter.zero_()
    optimizer.state.clear()
    sampler.manual_seed(0); trajectory.manual_seed(0); torch.manual_seed(0)
    saved = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    restore_trainable(net, saved["trainable"]); optimizer.load_state_dict(saved["optimizer"])
    sampler.set_state(saved["sampler_rng"]); trajectory.set_state(saved["trajectory_rng"])
    torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
    actual_parameters = trainable_state(net)
    if any(not torch.equal(value, actual_parameters[name]) for name, value in saved["trainable"].items()):
        raise ValueError("new parameter values differ after reload")
    actual_optimizer = optimizer.state_dict()
    if actual_optimizer["param_groups"] != saved["optimizer"]["param_groups"]:
        raise ValueError("optimizer groups differ after reload")
    for parameter, state in saved["optimizer"]["state"].items():
        restored = actual_optimizer["state"][parameter]
        if state.keys() != restored.keys() or any(not torch.equal(value.cpu(), restored[key].cpu()) for key, value in state.items()):
            raise ValueError("optimizer moments differ after reload")
    if not torch.equal(torch.get_rng_state(), saved["torch_rng"]) or any(not torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), saved["cuda_rng"])):
        raise ValueError("native RNG differs after reload")
    if probe(net, examples, tokenizer) != after or not torch.equal(sampler.get_state(), checkpoint["sampler_rng"]) or not torch.equal(trajectory.get_state(), checkpoint["trajectory_rng"]):
        raise ValueError("checkpoint/RNG output reload differs")
    if frozen_hash(net) != frozen or mechanism.tensor_digest({str(i): b for i, b in enumerate(net.books)}) != books:
        raise ValueError("original body or books mutated")
    verify(producer)
    measured = metrics(after)
    result = {"status": "systems_lifecycle_passed_not_quality_evidence" if pilot else "training_capacity_control_completed",
        "updates": updates, "before": metrics(before), "after": measured,
        "oracle_training_capacity_gate": None if pilot else measured["B"]["token_recovery"] >= .9,
        "original_body_books_selected_cache_unchanged": True, "checkpoint_optimizer_RNG_reload_exact": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "task_accuracy": None,
        "NCM_success_established": False, "development_or_final_opened": False, "speed_tested": False,
        "producer_fingerprint": producer["fingerprint"], "checkpoint_sha256": file_hash(root / "model.pt")}
    write_json(root / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pilot", action="store_true")
    main(parser.parse_args().pilot)
