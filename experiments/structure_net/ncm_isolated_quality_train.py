"""Clean native quality-teacher training in an isolated word adapter bank."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_isolated_quality_data as data
from experiments.structure_net import ncm_isolated_quality_cache_v2 as cache
from experiments.structure_net import ncm_separated_word_bank as bank
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_native_query_distillation import pack_prefix
from experiments.structure_net.ncm_native_readout import frozen_hash, restore_trainable, trainable_state
from experiments.structure_net.ncm_native_word_mechanism import tensor_digest
from experiments.structure_net.ncm_native_word_neighborhood import predicted_concepts
from experiments.structure_net.ncm_paper_data import ROOT, verify

PILOT = ROOT / "20261023_isolated_quality_train_pilot_v1"
TRAIN = ROOT / "20261023_isolated_quality_train_v1"


@dataclass(frozen=True)
class Config:
    updates: int = 512
    accumulation: int = 8
    batch: int = 1
    max_lr: float = 5e-5
    warmup: int = 32
    weight_decay: float = .01
    clip: float = 1.
    sampler_seed: int = 7142


def learning_rate(update, config=Config()):
    if not 1 <= update <= config.updates or not 0 < config.warmup < config.updates:
        raise ValueError("valid fixed update/schedule required")
    if update <= config.warmup:
        return config.max_lr * update / config.warmup
    phase = (update - config.warmup) / (config.updates - config.warmup)
    return config.max_lr * .5 * (1. + math.cos(math.pi * phase))


def producing(root, stage, inputs, config):
    p = data.producing(root, stage, inputs + [Path(__file__), Path(cache.__file__), cache.AMENDMENT],
        {**asdict(Config()), **config, "quality_teacher_control": True, "NCM_model": False,
         "word_loss": "hard_target_CE_only", "word_rank": 8, "word_initialization_seed": 5142,
         "old_forecaster_frozen": True, "module_mode": "eval_with_gradients", "gradient_checkpointing": False,
         "CPU_cache_loading": "private_memory_map_no_mutation"})
    for path in (Path(__file__), Path(cache.__file__), cache.AMENDMENT):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("quality_train_" + path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = data.run.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def model_and_tokenizer():
    net, tokenizer = data.run.model_and_tokenizer()
    restore_trainable(net, torch.load(data.run.TRAIN / "model.pt", map_location="cpu", weights_only=True)["trainable"])
    if frozen_hash(net) != data.run.q.read(data.run.TRAIN / "result.json")["original_frozen_hash_unchanged"]:
        raise ValueError("original native body differs")
    bank.separate(net)
    net.eval()
    if getattr(net.backbone.language.model, "gradient_checkpointing", False):
        raise ValueError("checkpointing would need explicit adapter context replay")
    word = trainable_state(net)
    if sum(v.numel() for v in word.values()) != 4587520 or any(not k.endswith(("word_a", "word_b")) for k in word):
        raise ValueError("independent word parameter scope differs")
    if any(torch.count_nonzero(v) for k, v in word.items() if k.endswith("word_b")):
        raise ValueError("fresh zero word bank required; pilot weights forbidden")
    return net, tokenizer


def cache_digest(examples):
    h = hashlib.sha256()
    for identifier, example in sorted(examples.items()):
        h.update(identifier.encode())
        values = [example["targets"], example["prefix"]["source_mask"], example["prefix"]["anchor_token"]]
        values += [value for pair in example["prefix"]["keys_values"] for value in pair]
        for value in values:
            h.update(str(tuple(value.shape)).encode()); h.update(str(value.dtype).encode())
            h.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_examples(manifest, rows, producer):
    result = {}
    for row in rows:
        entry = manifest["cases"][row["id"]]
        if producer["input_hashes"][entry["path"]] != entry["sha256"]:
            raise ValueError("cache hash differs from completed cache manifest")
        example = torch.load(entry["path"], map_location="cpu", weights_only=True, mmap=True)
        target = example["targets"]
        prefix = example["prefix"]
        if (example["id"] != row["id"] or target[target.ne(-100)].tolist() != row["target_ids"]
                or list(prefix["source_mask"].shape) != [1, row["source_frame"] - 1]
                or int(prefix["source_mask"].sum()) != row["native_prompt_tokens"] - 1
                or int(prefix["anchor_token"][0, 0]) != row["context_ids"][-1]):
            raise ValueError("cached training target/source alignment differs")
        result[row["id"]] = example
    return result


def case_tensors(example, eos):
    memory = pack_prefix([example["prefix"]], "cuda")
    targets = example["targets"][None].cuda()
    gold = targets.masked_fill(targets.eq(-100), eos)
    return memory, targets, gold


@torch.no_grad()
def probes(net, examples, ids, eos, require_parent=False):
    rows = []
    for identifier in ids:
        memory, targets, gold = case_tensors(examples[identifier], eos)
        logits = bank.word_logits(net, memory, gold)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite word probe")
        if require_parent and not torch.equal(logits, bank.parent_logits(net, memory, gold)):
            raise ValueError("zero word bank differs from original cached language logits")
        codes, prior = predicted_concepts(net, memory)
        rows.append({"id": identifier, "logits_sha256": tensor_digest({"word": logits}),
            "teacher_forced_ids": logits.argmax(-1).tolist()[0], "task_accuracy": None,
            "coarse_codes": [c.tolist()[0] for c in codes], "coarse_prior_sha256": tensor_digest({"prior": prior})})
    return rows


def save_checkpoint(path, net, optimizer, sampler, step, producer):
    if path.exists():
        raise ValueError("checkpoint overwrite forbidden")
    torch.save({"word": trainable_state(net), "optimizer": optimizer.state_dict(),
        "sampler_rng": sampler.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(), "step": step,
        "producer_fingerprint": producer["fingerprint"], "config": asdict(Config())}, path)


def exact_reload(path, net, optimizer, sampler, examples, ids, eos):
    before = probes(net, examples, ids, eos)
    parameters = [p for p in net.parameters() if p.requires_grad]
    with torch.no_grad():
        for p in parameters:
            p.zero_()
    optimizer.state.clear(); sampler.manual_seed(0); torch.manual_seed(0)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    restore_trainable(net, saved["word"]); optimizer.load_state_dict(saved["optimizer"])
    sampler.set_state(saved["sampler_rng"]); torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    actual = trainable_state(net)
    if any(not torch.equal(actual[name], value) for name, value in saved["word"].items()):
        raise ValueError("word parameter reload differs")
    restored = optimizer.state_dict()
    if restored["param_groups"] != saved["optimizer"]["param_groups"]:
        raise ValueError("optimizer groups reload differs")
    for index, values in saved["optimizer"]["state"].items():
        if any(not torch.equal(value.cpu(), restored["state"][index][key].cpu()) for key, value in values.items()):
            raise ValueError("optimizer moments reload differs")
    if (not torch.equal(sampler.get_state(), saved["sampler_rng"])
            or not torch.equal(torch.get_rng_state(), saved["torch_rng"])
            or any(not torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), saved["cuda_rng"]))):
        raise ValueError("sampler/native RNG reload differs")
    after = probes(net, examples, ids, eos)
    if before != after:
        raise ValueError("word/coarse output reload differs")
    return after


def main(pilot=False):
    data.run.require_gpu(); config = Config()
    data.terminal(data.DEST); data.terminal(bank.PILOT)
    cache_root = cache.PILOT if pilot else cache.CACHE
    data.terminal(cache_root)
    expected_status = "systems_cache_identity_passed_not_quality_evidence" if pilot else "mixed_length_source_caches_prepared"
    if data.run.q.read(cache_root / "result.json")["status"] != expected_status:
        raise ValueError("corrected completed source cache required")
    if not pilot:
        data.terminal(PILOT)
        p_result = data.run.q.read(PILOT / "result.json")
        if p_result["status"] != "systems_clean_quality_training_passed_not_quality_evidence" or p_result["max_source_frame_seen"] != 2304:
            raise ValueError("real longest-frame word training pilot required")
    metadata = data.run.q.read(data.DEST / "cases.json")
    if file_hash(data.DEST / "cases.json") != data.run.q.read(data.DEST / "result.json")["cases_sha256"]:
        raise ValueError("training metadata changed")
    pilot_ids = data.run.q.read(data.DEST / "selection.json")["pilot_ids"]
    rows = [next(r for r in metadata if r["id"] == key) for key in pilot_ids] if pilot else metadata
    manifest = data.run.q.read(cache_root / "manifest.json")
    if file_hash(cache_root / "manifest.json") != data.run.q.read(cache_root / "result.json")["manifest_sha256"]:
        raise ValueError("source-cache manifest changed")
    if len(rows) != (2 if pilot else 252) or manifest["ids"] != [r["id"] for r in rows]:
        raise ValueError("fixed training cohort/order differs")
    inputs = [data.DEST / n for n in ("producer.json", "result.json", "cases.json", "selection.json", "worker_terminal_receipt.json")]
    inputs += [cache_root / n for n in ("producer.json", "result.json", "manifest.json", "worker_terminal_receipt.json")]
    inputs += [Path(manifest["cases"][r["id"]]["path"]) for r in rows]
    inputs += [data.run.TRAIN / n for n in ("model.pt", "producer.json", "result.json")]
    inputs += [p for p in data.run.MODEL_PATH.iterdir() if p.is_file()]
    if not pilot:
        inputs += [PILOT / n for n in ("producer.json", "result.json", "worker_terminal_receipt.json")]
    root = PILOT if pilot else TRAIN
    producer = producing(root, "short_long_clean_word_training_pilot" if pilot else "isolated_clean_word_quality_training",
        inputs, {"pilot": pilot, "actual_updates": 2 if pilot else 512,
        "actual_accumulation": 1 if pilot else 8, "actual_cases": len(rows), "physical_gpu": 0})
    examples = load_examples(manifest, rows, producer)
    content_before = cache_digest(examples)
    net, tokenizer = model_and_tokenizer()
    frozen = frozen_hash(net)
    books = tensor_digest({str(i): b for i, b in enumerate(net.books)})
    initial_word_hash = tensor_digest(trainable_state(net))
    if not pilot and initial_word_hash != p_result["initial_word_parameters_sha256"]:
        raise ValueError("full run did not start from fresh pilot-matched word initialization")
    initial_probes = probes(net, examples, pilot_ids, tokenizer.eos_token_id, require_parent=True)
    write_json(root / "initial_probes.json", initial_probes)
    parameters = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.max_lr, weight_decay=config.weight_decay)
    sampler = torch.Generator().manual_seed(config.sampler_seed)
    updates, accumulation = (2, 1) if pilot else (config.updates, config.accumulation)
    (root / "checkpoints").mkdir()
    torch.cuda.reset_peak_memory_stats()
    for step in range(updates):
        rate = learning_rate(step + 1, config)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        selected = [step] if pilot else torch.randint(len(rows), (accumulation,), generator=sampler).tolist()
        losses = []
        for index in selected:
            case = rows[index]
            memory, targets, gold = case_tensors(examples[case["id"]], tokenizer.eos_token_id)
            logits = bank.word_logits(net, memory, gold)
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
            if not torch.isfinite(loss):
                raise ValueError("nonfinite native word objective")
            (loss / accumulation).backward()
            losses.append(float(loss.detach()))
            del logits, loss, memory
        if any(p.grad is not None for p in net.parameters() if not p.requires_grad):
            raise ValueError("frozen original/forecaster parameters received gradients")
        norm = torch.nn.utils.clip_grad_norm_(parameters, config.clip, error_if_nonfinite=True)
        if not float(norm) > 0:
            raise ValueError("missing new word gradients")
        optimizer.step()
        row = {"step": step + 1, "lr": rate, "loss": sum(losses) / len(losses), "micro_losses": losses,
            "ids": [rows[i]["id"] for i in selected], "frames": [rows[i]["source_frame"] for i in selected],
            "gradient_norm": float(norm)}
        with (root / "trace.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        if pilot or (step + 1) % 8 == 0:
            print(json.dumps(row), flush=True)
        if not pilot and (step + 1) % 128 == 0 and step + 1 < updates:
            save_checkpoint(root / "checkpoints" / f"step_{step+1:06d}.pt", net, optimizer, sampler, step + 1, producer)
    save_checkpoint(root / "model.pt", net, optimizer, sampler, updates, producer)
    reloaded = exact_reload(root / "model.pt", net, optimizer, sampler, examples, pilot_ids, tokenizer.eos_token_id)
    if any((a["coarse_codes"], a["coarse_prior_sha256"]) != (b["coarse_codes"], b["coarse_prior_sha256"]) for a, b in zip(initial_probes, reloaded)):
        raise ValueError("word training changed the frozen concept predictions")
    if frozen_hash(net) != frozen or tensor_digest({str(i): b for i, b in enumerate(net.books)}) != books:
        raise ValueError("original/forecaster/books mutated")
    if cache_digest(examples) != content_before:
        raise ValueError("CPU source/target tensors mutated")
    if tensor_digest(trainable_state(net)) == initial_word_hash:
        raise ValueError("word parameters did not change")
    verify(producer)
    write_json(root / "reload_probes.json", reloaded)
    result = {"status": "systems_clean_quality_training_passed_not_quality_evidence" if pilot else "quality_teacher_trained_pending_task_evaluation",
        "updates": updates, "accumulation": accumulation, "cases": len(rows),
        "max_source_frame_seen": max(r["source_frame"] for r in rows), "only_word_bank_trained": True,
        "zero_parent_identity_short_and_long_exact": True, "original_body_forecaster_books_unchanged": True,
        "CPU_cache_digest_unchanged": content_before, "checkpoint_optimizer_RNG_output_reload_exact": True,
        "initial_word_parameters_sha256": initial_word_hash, "word_parameters": sum(p.numel() for p in parameters),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "checkpoint_sha256": file_hash(root / "model.pt"),
        "quality_teacher_control": True, "task_accuracy": None, "NCM_success_established": False,
        "final_test_opened": False, "speed_tested": False, "producer_fingerprint": producer["fingerprint"]}
    write_json(root / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pilot", action="store_true")
    main(parser.parse_args().pilot)
