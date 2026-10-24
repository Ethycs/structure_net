"""Decoder-only adaptation on author TRAIN text; frozen codec representation."""
import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_lm1b_codec as native
from experiments.structure_net import ncm_lm1b_framed_codec as framed
from experiments.structure_net import ncm_lm1b_codec_semantics as semantics
from experiments.structure_net.ncm_memory_data import file_hash, write_json

ROOT, q = native.ROOT, native.q
HYPOTHESIS = "ncm-lm1b-frozen-book-decoder-repair-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-lm1b-decoder-repair-plan.md")
MODULE = "experiments.structure_net.ncm_lm1b_decoder_repair"
INPUTS = ROOT / "20261024_lm1b_decoder_training_inputs_v1"
DATA = ROOT / "20261024_lm1b_decoder_repair_data_v1"
PILOT = ROOT / "20261024_lm1b_decoder_repair_pilot_v1"
TRAIN = ROOT / "20261024_lm1b_decoder_repair_train_v1"
WORKFLOW = ROOT / "20261024_lm1b_decoder_repair_workflow_v1"
PARENT_SHA = "af2335b3a87c83509673f8e693aabf5682a71364f5350d5ee637a10ce9416337"


@dataclass(frozen=True)
class Config:
    updates: int = 1024
    pilot_updates: int = 2
    corpus_frames: int = 16384
    microbatch: int = 8
    accumulation: int = 4
    warmup: int = 64
    max_lr: float = 2e-5
    min_lr: float = 2e-6
    native_probability: float = .2
    corpus_seed: int = 79142
    sampler_seed: int = 80142
    augmentation_seed: int = 81142
    dropout_seed: int = 82142


def rate(step, config=Config()):
    if not 1 <= step <= config.updates:
        raise ValueError("step outside fixed budget")
    if step <= config.warmup:
        return config.max_lr * step / config.warmup
    fraction = (step - config.warmup) / (config.updates - config.warmup)
    return config.min_lr + (config.max_lr - config.min_lr) * .5 * (1 + math.cos(math.pi * fraction))


def augment(frame, rng, config=Config()):
    values = frame.tolist()
    if len(values) != 128 or values[0] != 101 or values[-1] != 101:
        raise ValueError("complete author-native CLS boundaries required")
    if torch.rand((), generator=rng).item() < config.native_probability:
        return frame.clone(), {"mode": "native", "length": 128}
    lexical = [v for v in values[1:-1] if v not in (0, 101, 102)]
    if not lexical:
        raise ValueError("nonempty native lexical content required")
    length = int(torch.randint(1, len(lexical) + 1, (), generator=rng))
    lexical = lexical[:length]
    packed = torch.tensor([101, *lexical, 101] + [0] * (126 - length), dtype=torch.long)
    return packed, {"mode": "short", "length": length}


def frozen_hash(codec):
    h = hashlib.sha256()
    for name, value in sorted(codec.state_dict().items()):
        if not name.startswith("decoder."):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def decoder_training(codec):
    codec.eval().requires_grad_(False)
    codec.decoder.requires_grad_(True).train()
    return codec


def producing(root, stage, inputs=(), config=None):
    files = [Path(__file__), PLAN, Path("tests/structure_net/test_ncm_lm1b_decoder_repair.py"),
        Path(framed.__file__), framed.PLAN, Path("experiments/structure_net/ncm_frozen_body.py")]
    common = [INPUTS / "manifest.json", native.RELEASE / "files/lm1b/tokenizer.pt",
        native.RELEASE / "files/lm1b/small/config.yaml", semantics.RESULTS / "campaign_results.json",
        semantics.RESULTS / (semantics.HYPOTHESIS + "_registry_readback.json"),
        framed.DATA / "cases.json", framed.DATA / "selection.json", framed.RECON / "outputs.json"]
    p = native.producing(root, stage, [*inputs, *common, *files], {**asdict(Config()), **(config or {}),
        "hypothesis_id": HYPOTHESIS, "training_source": "only author LM1B TRAIN frames",
        "application_targets_used_for_gradient": False, "validation_used_for_training": False,
        "original_parent_loaded": False, "original_parent_MLPs_frozen": True,
        "codec_encoder_codebooks_and_EMA_frozen": True, "codec_decoder_trainable": True,
        "objective": "native word cross entropy ignoring PAD0", "full_codes_only": True,
        "fine_level_dropout": False, "code_corruption": False, "checkpoint_selection": "fixed final1024",
        "source_prediction": False, "task_accuracy": None, "no_semantic_accuracy_claim": True})
    p["hypothesis_id"] = HYPOTHESIS
    p["config"]["schema"] = "nal.ncm-lm1b-frozen-book-decoder-repair.v1"
    p["config"]["model_training_executed"] = stage in ("systems_decoder_only_training_pilot", "fixed1024_decoder_only_repair")
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"decoder_repair_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def prepare():
    if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only unpaired corpus selection required")
    manifest = q.read(INPUTS / "manifest.json")
    if manifest["validation_used_for_training"] or manifest["dataset_revision"] != "6580ba41a4b0a976dde876fbe84d25327617b0e6":
        raise ValueError("pinned author TRAIN split required")
    for name, entry in manifest["files"].items():
        if not name.startswith("lm1b/train_") or file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("complete author TRAIN file identity differs")
    import numpy as np
    path = Path(manifest["files"]["lm1b/train_bert_30522.bin"]["path"])
    raw = np.memmap(path, mode="r", dtype=np.uint16).reshape(-1, 128)
    candidates = random.Random(Config().corpus_seed).sample(range(len(raw)), Config().corpus_frames * 2)
    ids, rejected = [], []
    for index in candidates:
        frame = raw[index]
        if frame[0] != 101 or frame[-1] != 101 or (frame >= 30522).any():
            raise ValueError("unexpected author-native frame format")
        if not any(v not in (0, 101, 102) for v in frame[1:-1]):
            rejected.append({"index": index, "reason": "no lexical content"})
            continue
        ids.append(index)
        if len(ids) == Config().corpus_frames:
            break
    if len(ids) != Config().corpus_frames:
        raise ValueError("complete fixed unpaired corpus required")
    p = producing(DATA, "author_TRAIN_only_fixed_frame_corpus", [path], {"CPU_only": True})
    frames = torch.tensor(raw[ids].astype('int64'), dtype=torch.long)
    torch.save(frames, DATA / "frames.pt")
    write_json(DATA / "selection.json", {"indices": ids, "rejected": rejected, "source_native_frames": len(raw)})
    native.vendor.verify(p)
    write_json(DATA / "result.json", {"status": "fixed_unpaired_author_TRAIN_corpus_prepared", "frames": len(frames),
        "native_width": 128, "frames_sha256": file_hash(DATA / "frames.pt"), "selection_sha256": file_hash(DATA / "selection.json"),
        "application_target_training": False, "validation_training": False, "producer_fingerprint": p["fingerprint"]})


@torch.no_grad()
def probes(codec):
    old_mode = codec.decoder.training
    codec.decoder.eval()
    values = {v["id"]: v for v in q.read(framed.RECON / "outputs.json")}
    outputs = []
    for identity in q.read(framed.DATA / "selection.json")["pilot_ids"]:
        codes = torch.tensor(values[identity]["codes"], device="cuda", dtype=torch.long)
        raw = codec.decoder(codec.vq.idx_to_z_q(codes)).argmax(-1).cpu().tolist()
        outputs.append({"id": identity, "raw_ids": [v for frame in raw for v in frame]})
    codec.decoder.train(old_mode)
    return outputs


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(same(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def checkpoint(path, codec, optimizer, sampler, augmentation, step, frozen, p):
    saved = {"step": step, "config": asdict(Config()), "decoder": {k: v.detach().cpu() for k, v in codec.decoder.state_dict().items()},
        "optimizer": optimizer.state_dict(), "sampler_rng": sampler.get_state(), "augmentation_rng": augmentation.get_state(),
        "CPU_rng": torch.get_rng_state(), "CUDA_rng": torch.cuda.get_rng_state_all(),
        "frozen_state_sha256": frozen, "producer_fingerprint": p["fingerprint"]}
    tmp = path.with_suffix(".tmp")
    torch.save(saved, tmp)
    tmp.replace(path)


def optimizer_for(codec):
    return torch.optim.AdamW(codec.decoder.parameters(), lr=Config().max_lr, betas=(.9, .95), eps=1e-8, weight_decay=.01)


def train(pilot=False):
    native.require_gpu()
    native.vendor.terminal(DATA)
    if file_hash(DATA / "frames.pt") != q.read(DATA / "result.json")["frames_sha256"]:
        raise ValueError("fixed TRAIN corpus changed")
    if not pilot:
        native.vendor.terminal(PILOT)
        if q.read(PILOT / "result.json")["status"] != "systems_decoder_only_gradients_and_reload_passed":
            raise ValueError("actual training/reload pilot required")
    root = PILOT if pilot else TRAIN
    inputs = [DATA / n for n in ("producer.json", "result.json", "frames.pt", "selection.json", "worker_terminal_receipt.json")]
    if not pilot:
        inputs += [PILOT / n for n in ("producer.json", "result.json", "initial_probes.json", "worker_terminal_receipt.json")]
    p = producing(root, "systems_decoder_only_training_pilot" if pilot else "fixed1024_decoder_only_repair", inputs,
        {"pilot": pilot, "physical_gpu": 2, "actual_updates": Config().pilot_updates if pilot else Config().updates,
            "actual_accumulation": 1 if pilot else Config().accumulation, "peak_limit_bytes": 4700000000})
    frames = torch.load(DATA / "frames.pt", map_location="cpu", weights_only=True)
    codec = decoder_training(native.load_codec())
    frozen = frozen_hash(codec)
    initial_hash = native.vendor.module_hash(codec)
    if initial_hash != "9aace450a49a7dfb44ba1eb00a19974a50d67062ad9f2202a6a2361ea4344e32":
        raise ValueError("fresh exact released initialization required")
    initial = probes(codec)
    originals = {r["id"]: r["raw_ids"] for r in q.read(framed.RECON / "outputs.json")}
    if any(v["raw_ids"] != originals[v["id"]] for v in initial):
        raise ValueError("released initial native decoder parity differs")
    if not pilot and initial != q.read(PILOT / "initial_probes.json"):
        raise ValueError("full repair must reset to fresh release, not pilot weights")
    write_json(root / "initial_probes.json", initial)
    torch.manual_seed(Config().dropout_seed)
    torch.cuda.manual_seed_all(Config().dropout_seed)
    sampler = torch.Generator().manual_seed(Config().sampler_seed)
    augmentation = torch.Generator().manual_seed(Config().augmentation_seed)
    optimizer = optimizer_for(codec)
    updates = Config().pilot_updates if pilot else Config().updates
    accumulation = 1 if pilot else Config().accumulation
    torch.cuda.reset_peak_memory_stats()
    with (root / "trace.jsonl").open("x") as trace:
        for step in range(1, updates + 1):
            if codec.encoder.training or codec.vq.training or not codec.decoder.training:
                raise ValueError("only decoder stochastic training mode allowed")
            optimizer.zero_grad(set_to_none=True)
            lr = rate(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            losses, samples = [], []
            for _ in range(accumulation):
                indices = torch.randint(len(frames), (Config().microbatch,), generator=sampler).tolist()
                generated = [augment(frames[i], augmentation) for i in indices]
                inputs = torch.stack([v[0] for v in generated]).cuda()
                with torch.no_grad():
                    codes = codec.encode(inputs)
                    latent = codec.vq.idx_to_z_q(codes)
                logits = codec.decoder(latent.detach())
                loss = F.cross_entropy(logits.float().flatten(0, 1), inputs.flatten(), ignore_index=0)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite decoder repair loss")
                (loss / accumulation).backward()
                losses.append(float(loss.detach()))
                samples += [{"row": i, **v[1]} for i, v in zip(indices, generated)]
            if any(v.grad is not None for n, v in codec.named_parameters() if not n.startswith("decoder.")):
                raise ValueError("frozen encoder/codebooks received gradients")
            norm = torch.nn.utils.clip_grad_norm_(codec.decoder.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError("nonfinite decoder gradients")
            optimizer.step()
            trace.write(json.dumps({"step": step, "lr": lr, "loss": sum(losses) / len(losses),
                "gradient_norm": float(norm), "samples": samples}, allow_nan=False) + "\n")
            trace.flush()
            if pilot or step % 32 == 0:
                print(json.dumps({"step": step, "updates": updates, "loss": sum(losses) / len(losses)}), flush=True)
            if not pilot and step == 512:
                checkpoint(root / "recovery_step512.pt", codec, optimizer, sampler, augmentation, step, frozen, p)
    if frozen_hash(codec) != frozen or native.vendor.module_hash(codec) == initial_hash:
        raise ValueError("frozen representation changed or decoder failed to update")
    before_reload = probes(codec)
    checkpoint(root / "model.pt", codec, optimizer, sampler, augmentation, updates, frozen, p)
    saved = torch.load(root / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    del codec, optimizer, logits, loss, latent
    gc.collect(); torch.cuda.empty_cache()
    codec = decoder_training(native.load_codec())
    codec.decoder.load_state_dict(saved["decoder"], strict=True)
    optimizer = optimizer_for(codec); optimizer.load_state_dict(saved["optimizer"])
    sampler.set_state(saved["sampler_rng"]); augmentation.set_state(saved["augmentation_rng"])
    torch.set_rng_state(saved["CPU_rng"]); torch.cuda.set_rng_state_all(saved["CUDA_rng"])
    if (frozen_hash(codec) != frozen or probes(codec) != before_reload or not same(optimizer.state_dict(), saved["optimizer"])
            or not torch.equal(sampler.get_state(), saved["sampler_rng"])
            or not torch.equal(augmentation.get_state(), saved["augmentation_rng"])
            or not torch.equal(torch.get_rng_state(), saved["CPU_rng"])
            or not same(torch.cuda.get_rng_state_all(), saved["CUDA_rng"])):
        raise ValueError("model/optimizer/RNG/exact inference reload failed")
    peak = torch.cuda.max_memory_allocated()
    if peak > 4700000000:
        raise ValueError("decoder repair exceeds planned memory")
    write_json(root / "reload_probes.json", before_reload)
    native.vendor.verify(p)
    result = {"status": "systems_decoder_only_gradients_and_reload_passed" if pilot else "decoder_repair_trained_reconstruction_and_semantics_pending",
        "updates": updates, "initial_codec_hash": initial_hash, "final_codec_hash": native.vendor.module_hash(codec),
        "frozen_state_sha256_unchanged": frozen, "decoder_trainable_parameters": sum(v.numel() for v in codec.decoder.parameters()),
        "exact_reload": True, "model_training_executed": True, "application_targets_used_for_gradient": False,
        "validation_used_for_training": False, "peak_allocated_bytes": peak, "checkpoint_sha256": file_hash(root / "model.pt"),
        "task_accuracy": None, "final_test_opened": False, "producer_fingerprint": p["fingerprint"]}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


def workflow():
    if file_hash(semantics.RESULTS / "campaign_results.json") != PARENT_SHA:
        raise ValueError("fixed failed semantic control required before repair")
    receipt = q.read(semantics.RESULTS / (semantics.HYPOTHESIS + "_registry_readback.json"))
    if receipt["record"]["campaign_sha256"] != PARENT_SHA:
        raise ValueError("registered parent failure required")
    if any(p.exists() for p in (WORKFLOW, DATA, PILOT, TRAIN)):
        raise ValueError("fresh repair training workflow required")
    p = producing(WORKFLOW, "unpaired_decoder_only_repair_training")
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [], "producer_fingerprint": p["fingerprint"]}
    try:
        for stage, root, gpu in (("prepare", DATA, False), ("pilot", PILOT, True), ("train", TRAIN, True)):
            native.vendor.verify(p)
            command = [sys.executable, "-u", "-m", MODULE, stage]
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "2" if gpu else ""})
            journal.update(active_stage=stage, child_pid=child.pid); write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            record = {"stage": stage, "exit_code": code, "child_pid": child.pid, "observer_pid": os.getpid(),
                "observer": "subprocess.wait", "command": command}
            write_json(WORKFLOW / "stage_receipts" / (stage + ".json"), record)
            if root.exists(): write_json(root / "worker_terminal_receipt.json", record)
            journal["stages"].append(record)
            if code: raise RuntimeError("decoder repair stage failed: " + stage)
        native.vendor.verify(p)
        journal.update(status="decoder_repair_training_completed_quality_unmeasured", active_stage=None, child_pid=None)
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "pilot", "train", "workflow"))
    args = parser.parse_args()
    {"prepare": prepare, "pilot": lambda: train(True), "train": train, "workflow": workflow}[args.stage]()
