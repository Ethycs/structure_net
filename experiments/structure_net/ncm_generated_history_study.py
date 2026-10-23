"""Fixed generated-history word training after a registered concept bootstrap."""
import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_generated_history_core as core
from experiments.structure_net import ncm_expanded_word_data as expanded
from experiments.structure_net import ncm_word_final_eval as evaluation
from experiments.structure_net import ncm_source_hierarchy_supervised as bootstrap
from experiments.neural_architecture_lab import finalize_ncm_source_hierarchy_bootstrap as archive
from experiments.structure_net.ncm_device_geometry import set_source_frame
from experiments.structure_net.ncm_memory_data import file_hash, write_json

data = evaluation.data
base = evaluation.recovery.base
q = evaluation.q
ROOT = data.ROOT
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-generated-history-word-plan.md")
HYPOTHESIS = "ncm-generated-history-word-v1"
CACHE_PILOT = ROOT / "20261023_generated_history_cache_pilot_v1"
CACHE = ROOT / "20261023_generated_history_cache_v1"
PILOT = ROOT / "20261023_generated_history_train_pilot_v1"
TRAIN = ROOT / "20261023_generated_history_train_v1"
SOURCE_FIELDS = (*evaluation.IDENTITY, "split", "source_ids", "source_frame",
    "alignment_sha256", "native_prompt_sha256")
_base_producing = evaluation.producing


@dataclass(frozen=True)
class Config:
    updates: int = 32768
    warmup: int = 256
    max_lr: float = 1e-4
    min_lr: float = 1e-5
    model_seed: int = 20142
    sampler_seed: int = 21142
    gold_weight: float = .25
    max_source_frame: int = 3328


def rate(step, config=Config()):
    if not 1 <= step <= config.updates:
        raise ValueError("step outside the fixed word budget")
    if step <= config.warmup:
        return config.max_lr * step / config.warmup
    return config.min_lr + (config.max_lr-config.min_lr)*.5*(1+math.cos(
        math.pi*(step-config.warmup)/(config.updates-config.warmup)))


def source_view(row):
    return {key: row[key] for key in SOURCE_FIELDS}


def source_batch(row, manifest, frame=None):
    if any(key in row for key in ("target_text", "target", "word_ids", "concept_ids", "text")):
        raise ValueError("forecaster inputs must contain no target or candidate labels")
    return base.batch_for(row, manifest, frame=frame)


def parameter_hash(net, frozen_only=False):
    return base.tensor_digest({name: value for name, value in net.named_parameters()
        if not frozen_only or not value.requires_grad})


def require_data():
    data.terminal(expanded.DEST)
    result = q.read(expanded.DEST / "result.json")
    if result["status"] != "expanded_whole_summary_TRAIN_data_prepared" or not result["all_first_decisions_replayed"]:
        raise ValueError("complete independently vetted expanded TRAIN data required")
    for name, expected in result["files"].items():
        if file_hash(expanded.DEST / name) != expected:
            raise ValueError("expanded data or canonical judgment cache changed")
    rows = q.read(expanded.DEST / "cases.json")["train"]
    manifest = q.read(expanded.DEST / "manifest.json")
    if not 252 < len(rows) <= 764 or len(rows) != result["cases"] or set(manifest) != {r["id"] for r in rows}:
        raise ValueError("all original and accepted additional TRAIN cases required")
    if max(r["source_frame"] for r in rows) > Config().max_source_frame:
        raise ValueError("source frame exceeds the prospective maximum")
    return rows, manifest


def require_forecaster():
    data.terminal(bootstrap.TRAIN)
    result = q.read(bootstrap.TRAIN / "result.json")
    receipt = q.read(archive.DEST / (archive.worker.HYPOTHESIS+"_registry_readback.json"))
    campaign = q.read(archive.DEST / "campaign_results.json")
    if (receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != file_hash(archive.DEST / "campaign_results.json")
            or receipt["record"]["confirmation_status"] != archive.CLASSIFICATION
            or result["updates"] != 16384 or not result["reload_exact"]
            or file_hash(bootstrap.TRAIN / "model.pt") != result["checkpoint_sha256"]):
        raise ValueError("registered fixed final concept forecaster required")
    for name in ("producer.json", "result.json", "model.pt", "worker_terminal_receipt.json"):
        path = bootstrap.TRAIN / name
        if campaign["artifacts"].get(str(path)) != file_hash(path):
            raise ValueError("registered forecaster evidence changed")
    return result


def load_forecaster(device="cuda"):
    result = require_forecaster()
    net, codec, _ = evaluation.core.make_model(device="cpu")
    state = torch.load(bootstrap.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    if state["step"] != 16384 or state["producer_fingerprint"] != result["producer_fingerprint"]:
        raise ValueError("forecaster checkpoint selection identity differs")
    set_source_frame(net.generator, state["source_frame"], maximum=8192)
    net.load_state_dict(state["model"], strict=True)
    codec.to(device).eval().requires_grad_(False)
    net.to(device).eval().requires_grad_(False)
    set_source_frame(net.generator, state["source_frame"], maximum=8192)
    if evaluation.core.released.module_hash(codec) != result["frozen_codec_sha256_unchanged"]:
        raise ValueError("original codec representation differs")
    return net, codec


def producing(root, stage, inputs, config):
    files = [Path(__file__), PLAN, Path(core.__file__), Path(expanded.__file__),
        expanded.PLAN, Path("experiments/structure_net/ncm_word_final_head.py"),
        Path("experiments/structure_net/ncm_device_geometry.py"), Path(archive.__file__), archive.PLAN]
    files += sorted(data.VENDOR.rglob("*.py"))
    inputs = list(inputs) + [bootstrap.TRAIN / name for name in
        ("producer.json", "result.json", "model.pt", "worker_terminal_receipt.json")]
    inputs += [archive.DEST / "campaign_results.json",
        archive.DEST / (archive.worker.HYPOTHESIS+"_registry_readback.json")]
    inputs += [expanded.DEST / name for name in
        ("producer.json", "result.json", "cases.json", "manifest.json", "judgment_cache.json", "worker_terminal_receipt.json")]
    producer = _base_producing(root, stage, inputs+files, {**asdict(Config()), **config,
        "hypothesis_id": HYPOTHESIS, "forecaster_frozen": True,
        "independent_word_branch": True, "parent_loaded": False, "task_accuracy": None,
        "original_parent_weights_frozen": True, "original_codec_frozen": True})
    producer["hypothesis_id"] = HYPOTHESIS
    auxiliary = bool(config.get("gold_history_auxiliary_TRAIN_only", False))
    producer["config"].update(source_prediction=True, source_only=not auxiliary,
        oracle_target_codes=auxiliary, teacher_forced_training_auxiliary=auxiliary)
    for index, path in enumerate(files):
        producer["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"generated_history_{index}_{path.name}").write_bytes(path.read_bytes())
    producer["fingerprint"] = q.digest({k: v for k, v in producer.items() if k != "fingerprint"})
    write_json(root / "producer.json", producer)
    return producer


def probes(rows):
    first = rows[0]
    longest = sorted(rows, key=lambda r: (-len(r["source_ids"]), r["id"]))[0]
    if first["id"] == longest["id"]:
        raise ValueError("distinct first and longest TRAIN pilot cases required")
    return [first, longest]


@torch.no_grad()
def forecast(net, row, manifest, frame=None):
    batch = source_batch(row, manifest, frame=frame)
    set_source_frame(net.generator, batch["source_ids"].shape[1], maximum=8192)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = core.own_history(net, batch["features"], batch["mask"], batch["source_ids"])
    codes = result["coarse_codes"]
    if codes.shape != (1, 608) or (codes < 0).any() or (codes >= 4096).any():
        raise ValueError("complete source-predicted coarse hierarchy required")
    return {"id": row["id"], "source_sha256": row["source_sha256"],
        "source_frame": batch["source_ids"].shape[1], "coarse_codes": codes[0].cpu().tolist(),
        "history_sha256": base.tensor_digest({"codes": codes})}


def cache(pilot=False):
    data.require_gpu()
    torch.backends.cudnn.allow_tf32 = False
    rows, manifest = require_data()
    require_forecaster()
    selected = probes(rows) if pilot else rows
    inputs = []
    for row in selected:
        entry = manifest[row["id"]]
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("frozen source feature cache changed")
        inputs.append(Path(entry["path"]))
    if not pilot:
        data.terminal(CACHE_PILOT)
        if q.read(CACHE_PILOT / "result.json")["status"] != "systems_own_history_cache_passed":
            raise ValueError("real source-only cache pilot required")
        inputs += [CACHE_PILOT / name for name in ("producer.json", "result.json", "histories.json", "worker_terminal_receipt.json")]
    root = CACHE_PILOT if pilot else CACHE
    producer = producing(root, "own_history_cache_pilot" if pilot else "complete_source_only_TRAIN_concept_cache",
        inputs, {"cases": len(selected), "training": False, "oracle_target_codes": False,
            "physical_gpu": 0, "precision": "FP32weights/BF16forward/FP32reconstruction"})
    net, codec = load_forecaster()
    before = parameter_hash(net)
    codec_before = evaluation.core.released.module_hash(codec)
    torch.cuda.reset_peak_memory_stats()
    histories = {} if pilot else q.read(CACHE_PILOT / "histories.json")
    for row in selected:
        view = source_view(row)
        if row["id"] in histories:
            item = histories[row["id"]]
            if item["source_sha256"] != row["source_sha256"] or item["source_frame"] != row["source_frame"]:
                raise ValueError("retained native-frame cache pilot identity differs")
            continue
        item = forecast(net, view, manifest)
        if pilot:
            if forecast(net, view, manifest) != item:
                raise ValueError("source-only concept generation must repeat exactly")
            batch = source_batch(view, manifest)
            set_source_frame(net.generator, row["source_frame"], maximum=8192)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                existing = net.generate(batch["features"], batch["mask"], batch["source_ids"])
            if item["coarse_codes"] != torch.cat(existing["coarse_codes"], dim=1)[0].cpu().tolist():
                raise ValueError("coarse-only rollout differs from existing generation")
        histories[row["id"]] = item
        write_json(root / "histories.json", histories)
        if pilot or len(histories) % 16 == 0 or len(histories) == len(selected):
            print(json.dumps({"cached": len(histories), "total": len(selected)}), flush=True)
    if pilot:
        maximum = forecast(net, source_view(selected[-1]), manifest, frame=Config().max_source_frame)
        write_json(root / "maximum_frame_probe.json", maximum)
    if (parameter_hash(net) != before or evaluation.core.released.module_hash(codec) != codec_before
            or any(p.grad is not None for p in net.parameters()) or torch.cuda.max_memory_allocated() > 8_000_000_000):
        raise ValueError("forecaster state, gradient or memory gate failed")
    data.verify(producer)
    write_json(root / "source_cases.json", [source_view(row) for row in selected])
    result = {"status": "systems_own_history_cache_passed" if pilot else "own_history_TRAIN_cache_prepared",
        "cases": len(histories), "forecaster_parameter_sha256": before,
        "codec_hash_unchanged": codec_before, "histories_sha256": file_hash(root / "histories.json"),
        "source_cases_sha256": file_hash(root / "source_cases.json"), "source_only": True,
        "target_labels_in_cache": False, "pilot_histories_retained_exactly": not pilot,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "task_accuracy": None,
        "producer_fingerprint": producer["fingerprint"]}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


def histories_for(rows):
    data.terminal(CACHE)
    result = q.read(CACHE / "result.json")
    if (result["status"] != "own_history_TRAIN_cache_prepared"
            or file_hash(CACHE / "histories.json") != result["histories_sha256"]
            or file_hash(CACHE / "source_cases.json") != result["source_cases_sha256"]):
        raise ValueError("complete qualified actual-history cache required")
    histories = q.read(CACHE / "histories.json")
    if q.read(CACHE / "source_cases.json") != [source_view(row) for row in rows] or set(histories) != {r["id"] for r in rows}:
        raise ValueError("cache must cover every prescribed source without target fields")
    for row in rows:
        validate_history(row, histories[row["id"]])
    return histories


def validate_history(row, item):
    codes = torch.tensor([item["coarse_codes"]], dtype=torch.long)
    if (codes.shape != (1, 608) or (codes < 0).any() or (codes >= 4096).any()
            or item["id"] != row["id"] or item["source_sha256"] != row["source_sha256"]
            or item["source_frame"] != row["source_frame"]
            or base.tensor_digest({"codes": codes}) != item["history_sha256"]):
        raise ValueError("cached history content or identity differs")


def paired_backward(branch, batch, own_codes, gold_weight=.25, pad=50257):
    losses = {}
    coarse_width = sum(branch.generator.levels[:-1])
    for name, codes, weight in (("own_word_CE", own_codes, 1.),
            ("gold_word_CE", batch["codes"][:, :coarse_width], gold_weight)):
        conditioning = core.conditioning_from_codes(branch.generator.vqvae_local[0], codes)
        with torch.autocast(batch["features"].device.type, dtype=torch.bfloat16):
            logits = core.word_logits(branch, batch["features"], batch["mask"], conditioning, batch["source_ids"])
        loss = core.word_loss(logits, batch["words"], pad=pad)
        (weight*loss).backward()
        losses[name] = float(loss.detach())
        del conditioning, logits, loss
    return losses


@torch.no_grad()
def probe(forecaster, branch, rows, manifest):
    before_mode = branch.training
    frames = forecaster.generator.prefix_len, branch.generator.prefix_len
    forecaster.cuda().eval()
    branch.eval()
    output = []
    try:
        for row in rows:
            batch = source_batch(source_view(row), manifest)
            for net in (forecaster, branch):
                set_source_frame(net.generator, batch["source_ids"].shape[1], maximum=8192)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                history = core.own_history(forecaster, batch["features"], batch["mask"], batch["source_ids"])
                logits = core.word_logits(branch, batch["features"], batch["mask"], history["word_conditioning"], batch["source_ids"])
            output.append({"id": row["id"], "coarse_codes": history["coarse_codes"].cpu().tolist(),
                "raw_word_ids": logits.argmax(-1).cpu().tolist()})
    finally:
        for net, frame in zip((forecaster, branch), frames):
            set_source_frame(net.generator, frame, maximum=8192)
        branch.train(before_mode)
    return output


def save(path, branch, optimizer, sampler, step, producer):
    if path.exists():
        raise ValueError("checkpoint overwrite prohibited")
    torch.save({"model": branch.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
        "source_frame": branch.generator.prefix_len, "sampler_rng": sampler.get_state(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "config": asdict(Config()), "forecaster_checkpoint_sha256": require_forecaster()["checkpoint_sha256"],
        "producer_fingerprint": producer["fingerprint"]}, path)


def reload_check(path, forecaster, branch, optimizer, sampler, rows, manifest):
    before = probe(forecaster, branch, rows, manifest)
    state = torch.load(path, map_location="cpu", weights_only=True)
    with torch.no_grad():
        for value in branch.state_dict().values():
            value.zero_()
    optimizer.state.clear()
    sampler.manual_seed(0)
    torch.manual_seed(0)
    set_source_frame(branch.generator, state["source_frame"], maximum=8192)
    branch.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    sampler.set_state(state["sampler_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    if any(not torch.equal(value.cpu(), state["model"][name]) for name, value in branch.state_dict().items()):
        raise ValueError("word branch checkpoint reload differs")
    actual = optimizer.state_dict()
    if actual["param_groups"] != state["optimizer"]["param_groups"]:
        raise ValueError("optimizer parameter groups differ")
    for index, values in state["optimizer"]["state"].items():
        for key, value in values.items():
            if not torch.equal(value.cpu(), actual["state"][index][key].cpu()):
                raise ValueError("optimizer moment reload differs")
    if (not torch.equal(sampler.get_state(), state["sampler_rng"])
            or not torch.equal(torch.get_rng_state(), state["torch_rng"])
            or any(not torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), state["cuda_rng"]))):
        raise ValueError("sampler or CPU/CUDA RNG reload differs")
    if probe(forecaster, branch, rows, manifest) != before:
        raise ValueError("composed source-only inference differs after reload")
    return before


def train(pilot=False):
    data.require_gpu()
    torch.backends.cudnn.allow_tf32 = False
    rows, manifest = require_data()
    histories = histories_for(rows)
    selected = probes(rows)
    used = selected if pilot else rows
    inputs = [CACHE / name for name in ("producer.json", "result.json", "histories.json", "source_cases.json", "worker_terminal_receipt.json")]
    for row in used:
        entry = manifest[row["id"]]
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("frozen source feature changed")
        inputs.append(Path(entry["path"]))
    if not pilot:
        data.terminal(PILOT)
        if q.read(PILOT / "result.json")["status"] != "systems_generated_history_word_training_passed":
            raise ValueError("actual maximum-frame two-model gradient/reload pilot required")
        inputs += [PILOT / name for name in ("producer.json", "result.json", "worker_terminal_receipt.json")]
    root = PILOT if pilot else TRAIN
    producer = producing(root, "generated_history_word_gradient_reload_pilot" if pilot else "fixed_generated_history_word_training",
        inputs, {"actual_updates": 2 if pilot else Config().updates, "batch": 1,
            "cases": len(used), "training_pool_cases": len(rows), "gold_history_auxiliary_TRAIN_only": True,
            "physical_gpu": 0, "precision": "FP32masters/BF16forward/FP32loss_and_reconstruction",
            "matmul_TF32": False, "cuDNN_TF32": False})
    config = Config()
    torch.manual_seed(config.model_seed)
    torch.cuda.manual_seed_all(config.model_seed)
    forecaster, codec = load_forecaster(device="cpu")
    forecaster_before = parameter_hash(forecaster)
    if forecaster_before != q.read(CACHE / "result.json")["forecaster_parameter_sha256"]:
        raise ValueError("word training forecaster differs from cached histories")
    branch = core.independent_word_branch(forecaster)
    codec.cuda()
    branch.cuda()
    set_source_frame(branch.generator, branch.generator.prefix_len, maximum=8192)
    initial = parameter_hash(branch)
    frozen = parameter_hash(branch, frozen_only=True)
    codec_before = evaluation.core.released.module_hash(codec)
    if not pilot and initial != q.read(PILOT / "result.json")["initial_parameter_sha256"]:
        raise ValueError("full word training must start fresh from the same registered forecaster")
    parameters = [p for p in branch.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.max_lr, betas=(.9, .95), weight_decay=.01)
    sampler = torch.Generator().manual_seed(config.sampler_seed)
    torch.manual_seed(config.model_seed)
    torch.cuda.manual_seed_all(config.model_seed)
    branch.train()
    torch.cuda.reset_peak_memory_stats()
    (root / "checkpoints").mkdir()
    updates = 2 if pilot else config.updates
    for step in range(1, updates+1):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = rate(step)
        row = selected[step-1] if pilot else rows[int(torch.randint(len(rows), (1,), generator=sampler))]
        batch = base.batch_for(row, manifest, frame=config.max_source_frame if pilot and step == 2 else None)
        set_source_frame(branch.generator, batch["source_ids"].shape[1], maximum=8192)
        own_codes = torch.tensor([histories[row["id"]]["coarse_codes"]], dtype=torch.long, device="cuda")
        losses = paired_backward(branch, batch, own_codes, config.gold_weight)
        if (any(p.grad is not None for p in forecaster.parameters()) or any(p.grad is not None for p in codec.parameters())
                or any(p.grad is not None for p in branch.parameters() if not p.requires_grad)
                or branch.source_head.source_projection.weight.grad is None
                or not torch.count_nonzero(branch.source_head.source_projection.weight.grad)
                or branch.word_projection.weight.grad is None or not torch.count_nonzero(branch.word_projection.weight.grad)):
            raise ValueError("independent word/source versus frozen gradient routing failed")
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        if torch.cuda.max_memory_allocated() > 8_000_000_000:
            raise ValueError("word training allocation exceeds the fixed8GBceiling")
        entry = {"step": step, "id": row["id"], "source_frame": branch.generator.prefix_len,
            "history_sha256": histories[row["id"]]["history_sha256"], "lr": rate(step),
            "gradient_norm": float(norm), **losses}
        with (root / "trace.jsonl").open("a") as handle:
            handle.write(json.dumps(entry, allow_nan=False)+"\n")
        if pilot or step % 32 == 0:
            print(json.dumps(entry), flush=True)
        del batch, own_codes
        if not pilot and step % 4096 == 0 and step < updates:
            if (parameter_hash(branch, frozen_only=True) != frozen or parameter_hash(forecaster) != forecaster_before
                    or evaluation.core.released.module_hash(codec) != codec_before):
                raise ValueError("frozen word-branch/forecaster/codec state changed")
            save(root / "checkpoints" / f"step_{step:06d}.pt", branch, optimizer, sampler, step, producer)
    save(root / "model.pt", branch, optimizer, sampler, updates, producer)
    reloaded = reload_check(root / "model.pt", forecaster, branch, optimizer, sampler, selected, manifest)
    if (parameter_hash(branch) == initial or parameter_hash(branch, frozen_only=True) != frozen
            or parameter_hash(forecaster) != forecaster_before or evaluation.core.released.module_hash(codec) != codec_before
            or torch.cuda.max_memory_allocated() > 8_000_000_000):
        raise ValueError("word branch must change while frozen modules and memory remain qualified")
    data.verify(producer)
    write_json(root / "source_only_reload_probe.json", reloaded)
    result = {"status": "systems_generated_history_word_training_passed" if pilot else "generated_history_word_trained_pending_task_evaluation",
        "updates": updates, "cases": len(used), "training_pool_cases": len(rows),
        "initial_parameter_sha256": initial, "frozen_word_branch_parameter_sha256": frozen,
        "forecaster_parameter_sha256_unchanged": forecaster_before, "codec_hash_unchanged": codec_before,
        "checkpoint_sha256": file_hash(root / "model.pt"), "reload_exact": True,
        "source_only_composed_reload_exact": True, "parent_loaded": False,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "task_accuracy": None,
        "NCM_success_established": False, "final_test_opened": False, "speed_tested": False,
        "producer_fingerprint": producer["fingerprint"]}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["cache-pilot", "cache", "pilot", "train"])
    stage = parser.parse_args().stage
    cache(stage == "cache-pilot") if stage in ("cache-pilot", "cache") else train(stage == "pilot")
