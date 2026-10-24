"""Conditional source-only NCM word fitting with verified dense TRAIN targets."""
import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from experiments.structure_net import ncm_decoder_latent_convergence as teacher
from experiments.structure_net import ncm_dense_latent_word as core
from experiments.structure_net import ncm_clean_word_dropout as dropout
from experiments.structure_net.ncm_device_geometry import set_source_frame
from experiments.structure_net.ncm_expanded_word_data_workflow import kernel_identity
from experiments.structure_net.ncm_memory_data import file_hash, write_json

clean = teacher.prior.clean
ROOT = clean.ROOT
HYPOTHESIS = "ncm-dense-latent-source-word-fit-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-dense-word-fit-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-dense-word-fit.md")
WORKFLOW = ROOT / "20261023_dense_word_fit_workflow_v1"
PILOT = ROOT / "20261023_dense_word_fit_pilot_v1"
TRAIN = ROOT / "20261023_dense_word_fit_train_v1"
FIT = ROOT / "20261023_dense_word_fit_measure_v1"
RESULTS = ROOT / "20261023_dense_word_fit_results_v1"
MODULE = "experiments.structure_net.ncm_dense_word_fit"
AUDITOR = "experiments.neural_architecture_lab.finalize_ncm_dense_word_fit"
AUDITOR_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_dense_word_fit.py")
TEACHER_SHA = "1a5587221af8048f3052296bb17d6c6ab40afdf7e62d689088b7b0061070a834"
INITIAL_WORD = "c90ee2ae272c298022d50d7a9b4182f06d8d995fadf8ecceb6f091e35b8b5223"


@dataclass(frozen=True)
class Config:
    updates: int = 2048
    warmup: int = 64
    max_lr: float = 1e-4
    min_lr: float = 1e-5
    model_seed: int = 45142
    sampler_seed: int = 46142
    latent_weight: float = 1.
    word_weight: float = .1
    max_source_frame: int = 3328
    snapshot_interval: int = 1024
    batch: int = 1
    beta1: float = .9
    beta2: float = .95
    optimizer_eps: float = 1e-8
    weight_decay: float = .01
    gradient_clip: float = 1.

    def __post_init__(self):
        if (not 1 <= self.warmup < self.updates or not 0 < self.min_lr <= self.max_lr
                or self.model_seed == self.sampler_seed or self.batch != 1
                or self.snapshot_interval <= 0 or self.updates % self.snapshot_interval
                or self.latent_weight <= 0 or self.word_weight <= 0
                or not 0 < self.beta1 < 1 or not 0 < self.beta2 < 1
                or self.optimizer_eps <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0):
            raise ValueError("invalid fixed dense fitting configuration")


def rate(step, config=Config()):
    if not 1 <= step <= config.updates:
        raise ValueError("step outside fixed dense fitting budget")
    if step <= config.warmup:
        return config.max_lr * step / config.warmup
    return config.min_lr + (config.max_lr - config.min_lr) * .5 * (
        1 + math.cos(math.pi * (step - config.warmup) / (config.updates - config.warmup)))


def admission_result(campaign):
    gates = campaign["fit_gates"]
    if set(gates) != set(dropout.ARMS) or any(type(v) is not bool for v in gates.values()):
        raise ValueError("both fixed dropout decisions required")
    expected = { (False, False): "both_dropout_fit_gates_failed",
        (False, True): "zero_dropout_fit_passed_control_failed",
        (True, False): "original_dropout_only_fit_passed",
        (True, True): "both_continuations_fit_passed"}[tuple(gates[a] for a in dropout.ARMS)]
    if campaign["classification"] != expected:
        raise ValueError("dropout classification and gates disagree")
    return expected == "both_dropout_fit_gates_failed"


def admission():
    if (dropout.WORKFLOW / "failure.json").exists():
        raise ValueError("upstream dropout infrastructure failure must remain preserved")
    if not (dropout.WORKFLOW / "result.json").exists():
        return None
    q = clean.recovery.study.q
    campaign_path = dropout.RESULTS / "campaign_results.json"
    campaign = q.read(campaign_path)
    receipt = q.read(dropout.RESULTS / (dropout.HYPOTHESIS + "_registry_readback.json"))
    finished = q.read(dropout.WORKFLOW / "result.json")
    if (receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != file_hash(campaign_path)
            or receipt["exact"]["campaign_sha256"] != file_hash(campaign_path)
            or finished["status"] != "dropout_fitting_comparison_completed"
            or finished["classification"] != campaign["classification"]
            or len(finished["stages"]) != 8 or any(r["exit_code"] != 0 for r in finished["stages"])):
        raise ValueError("complete registered dropout comparison and8zero exits required")
    return admission_result(campaign)


def inputs(require_gate=True):
    rows, _ = teacher.inputs()
    q = clean.recovery.study.q
    archive = teacher.RESULTS / "campaign_results.json"
    campaign = q.read(archive)
    receipt = q.read(teacher.RESULTS / (teacher.HYPOTHESIS + "_registry_readback.json"))
    if (file_hash(archive) != TEACHER_SHA or not campaign["latent_fit_gate"]
            or receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != TEACHER_SHA
            or receipt["exact"]["campaign_sha256"] != TEACHER_SHA):
        raise ValueError("verified fixed256-step teacher labels required")
    for name in ("producer.json", "result.json", "latents.pt", "case_measurements.json", "worker_terminal_receipt.json"):
        p = teacher.MEASURE / name
        if campaign["artifacts"].get(str(p)) != file_hash(p):
            raise ValueError("bound teacher numerical evidence differs")
    labels = torch.load(teacher.MEASURE / "latents.pt", map_location="cpu", weights_only=True)
    measured = q.read(teacher.MEASURE / "case_measurements.json")
    for row, value in zip(rows, measured):
        if value["arms"][teacher.prior.ARMS[1]]["correct"] != value["arms"][teacher.prior.ARMS[1]]["total"]:
            raise ValueError("all32 teacher targets must reconstruct completely")
        if labels[row["id"]]["refined"].shape != (1, 192, 384):
            raise ValueError("complete dense teacher target required")
    study = clean.recovery.study
    manifest_path = study.expanded.DEST / "manifest.json"
    upstream = q.read(teacher.prior.continuous.WORKFLOW / "producer.json")
    if upstream["input_hashes"].get(str(manifest_path)) != file_hash(manifest_path):
        raise ValueError("unchanged source feature manifest required")
    manifest = q.read(manifest_path)
    cache = q.read(study.CACHE / "result.json")
    if file_hash(study.CACHE / "histories.json") != cache["histories_sha256"]:
        raise ValueError("unchanged actual source-only concept history cache required")
    histories = q.read(study.CACHE / "histories.json")
    prior_campaign = q.read(clean.diagnostic.ARCHIVE / "campaign_results.json")
    for name in ("histories.json", "result.json", "source_cases.json"):
        item = study.CACHE / name
        if prior_campaign["artifacts"].get(str(item)) != file_hash(item):
            raise ValueError("registered source-only history binding changed")
    for row in rows:
        study.validate_history(row, histories[row["id"]])
    if require_gate and admission() is not True:
        raise ValueError("both registered dropout fitting failures required before GPU work")
    return rows, manifest, histories, labels


def producing(destination, stage, extra=(), pending=False):
    study = clean.recovery.study
    files = [Path(__file__), PLAN, AUDITOR_FILE, Path(core.__file__), Path(dropout.__file__), dropout.PLAN]
    common = [teacher.RESULTS / "campaign_results.json",
        teacher.RESULTS / (teacher.HYPOTHESIS + "_registry_readback.json")]
    common += [teacher.MEASURE / n for n in ("producer.json", "result.json", "latents.pt", "case_measurements.json", "worker_terminal_receipt.json")]
    common += [clean.TRAIN / n for n in ("model.pt", "result.json", "producer.json", "worker_terminal_receipt.json")]
    common += [study.CACHE / n for n in ("histories.json", "result.json", "producer.json", "source_cases.json", "worker_terminal_receipt.json")]
    common += [study.expanded.DEST / "manifest.json", dropout.WORKFLOW / "producer.json"]
    if not pending:
        common += [dropout.WORKFLOW / "result.json", dropout.RESULTS / "campaign_results.json",
            dropout.RESULTS / (dropout.HYPOTHESIS + "_registry_readback.json")]
    p = teacher.producing(destination, stage, [*extra, *common, *files])
    p["hypothesis_id"] = HYPOTHESIS
    p["config"] = {**asdict(Config()), "hypothesis_id": HYPOTHESIS, "stage": stage,
        "TRAIN_cases": 32, "full_available_TRAIN_cases": 690, "physical_gpu": 0,
        "initial_word_checkpoint_sha256": INITIAL_WORD, "teacher_campaign_sha256": TEACHER_SHA,
        "precision": "FP32masters_optimizer_loss_decoder_BF16source_transformer_projection",
        "matmul_TF32": False, "cuDNN_TF32": False, "parent_loaded": False,
        "body_MLPs_forecaster_codec_original_decoder_and_unused_heads_frozen": True,
        "conditioning": "actual_cached_source_only_forecaster_history",
        "teacher_latents_and_words_are_labels_only": True, "training_dropout": 0,
        "task_accuracy": None, "final_test_opened": False, "speed_tested": False}
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (destination / "producer_source" / f"dense_fit_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = study.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(destination / "producer.json", p)
    return p


def model(path=None, expected_step=None):
    study = clean.recovery.study
    if file_hash(clean.TRAIN / "model.pt") != INITIAL_WORD:
        raise ValueError("unchanged clean-word initializer required")
    fixed, branch, codec = clean.model(clean.TRAIN / "model.pt", clean.Config().updates)
    core.prepare_branch(branch)
    if path is not None:
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if saved["config"] != asdict(Config()) or saved["step"] != expected_step:
            raise ValueError("fixed dense checkpoint identity differs")
        set_source_frame(branch.generator, saved["source_frame"], maximum=8192)
        branch.load_state_dict(saved["model"], strict=True)
    if any(not torch.equal(v, codec.decoder.state_dict()[k])
           for k, v in branch.word_decoder.state_dict().items()):
        raise ValueError("dense model original decoder differs")
    return fixed, branch, codec


@torch.no_grad()
def predict(branch, batch, codes):
    conditioning = clean.recovery.study.core.conditioning_from_codes(branch.generator.vqvae_local[0], codes)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        latent, logits = core.prediction(branch, batch["features"], batch["mask"], conditioning, batch["source_ids"])
    return latent, logits


@torch.no_grad()
def probe(fixed, branch, rows, manifest, histories):
    study = clean.recovery.study
    was_training = branch.training
    frame = branch.generator.prefix_len
    fixed.cuda().eval()
    core.train_mode(branch, False)
    outputs = []
    try:
        for row in rows:
            batch = study.source_batch(study.source_view(row), manifest)
            for net in (fixed, branch):
                set_source_frame(net.generator, row["source_frame"], maximum=8192)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                history = study.core.own_history(fixed, batch["features"], batch["mask"], batch["source_ids"])
            if history["coarse_codes"][0].tolist() != histories[row["id"]]["coarse_codes"]:
                raise ValueError("fresh actual concept history differs from cache")
            latent, logits = predict(branch, batch, history["coarse_codes"])
            outputs.append({"id": row["id"], "coarse_codes": history["coarse_codes"].cpu().tolist(),
                "raw_word_ids": logits.argmax(-1).cpu().tolist(),
                "latent_sha256": study.base.tensor_digest({"latent": latent})})
    finally:
        fixed.cpu()
        set_source_frame(branch.generator, frame, maximum=8192)
        core.train_mode(branch, was_training)
    return outputs


def save(path, branch, optimizer, sampler, step, producer):
    if path.exists():
        raise ValueError("checkpoint overwrite prohibited")
    torch.save({"model": branch.state_dict(), "optimizer": optimizer.state_dict(),
        "parameter_names": [n for n, _ in branch.named_parameters()],
        "trainable_parameter_names": [n for n, p in branch.named_parameters() if p.requires_grad],
        "step": step, "source_frame": branch.generator.prefix_len, "config": asdict(Config()),
        "sampler_rng": sampler.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(), "producer_fingerprint": producer["fingerprint"]}, path)


def reload_check(path, fixed, branch, optimizer, sampler, rows, manifest, histories):
    before = probe(fixed, branch, rows, manifest, histories)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    with torch.no_grad():
        for value in branch.state_dict().values():
            value.zero_()
    optimizer.state.clear()
    sampler.manual_seed(0)
    torch.manual_seed(0)
    set_source_frame(branch.generator, saved["source_frame"], maximum=8192)
    branch.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    sampler.set_state(saved["sampler_rng"])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    if (not teacher.prior.same({k: v.cpu() for k, v in branch.state_dict().items()}, saved["model"])
            or not teacher.prior.same(teacher.prior.to_cpu(optimizer.state_dict()), saved["optimizer"])
            or not torch.equal(sampler.get_state(), saved["sampler_rng"])
            or not torch.equal(torch.get_rng_state(), saved["torch_rng"])
            or not teacher.prior.same(torch.cuda.get_rng_state_all(), saved["cuda_rng"])):
        raise ValueError("exact dense model/optimizer/sampler/RNG reload required")
    if before != probe(fixed, branch, rows, manifest, histories):
        raise ValueError("composed source-only dense predictions changed after reload")
    return before


def train(pilot=False):
    rows, manifest, histories, labels = inputs()
    study = clean.recovery.study
    study.data.require_gpu()
    torch.backends.cudnn.allow_tf32 = False
    config = Config()
    probes = clean.training_probes(rows)
    used = probes if pilot else rows
    extra = []
    for row in used:
        item = manifest[row["id"]]
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("unchanged full source features required")
        extra.append(Path(item["path"]))
    if not pilot:
        study.data.terminal(PILOT)
        extra += [PILOT / n for n in ("producer.json", "result.json", "worker_terminal_receipt.json")]
    destination = PILOT if pilot else TRAIN
    p = producing(destination, "systems_lifecycle_only_not_quality_evidence" if pilot else "fixed32_source_only_dense_training", extra)
    torch.manual_seed(config.model_seed)
    torch.cuda.manual_seed_all(config.model_seed)
    fixed, branch, codec = model()
    initial = study.parameter_hash(branch)
    frozen = study.parameter_hash(branch, frozen_only=True)
    fixed_hash = study.parameter_hash(fixed)
    codec_hash = teacher.prior.module_hash(codec)
    maps = dropout.dropout_map(branch)
    if any(maps.values()):
        raise ValueError("dense branch must have zero module and functional dropout")
    if not pilot and initial != study.q.read(PILOT / "result.json")["initial_parameter_sha256"]:
        raise ValueError("full dense training must restart the exact pilot initializer")
    parameters = [v for v in branch.parameters() if v.requires_grad]
    opt = torch.optim.AdamW(parameters, lr=config.max_lr, betas=(config.beta1, config.beta2),
        eps=config.optimizer_eps, weight_decay=config.weight_decay)
    sampler = torch.Generator().manual_seed(config.sampler_seed)
    torch.manual_seed(config.model_seed)
    torch.cuda.manual_seed_all(config.model_seed)
    core.train_mode(branch)
    torch.cuda.reset_peak_memory_stats()
    (destination / "checkpoints").mkdir()
    updates = 2 if pilot else config.updates
    for step in range(1, updates + 1):
        row = probes[step - 1] if pilot else rows[int(torch.randint(len(rows), (1,), generator=sampler))]
        batch = study.source_batch(study.source_view(row), manifest,
            frame=config.max_source_frame if pilot and step == 2 else None)
        codes = torch.tensor([histories[row["id"]]["coarse_codes"]], device="cuda")
        words = torch.tensor([row["word_ids"]], device="cuda")
        target = labels[row["id"]]["refined"].cuda()
        set_source_frame(branch.generator, batch["source_ids"].shape[1], maximum=8192)
        conditioning = study.core.conditioning_from_codes(branch.generator.vqvae_local[0], codes)
        lr = config.max_lr if pilot else rate(step)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent, logits = core.prediction(branch, batch["features"], batch["mask"], conditioning, batch["source_ids"])
        loss, parts = core.objective(latent, logits, target, words,
            latent_weight=config.latent_weight, word_weight=config.word_weight)
        loss.backward()
        if (any(v.grad is not None for v in fixed.parameters()) or any(v.grad is not None for v in codec.parameters())
                or any(v.grad is not None for v in branch.parameters() if not v.requires_grad)):
            raise ValueError("dense source training changed frozen gradient routing")
        if pilot and (not branch.source_head.source_projection.weight.grad.abs().sum()
                or not branch.word_projection.weight.grad.abs().sum()):
            raise ValueError("positive source and word projection gradients required")
        norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip, error_if_nonfinite=True)
        opt.step()
        record = {"step": step, "id": row["id"], "source_frame": branch.generator.prefix_len, "lr": lr,
            "actual_history_sha256": study.base.tensor_digest({"codes": codes}),
            "target_word_sha256": study.base.tensor_digest({"words": words}),
            "teacher_latent_sha256": study.base.tensor_digest({"latent": target}),
            **parts, "gradient_norm": float(norm)}
        with (destination / "trace.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        if pilot or step % 32 == 0:
            print(json.dumps(record), flush=True)
        if torch.cuda.max_memory_allocated() > 8_000_000_000:
            raise ValueError("dense source training exceeds8GB allocation limit")
        del latent, logits, loss, batch, target
        if not pilot and step % config.snapshot_interval == 0 and step < updates:
            save(destination / "checkpoints" / f"step_{step:06d}.pt", branch, opt, sampler, step, p)
    save(destination / "model.pt", branch, opt, sampler, updates, p)
    outputs = reload_check(destination / "model.pt", fixed, branch, opt, sampler, probes, manifest, histories)
    if (study.parameter_hash(branch) == initial or study.parameter_hash(branch, frozen_only=True) != frozen
            or study.parameter_hash(fixed) != fixed_hash or teacher.prior.module_hash(codec) != codec_hash
            or dropout.dropout_map(branch) != maps
            or torch.cuda.max_memory_allocated() > 8_000_000_000):
        raise ValueError("dense original frozen modules, update or memory contract differs")
    write_json(destination / "source_only_reload_probe.json", outputs)
    study.data.verify(p)
    result = {"status": "systems_dense_gradient_reload_passed" if pilot else "dense_source_word_trained",
        "updates": updates, "training_population": 32, "cases": len(used), "initial_parameter_sha256": initial,
        "frozen_parameter_sha256": frozen, "forecaster_parameter_sha256": fixed_hash, "codec_hash": codec_hash,
        "reload_exact": True, "source_only_composed_reload_exact": True, "original_decoder_frozen": True,
        "dropout_map": maps,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "checkpoint_sha256": file_hash(destination / "model.pt"),
        "producer_fingerprint": p["fingerprint"], "task_accuracy": None, "semantic_quality_measured": False,
        "final_test_opened": False, "speed_tested": False}
    write_json(destination / "result.json", result)
    print(json.dumps(result), flush=True)


def aggregate(values):
    result = {}
    for group in ("original_16", "additional_16", "all_32"):
        selected = values if group == "all_32" else [r for r in values if r["diagnostic_group"] == group]
        correct, total = (sum(r[k] for r in selected) for k in ("correct", "total"))
        result[group] = {"cases": len(selected), "correct": correct, "total": total,
            "token_agreement": correct / total if total else None,
            "exact_target_texts": sum(r["exact_target_text"] for r in selected)}
    return result


def fitting_gate(components):
    converted = {name: {"cases": row["cases"], "arms": {"correct_history": row}} for name, row in components.items()}
    return clean.fitting_gate(converted)


def fit():
    rows, manifest, histories, labels = inputs()
    study = clean.recovery.study
    study.data.require_gpu()
    torch.backends.cudnn.allow_tf32 = False
    study.data.terminal(TRAIN)
    trained = study.q.read(TRAIN / "result.json")
    if file_hash(TRAIN / "model.pt") != trained["checkpoint_sha256"]:
        raise ValueError("fixed final dense student checkpoint required")
    p = producing(FIT, "fixed32_actual_source_only_TRAIN_word_fitting",
        [TRAIN / n for n in ("model.pt", "producer.json", "result.json", "worker_terminal_receipt.json")])
    fixed, branch, codec = model(TRAIN / "model.pt", Config().updates)
    branch.eval().requires_grad_(False)
    before = study.parameter_hash(branch), study.parameter_hash(fixed), teacher.prior.module_hash(codec)
    outputs = probe(fixed, branch, rows, manifest, histories)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(study.data.GPT2, local_files_only=True)
    values = []
    for row, output in zip(rows, outputs):
        raw = output["raw_word_ids"][0]
        text, end = clean.recovery.evaluation.reference.decode_words(raw, tokenizer)
        values.append({"id": row["id"], "source_sha256": row["source_sha256"], "diagnostic_group": row["diagnostic_group"],
            "target_word_ids": row["word_ids"], "target_text": row["target_text"], "raw_word_ids": raw,
            "actual_coarse_codes": output["coarse_codes"][0], "latent_sha256": output["latent_sha256"],
            **clean.diagnostic.counts(torch.tensor([raw]), torch.tensor([row["word_ids"]])),
            "text": text, "global_EOS_position": end if end < 192 else None, "exact_target_text": text == row["target_text"]})
    if before != (study.parameter_hash(branch), study.parameter_hash(fixed), teacher.prior.module_hash(codec)):
        raise ValueError("read-only fitting changed dense model or original frozen state")
    components = aggregate(values)
    write_json(FIT / "case_measurements.json", values)
    study.data.verify(p)
    result = {"status": "dense_source_TRAIN_fitting_measured", "cases": 32, "components": components,
        "actual_history_fit_gate": fitting_gate(components), "checkpoint_sha256": trained["checkpoint_sha256"],
        "parameters_unchanged": True, "fresh_native_cache_parity": True,
        "measurements_sha256": file_hash(FIT / "case_measurements.json"), "task_accuracy": None,
        "final_test_opened": False, "speed_tested": False, "producer_fingerprint": p["fingerprint"]}
    write_json(FIT / "result.json", result)
    print(json.dumps(result), flush=True)


def workflow():
    inputs(require_gate=False)
    if any(p.exists() for p in (WORKFLOW, PILOT, TRAIN, FIT, RESULTS, REPORT,
            Path("data/hypothesis_registry") / (HYPOTHESIS + ".json"))):
        raise ValueError("fresh dense fitting destinations required")
    progress = clean.recovery.study.q.read(dropout.WORKFLOW / "progress.json")
    identity = kernel_identity(progress["observer_pid"])
    if identity is None and admission() is None:
        raise ValueError("unfinished dropout comparison must be authoritatively live")
    p = producing(WORKFLOW, "wait_for_failed_dropout_fit_then_dense_source_fit", pending=True)
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "waiting_for_dropout_comparison", "observer_pid": os.getpid(),
        "upstream_kernel_identity": identity, "stages": [], "producer_fingerprint": p["fingerprint"], "final_test_opened": False}
    write_json(WORKFLOW / "progress.json", journal)
    try:
        while True:
            ready = admission()
            alive = identity is not None and kernel_identity(identity["pid"]) == identity
            if ready is False:
                journal.update(status="not_admitted_dropout_fitting_passed", model_training_executed=False, task_accuracy=None)
                write_json(WORKFLOW / "result.json", journal)
                return
            if ready is True and not alive:
                break
            if ready is None and not alive:
                raise RuntimeError("upstream terminal without required registered decisions")
            time.sleep(5)
        for stage, dest in (("pilot", PILOT), ("train", TRAIN), ("fit", FIT), ("finalize", RESULTS), ("register", RESULTS)):
            if dest.exists() != (stage == "register"):
                raise ValueError("fresh dense stage destination required")
            clean.recovery.study.data.verify(p)
            command = [sys.executable, "-u", "-m", AUDITOR if stage in ("finalize", "register") else MODULE, stage]
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "" if stage in ("finalize", "register") else "0"})
            journal.update(status="running", active_stage=stage, child_pid=child.pid)
            write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            receipt = {"status": "observed_worker_terminal", "stage": stage, "exit_code": code,
                "observer_pid": os.getpid(), "child_pid": child.pid, "observer": "subprocess.wait", "command": command, "destination": str(dest)}
            write_json(WORKFLOW / "stage_receipts" / (stage + ".json"), receipt)
            if dest.exists():
                path = dest / ("registry_terminal_receipt.json" if stage == "register" else "worker_terminal_receipt.json")
                if path.exists():
                    raise ValueError("terminal receipt overwrite prohibited")
                write_json(path, receipt)
            journal["stages"].append(receipt)
            if code:
                raise RuntimeError("actual dense stage failure: " + stage)
            clean.recovery.study.data.verify(p)
        path = RESULTS / "campaign_results.json"
        receipt = clean.recovery.study.q.read(RESULTS / (HYPOTHESIS + "_registry_readback.json"))
        if receipt["status"] != "registered_and_read_back" or receipt["exact"]["campaign_sha256"] != file_hash(path):
            raise ValueError("exact dense fitting registry required")
        journal.update(status="dense_source_fitting_completed", active_stage=None,
            classification=clean.recovery.study.q.read(path)["classification"], task_accuracy=None, model_training_executed=True)
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("check", "pilot", "train", "fit", "workflow"))
    args = parser.parse_args()
    if args.stage == "check":
        rows, *_ = inputs(require_gate=False)
        sampler = torch.Generator().manual_seed(Config().sampler_seed)
        counts = [0] * len(rows)
        for _ in range(Config().updates):
            counts[int(torch.randint(len(rows), (1,), generator=sampler))] += 1
        if min(counts) == 0:
            raise ValueError("fixed sampler must cover every32 TRAIN case")
        print(json.dumps({"TRAIN_cases": len(rows), "minimum_samples": min(counts), "maximum_samples": max(counts), "config": asdict(Config()), "admission": admission()}))
    elif args.stage == "workflow":
        workflow()
    elif args.stage == "fit":
        fit()
    else:
        train(args.stage == "pilot")
