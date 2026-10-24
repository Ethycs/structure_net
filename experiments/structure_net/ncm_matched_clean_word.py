"""Restore correct-history word fitting before matched-population forecasting."""
import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_reviewed_draft_recovery as recovery
from experiments.structure_net import ncm_reviewed_word_train_diagnostic as diagnostic
from experiments.structure_net.ncm_device_geometry import set_source_frame
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_native_label_quality import dependencies

ROOT = recovery.ROOT
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-matched-clean-word-plan.md")
HYPOTHESIS = "ncm-matched-population-clean-word-v1"
PILOT = ROOT / "20261023_matched_clean_word_pilot_v1"
FIT_PILOT = ROOT / "20261023_matched_clean_word_fit_pilot_v1"
TRAIN = ROOT / "20261023_matched_clean_word_train_v1"
FIT = ROOT / "20261023_matched_clean_word_fit_v1"
WORKFLOW = ROOT / "20261023_matched_clean_word_workflow_v1"
RESULTS = ROOT / "20261023_matched_clean_word_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-matched-clean-word.md")
MODULE = "experiments.structure_net.ncm_matched_clean_word"
FINALIZER_MODULE = "experiments.neural_architecture_lab.finalize_ncm_matched_clean_word"
FINALIZER_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_matched_clean_word.py")
INITIAL_WORD = diagnostic.CHECKPOINT
INITIAL_FORECASTER = "27f835d1391d7e14439716a0514a86d5b301bee79341199ab9dcfe88748e454d"
FROZEN_WORD = "713f63ea5a7778dc5d17e2674d0d123d17302efb76d8828ac951051485de21c9"


@dataclass(frozen=True)
class Config:
    updates: int = 32768
    warmup: int = 256
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    model_seed: int = 24142
    sampler_seed: int = 25142
    max_source_frame: int = 3328
    fit_numerator: int = 199
    fit_denominator: int = 200

    def __post_init__(self):
        if (not 1 <= self.warmup <= self.updates
                or not 0 < self.min_lr <= self.max_lr
                or not 0 < self.fit_numerator <= self.fit_denominator
                or self.model_seed == self.sampler_seed):
            raise ValueError("invalid fixed clean-word configuration")


def rate(step, config):
    if not 1 <= step <= config.updates:
        raise ValueError("step outside fixed clean-word budget")
    if step <= config.warmup:
        return config.max_lr * step / config.warmup
    return config.min_lr + (config.max_lr - config.min_lr) * .5 * (
        1 + math.cos(math.pi * (step - config.warmup) / (config.updates - config.warmup)))


def fitting_gate(components, config=Config()):
    groups = {"all_32": 32, "original_16": 16, "additional_16": 16}
    counts = {}
    for group, cases in groups.items():
        values = components[group]
        word = values["arms"]["correct_history"]
        correct, total = word["correct"], word["total"]
        if (values["cases"] != cases or not isinstance(correct, int)
                or not isinstance(total, int) or not 0 <= correct <= total or total == 0):
            raise ValueError("complete fixed TRAIN probe groups and valid counts required")
        counts[group] = (correct, total)
    if any(counts["all_32"][i] != counts["original_16"][i] + counts["additional_16"][i] for i in (0, 1)):
        raise ValueError("probe group denominators or counts do not compose")
    return all(config.fit_denominator * correct >= config.fit_numerator * total
               for correct, total in counts.values())


def require_inputs():
    dependencies()
    selected, manifest, histories = diagnostic.inputs_and_selection()
    study = recovery.study
    proof = study.q.read(diagnostic.DEST / "result.json")
    if (file_hash(diagnostic.DEST / "result.json") !=
            "7e17101af06d30d4ef490cc740225aaada14128e315c47870d33263c7aed837f"
            or proof["cases"] != 32 or not proof["fresh_histories_match_training_cache"]
            or proof["task_accuracy"] is not None):
        raise ValueError("completed fixed TRAIN diagnosis required")
    rows = study.q.read(study.expanded.DEST / "cases.json")["train"]
    if len(rows) != 690 or any(row["source_frame"] > Config().max_source_frame for row in rows):
        raise ValueError("same complete690 TRAIN population and full source frames required")
    return rows, manifest, histories, selected


def model(path=None, expected_step=None):
    """Use existing learned weights; only the new optimizer starts from scratch."""
    recovery.configure()
    recovery.evaluation.configure()
    study = recovery.study
    forecaster, branch, codec, _ = recovery.evaluation.checkpoint()
    if study.require_forecaster()["checkpoint_sha256"] != INITIAL_FORECASTER:
        raise ValueError("initial frozen forecaster changed")
    if path is not None:
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if (saved["step"] != expected_step or saved["config"] != asdict(Config())
                or saved["initial_word_checkpoint_sha256"] != INITIAL_WORD
                or saved["forecaster_checkpoint_sha256"] != INITIAL_FORECASTER):
            raise ValueError("fixed clean-word checkpoint identity differs")
        set_source_frame(branch.generator, saved["source_frame"], maximum=8192)
        branch.load_state_dict(saved["model"], strict=True)
    if study.parameter_hash(branch, frozen_only=True) != FROZEN_WORD:
        raise ValueError("unused word-branch roots or concept heads changed")
    forecaster.cpu().eval().requires_grad_(False)
    branch.cuda()
    set_source_frame(branch.generator, branch.generator.prefix_len, maximum=8192)
    return forecaster, branch, codec


def producing(root, stage, inputs, config):
    study = recovery.study
    files = [Path(__file__), PLAN, FINALIZER_FILE, Path(diagnostic.__file__)]
    inputs = list(inputs) + files
    inputs += [diagnostic.ARCHIVE / "campaign_results.json",
        diagnostic.ARCHIVE / (study.HYPOTHESIS + "_registry_readback.json"),
        diagnostic.DEST / "result.json", diagnostic.DEST / "case_measurements.json"]
    inputs += [study.TRAIN / name for name in
        ("producer.json", "result.json", "model.pt", "worker_terminal_receipt.json")]
    p = study.producing(root, stage, inputs, {**asdict(Config()), **config,
        "gold_weight": 1., "own_weight": 0., "objective": "correct_history_word_cross_entropy",
        "initial_word_checkpoint_sha256": INITIAL_WORD,
        "forecaster_checkpoint_sha256": INITIAL_FORECASTER,
        "TRAIN_population": 690, "task_accuracy": None, "final_test_opened": False,
        "precision": "FP32masters/BF16forward/FP32loss", "matmul_TF32": False, "cuDNN_TF32": False})
    p["hypothesis_id"] = HYPOTHESIS
    p["config"].update(hypothesis_id=HYPOTHESIS, source_only=False,
        oracle_target_codes=True, teacher_forced_coarse_history=True,
        teacher_forced_training_auxiliary=False, word_targets_in_prediction=False,
        finest_target_codes_in_prediction=False)
    for index, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"matched_clean_{index}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = study.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def clean_backward(branch, batch):
    study = recovery.study
    width = sum(branch.generator.levels[:-1])
    conditioning = study.core.conditioning_from_codes(
        branch.generator.vqvae_local[0], batch["codes"][:, :width])
    with torch.autocast(batch["features"].device.type, dtype=torch.bfloat16):
        logits = study.core.word_logits(branch, batch["features"], batch["mask"],
            conditioning, batch["source_ids"])
    value = study.core.word_loss(logits, batch["words"])
    value.backward()
    return float(value.detach())


def save(path, branch, optimizer, sampler, step, producer):
    if path.exists():
        raise ValueError("checkpoint overwrite prohibited")
    torch.save({"model": branch.state_dict(), "optimizer": optimizer.state_dict(),
        "step": step, "source_frame": branch.generator.prefix_len,
        "sampler_rng": sampler.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(), "config": asdict(Config()),
        "initial_word_checkpoint_sha256": INITIAL_WORD,
        "forecaster_checkpoint_sha256": INITIAL_FORECASTER,
        "producer_fingerprint": producer["fingerprint"]}, path)


def training_probes(rows):
    return [rows[0], sorted(rows, key=lambda row: (-len(row["source_ids"]), row["id"]))[0]]


def train(pilot=False):
    rows, manifest, histories, _ = require_inputs()
    study = recovery.study
    study.data.require_gpu()
    torch.backends.cudnn.allow_tf32 = False
    config = Config()
    probes = training_probes(rows)
    used = probes if pilot else rows
    inputs = []
    for row in used:
        entry = manifest[row["id"]]
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("frozen full source feature changed")
        inputs.append(Path(entry["path"]))
    if not pilot:
        for root in (PILOT, FIT_PILOT):
            study.data.terminal(root)
            inputs += [root / name for name in ("producer.json", "result.json", "worker_terminal_receipt.json")]
        if (study.q.read(PILOT / "result.json")["status"] != "systems_clean_word_training_passed"
                or study.q.read(FIT_PILOT / "result.json")["status"] != "systems_clean_word_fit_path_passed"):
            raise ValueError("actual training and fit-path CUDA pilots required")
    root = PILOT if pilot else TRAIN
    p = producing(root, "clean_word_maximum_frame_gradient_reload_pilot" if pilot else "fixed_complete690_clean_word_training",
        inputs, {"actual_updates": 2 if pilot else config.updates, "batch": 1,
            "cases": len(used), "physical_gpu": 0,
            "pilot_uses_maximum_learning_rate": pilot, "recovery_snapshot_interval": 4096})
    torch.manual_seed(config.model_seed)
    torch.cuda.manual_seed_all(config.model_seed)
    forecaster, branch, codec = model()
    initial = study.parameter_hash(branch)
    frozen = study.parameter_hash(branch, frozen_only=True)
    forecaster_hash = study.parameter_hash(forecaster)
    codec_hash = recovery.evaluation.reference.core.released.module_hash(codec)
    if not pilot and initial != study.q.read(PILOT / "result.json")["initial_parameter_sha256"]:
        raise ValueError("full training must start from the same learned checkpoint as pilot")
    parameters = [value for value in branch.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.max_lr, betas=(.9, .95), weight_decay=.01)
    sampler = torch.Generator().manual_seed(config.sampler_seed)
    torch.manual_seed(config.model_seed)
    torch.cuda.manual_seed_all(config.model_seed)
    branch.train()
    torch.cuda.reset_peak_memory_stats()
    (root / "checkpoints").mkdir()
    updates = 2 if pilot else config.updates
    for step in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        learning_rate = config.max_lr if pilot else rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        row = probes[step - 1] if pilot else rows[int(torch.randint(len(rows), (1,), generator=sampler))]
        batch = study.base.batch_for(row, manifest,
            frame=config.max_source_frame if pilot and step == 2 else None)
        set_source_frame(branch.generator, batch["source_ids"].shape[1], maximum=8192)
        loss = clean_backward(branch, batch)
        if (any(value.grad is not None for value in forecaster.parameters())
                or any(value.grad is not None for value in codec.parameters())
                or any(value.grad is not None for value in branch.parameters() if not value.requires_grad)
                or branch.source_head.source_projection.weight.grad is None
                or branch.word_projection.weight.grad is None):
            raise ValueError("clean-word versus frozen gradient routing failed")
        if pilot and (not torch.count_nonzero(branch.source_head.source_projection.weight.grad)
                or not torch.count_nonzero(branch.word_projection.weight.grad)
                or not any(value.grad is not None and torch.count_nonzero(value.grad)
                           for value in branch.word_decoder.parameters())):
            raise ValueError("pilot requires positive source, word projection and decoder gradients")
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        if torch.cuda.max_memory_allocated() > 8_000_000_000:
            raise ValueError("clean-word head allocation exceeds8GB")
        entry = {"step": step, "id": row["id"], "source_frame": branch.generator.prefix_len,
            "correct_history_sha256": study.base.tensor_digest({"codes": batch["codes"][:, :608]}),
            "target_word_sha256": study.base.tensor_digest({"words": batch["words"]}),
            "lr": learning_rate, "clean_word_CE": loss, "gradient_norm": float(norm)}
        with (root / "trace.jsonl").open("a") as handle:
            handle.write(json.dumps(entry, allow_nan=False) + "\n")
        if pilot or step % 32 == 0:
            print(json.dumps(entry), flush=True)
        del batch
        if not pilot and step % 4096 == 0 and step < updates:
            if (study.parameter_hash(branch, frozen_only=True) != frozen
                    or study.parameter_hash(forecaster) != forecaster_hash
                    or recovery.evaluation.reference.core.released.module_hash(codec) != codec_hash):
                raise ValueError("frozen modules changed before recovery save")
            save(root / "checkpoints" / f"step_{step:06d}.pt", branch, optimizer, sampler, step, p)
    save(root / "model.pt", branch, optimizer, sampler, updates, p)
    probes_after_reload = study.reload_check(root / "model.pt", forecaster, branch, optimizer, sampler, probes, manifest)
    if any(probe["coarse_codes"] != [histories[probe["id"]]["coarse_codes"]] for probe in probes_after_reload):
        raise ValueError("source-only reload probes differ from the frozen forecaster cache")
    if (study.parameter_hash(branch) == initial
            or study.parameter_hash(branch, frozen_only=True) != frozen
            or study.parameter_hash(forecaster) != forecaster_hash
            or recovery.evaluation.reference.core.released.module_hash(codec) != codec_hash
            or torch.cuda.max_memory_allocated() > 8_000_000_000):
        raise ValueError("clean training changed frozen state or exceeded its memory contract")
    study.data.verify(p)
    write_json(root / "source_only_reload_probe.json", probes_after_reload)
    result = {"status": "systems_clean_word_training_passed" if pilot else "clean_word_trained_pending_fitting_check",
        "updates": updates, "cases": len(used), "training_pool_cases": 690,
        "initial_parameter_sha256": initial, "frozen_word_branch_parameter_sha256": frozen,
        "forecaster_parameter_sha256_unchanged": forecaster_hash, "codec_hash_unchanged": codec_hash,
        "checkpoint_sha256": file_hash(root / "model.pt"), "reload_exact": True,
        "source_only_composed_reload_exact": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "task_accuracy": None, "NCM_success_established": False, "final_test_opened": False,
        "speed_tested": False, "producer_fingerprint": p["fingerprint"]}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


def fit(pilot=False):
    _, manifest, histories, selected = require_inputs()
    study = recovery.study
    study.data.require_gpu()
    torch.backends.cudnn.allow_tf32 = False
    source = PILOT if pilot else TRAIN
    root = FIT_PILOT if pilot else FIT
    study.data.terminal(source)
    trained = study.q.read(source / "result.json")
    expected_step = 2 if pilot else Config().updates
    if trained["updates"] != expected_step or not trained["reload_exact"]:
        raise ValueError("observed complete clean-word model required")
    inputs = [source / name for name in ("producer.json", "result.json", "model.pt", "worker_terminal_receipt.json")]
    for row in selected:
        entry = manifest[row["id"]]
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("frozen probe features changed")
        inputs.append(Path(entry["path"]))
    p = producing(root, "clean_word_complete_TRAIN_fit_path_pilot" if pilot else "clean_word_fixed32_TRAIN_fitting",
        inputs, {"actual_updates": 0, "model_checkpoint_updates": expected_step,
            "cases": 32, "physical_gpu": 0, "systems_pilot": pilot,
            "semantic_quality_measured": False})
    if file_hash(source / "model.pt") != trained["checkpoint_sha256"]:
        raise ValueError("selected clean-word model bytes changed")
    forecaster, branch, codec = model(source / "model.pt", expected_step)
    forecaster.cuda().eval().requires_grad_(False)
    branch.eval().requires_grad_(False)
    before = study.parameter_hash(forecaster), study.parameter_hash(branch)
    codec_hash = recovery.evaluation.reference.core.released.module_hash(codec)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(study.data.GPT2, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    measurements = []
    for row in selected:
        measurements.append(diagnostic.measure(forecaster, branch, row, manifest, histories, tokenizer))
        write_json(root / "case_measurements.json", measurements)
        if len(measurements) % 8 == 0:
            print(json.dumps({"measured": len(measurements), "total": 32}), flush=True)
    components = diagnostic.aggregate(measurements)
    gate = None if pilot else fitting_gate(components)
    if (before != (study.parameter_hash(forecaster), study.parameter_hash(branch))
            or recovery.evaluation.reference.core.released.module_hash(codec) != codec_hash
            or torch.cuda.max_memory_allocated() > 8_000_000_000):
        raise ValueError("read-only fit path state or memory gate failed")
    study.data.verify(p)
    result = {"status": "systems_clean_word_fit_path_passed" if pilot else "clean_word_fitting_measured",
        "cases": 32, "components": components, "clean_fit_gate": gate,
        "checkpoint_sha256": trained["checkpoint_sha256"], "parameters_and_codec_unchanged": True,
        "fresh_histories_match_training_cache": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "task_accuracy": None, "semantic_quality_measured": False,
        "NCM_success_established": False, "final_test_opened": False, "speed_tested": False,
        "measurements_sha256": file_hash(root / "case_measurements.json"),
        "producer_fingerprint": p["fingerprint"]}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


def run_stage(name, command, destination, journal, producer, *, existing=False, cpu=False, numerical=True):
    study = recovery.study
    if destination.exists() != existing:
        raise ValueError("freshness contract differs: " + name)
    study.data.verify(producer)
    env = dict(os.environ)
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    child = subprocess.Popen(command, env=env)
    journal.update(status="running", active_stage=name, child_pid=child.pid, observer_pid=os.getpid())
    write_json(WORKFLOW / "progress.json", journal)
    code = child.wait()
    receipt = {"status": "observed_worker_terminal", "stage": name, "child_pid": child.pid,
        "observer_pid": os.getpid(), "observer": "subprocess.wait", "exit_code": code,
        "command": command, "destination": str(destination)}
    write_json(WORKFLOW / "stage_receipts" / (name + ".json"), receipt)
    if destination.exists():
        path = destination / ("registry_terminal_receipt.json" if name == "register" else "worker_terminal_receipt.json")
        if path.exists():
            raise ValueError("terminal receipt must never be overwritten")
        write_json(path, receipt)
    journal["stages"].append(receipt)
    write_json(WORKFLOW / "progress.json", journal)
    if code != 0:
        raise RuntimeError("stage failed with observed exit: " + name)
    if numerical:
        study.data.terminal(destination)
    study.data.verify(producer)


def workflow():
    recovery.configure()
    registry = Path("data/hypothesis_registry") / (HYPOTHESIS + ".json")
    if any(path.exists() for path in (WORKFLOW, PILOT, FIT_PILOT, TRAIN, FIT, RESULTS, registry)) or REPORT.exists():
        raise ValueError("fresh clean-word workflow and all destinations required")
    p = producing(WORKFLOW, "supervise_complete690_clean_word_fitting", [],
        {"actual_updates": 0, "child_training_updates": Config().updates + 2,
         "all_stages": ["train_pilot", "fit_pilot", "train", "fit", "finalize", "register"]})
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "stages": [], "final_test_opened": False,
               "task_accuracy": None, "producer_fingerprint": p["fingerprint"]}
    try:
        stages = [("train_pilot", "pilot", PILOT), ("fit_pilot", "fit-pilot", FIT_PILOT),
                  ("train", "train", TRAIN), ("fit", "fit", FIT)]
        for name, stage, root in stages:
            run_stage(name, [sys.executable, "-u", "-m", MODULE, stage], root, journal, p)
        for stage in ("finalize", "register"):
            run_stage(stage, [sys.executable, "-u", "-m", FINALIZER_MODULE, stage],
                RESULTS, journal, p, existing=stage == "register", cpu=True, numerical=False)
        campaign = recovery.study.q.read(RESULTS / "campaign_results.json")
        receipt = recovery.study.q.read(RESULTS / (HYPOTHESIS + "_registry_readback.json"))
        if (receipt["status"] != "registered_and_read_back"
                or receipt["record"]["campaign_sha256"] != file_hash(RESULTS / "campaign_results.json")
                or receipt["record"]["confirmation_status"] != campaign["classification"]):
            raise ValueError("exact component registry readback required")
        journal.update(status="clean_word_component_completed", active_stage=None,
            clean_fit_gate=campaign["clean_fit_gate"], classification=campaign["classification"],
            forecaster_matched_population_training_still_required=True,
            semantic_quality_measured=False, NCM_success_established=False,
            campaign_sha256=file_hash(RESULTS / "campaign_results.json"))
        write_json(WORKFLOW / "result.json", journal)
        print(json.dumps(journal), flush=True)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["check", "pilot", "fit-pilot", "train", "fit", "workflow"])
    stage = parser.parse_args().stage
    if stage == "check":
        rows, _, _, selected = require_inputs()
        sampler = torch.Generator().manual_seed(Config().sampler_seed)
        sample_counts = [0] * len(rows)
        for _ in range(Config().updates):
            sample_counts[int(torch.randint(len(rows), (1,), generator=sampler))] += 1
        if not min(sample_counts):
            raise ValueError("fixed full training stream must cover every TRAIN case")
        print(json.dumps({"TRAIN_cases": len(rows), "fixed_probe_cases": len(selected),
            "minimum_sample_count": min(sample_counts), "maximum_sample_count": max(sample_counts),
            "config": asdict(Config()), "task_accuracy": None}))
    elif stage == "workflow":
        workflow()
    elif stage in ("pilot", "train"):
        train(stage == "pilot")
    else:
        fit(stage == "fit-pilot")
