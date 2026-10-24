"""Matched whole-TRAIN dropout comparison, conditional on clean-word fit failure."""
import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from experiments.structure_net import ncm_matched_clean_word as clean
from experiments.structure_net import ncm_matched_forecaster as forecaster
from experiments.structure_net.ncm_device_geometry import set_source_frame
from experiments.structure_net.ncm_expanded_word_data_workflow import kernel_identity
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_native_label_quality import dependencies

ROOT = clean.ROOT
HYPOTHESIS = "ncm-clean-word-dropout-fit-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-clean-word-dropout-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-clean-word-dropout.md")
WORKFLOW = ROOT / "20261023_clean_word_dropout_workflow_v1"
RESULTS = ROOT / "20261023_clean_word_dropout_results_v1"
MODULE = "experiments.structure_net.ncm_clean_word_dropout"
AUDITOR = "experiments.neural_architecture_lab.finalize_ncm_clean_word_dropout"
AUDITOR_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_clean_word_dropout.py")
ARMS = ("original_dropout", "zero_dropout")


@dataclass(frozen=True)
class Config:
    updates: int = 8192
    warmup: int = 128
    max_lr: float = 3e-5
    model_seed: int = 30142
    sampler_seed: int = 31142
    max_source_frame: int = 3328
    snapshot_interval: int = 4096
    batch: int = 1
    beta1: float = .9
    beta2: float = .95
    optimizer_eps: float = 1e-8
    weight_decay: float = .01
    gradient_clip: float = 1.

    def __post_init__(self):
        if (not 1 <= self.warmup <= self.updates or self.max_lr <= 0
                or self.model_seed == self.sampler_seed or self.batch != 1
                or self.snapshot_interval < 1 or self.updates % self.snapshot_interval
                or not 0 < self.beta1 < 1 or not 0 < self.beta2 < 1
                or self.optimizer_eps <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0):
            raise ValueError("invalid fixed dropout comparison configuration")


def root(arm, stage):
    if arm not in ARMS or stage not in ("pilot", "train", "fit"):
        raise ValueError("unknown fixed arm or stage")
    return ROOT / f"20261023_clean_word_dropout_{arm}_{stage}_v1"


def rate(step, config=Config()):
    if not 1 <= step <= config.updates:
        raise ValueError("step outside fixed comparison budget")
    return config.max_lr * min(step / config.warmup, 1.)


def dropout_map(branch):
    result = {}
    for scope, net in (("transformer", branch.generator.transformer), ("decoder", branch.word_decoder)):
        for name, module in net.named_modules():
            prefix = scope + ("." + name if name else "")
            if isinstance(module, torch.nn.Dropout):
                result[prefix + ".p"] = float(module.p)
            value = getattr(module, "dropout", None)
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                result[prefix + ".dropout"] = float(value)
    if not result or any(not 0 <= value < 1 for value in result.values()):
        raise ValueError("complete valid module and functional attention dropout rates required")
    return result


def apply_dropout(branch, arm, original=None):
    if arm not in ARMS:
        raise ValueError("unknown dropout arm")
    before = dropout_map(branch)
    if original is not None and before != original:
        raise ValueError("fresh model original dropout map differs")
    if arm == "zero_dropout":
        for net in (branch.generator.transformer, branch.word_decoder):
            for module in net.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.
                value = getattr(module, "dropout", None)
                if isinstance(value, (float, int)) and not isinstance(value, bool):
                    module.dropout = 0.
    after = dropout_map(branch)
    if after != {name: value if arm == ARMS[0] else 0. for name, value in before.items()}:
        raise ValueError("dropout intervention coverage differs")
    return {"original": before, "applied": after}


def admission():
    """True: registered word-fit miss and forecaster declined. False: word passed."""
    value = forecaster.admission()
    if value is None:
        return None
    if value:
        return False
    path = forecaster.WORKFLOW / "result.json"
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if (result["status"] != "not_admitted_clean_word_fit_gate_failed"
            or result["stages"] or result["model_training_executed"]):
        raise ValueError("queued forecaster must decline the failed clean-word component")
    return True


def data_inputs(require_gate=True):
    dependencies()
    clean.recovery.configure()
    if require_gate and admission() is not True:
        raise ValueError("registered clean-word fit miss and declined forecaster required")
    rows, manifest, histories, selected = clean.require_inputs()
    study = clean.recovery.study
    upstream = study.q.read(clean.WORKFLOW / "producer.json")["input_hashes"]
    for name in ("cases.json", "manifest.json", "judgment_cache.json"):
        path = study.expanded.DEST / name
        if upstream.get(str(path)) != file_hash(path):
            raise ValueError("same actual complete690 population required")
    return rows, manifest, histories, selected


def initial_result():
    if admission() is not True:
        raise ValueError("negative upstream gate must be complete")
    study = clean.recovery.study
    trained = study.q.read(clean.TRAIN / "result.json")
    campaign = study.q.read(clean.RESULTS / "campaign_results.json")
    if (campaign["training"] != trained or campaign["classification"] != "clean_word_fit_gate_failed"
            or file_hash(clean.TRAIN / "model.pt") != trained["checkpoint_sha256"]):
        raise ValueError("unchanged final registered clean-word initializer required")
    return trained


def producing(destination, stage, inputs, config, *, pending=False):
    study = clean.recovery.study
    upstream = study.q.read(clean.WORKFLOW / "producer.json")
    paths = [Path(__file__), PLAN, AUDITOR_FILE, Path(forecaster.__file__), forecaster.PLAN]
    paths += [Path(name) for name in upstream["source_hashes"]]
    paths += sorted(study.data.VENDOR.rglob("*.py"))
    files = list({str(path.resolve()): path for path in paths}.values())
    common = [clean.WORKFLOW / "producer.json", forecaster.WORKFLOW / "producer.json"]
    common += [study.expanded.DEST / name for name in
        ("producer.json", "result.json", "cases.json", "manifest.json", "judgment_cache.json", "worker_terminal_receipt.json")]
    if not pending:
        trained = initial_result()
        common += [clean.WORKFLOW / "result.json", forecaster.WORKFLOW / "result.json",
            clean.RESULTS / "campaign_results.json", clean.RESULTS / (clean.HYPOTHESIS + "_registry_readback.json")]
        common += [clean.TRAIN / name for name in
            ("producer.json", "result.json", "model.pt", "worker_terminal_receipt.json")]
        config = {**config, "initial_word_checkpoint_sha256": trained["checkpoint_sha256"]}
    p = study.data.freeze(destination, stage, list(inputs) + common + files,
        {**asdict(Config()), **config, "hypothesis_id": HYPOTHESIS,
         "TRAIN_population": 690, "precision": "FP32masters/BF16forward/FP32loss",
         "matmul_TF32": False, "cuDNN_TF32": False, "word_objective": "correct_coarse_history_CE_only",
         "body_MLPs_codec_forecaster_and_unused_heads_frozen": True,
         "finest_labels_or_word_targets_in_prediction": False,
         "parent_loaded": False, "task_accuracy": None, "semantic_quality_measured": False,
         "final_test_opened": False, "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (destination / "producer_source" / f"dropout_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = study.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(destination / "producer.json", p)
    return p


def model(arm, path=None, expected_step=None):
    trained = initial_result()
    study = clean.recovery.study
    fixed, branch, codec = clean.model(clean.TRAIN / "model.pt", clean.Config().updates)
    original = dropout_map(branch)
    if set(original.values()) != {.1, .3}:
        raise ValueError("actual released transformer/decoder original dropout changed")
    initial = study.parameter_hash(branch)
    maps = apply_dropout(branch, arm)
    if study.parameter_hash(branch) != initial:
        raise ValueError("dropout intervention changed parameter bytes")
    if path is not None:
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        validate_checkpoint(saved, arm, maps, expected_step, trained["checkpoint_sha256"])
        set_source_frame(branch.generator, saved["source_frame"], maximum=8192)
        branch.load_state_dict(saved["model"], strict=True)
        if study.parameter_hash(branch, frozen_only=True) != clean.FROZEN_WORD:
            raise ValueError("selected model changed unused roots or heads")
    return fixed, branch, codec, maps, initial


def validate_checkpoint(saved, arm, maps, expected_step, initial_sha):
    if (saved["step"] != expected_step or saved["config"] != asdict(Config())
            or saved["arm"] != arm or saved["dropout_maps"] != maps
            or saved["initial_word_checkpoint_sha256"] != initial_sha
            or saved["forecaster_checkpoint_sha256"] != clean.INITIAL_FORECASTER):
        raise ValueError("fixed selected dropout checkpoint identity differs")


def save(path, arm, branch, optimizer, sampler, step, maps, producer):
    if path.exists():
        raise ValueError("checkpoint overwrite prohibited")
    torch.save({"model": branch.state_dict(), "optimizer": optimizer.state_dict(),
        "parameter_names": [name for name, _ in branch.named_parameters()],
        "step": step, "arm": arm, "source_frame": branch.generator.prefix_len,
        "dropout_maps": maps, "config": asdict(Config()), "sampler_rng": sampler.get_state(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "initial_word_checkpoint_sha256": producer["config"]["initial_word_checkpoint_sha256"],
        "forecaster_checkpoint_sha256": clean.INITIAL_FORECASTER,
        "producer_fingerprint": producer["fingerprint"]}, path)


def train(arm, pilot=False):
    rows, manifest, histories, _ = data_inputs()
    study = clean.recovery.study
    study.data.require_gpu()
    torch.backends.cudnn.allow_tf32 = False
    config = Config()
    probes = clean.training_probes(rows)
    used = probes if pilot else rows
    inputs = []
    for row in used:
        entry = manifest[row["id"]]
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("unchanged complete TRAIN features required")
        inputs.append(Path(entry["path"]))
    if not pilot:
        for name in ARMS:
            location = root(name, "pilot")
            study.data.terminal(location)
            if study.q.read(location / "result.json")["status"] != "systems_dropout_gradient_reload_passed":
                raise ValueError("both actual successful arm pilots required")
            inputs += [location / f for f in ("producer.json", "result.json", "worker_terminal_receipt.json")]
    destination = root(arm, "pilot" if pilot else "train")
    updates = 2 if pilot else config.updates
    p = producing(destination, "matched_dropout_training_pilot" if pilot else "matched_dropout_whole_TRAIN_training",
        inputs, {"arm": arm, "actual_updates": updates, "cases": len(used), "physical_gpu": 0,
                 "systems_only": pilot, "pilot_uses_maximum_learning_rate": pilot})
    torch.manual_seed(config.model_seed)
    torch.cuda.manual_seed_all(config.model_seed)
    fixed, branch, codec, maps, initial = model(arm)
    frozen = study.parameter_hash(branch, frozen_only=True)
    fixed_hash = study.parameter_hash(fixed)
    codec_hash = clean.recovery.evaluation.reference.core.released.module_hash(codec)
    parameters = [v for v in branch.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.max_lr, betas=(config.beta1, config.beta2),
        eps=config.optimizer_eps, weight_decay=config.weight_decay)
    sampler = torch.Generator().manual_seed(config.sampler_seed)
    torch.manual_seed(config.model_seed)
    torch.cuda.manual_seed_all(config.model_seed)
    branch.train()
    torch.cuda.reset_peak_memory_stats()
    (destination / "checkpoints").mkdir()
    for step in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        lr = config.max_lr if pilot else rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        row = probes[step - 1] if pilot else rows[int(torch.randint(len(rows), (1,), generator=sampler))]
        batch = study.base.batch_for(row, manifest,
            frame=config.max_source_frame if pilot and step == 2 else None)
        set_source_frame(branch.generator, batch["source_ids"].shape[1], maximum=8192)
        loss = clean.clean_backward(branch, batch)
        if (any(v.grad is not None for v in fixed.parameters())
                or any(v.grad is not None for v in codec.parameters())
                or any(v.grad is not None for v in branch.parameters() if not v.requires_grad)):
            raise ValueError("frozen modules received gradients")
        if pilot and (not branch.source_head.source_projection.weight.grad.abs().sum()
                or not branch.word_projection.weight.grad.abs().sum()
                or not any(v.grad is not None and v.grad.abs().sum() for v in branch.word_decoder.parameters())):
            raise ValueError("positive source/projection/decoder gradients required")
        norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        if dropout_map(branch) != maps["applied"] or torch.cuda.max_memory_allocated() > 8_000_000_000:
            raise ValueError("dropout configuration or8GB memory contract changed")
        item = {"step": step, "id": row["id"], "source_frame": branch.generator.prefix_len,
            "correct_history_sha256": study.base.tensor_digest({"codes": batch["codes"][:, :608]}),
            "target_word_sha256": study.base.tensor_digest({"words": batch["words"]}),
            "lr": lr, "clean_word_CE": loss, "gradient_norm": float(norm)}
        with (destination / "trace.jsonl").open("a") as f:
            f.write(json.dumps(item, allow_nan=False) + "\n")
        if pilot or step % 32 == 0:
            print(json.dumps(item), flush=True)
        del batch
        if not pilot and step % config.snapshot_interval == 0 and step < updates:
            save(destination / "checkpoints" / f"step_{step:06d}.pt", arm,
                branch, optimizer, sampler, step, maps, p)
    save(destination / "model.pt", arm, branch, optimizer, sampler, updates, maps, p)
    probes_out = study.reload_check(destination / "model.pt", fixed, branch, optimizer, sampler, probes, manifest)
    if (dropout_map(branch) != maps["applied"] or study.parameter_hash(branch) == initial
            or study.parameter_hash(branch, frozen_only=True) != frozen
            or study.parameter_hash(fixed) != fixed_hash
            or clean.recovery.evaluation.reference.core.released.module_hash(codec) != codec_hash
            or torch.cuda.max_memory_allocated() > 8_000_000_000
            or any(item["coarse_codes"] != [histories[item["id"]]["coarse_codes"]] for item in probes_out)):
        raise ValueError("frozen state, native cache parity, reload dropout or memory differs")
    write_json(destination / "source_only_reload_probe.json", probes_out)
    study.data.verify(p)
    result = {"status": "systems_dropout_gradient_reload_passed" if pilot else "matched_dropout_word_trained",
        "arm": arm, "updates": updates, "cases": len(used), "training_population": 690,
        "initial_parameter_sha256": initial, "frozen_word_parameter_sha256": frozen,
        "frozen_forecaster_parameter_sha256": fixed_hash, "codec_hash_unchanged": codec_hash,
        "dropout_maps": maps, "reload_exact": True, "source_only_composed_reload_exact": True,
        "checkpoint_sha256": file_hash(destination / "model.pt"),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "task_accuracy": None,
        "semantic_quality_measured": False, "final_test_opened": False, "speed_tested": False,
        "producer_fingerprint": p["fingerprint"]}
    write_json(destination / "result.json", result)
    print(json.dumps(result), flush=True)


def fit(arm):
    _, manifest, histories, selected = data_inputs()
    study = clean.recovery.study
    study.data.require_gpu()
    torch.backends.cudnn.allow_tf32 = False
    source = root(arm, "train")
    study.data.terminal(source)
    trained = study.q.read(source / "result.json")
    if (trained["updates"] != Config().updates or not trained["reload_exact"]
            or file_hash(source / "model.pt") != trained["checkpoint_sha256"]):
        raise ValueError("exact final arm checkpoint required")
    inputs = [source / f for f in ("producer.json", "result.json", "model.pt", "worker_terminal_receipt.json")]
    inputs += [Path(manifest[row["id"]]["path"]) for row in selected]
    destination = root(arm, "fit")
    p = producing(destination, "matched_dropout_fixed32_TRAIN_fitting", inputs,
        {"arm": arm, "actual_updates": 0, "cases": 32, "physical_gpu": 0})
    fixed, branch, codec, maps, _ = model(arm, source / "model.pt", Config().updates)
    fixed.cuda().eval().requires_grad_(False)
    branch.eval().requires_grad_(False)
    before = study.parameter_hash(fixed), study.parameter_hash(branch)
    codec_hash = clean.recovery.evaluation.reference.core.released.module_hash(codec)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(study.data.GPT2, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    measured = []
    for row in selected:
        item = clean.diagnostic.measure(fixed, branch, row, manifest, histories, tokenizer)
        if len(measured) == 0 and clean.diagnostic.measure(fixed, branch, row, manifest, histories, tokenizer) != item:
            raise ValueError("fresh arm fitting must repeat exactly")
        measured.append(item)
        write_json(destination / "case_measurements.json", measured)
        if len(measured) % 8 == 0:
            print(json.dumps({"arm": arm, "measured": len(measured), "total": 32}), flush=True)
    if (before != (study.parameter_hash(fixed), study.parameter_hash(branch))
            or clean.recovery.evaluation.reference.core.released.module_hash(codec) != codec_hash
            or dropout_map(branch) != maps["applied"] or torch.cuda.max_memory_allocated() > 8_000_000_000):
        raise ValueError("read-only arm fitting state or memory contract differs")
    components = clean.diagnostic.aggregate(measured)
    result = {"status": "matched_dropout_TRAIN_fitting_measured", "arm": arm,
        "cases": 32, "components": components, "clean_fit_gate": clean.fitting_gate(components),
        "checkpoint_sha256": trained["checkpoint_sha256"], "dropout_maps": maps,
        "parameters_and_codec_unchanged": True, "fresh_histories_match_training_cache": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "task_accuracy": None,
        "semantic_quality_measured": False, "final_test_opened": False, "speed_tested": False,
        "measurements_sha256": file_hash(destination / "case_measurements.json"),
        "producer_fingerprint": p["fingerprint"]}
    study.data.verify(p)
    write_json(destination / "result.json", result)
    print(json.dumps(result), flush=True)


def run_stage(name, command, destination, journal, producer, *, existing=False, cpu=False, numerical=True):
    study = clean.recovery.study
    if destination.exists() != existing:
        raise ValueError("fresh stage destination required: " + name)
    study.data.verify(producer)
    env = dict(os.environ)
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    child = subprocess.Popen(command, env=env)
    journal.update(status="running", active_stage=name, child_pid=child.pid)
    write_json(WORKFLOW / "progress.json", journal)
    code = child.wait()
    receipt = {"status": "observed_worker_terminal", "stage": name, "child_pid": child.pid,
        "observer_pid": os.getpid(), "observer": "subprocess.wait", "exit_code": code,
        "command": command, "destination": str(destination)}
    write_json(WORKFLOW / "stage_receipts" / (name + ".json"), receipt)
    if destination.exists():
        path = destination / ("registry_terminal_receipt.json" if name == "register" else "worker_terminal_receipt.json")
        if path.exists():
            raise ValueError("terminal receipt overwrite prohibited")
        write_json(path, receipt)
    journal["stages"].append(receipt)
    write_json(WORKFLOW / "progress.json", journal)
    if code != 0:
        raise RuntimeError("stage failed with observed exit: " + name)
    if numerical:
        study.data.terminal(destination)
    study.data.verify(producer)


def workflow():
    dependencies()
    clean.recovery.configure()
    targets = [WORKFLOW, RESULTS, REPORT, Path("data/hypothesis_registry") / (HYPOTHESIS + ".json")]
    targets += [root(arm, stage) for arm in ARMS for stage in ("pilot", "train", "fit")]
    if any(path.exists() for path in targets):
        raise ValueError("fresh comparison workflow and all destinations required")
    identities = []
    for location in (clean.WORKFLOW, forecaster.WORKFLOW):
        if (location / "progress.json").exists():
            value = clean.recovery.study.q.read(location / "progress.json")
            identity = kernel_identity(value["observer_pid"])
            if identity is None and not (location / "result.json").exists():
                raise ValueError("unfinished upstream supervisor must be authoritatively live")
            if identity is not None:
                identities.append({"workflow": str(location), **identity})
        elif not (location / "result.json").exists():
            raise ValueError("unfinished upstream supervisor requires live progress identity")
    p = producing(WORKFLOW, "observe_word_fit_miss_then_compare_dropout", [],
        {"actual_updates": 0, "upstream_kernel_identities": identities, "poll_seconds": 5,
         "upstream_workers_restarted": False, "training": False}, pending=True)
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "waiting_for_clean_word_fitting_and_forecaster_decline",
        "observer_pid": os.getpid(), "upstream_kernel_identities": identities,
        "stages": [], "final_test_opened": False, "producer_fingerprint": p["fingerprint"]}
    write_json(WORKFLOW / "progress.json", journal)
    try:
        while True:
            if any((location / "failure.json").exists() for location in (clean.WORKFLOW, forecaster.WORKFLOW)):
                raise RuntimeError("upstream failure preserved")
            ready = admission()
            alive = any(kernel_identity(item["pid"]) == {k: item[k] for k in ("pid", "start_ticks")}
                        for item in identities)
            if ready is False:
                journal.update(status="not_admitted_clean_word_fit_passed", model_training_executed=False,
                    task_accuracy=None, active_stage=None)
                write_json(WORKFLOW / "result.json", journal)
                return
            if ready is True and not alive:
                break
            if ready is None and not alive:
                raise RuntimeError("upstream terminal without required registered negative-branch evidence")
            time.sleep(5)
        for arm in ARMS:
            name = arm + "_pilot"
            run_stage(name, [sys.executable, "-u", "-m", MODULE, "pilot", "--arm", arm],
                root(arm, "pilot"), journal, p)
        for arm in ARMS:
            for stage in ("train", "fit"):
                run_stage(arm + "_" + stage, [sys.executable, "-u", "-m", MODULE, stage, "--arm", arm],
                    root(arm, stage), journal, p)
        for stage in ("finalize", "register"):
            run_stage(stage, [sys.executable, "-u", "-m", AUDITOR, stage], RESULTS, journal, p,
                existing=stage == "register", cpu=True, numerical=False)
        study = clean.recovery.study
        path = RESULTS / "campaign_results.json"
        campaign = study.q.read(path)
        receipt = study.q.read(RESULTS / (HYPOTHESIS + "_registry_readback.json"))
        if (receipt["status"] != "registered_and_read_back"
                or receipt["record"]["campaign_sha256"] != file_hash(path)
                or receipt["record"]["confirmation_status"] != campaign["classification"]):
            raise ValueError("exact conservative dropout comparison registry readback required")
        journal.update(status="dropout_fitting_comparison_completed", active_stage=None,
            classification=campaign["classification"], model_training_executed=True,
            task_accuracy=None, semantic_quality_measured=False, NCM_success_established=False,
            campaign_sha256=file_hash(path))
        write_json(WORKFLOW / "result.json", journal)
        print(json.dumps(journal), flush=True)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("check", "pilot", "train", "fit", "workflow"))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.stage in ("pilot", "train", "fit") and args.arm is None:
        parser.error("an explicit fixed arm is required")
    if args.stage == "check":
        rows, _, _, selected = data_inputs(require_gate=False)
        sampler = torch.Generator().manual_seed(Config().sampler_seed)
        counts = [0] * len(rows)
        for _ in range(Config().updates):
            counts[int(torch.randint(len(rows), (1,), generator=sampler))] += 1
        if not min(counts):
            raise ValueError("fixed stream must cover every690 TRAIN row")
        print(json.dumps({"TRAIN_cases": len(rows), "fixed_probes": len(selected),
            "minimum_sample_count": min(counts), "maximum_sample_count": max(counts),
            "config": asdict(Config()), "upstream_admission": admission(), "task_accuracy": None}))
    elif args.stage == "workflow":
        workflow()
    elif args.stage == "fit":
        fit(args.arm)
    else:
        train(args.arm, args.stage == "pilot")


if __name__ == "__main__":
    main()
