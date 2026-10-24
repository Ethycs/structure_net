"""Conditionally train NCM concepts on the complete matched TRAIN population."""
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
from experiments.structure_net import ncm_source_hierarchy_core as core
from experiments.structure_net import ncm_word_final_train as legacy
from experiments.structure_net.ncm_device_geometry import set_source_frame
from experiments.structure_net.ncm_expanded_word_data_workflow import kernel_identity
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_native_label_quality import dependencies

ROOT = clean.ROOT
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-matched-forecaster-plan.md")
HYPOTHESIS = "ncm-matched-population-forecaster-v1"
PILOT = ROOT / "20261023_matched_forecaster_pilot_v1"
FIT_PILOT = ROOT / "20261023_matched_forecaster_fit_pilot_v1"
TRAIN = ROOT / "20261023_matched_forecaster_train_v1"
FIT = ROOT / "20261023_matched_forecaster_fit_v1"
WORKFLOW = ROOT / "20261023_matched_forecaster_workflow_v1"
RESULTS = ROOT / "20261023_matched_forecaster_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-matched-forecaster.md")
MODULE = "experiments.structure_net.ncm_matched_forecaster"
FINALIZER_MODULE = "experiments.neural_architecture_lab.finalize_ncm_matched_forecaster"
FINALIZER_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_matched_forecaster.py")
WIDTHS = (16, 24, 40, 56, 80, 104, 128, 160)


@dataclass(frozen=True)
class Config:
    updates: int = 49152
    warmup: int = 256
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    batch: int = 1
    beta1: float = .9
    beta2: float = .95
    weight_decay: float = .01
    optimizer_eps: float = 1e-8
    gradient_clip: float = 1.
    model_seed: int = 26142
    sampler_seed: int = 27142
    noise_seed: int = 28142
    corrupt_probability: float = .5
    maximum_uniform_rate: float = .35
    root_weight: float = 4.
    padded_weight: float = .25
    max_source_frame: int = 3328
    snapshot_interval: int = 8192
    fit_numerator: int = 99
    fit_denominator: int = 100

    def __post_init__(self):
        if (not 1 <= self.warmup <= self.updates or not 0 < self.min_lr <= self.max_lr
                or not 0 <= self.corrupt_probability <= 1
                or not 0 <= self.maximum_uniform_rate <= 1
                or self.snapshot_interval < 1 or self.updates % self.snapshot_interval
                or self.batch != 1 or not 0 < self.beta1 < 1 or not 0 < self.beta2 < 1
                or self.weight_decay < 0 or self.optimizer_eps <= 0 or self.gradient_clip <= 0
                or len({self.model_seed, self.sampler_seed, self.noise_seed}) != 3
                or not 0 < self.fit_numerator <= self.fit_denominator):
            raise ValueError("invalid fixed matched-forecaster configuration")


def admission():
    """None means pending; False is a completed valid upstream gate miss."""
    path = clean.WORKFLOW / "result.json"
    if not path.exists():
        return None
    try:
        parent = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    study = clean.recovery.study
    if (parent["status"] != "clean_word_component_completed"
            or len(parent["stages"]) != 6 or any(row["exit_code"] != 0 for row in parent["stages"])):
        raise ValueError("completed upstream workflow with six observed zero exits required")
    campaign_path = clean.RESULTS / "campaign_results.json"
    campaign = study.q.read(campaign_path)
    receipt = study.q.read(clean.RESULTS / (clean.HYPOTHESIS + "_registry_readback.json"))
    if (receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != file_hash(campaign_path)
            or parent["campaign_sha256"] != file_hash(campaign_path)
            or receipt["record"]["confirmation_status"] != campaign["classification"]
            or campaign["training_cases"] != 690 or campaign["semantic_quality_measured"]
            or campaign["final_test_opened"]
            or parent["clean_fit_gate"] != campaign["clean_fit_gate"]
            or receipt["record"]["clean_fit_gate"] != campaign["clean_fit_gate"]
            or campaign["clean_fit_gate"] != clean.fitting_gate(campaign["fitting"]["components"])):
        raise ValueError("exact registered complete690 clean-word component required")
    return bool(campaign["clean_fit_gate"] and parent["clean_fit_gate"]
                and receipt["record"]["clean_fit_gate"])


def require_admission():
    if admission() is not True:
        raise ValueError("registered upstream clean-word fitting gate must pass before model work")


def data_inputs(require_gate=True):
    dependencies()
    clean.recovery.configure()
    if require_gate:
        require_admission()
    study = clean.recovery.study
    rows, manifest = study.require_data()
    upstream_data = study.q.read(clean.WORKFLOW / "producer.json")["input_hashes"]
    for name in ("cases.json", "manifest.json", "judgment_cache.json"):
        path = study.expanded.DEST / name
        if upstream_data.get(str(path)) != file_hash(path):
            raise ValueError("forecaster TRAIN data must match the actual clean-word population exactly")
    if len(rows) != 690 or max(row["source_frame"] for row in rows) > Config().max_source_frame:
        raise ValueError("same complete690 TRAIN population and full source frames required")
    original = study.q.read(study.data.DATA / "cases.json")["train"]
    selected = clean.diagnostic.selection(rows, {row["id"] for row in original},
        study.q.read(clean.diagnostic.OLD_DIAGNOSTIC / "selection.json"))
    if study.require_forecaster()["checkpoint_sha256"] != clean.INITIAL_FORECASTER:
        raise ValueError("unchanged registered initial concept forecaster required")
    return rows, manifest, selected


def model(path=None, expected_step=None):
    study = clean.recovery.study
    net, codec = study.load_forecaster()
    core.concept_scope(net)
    expected_frozen = study.q.read(study.bootstrap.TRAIN / "result.json")["frozen_parameter_sha256_unchanged"]
    if study.parameter_hash(net, frozen_only=True) != expected_frozen:
        raise ValueError("word modules and finest head differ from the frozen initializer")
    if path is not None:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if (state["step"] != expected_step or state["config"] != asdict(Config())
                or state["initial_forecaster_checkpoint_sha256"] != clean.INITIAL_FORECASTER):
            raise ValueError("selected matched forecaster identity differs")
        set_source_frame(net.generator, state["source_frame"], maximum=8192)
        net.load_state_dict(state["model"], strict=True)
    if study.parameter_hash(net, frozen_only=True) != expected_frozen:
        raise ValueError("selected checkpoint changed frozen word modules or finest head")
    return net, codec, expected_frozen


def producing(root, stage, inputs, config, *, pending=False):
    study = clean.recovery.study
    files = [Path(__file__), PLAN, FINALIZER_FILE, Path(clean.__file__), clean.PLAN,
        Path(core.__file__), Path(legacy.__file__), Path(clean.recovery.__file__),
        Path(clean.diagnostic.__file__), Path("experiments/structure_net/ncm_device_geometry.py"),
        Path("experiments/structure_net/ncm_prefix_geometry.py"),
        Path("experiments/structure_net/ncm_expanded_word_data_workflow.py"),
        Path("experiments/structure_net/ncm_generated_history_study.py"),
        Path("experiments/structure_net/ncm_generated_history_core.py"),
        Path("experiments/structure_net/ncm_generated_history_eval.py"),
        Path("experiments/structure_net/ncm_word_final_eval.py"),
        Path("experiments/structure_net/ncm_word_final_data.py"),
        Path("experiments/structure_net/ncm_native_word_mechanism.py"),
        Path("experiments/structure_net/ncm_native_label_quality.py"),
        Path("experiments/structure_net/ncm_reviewed_draft_data.py"),
        Path("experiments/structure_net/ncm_expanded_word_data.py"),
        Path("experiments/structure_net/ncm_word_final_training.py"),
        Path("experiments/structure_net/ncm_word_final_head.py"),
        Path("experiments/structure_net/ncm_frozen_source_head.py"),
        Path("experiments/structure_net/ncm_blockwise_codec.py"),
        Path("experiments/structure_net/ncm_broad_learned_codec.py")]
    files += sorted(study.data.VENDOR.rglob("*.py"))
    common = [study.bootstrap.TRAIN / name for name in
        ("producer.json", "result.json", "model.pt", "worker_terminal_receipt.json")]
    common += [study.archive.DEST / "campaign_results.json",
        study.archive.DEST / (study.archive.worker.HYPOTHESIS + "_registry_readback.json")]
    common += [study.expanded.DEST / name for name in
        ("producer.json", "result.json", "cases.json", "manifest.json", "judgment_cache.json", "worker_terminal_receipt.json")]
    common += [clean.WORKFLOW / "producer.json",
        clean.diagnostic.OLD_DIAGNOSTIC / "selection.json", study.data.DATA / "cases.json"]
    common += [study.data.INITIAL / name for name in ("tokenizer.pt", "generator.pt", "config.yaml")]
    if not pending:
        require_admission()
        common += [clean.WORKFLOW / "result.json", clean.RESULTS / "campaign_results.json",
            clean.RESULTS / (clean.HYPOTHESIS + "_registry_readback.json"), clean.FIT / "result.json"]
    p = study.data.freeze(root, stage, list(inputs) + common + files,
        {**asdict(Config()), **config, "hypothesis_id": HYPOTHESIS,
         "initial_forecaster_checkpoint_sha256": clean.INITIAL_FORECASTER,
         "TRAIN_population": 690, "word_losses": False,
         "word_modules_finest_head_and_codec_frozen": True,
         "original_parent_weights_and_MLPs_frozen": True, "parent_loaded": False,
         "word_targets_in_prediction": False, "source_prediction": True,
         "finest_target_codes_in_prediction": False, "finest_prediction_history": "constant_zero",
         "precision": "FP32masters/BF16forward/FP32loss_and_reconstruction",
         "matmul_TF32": False, "cuDNN_TF32": False, "task_accuracy": None,
         "semantic_quality_measured": False, "final_test_opened": False, "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for index, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"matched_forecaster_{index}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = study.q.digest({key: value for key, value in p.items() if key != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def save(path, net, optimizer, sampler, noise, step, producer):
    if path.exists():
        raise ValueError("checkpoint overwrite prohibited")
    torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
        "source_frame": net.generator.prefix_len, "sampler_rng": sampler.get_state(),
        "noise_rng": noise.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(), "config": asdict(Config()),
        "frozen_parameter_names": [name for name, value in net.named_parameters() if not value.requires_grad],
        "initial_forecaster_checkpoint_sha256": clean.INITIAL_FORECASTER,
        "producer_fingerprint": producer["fingerprint"]}, path)


def prediction_history(history):
    if history.ndim != 2 or history.shape[1] != 800:
        raise ValueError("complete hierarchy frame required")
    # The existing quantizer interface takes a full frame; its finest inputs
    # are unused by coarse prediction. Supply constants instead of labels.
    return torch.cat((history[:, :608], torch.zeros_like(history[:, 608:])), dim=1)


def train(pilot=False):
    rows, manifest, _ = data_inputs()
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
            raise ValueError("frozen matched source feature changed")
        inputs.append(Path(entry["path"]))
    if not pilot:
        for root in (PILOT, FIT_PILOT):
            study.data.terminal(root)
            inputs += [root / name for name in ("producer.json", "result.json", "worker_terminal_receipt.json")]
        if (study.q.read(PILOT / "result.json")["status"] != "systems_matched_forecaster_training_passed"
                or study.q.read(FIT_PILOT / "result.json")["status"] != "systems_matched_forecaster_fit_path_passed"):
            raise ValueError("actual gradient/reload and native-generation pilots required")
    root = PILOT if pilot else TRAIN
    p = producing(root, "matched_forecaster_gradient_reload_pilot" if pilot else "fixed_complete690_concept_training",
        inputs, {"actual_updates": 2 if pilot else config.updates, "batch": 1, "cases": len(used),
            "training": True, "oracle_coarse_history_in_training": True,
            "pilot_uses_maximum_learning_rate": pilot, "physical_gpu": 0})
    torch.manual_seed(config.model_seed)
    torch.cuda.manual_seed_all(config.model_seed)
    net, codec, frozen = model()
    initial = study.parameter_hash(net)
    codec_hash = clean.recovery.evaluation.reference.core.released.module_hash(codec)
    if not pilot and initial != study.q.read(PILOT / "result.json")["initial_parameter_sha256"]:
        raise ValueError("full forecaster must start fresh from the same learned initializer")
    parameters = [value for value in net.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.max_lr, betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay, eps=config.optimizer_eps)
    sampler = torch.Generator().manual_seed(config.sampler_seed)
    noise = torch.Generator().manual_seed(config.noise_seed)
    torch.manual_seed(config.model_seed)
    torch.cuda.manual_seed_all(config.model_seed)
    net.train()
    torch.cuda.reset_peak_memory_stats()
    (root / "checkpoints").mkdir()
    updates = 2 if pilot else config.updates
    for step in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        learning_rate = config.max_lr if pilot else clean.rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        row = probes[step - 1] if pilot else rows[int(torch.randint(len(rows), (1,), generator=sampler))]
        batch = study.base.batch_for(row, manifest, frame=config.max_source_frame if pilot and step == 2 else None)
        set_source_frame(net.generator, batch["source_ids"].shape[1], maximum=8192)
        drawn = bool(torch.rand((), generator=noise) < config.corrupt_probability)
        corrupted = step == 2 if pilot else drawn
        history = core.uniform_history(batch["codes"], net.generator.levels, noise,
            config.maximum_uniform_rate) if corrupted else batch["codes"]
        history = prediction_history(history)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = core.concept_logits(net, batch["features"], batch["mask"], history, batch["source_ids"])
        loss, levels = core.concept_loss(logits, batch["codes"], batch["words"], net.generator.levels,
            root_weight=config.root_weight, pad_weight=config.padded_weight)
        loss.backward()
        if (any(value.grad is not None for value in net.parameters() if not value.requires_grad)
                or any(value.grad is not None for value in codec.parameters())
                or net.source_head.source_projection.weight.grad is None
                or net.generator.l0_queries.grad is None
                or any(head.proj.weight.grad is None for head in net.generator.lm_heads[:-1])):
            raise ValueError("source/coarse versus frozen word/codec gradient routing failed")
        if pilot and (not torch.count_nonzero(net.source_head.source_projection.weight.grad)
                or not torch.count_nonzero(net.generator.l0_queries.grad)
                or any(not torch.count_nonzero(head.proj.weight.grad) for head in net.generator.lm_heads[:-1])):
            raise ValueError("pilot requires positive gradients in every source/coarse path")
        norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        if torch.cuda.max_memory_allocated() > 8_000_000_000:
            raise ValueError("matched forecaster exceeds8GBhead allocation")
        item = {"step": step, "id": row["id"], "source_frame": net.generator.prefix_len,
            "corrupted": corrupted, "history_sha256": study.base.tensor_digest({"codes": history}),
            "target_concept_sha256": study.base.tensor_digest({"codes": batch["codes"]}),
            "target_word_sha256": study.base.tensor_digest({"words": batch["words"]}),
            "lr": learning_rate, "loss": float(loss.detach()),
            "gradient_norm": float(norm), "levels": levels}
        with (root / "trace.jsonl").open("a") as handle:
            handle.write(json.dumps(item, allow_nan=False) + "\n")
        if pilot or step % 32 == 0:
            print(json.dumps(item), flush=True)
        del batch, history, logits, loss
        if not pilot and step % config.snapshot_interval == 0 and step < updates:
            if (study.parameter_hash(net, frozen_only=True) != frozen
                    or clean.recovery.evaluation.reference.core.released.module_hash(codec) != codec_hash):
                raise ValueError("frozen word modules, finest head or codec changed")
            save(root / "checkpoints" / f"step_{step:06d}.pt", net, optimizer, sampler, noise, step, p)
    save(root / "model.pt", net, optimizer, sampler, noise, updates, p)
    # The original helper's sole runtime replacement is the established device-aware geometry.
    legacy.set_source_frame = set_source_frame
    reload_probe = legacy.reload_check(root / "model.pt", net, optimizer, sampler, noise, probes, manifest)
    if (study.parameter_hash(net) == initial or study.parameter_hash(net, frozen_only=True) != frozen
            or clean.recovery.evaluation.reference.core.released.module_hash(codec) != codec_hash
            or torch.cuda.max_memory_allocated() > 8_000_000_000):
        raise ValueError("trained source state, frozen modules or memory contract differs")
    study.data.verify(p)
    write_json(root / "source_only_reload_probe.json", reload_probe)
    result = {"status": "systems_matched_forecaster_training_passed" if pilot else "matched_forecaster_trained_pending_fitting",
        "updates": updates, "cases": len(used), "training_pool_cases": 690,
        "initial_parameter_sha256": initial, "frozen_parameter_sha256_unchanged": frozen,
        "frozen_codec_sha256_unchanged": codec_hash, "word_modules_finest_head_and_codec_frozen": True,
        "reload_exact": True, "source_only_reload_probe_exact": True,
        "checkpoint_sha256": file_hash(root / "model.pt"),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "task_accuracy": None,
        "NCM_success_established": False, "final_test_opened": False, "speed_tested": False,
        "producer_fingerprint": p["fingerprint"]}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


def count_codes(codes, row):
    if len(codes) != 608 or any(not isinstance(value, int) or isinstance(value, bool)
                              or not 0 <= value < 4096 for value in codes):
        raise ValueError("all608 actual coarse codes required")
    words = row["word_ids"]
    if len(words) != 192 or not any(value != 50257 for value in words):
        raise ValueError("complete nonempty target frame required for component scoring")
    active = [any(value != 50257 for value in words[block * 48:(block + 1) * 48]) for block in range(4)]
    output, position = [], 0
    for level, width in enumerate(WIDTHS):
        indices = [i for i in range(width) if active[i // (width // 4)]]
        output.append({"level": level, "active_correct": sum(
            codes[position + i] == row["concept_ids"][position + i] for i in indices),
            "active_total": len(indices)})
        position += width
    return output


def aggregate(measurements):
    output = {}
    for group in ("all_32", "original_16", "additional_16"):
        selected = measurements if group == "all_32" else [row for row in measurements if row["diagnostic_group"] == group]
        output[group] = {"cases": len(selected), "levels": [
            {"level": level, "active_correct": sum(row["levels"][level]["active_correct"] for row in selected),
             "active_total": sum(row["levels"][level]["active_total"] for row in selected)}
            for level in range(8)]}
    return output


def fitting_gate(summary, config=Config()):
    for group, cases in (("all_32", 32), ("original_16", 16), ("additional_16", 16)):
        if summary[group]["cases"] != cases or len(summary[group]["levels"]) != 8:
            raise ValueError("complete fixed32 TRAIN probe groups and eight levels required")
        for level, values in enumerate(summary[group]["levels"]):
            correct, total = values["active_correct"], values["active_total"]
            if (values["level"] != level or not isinstance(correct, int) or not isinstance(total, int)
                    or not 0 <= correct <= total or total == 0):
                raise ValueError("valid active code fitting counts required")
    for level in range(8):
        for key in ("active_correct", "active_total"):
            if summary["all_32"]["levels"][level][key] != sum(
                    summary[group]["levels"][level][key] for group in ("original_16", "additional_16")):
                raise ValueError("group counts do not compose")
    return all(config.fit_denominator * value["active_correct"] >= config.fit_numerator * value["active_total"]
        for group in summary.values() for value in group["levels"])


@torch.no_grad()
def predict(net, row, manifest):
    study = clean.recovery.study
    batch = study.source_batch(study.source_view(row), manifest)
    set_source_frame(net.generator, row["source_frame"], maximum=8192)
    handles = [head.register_forward_hook(require_finite_logits) for head in net.generator.lm_heads[:-1]]
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = study.core.own_history(net, batch["features"], batch["mask"], batch["source_ids"])
    finally:
        for handle in handles:
            handle.remove()
    codes = output["coarse_codes"][0].cpu().tolist()
    return {"id": row["id"], "source_id": row["source_id"], "source_sha256": row["source_sha256"],
        "source_frame": row["source_frame"], "diagnostic_group": row["diagnostic_group"],
        "coarse_codes": codes, "levels": count_codes(codes, row)}


def require_finite_logits(module, inputs, output):
    if not torch.isfinite(output).all():
        raise ValueError("nonfinite source-only coarse logits")


def fit(pilot=False):
    _, manifest, selected = data_inputs()
    study = clean.recovery.study
    study.data.require_gpu()
    torch.backends.cudnn.allow_tf32 = False
    source, root = (PILOT, FIT_PILOT) if pilot else (TRAIN, FIT)
    study.data.terminal(source)
    trained = study.q.read(source / "result.json")
    expected_step = 2 if pilot else Config().updates
    if (trained["updates"] != expected_step or not trained["reload_exact"]
            or file_hash(source / "model.pt") != trained["checkpoint_sha256"]):
        raise ValueError("complete unchanged matched forecaster required")
    inputs = [source / name for name in ("producer.json", "result.json", "model.pt", "worker_terminal_receipt.json")]
    for row in selected:
        entry = manifest[row["id"]]
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("frozen matched probe features changed")
        inputs.append(Path(entry["path"]))
    p = producing(root, "matched_forecaster_native_fit_path_pilot" if pilot else "matched_forecaster_fixed32_TRAIN_fitting",
        inputs, {"actual_updates": 0, "model_checkpoint_updates": expected_step, "training": False,
                 "source_only": True, "cases": 32, "physical_gpu": 0})
    net, codec, _ = model(source / "model.pt", expected_step)
    net.eval().requires_grad_(False)
    before = study.parameter_hash(net)
    codec_hash = clean.recovery.evaluation.reference.core.released.module_hash(codec)
    torch.cuda.reset_peak_memory_stats()
    values = []
    for i, row in enumerate(selected):
        actual = predict(net, row, manifest)
        if pilot and i in (0, max(range(len(selected)), key=lambda j: len(selected[j]["source_ids"]))):
            if predict(net, row, manifest) != actual:
                raise ValueError("native source-only coarse rollout must repeat exactly")
            batch = study.source_batch(study.source_view(row), manifest)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                native = net.generate(batch["features"], batch["mask"], batch["source_ids"])
            if torch.cat(native["coarse_codes"], dim=1)[0].cpu().tolist() != actual["coarse_codes"]:
                raise ValueError("coarse-only rollout differs from native nine-pass generation")
        values.append(actual)
        write_json(root / "case_measurements.json", values)
        if len(values) % 8 == 0:
            print(json.dumps({"measured": len(values), "total": 32}), flush=True)
    summary = aggregate(values)
    gate = None if pilot else fitting_gate(summary)
    if (study.parameter_hash(net) != before
            or clean.recovery.evaluation.reference.core.released.module_hash(codec) != codec_hash
            or torch.cuda.max_memory_allocated() > 8_000_000_000):
        raise ValueError("read-only forecaster fitting state or memory gate differs")
    study.data.verify(p)
    result = {"status": "systems_matched_forecaster_fit_path_passed" if pilot else "matched_forecaster_fitting_measured",
        "cases": 32, "components": summary, "forecaster_fit_gate": gate,
        "checkpoint_sha256": trained["checkpoint_sha256"], "parameters_and_codec_unchanged": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "task_accuracy": None,
        "semantic_quality_measured": False, "NCM_success_established": False,
        "final_test_opened": False, "speed_tested": False,
        "measurements_sha256": file_hash(root / "case_measurements.json"),
        "producer_fingerprint": p["fingerprint"]}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


def run_stage(name, command, destination, journal, producer, *, existing=False, cpu=False, numerical=True):
    study = clean.recovery.study
    if destination.exists() != existing:
        raise ValueError("freshness contract differs: " + name)
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
        target = destination / ("registry_terminal_receipt.json" if name == "register" else "worker_terminal_receipt.json")
        if target.exists():
            raise ValueError("terminal receipt must never be overwritten")
        write_json(target, receipt)
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
    registry = Path("data/hypothesis_registry") / (HYPOTHESIS + ".json")
    if any(path.exists() for path in (WORKFLOW, PILOT, FIT_PILOT, TRAIN, FIT, RESULTS, registry)) or REPORT.exists():
        raise ValueError("fresh matched forecaster workflow and all destinations required")
    pending = admission() is None
    identity = None
    if pending:
        progress = clean.recovery.study.q.read(clean.WORKFLOW / "progress.json")
        identity = kernel_identity(progress["observer_pid"])
        if identity is None or (clean.WORKFLOW / "failure.json").exists():
            raise ValueError("unfinished clean-word supervisor must be authoritatively live")
    p = producing(WORKFLOW, "observe_clean_word_gate_then_train_complete690_forecaster", [],
        {"actual_updates": 0, "upstream_kernel_identity": identity, "poll_seconds": 5,
         "upstream_worker_restarted": False, "upstream_gate_pending_at_launch": pending,
         "training": False}, pending=True)
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "waiting_for_registered_clean_word_fitting",
        "observer_pid": os.getpid(), "upstream_kernel_identity": identity,
        "stages": [], "final_test_opened": False, "producer_fingerprint": p["fingerprint"]}
    write_json(WORKFLOW / "progress.json", journal)
    try:
        while True:
            if (clean.WORKFLOW / "failure.json").exists():
                raise RuntimeError("upstream clean-word failure preserved")
            ready = admission()
            alive = identity is not None and kernel_identity(identity["pid"]) == identity
            if ready is not None and not alive:
                break
            if ready is None and not alive:
                raise RuntimeError("upstream terminal without required successful component evidence")
            time.sleep(5)
        if not ready:
            journal.update(status="not_admitted_clean_word_fit_gate_failed",
                active_stage=None, model_training_executed=False, task_accuracy=None,
                NCM_success_established=False)
            write_json(WORKFLOW / "result.json", journal)
            return
        require_admission()
        for name, stage, root in (("train_pilot", "pilot", PILOT), ("fit_pilot", "fit-pilot", FIT_PILOT),
                                  ("train", "train", TRAIN), ("fit", "fit", FIT)):
            run_stage(name, [sys.executable, "-u", "-m", MODULE, stage], root, journal, p)
        for stage in ("finalize", "register"):
            run_stage(stage, [sys.executable, "-u", "-m", FINALIZER_MODULE, stage],
                RESULTS, journal, p, existing=stage == "register", cpu=True, numerical=False)
        campaign = clean.recovery.study.q.read(RESULTS / "campaign_results.json")
        receipt = clean.recovery.study.q.read(RESULTS / (HYPOTHESIS + "_registry_readback.json"))
        if (receipt["status"] != "registered_and_read_back"
                or receipt["record"]["campaign_sha256"] != file_hash(RESULTS / "campaign_results.json")
                or receipt["record"]["confirmation_status"] != campaign["classification"]):
            raise ValueError("exact matched forecaster registry readback required")
        journal.update(status="matched_forecaster_component_completed", active_stage=None,
            forecaster_fit_gate=campaign["forecaster_fit_gate"], model_training_executed=True,
            predicted_history_word_adaptation_still_required=True,
            semantic_quality_measured=False, task_accuracy=None,
            NCM_success_established=False, campaign_sha256=file_hash(RESULTS / "campaign_results.json"))
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
        rows, _, selected = data_inputs(require_gate=False)
        sampler = torch.Generator().manual_seed(Config().sampler_seed)
        counts = [0] * len(rows)
        for _ in range(Config().updates):
            counts[int(torch.randint(len(rows), (1,), generator=sampler))] += 1
        if not min(counts):
            raise ValueError("fixed forecaster stream must cover every TRAIN case")
        print(json.dumps({"TRAIN_cases": len(rows), "fixed_probe_cases": len(selected),
            "minimum_sample_count": min(counts), "maximum_sample_count": max(counts),
            "config": asdict(Config()), "upstream_admission": admission(), "task_accuracy": None}))
    elif stage == "workflow":
        workflow()
    elif stage in ("pilot", "train"):
        train(stage == "pilot")
    else:
        fit(stage == "fit-pilot")
