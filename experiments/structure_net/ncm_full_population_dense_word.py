"""Train the dense word branch on all690 targets and matched source rollouts."""
import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from experiments.structure_net import ncm_dense_word_fit as legacy
from experiments.structure_net import ncm_dense_checkpoint_loader as loader
from experiments.structure_net import ncm_dense_matched_forecaster as forecaster
from experiments.structure_net import ncm_clean_development_features as development
from experiments.structure_net.ncm_device_geometry import set_source_frame
from experiments.structure_net.ncm_expanded_word_data_workflow import kernel_identity
from experiments.structure_net.ncm_memory_data import file_hash, write_json

clean, core, targets = legacy.clean, legacy.core, forecaster.targets
dropout, save, reload_check = legacy.dropout, legacy.save, legacy.reload_check
aggregate, fitting_gate = legacy.aggregate, legacy.fitting_gate
ROOT = legacy.ROOT
HYPOTHESIS = "ncm-full-population-dense-word-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-full-population-dense-word-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-full-population-dense-word.md")
WORKFLOW = ROOT / "20261024_full_population_dense_word_workflow_v1"
PILOT = ROOT / "20261024_full_population_dense_word_pilot_v1"
TRAIN = ROOT / "20261024_full_population_dense_word_train_v1"
FIT = ROOT / "20261024_full_population_dense_word_fit_v1"
RESULTS = ROOT / "20261024_full_population_dense_word_results_v1"
MODULE = "experiments.structure_net.ncm_full_population_dense_word"
AUDITOR = "experiments.neural_architecture_lab.finalize_ncm_full_population_dense_word"
AUDITOR_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_full_population_dense_word.py")
INITIAL_PATH = ROOT / "20261023_dense_word_fit_train_v1/model.pt"
INITIAL_WORD = "0b8afde259c1165ddc4ebb14110568b0695228850dd09eaeadf645095550e266"
INITIAL_PARAMETERS = "74ba3d21b957680e4614ed3679423798163795be7b394de8b658322deefc1880"
TARGET_MANIFEST = "32391a18808729bd6a57aba7e3a3163439f97aba6e54f528957a289475cfd726"
q = clean.recovery.study.q


@dataclass(frozen=True)
class Config(legacy.Config):
    updates: int = 32768
    warmup: int = 256
    model_seed: int = 55142
    sampler_seed: int = 56142
    snapshot_interval: int = 8192


def configure():
    clean.dependencies()
    clean.recovery.configure()
    # The retained numerical save/reload/probe helpers resolve this process's config.
    legacy.Config = Config


def rate(step):
    return legacy.rate(step, Config())


def completed(workflow, results, hypothesis, status, stages):
    if (workflow / "failure.json").exists():
        raise RuntimeError("upstream failure preserved: " + str(workflow))
    if not (workflow / "result.json").exists():
        return False
    done = q.read(workflow / "result.json")
    path = results / "campaign_results.json"
    receipt = q.read(results / (hypothesis + "_registry_readback.json"))
    digest = file_hash(path)
    if (done["status"] != status or [(s["stage"], s["exit_code"]) for s in done["stages"]] != [(s, 0) for s in stages]
            or receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != digest or receipt["exact"]["campaign_sha256"] != digest):
        raise ValueError("complete observed upstream lifecycle and exact registry required")
    return True


def upstreams():
    return ((forecaster.WORKFLOW, forecaster.RESULTS, forecaster.HYPOTHESIS,
        "dense_matched_forecaster_completed", ("train_pilot", "fit_pilot", "train", "fit", "cache", "finalize", "register"), forecaster.MODULE),
        (development.WORKFLOW, development.RESULTS, development.HYPOTHESIS,
        "clean_development_features_completed", ("prepare", "feature-pilot", "features", "finalize", "register"), development.MODULE))


def validate_label(row, entry, campaign):
    study = clean.recovery.study
    path, measured = Path(entry["state"]), Path(entry["measurement"])
    if (entry["updates"] != 256 or entry["target_word_sha256"] != study.base.tensor_digest({"target": torch.tensor([row["word_ids"]])})
            or campaign["artifacts"].get(str(path)) != entry["state_sha256"]
            or campaign["artifacts"].get(str(measured)) != entry["measurement_sha256"]
            or file_hash(path) != entry["state_sha256"] or file_hash(measured) != entry["measurement_sha256"]):
        raise ValueError("exact registered complete TRAIN latent/word pairing required")
    value = q.read(measured)
    if (value["id"] != row["id"] or value["source_sha256"] != row["source_sha256"]
            or value["target_word_ids"] != row["word_ids"] or value["target_text"] != row["target_text"]):
        raise ValueError("dense labels must retain their own complete source and summary")
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["refined"]
    if saved.shape != (1, 192, 384) or saved.dtype != torch.float32 or not torch.isfinite(saved).all():
        raise ValueError("finite complete FP32 dense target required")
    return {"refined": saved.detach()}


def static_inputs():
    rows, manifest, selected = forecaster.data_inputs()
    path = targets.TARGETS / "manifest.json"
    if file_hash(path) != TARGET_MANIFEST or file_hash(INITIAL_PATH) != INITIAL_WORD:
        raise ValueError("fixed all690 target manifest and learned word initializer required")
    labels_manifest = q.read(path)
    if set(labels_manifest) != {r["id"] for r in rows}:
        raise ValueError("all690 target identities required")
    campaign = q.read(targets.RESULTS / "campaign_results.json")
    labels = {r["id"]: validate_label(r, labels_manifest[r["id"]], campaign) for r in rows}
    return rows, manifest, selected, labels


def inputs():
    if not all(completed(*item[:5]) for item in upstreams()):
        raise ValueError("both upstream workflows must finish before numerical word work")
    rows, manifest, _, labels = static_inputs()
    cached = q.read(forecaster.CACHE / "result.json")
    campaign = q.read(forecaster.RESULTS / "campaign_results.json")
    for name in ("producer.json", "result.json", "histories.json", "source_cases.json", "worker_terminal_receipt.json"):
        path = forecaster.CACHE / name
        if campaign["artifacts"].get(str(path)) != file_hash(path):
            raise ValueError("registered matched forecaster cache bytes differ")
    histories = q.read(forecaster.CACHE / "histories.json")
    if (set(histories) != {r["id"] for r in rows} or cached["cases"] != 690 or not cached["source_only"]
            or cached["target_labels_in_prediction"] or cached["histories_sha256"] != file_hash(forecaster.CACHE / "histories.json")
            or cached["checkpoint_sha256"] != file_hash(forecaster.TRAIN / "model.pt")
            or q.read(forecaster.CACHE / "source_cases.json") != [clean.recovery.study.source_view(r) for r in rows]):
        raise ValueError("all690 source-only histories from fixed final matched forecaster required")
    for row in rows:
        clean.recovery.study.validate_history(row, histories[row["id"]])
    admission = q.read(WORKFLOW / "admission.json")
    if (admission["forecaster_checkpoint_sha256"] != cached["checkpoint_sha256"]
            or admission["histories_sha256"] != cached["histories_sha256"]):
        raise ValueError("immutable prospective forecaster selection changed")
    return rows, manifest, histories, labels


def model(path=None, expected_step=None, device="cuda", *, pending=False):
    if pending and (device != "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES") != ""):
        raise ValueError("pending-forecaster constructor check must stay CPU only")
    fixed, branch, codec, _ = loader.load_models(device="cpu")
    initial = loader.checked_checkpoint(INITIAL_PATH, INITIAL_WORD, 2048)
    set_source_frame(branch.generator, initial["source_frame"], maximum=8192)
    branch.load_state_dict(initial["model"], strict=True)
    core.prepare_branch(branch)
    if (clean.recovery.study.parameter_hash(branch) != INITIAL_PARAMETERS
            or clean.recovery.study.parameter_hash(branch, frozen_only=True) != loader.DENSE_FROZEN_PARAMETERS):
        raise ValueError("complete selected word initializer or original frozen parameters changed")
    if not pending:
        selected = q.read(WORKFLOW / "admission.json")
        forecast = loader.checked_checkpoint(forecaster.TRAIN / "model.pt", selected["forecaster_checkpoint_sha256"], forecaster.Config().updates)
        if forecast["config"] != asdict(forecaster.Config()):
            raise ValueError("fixed final matched forecaster configuration required")
        set_source_frame(fixed.generator, forecast["source_frame"], maximum=8192)
        fixed.load_state_dict(forecast["model"], strict=True)
        if clean.recovery.study.parameter_hash(fixed) != selected["forecaster_parameter_sha256"]:
            raise ValueError("actual final forecaster parameter identity differs")
    fixed.eval().requires_grad_(False)
    if path is not None:
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if saved["config"] != asdict(Config()) or saved["step"] != expected_step:
            raise ValueError("fixed final full-population word checkpoint required")
        set_source_frame(branch.generator, saved["source_frame"], maximum=8192)
        branch.load_state_dict(saved["model"], strict=True)
    if (loader.module_hash(codec) != loader.ORIGINAL_CODEC or any(not torch.equal(v, codec.decoder.state_dict()[k])
            for k, v in branch.word_decoder.state_dict().items())):
        raise ValueError("original shared codec and frozen word decoder differ")
    codec.to(device).eval().requires_grad_(False)
    branch.to(device).float().eval()
    set_source_frame(branch.generator, branch.generator.prefix_len, maximum=8192)
    return fixed, branch, codec


def producing(root, stage, extra=(), *, pending=False, identities=None):
    forecaster.admission()
    source = q.read(forecaster.WORKFLOW / "producer.json")
    initial_source = q.read(legacy.WORKFLOW / "producer.json")
    files = list(dict.fromkeys([Path(__file__), PLAN, AUDITOR_FILE, Path(legacy.__file__),
        Path(loader.__file__), Path(core.__file__), Path(legacy.dropout.__file__),
        Path(legacy.teacher.prior.__file__), Path(development.__file__), development.PLAN,
        Path(kernel_identity.__code__.co_filename),
        Path("experiments/structure_net/ncm_generated_history_core.py"),
        *[Path(p) for p in source["source_hashes"]], *[Path(p) for p in initial_source["source_hashes"]]]))
    common = [targets.CASES, targets.CASES.parent / "manifest.json", targets.TARGETS / "manifest.json",
        targets.RESULTS / "campaign_results.json", targets.RESULTS / (targets.HYPOTHESIS + "_registry_readback.json"),
        INITIAL_PATH, INITIAL_PATH.parent / "result.json", legacy.RESULTS / "campaign_results.json",
        forecaster.WORKFLOW / "producer.json", development.WORKFLOW / "producer.json"]
    for entry in q.read(targets.TARGETS / "manifest.json").values():
        common += [Path(entry["state"]), Path(entry["measurement"])]
    if not pending:
        common += [WORKFLOW / "admission.json"]
        for workflow, results, hypothesis, _, _, _ in upstreams():
            common += [workflow / "result.json", results / "campaign_results.json", results / (hypothesis + "_registry_readback.json")]
        common += [forecaster.TRAIN / "model.pt", forecaster.CACHE / "histories.json", forecaster.CACHE / "result.json"]
    p = clean.recovery.study.data.freeze(root, stage, [*extra, *common, *files], {
        **asdict(Config()), "hypothesis_id": HYPOTHESIS, "schema": "nal.ncm-full-population-dense-word.v1",
        "TRAIN_cases": 690, "initial_word_checkpoint_sha256": INITIAL_WORD,
        "selected_forecaster_update": forecaster.Config().updates, "selected_word_update": Config().updates,
        "teacher_latents_and_words_are_labels_only": True, "conditioning": "actual_matched_source_only_forecaster_history",
        "training_dropout": 0, "body_MLPs_forecaster_codec_original_decoder_and_unused_heads_frozen": True,
        "precision": "FP32masters_optimizer_decoder_loss_BF16source_transformer_projection",
        "matmul_TF32": False, "cuDNN_TF32": False, "parent_loaded": False,
        "physical_gpu": 0, "torch": torch.__version__, "cuda": torch.version.cuda,
        "upstream_kernel_identities": identities if pending else q.read(WORKFLOW / "producer.json")["config"]["upstream_kernel_identities"],
        "pending_upstreams": pending, "task_accuracy": None, "final_test_opened": False, "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"full_word_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def fit():
    rows, manifest, histories, _ = inputs()
    _, _, selected = forecaster.data_inputs()
    study = clean.recovery.study
    study.data.require_gpu()
    torch.backends.cudnn.allow_tf32 = False
    study.data.terminal(TRAIN)
    trained = q.read(TRAIN / "result.json")
    if file_hash(TRAIN / "model.pt") != trained["checkpoint_sha256"]:
        raise ValueError("fixed final full-population dense word checkpoint required")
    p = producing(FIT, "fixed32_TRAIN_diagnostic_after_all690_word_training", [TRAIN / n for n in
        ("model.pt", "producer.json", "result.json", "worker_terminal_receipt.json")])
    fixed, branch, codec = model(TRAIN / "model.pt", Config().updates)
    branch.eval().requires_grad_(False)
    before = study.parameter_hash(branch), study.parameter_hash(fixed), loader.module_hash(codec)
    outputs = legacy.probe(fixed, branch, selected, manifest, histories)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(study.data.GPT2, local_files_only=True)
    values = []
    for row, output in zip(selected, outputs):
        raw = output["raw_word_ids"][0]
        text, end = clean.recovery.evaluation.reference.decode_words(raw, tokenizer)
        values.append({"id": row["id"], "source_sha256": row["source_sha256"], "diagnostic_group": row["diagnostic_group"],
            "target_word_ids": row["word_ids"], "target_text": row["target_text"], "raw_word_ids": raw,
            "actual_coarse_codes": output["coarse_codes"][0], "latent_sha256": output["latent_sha256"],
            **clean.diagnostic.counts(torch.tensor([raw]), torch.tensor([row["word_ids"]])),
            "text": text, "global_EOS_position": end if end < 192 else None, "exact_target_text": text == row["target_text"]})
    if before != (study.parameter_hash(branch), study.parameter_hash(fixed), loader.module_hash(codec)):
        raise ValueError("read-only fitting changed model parameters")
    components = legacy.aggregate(values)
    write_json(FIT / "case_measurements.json", values)
    study.data.verify(p)
    result = {"status": "full_population_dense_TRAIN_fitting_measured", "cases": 32,
        "training_population": len(rows), "components": components,
        "actual_history_fit_gate": legacy.fitting_gate(components), "checkpoint_sha256": trained["checkpoint_sha256"],
        "parameters_unchanged": True, "fresh_native_cache_parity": True,
        "measurements_sha256": file_hash(FIT / "case_measurements.json"), "task_accuracy": None,
        "final_test_opened": False, "speed_tested": False, "producer_fingerprint": p["fingerprint"]}
    write_json(FIT / "result.json", result)
    print(json.dumps(result), flush=True)


def workflow():
    static_inputs()
    if any(p.exists() for p in (WORKFLOW, PILOT, TRAIN, FIT, RESULTS, REPORT,
            Path("data/hypothesis_registry") / (HYPOTHESIS + ".json"))):
        raise ValueError("fresh full-population dense word study required")
    identities = {}
    for workflow, results, hypothesis, status, stages, module in upstreams():
        ready = completed(workflow, results, hypothesis, status, stages)
        pid = q.read(workflow / "progress.json")["observer_pid"]
        identity = kernel_identity(pid)
        if (not ready and identity is None) or (identity is not None
                and module.encode() not in Path(f"/proc/{pid}/cmdline").read_bytes()):
            raise ValueError("same existing upstream supervisor must be authoritatively live")
        identities[hypothesis] = identity
    p = producing(WORKFLOW, "observe_upstreams_then_train_all690_dense_word_branch", pending=True, identities=identities)
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "waiting_for_registered_forecaster_and_DEV_features", "observer_pid": os.getpid(),
        "stages": [], "upstream_kernel_identities": identities, "producer_fingerprint": p["fingerprint"]}
    write_json(WORKFLOW / "progress.json", journal)
    try:
        while True:
            ready = [completed(*item[:5]) for item in upstreams()]
            alive = [identity is not None and kernel_identity(identity["pid"]) == identity for identity in identities.values()]
            if all(ready) and not any(alive):
                break
            if any(not r and not a for r, a in zip(ready, alive)):
                raise RuntimeError("upstream terminal without required complete registered lifecycle")
            time.sleep(5)
        cached = q.read(forecaster.CACHE / "result.json")
        write_json(WORKFLOW / "admission.json", {"forecaster_checkpoint_sha256": cached["checkpoint_sha256"],
            "forecaster_parameter_sha256": cached["forecaster_parameter_sha256"], "histories_sha256": cached["histories_sha256"],
            "upstreams_complete_registered_and_same_kernels_terminal": True, "original_word_initializer_sha256": INITIAL_WORD,
            "upstream_campaign_hashes": {h: file_hash(r / "campaign_results.json") for _, r, h, _, _, _ in upstreams()}})
        inputs()
        for stage, root in (("pilot", PILOT), ("train", TRAIN), ("fit", FIT), ("finalize", RESULTS), ("register", RESULTS)):
            clean.recovery.study.data.verify(p)
            command = [sys.executable, "-u", "-m", AUDITOR if stage in ("finalize", "register") else MODULE, stage]
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "" if stage in ("finalize", "register") else "0"})
            journal.update(status="running", active_stage=stage, child_pid=child.pid)
            write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            receipt = {"stage": stage, "exit_code": code, "child_pid": child.pid,
                "observer_pid": os.getpid(), "observer": "subprocess.wait", "command": command}
            write_json(WORKFLOW / "stage_receipts" / (stage + ".json"), receipt)
            if root.exists():
                write_json(root / ("registry_terminal_receipt.json" if stage == "register" else "worker_terminal_receipt.json"), receipt)
            journal["stages"].append(receipt)
            if code:
                raise RuntimeError("full-population dense word stage failed: " + stage)
        clean.recovery.study.data.verify(p)
        journal.update(status="full_population_dense_word_completed", active_stage=None,
            classification=q.read(RESULTS / "campaign_results.json")["classification"])
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


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
    p = producing(destination, "systems_lifecycle_only_not_quality_evidence" if pilot else "all690_source_only_dense_training", extra)
    torch.manual_seed(config.model_seed)
    torch.cuda.manual_seed_all(config.model_seed)
    fixed, branch, codec = model()
    initial = study.parameter_hash(branch)
    frozen = study.parameter_hash(branch, frozen_only=True)
    fixed_hash = study.parameter_hash(fixed)
    codec_hash = loader.module_hash(codec)
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
            **parts, "objective_loss": float(loss.detach()), "gradient_norm": float(norm)}
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
            or study.parameter_hash(fixed) != fixed_hash or loader.module_hash(codec) != codec_hash
            or dropout.dropout_map(branch) != maps
            or torch.cuda.max_memory_allocated() > 8_000_000_000):
        raise ValueError("dense original frozen modules, update or memory contract differs")
    write_json(destination / "source_only_reload_probe.json", outputs)
    study.data.verify(p)
    result = {"status": "systems_dense_gradient_reload_passed" if pilot else "dense_source_word_trained",
        "updates": updates, "training_population": len(rows), "cases": len(used), "initial_parameter_sha256": initial,
        "frozen_parameter_sha256": frozen, "forecaster_parameter_sha256": fixed_hash, "codec_hash": codec_hash,
        "reload_exact": True, "source_only_composed_reload_exact": True, "original_decoder_frozen": True,
        "dropout_map": maps,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "checkpoint_sha256": file_hash(destination / "model.pt"),
        "producer_fingerprint": p["fingerprint"], "task_accuracy": None, "semantic_quality_measured": False,
        "final_test_opened": False, "speed_tested": False}
    write_json(destination / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("check", "cpu-model", "pilot", "train", "fit", "workflow"))
    args = parser.parse_args()
    configure()
    if args.stage == "check":
        rows, _, selected, labels = static_inputs()
        print(json.dumps({"TRAIN_cases": len(rows), "dense_labels": len(labels), "diagnostic_cases": len(selected), "config": asdict(Config()), "task_accuracy": None}))
    elif args.stage == "cpu-model":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
            raise ValueError("constructor check must stay CPU only")
        _, branch, codec = model(device="cpu", pending=True)
        print(json.dumps({"status": "systems_dense_word_initializer_exact_not_quality", "initial_checkpoint_sha256": INITIAL_WORD,
            "parameter_sha256": clean.recovery.study.parameter_hash(branch), "frozen_parameter_sha256": clean.recovery.study.parameter_hash(branch, frozen_only=True),
            "codec_sha256": loader.module_hash(codec), "trainable_parameters": sum(v.numel() for v in branch.parameters() if v.requires_grad),
            "new_forecaster_selection_pending": True, "CUDA_initialized": torch.cuda.is_initialized()}))
    elif args.stage == "workflow":
        workflow()
    elif args.stage == "fit":
        fit()
    else:
        train(args.stage == "pilot")
