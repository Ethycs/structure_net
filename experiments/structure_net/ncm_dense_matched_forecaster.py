"""Exploratory full-population forecasting after dense TRAIN targets qualify.

The earlier conditional forecaster remains declined. This new study reuses its
tested numerical functions in a separate process with new paths and admission.
"""
import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_matched_forecaster as old
from experiments.structure_net import ncm_dense_checkpoint_loader as loader
from experiments.structure_net import ncm_dense_training_targets as targets
from experiments.structure_net.ncm_device_geometry import set_source_frame
from experiments.structure_net.ncm_memory_data import file_hash, write_json

ROOT = old.ROOT
HYPOTHESIS = "ncm-dense-matched-forecaster-v2"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-dense-matched-forecaster-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-dense-matched-forecaster.md")
WORKFLOW = ROOT / "20261024_dense_matched_forecaster_workflow_v2"
PILOT = ROOT / "20261024_dense_matched_forecaster_pilot_v2"
FIT_PILOT = ROOT / "20261024_dense_matched_forecaster_fit_pilot_v2"
TRAIN = ROOT / "20261024_dense_matched_forecaster_train_v2"
FIT = ROOT / "20261024_dense_matched_forecaster_fit_v2"
CACHE = ROOT / "20261024_dense_matched_forecaster_cache_v2"
RESULTS = ROOT / "20261024_dense_matched_forecaster_results_v2"
MODULE = "experiments.structure_net.ncm_dense_matched_forecaster"
AUDITOR = "experiments.neural_architecture_lab.finalize_ncm_dense_matched_forecaster"
AUDITOR_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_dense_matched_forecaster.py")
TARGET_CAMPAIGN = "618ce743575e8ac962ac57150546d7dc905201a62ca35bd0305027fa56edc575"
MANIFEST_SHA = "dc831896063be90fcdaaca71a2bfb08a17bb417123ae0c7c87361c7e0167a5c7"


@dataclass(frozen=True)
class Config(old.Config):
    updates: int = 32768
    model_seed: int = 52142
    sampler_seed: int = 53142
    noise_seed: int = 54142


def configure():
    old.dependencies()
    old.clean.recovery.configure()
    old.Config = Config
    for name in ("HYPOTHESIS", "PLAN", "REPORT", "WORKFLOW", "PILOT", "FIT_PILOT", "TRAIN", "FIT", "RESULTS", "MODULE"):
        setattr(old, name, globals()[name])
    old.FINALIZER_MODULE, old.FINALIZER_FILE = AUDITOR, AUDITOR_FILE
    old.data_inputs, old.model, old.producing = data_inputs, model, producing


def admission():
    q = old.clean.recovery.study.q
    path = targets.RESULTS / "campaign_results.json"
    campaign = q.read(path)
    receipt = q.read(targets.RESULTS / (targets.HYPOTHESIS + "_registry_readback.json"))
    done = q.read(targets.WORKFLOW / "result.json")
    if (file_hash(path) != TARGET_CAMPAIGN or not campaign["latent_fit_gate"]
            or campaign["TRAIN_cases"] != 690 or campaign["model_parameters_trained"]
            or receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != TARGET_CAMPAIGN
            or receipt["exact"]["campaign_sha256"] != TARGET_CAMPAIGN
            or done["status"] != "dense_training_targets_completed"
            or len(done["stages"]) != 4 or any(r["exit_code"] for r in done["stages"])):
        raise ValueError("registered complete690 dense targets and actual successful exits required")


def data_inputs(require_gate=True):
    old.dependencies()
    old.clean.recovery.configure()
    if require_gate:
        admission()
    study = old.clean.recovery.study
    cases = targets.CASES
    manifest_path = cases.parent / "manifest.json"
    if file_hash(cases) != targets.CASES_SHA or file_hash(manifest_path) != MANIFEST_SHA:
        raise ValueError("same complete690 sources, labels and original-body feature manifest required")
    rows, manifest = study.q.read(cases)["train"], study.q.read(manifest_path)
    if (len(rows) != 690 or set(manifest) != {r["id"] for r in rows}
            or max(r["source_frame"] for r in rows) > Config().max_source_frame
            or any(r["split"] != "train" or len(r["concept_ids"]) != 800 or len(r["word_ids"]) != 192 for r in rows)):
        raise ValueError("complete matched TRAIN population and full frames required")
    identities = study.q.read(old.clean.diagnostic.DEST / "selection.json")
    by_id = {r["id"]: r for r in rows}
    selected = []
    for identity in identities:
        row = by_id[identity["id"]]
        if any(row[k] != v for k, v in identity.items() if k in row):
            raise ValueError("unchanged fixed32 TRAIN diagnostic identities required")
        selected.append({**row, "diagnostic_group": identity["diagnostic_group"]})
    if (len(selected) != 32 or any(sum(r["diagnostic_group"] == group for r in selected) != 16
            for group in ("original_16", "additional_16"))):
        raise ValueError("same two16-case diagnostic groups required")
    return rows, manifest, selected


def model(path=None, expected_step=None, device="cuda"):
    study = old.clean.recovery.study
    saved = loader.checked_checkpoint(study.bootstrap.TRAIN / "model.pt", loader.FORECASTER_CHECKPOINT, 16384)
    net, codec, _ = loader.factory.make_model(device="cpu")
    set_source_frame(net.generator, saved["source_frame"], maximum=8192)
    net.load_state_dict(saved["model"], strict=True)
    if study.parameter_hash(net) != loader.FORECASTER_PARAMETERS or loader.module_hash(codec) != loader.ORIGINAL_CODEC:
        raise ValueError("complete registered forecaster or original codec identity differs")
    old.core.concept_scope(net)
    expected = study.q.read(study.bootstrap.TRAIN / "result.json")["frozen_parameter_sha256_unchanged"]
    if path is not None:
        selected = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if (selected["step"] != expected_step or selected["config"] != asdict(Config())
                or selected["initial_forecaster_checkpoint_sha256"] != loader.FORECASTER_CHECKPOINT):
            raise ValueError("only the selected fixed new forecaster checkpoint is allowed")
        set_source_frame(net.generator, selected["source_frame"], maximum=8192)
        net.load_state_dict(selected["model"], strict=True)
    if (study.parameter_hash(net, frozen_only=True) != expected
            or any(not torch.equal(head.E, net.generator.vqvae_local[0].vq.embeddings[i].weight)
                   for i, head in enumerate(net.generator.lm_heads))):
        raise ValueError("original frozen word/finest state or codebook buffers differ")
    codec.to(device).eval().requires_grad_(False)
    net.to(device).float().eval()
    return net, codec, expected


def producing(root, stage, inputs, config, *, pending=False):
    study = old.clean.recovery.study
    admission()
    files = [Path(__file__), PLAN, AUDITOR_FILE, Path(old.__file__), Path(loader.__file__),
        Path(old.core.__file__), Path(old.legacy.__file__), Path(loader.factory.__file__),
        Path(loader.factory.released.__file__), Path(targets.__file__),
        Path(study.__file__), Path(study.core.__file__), Path(study.base.tensor_digest.__code__.co_filename),
        Path("experiments/structure_net/ncm_word_final_head.py"),
        Path("experiments/structure_net/ncm_frozen_source_head.py"), Path("experiments/structure_net/ncm_blockwise_codec.py"),
        Path("experiments/structure_net/ncm_device_geometry.py"), Path("experiments/structure_net/ncm_native_label_quality.py"),
        Path("experiments/neural_architecture_lab/finalize_ncm_matched_forecaster.py"),
        Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    files += sorted(study.data.VENDOR.rglob("*.py"))
    common = [targets.RESULTS / "campaign_results.json", targets.RESULTS / (targets.HYPOTHESIS + "_registry_readback.json"),
        targets.WORKFLOW / "result.json", targets.CASES, targets.CASES.parent / "manifest.json",
        old.clean.diagnostic.DEST / "selection.json"]
    common += [study.bootstrap.TRAIN / n for n in ("model.pt", "producer.json", "result.json", "worker_terminal_receipt.json")]
    common += [loader.factory.released.INITIAL / n for n in ("tokenizer.pt", "generator.pt", "config.yaml")]
    p = study.data.freeze(root, stage, [*inputs, *common, *files],
        {**asdict(Config()), **config, "hypothesis_id": HYPOTHESIS,
         "admission": "new_dense_TRAIN_target_fidelity_not_old_failed_clean_word_gate",
         "TRAIN_population": 690, "initial_forecaster_checkpoint_sha256": loader.FORECASTER_CHECKPOINT,
         "word_modules_finest_head_and_codec_frozen": True, "parent_loaded": False,
         "original_parent_weights_and_MLPs_frozen": True, "word_losses": False,
         "finest_prediction_history": "constant_zero", "word_targets_in_prediction": False,
         "precision": "FP32masters_optimizer_loss_BF16forward_FP32reconstruction",
         "matmul_TF32": False, "cuDNN_TF32": False,
         "semantic_quality_measured": False, "task_accuracy": None, "final_test_opened": False, "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"dense_forecaster_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = study.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


@torch.no_grad()
def cache():
    rows, manifest, _ = data_inputs()
    study = old.clean.recovery.study
    study.data.require_gpu()
    torch.backends.cudnn.allow_tf32 = False
    study.data.terminal(TRAIN)
    inputs_ = [TRAIN / n for n in ("model.pt", "producer.json", "result.json", "worker_terminal_receipt.json")]
    for row in rows:
        entry = manifest[row["id"]]
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("complete frozen source feature bytes changed")
        inputs_.append(Path(entry["path"]))
    p = producing(CACHE, "complete690_source_only_new_forecaster_histories", inputs_, {"training": False, "physical_gpu": 0})
    net, codec, _ = model(TRAIN / "model.pt", Config().updates)
    net.eval().requires_grad_(False)
    before = study.parameter_hash(net)
    histories = {}
    probes = {r["id"] for r in old.clean.training_probes(rows)}
    hooks = [head.register_forward_hook(old.require_finite_logits) for head in net.generator.lm_heads[:-1]]
    for row in rows:
        item = study.forecast(net, study.source_view(row), manifest)
        study.validate_history(row, item)
        if row["id"] in probes and study.forecast(net, study.source_view(row), manifest) != item:
            raise ValueError("full native-frame source-only cache must repeat exactly")
        histories[row["id"]] = item
        write_json(CACHE / "histories.json", histories)
        if len(histories) % 16 == 0:
            print(json.dumps({"cached": len(histories), "total": len(rows)}), flush=True)
    for handle in hooks:
        handle.remove()
    sources = [study.source_view(r) for r in rows]
    write_json(CACHE / "source_cases.json", sources)
    if (study.parameter_hash(net) != before or loader.module_hash(codec) != loader.ORIGINAL_CODEC
            or any(p.grad is not None for p in net.parameters()) or torch.cuda.max_memory_allocated() > 8_000_000_000):
        raise ValueError("source-only cache modified frozen state or exceeded memory")
    study.data.verify(p)
    write_json(CACHE / "result.json", {"status": "matched_dense_TRAIN_histories_prepared", "cases": 690,
        "forecaster_parameter_sha256": before, "source_only": True, "target_labels_in_prediction": False,
        "histories_sha256": file_hash(CACHE / "histories.json"), "source_cases_sha256": file_hash(CACHE / "source_cases.json"),
        "checkpoint_sha256": file_hash(TRAIN / "model.pt"), "codec_unchanged": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "task_accuracy": None,
        "final_test_opened": False, "speed_tested": False, "producer_fingerprint": p["fingerprint"]})


def workflow():
    data_inputs()
    if any(p.exists() for p in (WORKFLOW, PILOT, FIT_PILOT, TRAIN, FIT, CACHE, RESULTS, REPORT,
            Path("data/hypothesis_registry") / (HYPOTHESIS + ".json"))):
        raise ValueError("fresh dense-matched-forecaster study required")
    p = producing(WORKFLOW, "supervise_new690_forecaster_and_source_only_cache", [], {"training": False})
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [], "producer_fingerprint": p["fingerprint"]}
    try:
        for name, stage, root in (("train_pilot", "pilot", PILOT), ("fit_pilot", "fit-pilot", FIT_PILOT),
                ("train", "train", TRAIN), ("fit", "fit", FIT), ("cache", "cache", CACHE),
                ("finalize", "finalize", RESULTS), ("register", "register", RESULTS)):
            old.clean.recovery.study.data.verify(p)
            command = [sys.executable, "-u", "-m", AUDITOR if stage in ("finalize", "register") else MODULE, stage]
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "" if stage in ("finalize", "register") else "0"})
            journal.update(active_stage=name, child_pid=child.pid)
            write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            receipt = {"stage": name, "exit_code": code, "child_pid": child.pid, "observer_pid": os.getpid(),
                "observer": "subprocess.wait", "command": command}
            write_json(WORKFLOW / "stage_receipts" / (name + ".json"), receipt)
            if root.exists():
                write_json(root / ("registry_terminal_receipt.json" if name == "register" else "worker_terminal_receipt.json"), receipt)
            journal["stages"].append(receipt)
            if code:
                raise RuntimeError("dense matched forecaster stage failed: " + name)
        old.clean.recovery.study.data.verify(p)
        journal.update(status="dense_matched_forecaster_completed", active_stage=None,
            classification=old.clean.recovery.study.q.read(RESULTS / "campaign_results.json")["classification"])
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


def __getattr__(name):
    return getattr(old, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("check", "cpu-model", "pilot", "fit-pilot", "train", "fit", "cache", "workflow"))
    args = parser.parse_args()
    configure()
    if args.stage == "check":
        rows, _, selected = data_inputs()
        print(json.dumps({"cases": len(rows), "diagnostic_cases": len(selected), "config": asdict(Config()), "task_accuracy": None}))
    elif args.stage == "cpu-model":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
            raise ValueError("constructor parity check must stay CPU only")
        net, _, frozen = model(device="cpu")
        print(json.dumps({"status": "systems_exact_forecaster_constructor", "initial_parameters": old.clean.recovery.study.parameter_hash(net),
            "frozen_parameters": frozen, "trainable_parameters": sum(p.numel() for p in net.parameters() if p.requires_grad)}))
    elif args.stage == "workflow":
        workflow()
    elif args.stage == "cache":
        cache()
    elif args.stage in ("pilot", "train"):
        old.train(args.stage == "pilot")
    else:
        old.fit(args.stage == "fit-pilot")
