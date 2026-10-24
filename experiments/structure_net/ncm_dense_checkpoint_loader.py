"""Load exact dense NCM initial weights without replaying historical data audits.

Callers must independently validate the data used by their new experiment.
This loader checks the actual model files and complete parameter/codec identity.
It is not used by already-running, fingerprint-frozen experiments.
"""
import argparse
import json
import os

import torch

from experiments.structure_net import ncm_matched_clean_word as clean
from experiments.structure_net import ncm_word_final_training as factory
from experiments.structure_net import ncm_dense_latent_word as core
from experiments.structure_net.ncm_device_geometry import set_source_frame
from experiments.structure_net.ncm_generated_history_core import independent_word_branch
from experiments.structure_net.ncm_frozen_body import module_hash
from experiments.structure_net.ncm_memory_data import file_hash

FORECASTER_CHECKPOINT = "27f835d1391d7e14439716a0514a86d5b301bee79341199ab9dcfe88748e454d"
FORECASTER_PARAMETERS = "5357dde0aa21c3c67cc281dcdb09ce74c380f6d29027eaa2853fe9a1aa863bf2"
WORD_CHECKPOINT = "c90ee2ae272c298022d50d7a9b4182f06d8d995fadf8ecceb6f091e35b8b5223"
DENSE_INITIAL_PARAMETERS = "16dd5f252d59ae731d25ea94645c1e300578708a4affd170f4ddc243d9a4812b"
DENSE_FROZEN_PARAMETERS = "6755c79596a36582c98d47e200fab5d72ea7080866b47dd1b153eb1e705b3a09"
ORIGINAL_CODEC = "50fc5680ce0fb3c82764d865243a2e509d8f16dd8c051113b16080b5d763d680"


def checked_checkpoint(path, expected_sha, expected_step):
    if file_hash(path) != expected_sha:
        raise ValueError("immutable model file identity differs: " + str(path))
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if saved["step"] != expected_step:
        raise ValueError("fixed selected model update differs")
    return saved


def load_models(device="cpu"):
    clean.recovery.configure()
    study = clean.recovery.study
    forecast_path = study.bootstrap.TRAIN / "model.pt"
    word_path = clean.TRAIN / "model.pt"
    forecast = checked_checkpoint(forecast_path, FORECASTER_CHECKPOINT, 16384)
    word = checked_checkpoint(word_path, WORD_CHECKPOINT, 32768)
    fixed, codec, _ = factory.make_model(device="cpu")
    set_source_frame(fixed.generator, forecast["source_frame"], maximum=8192)
    fixed.load_state_dict(forecast["model"], strict=True)
    fixed.eval().requires_grad_(False)
    if study.parameter_hash(fixed) != FORECASTER_PARAMETERS or module_hash(codec) != ORIGINAL_CODEC:
        raise ValueError("complete forecaster or original codec parameter identity differs")
    branch = independent_word_branch(fixed).eval()
    set_source_frame(branch.generator, word["source_frame"], maximum=8192)
    branch.load_state_dict(word["model"], strict=True)
    core.prepare_branch(branch)
    if (study.parameter_hash(branch) != DENSE_INITIAL_PARAMETERS
            or study.parameter_hash(branch, frozen_only=True) != DENSE_FROZEN_PARAMETERS):
        raise ValueError("all dense initializer or frozen parameter bytes differ")
    for net in (fixed, branch):
        facade = net.generator.vqvae_local[0]
        if facade.original is not codec or any(not torch.equal(head.E, facade.vq.embeddings[i].weight)
                for i, head in enumerate(net.generator.lm_heads)):
            raise ValueError("shared original representation or nonpersistent codebook buffers differ")
    metadata = {"forecaster_checkpoint_sha256": FORECASTER_CHECKPOINT,
        "word_checkpoint_sha256": WORD_CHECKPOINT, "forecaster_parameter_sha256": FORECASTER_PARAMETERS,
        "initial_dense_parameter_sha256": DENSE_INITIAL_PARAMETERS,
        "frozen_dense_parameter_sha256": DENSE_FROZEN_PARAMETERS, "codec_sha256": ORIGINAL_CODEC,
        "dense_parameter_names": len(list(branch.named_parameters())),
        "trainable_dense_parameters": sum(p.numel() for p in branch.parameters() if p.requires_grad),
        "historical_source_features_read": False, "parent_body_loaded": False,
        "model_training_executed": False, "task_accuracy": None}
    codec.to(device).eval().requires_grad_(False)
    branch.to(device)
    set_source_frame(branch.generator, word["source_frame"], maximum=8192)
    return fixed, branch, codec, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("check",))
    parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("loader parity check must be explicitly CPU only")
    torch.set_num_threads(2)
    _, _, _, metadata = load_models()
    if torch.cuda.is_initialized():
        raise ValueError("CPU constructor check initialized CUDA")
    print(json.dumps({"status": "systems_exact_CPU_model_loader_parity", **metadata}), flush=True)
