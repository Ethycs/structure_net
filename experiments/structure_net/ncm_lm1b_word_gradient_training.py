"""Fixed warm continuation with word gradients across native hard concept codes."""
import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import torch

from experiments.structure_net import ncm_lm1b_grounded_training as engine
from experiments.structure_net import ncm_lm1b_word_gradient as objective
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, q, ROOT = engine.native, engine.q, engine.ROOT
HYPOTHESIS = "ncm-lm1b-full690-word-gradient-neighborhood-v1"
MODULE = "experiments.structure_net.ncm_lm1b_word_gradient_training"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-lm1b-word-gradient-training-plan.md")
TEST = Path("tests/structure_net/test_ncm_lm1b_word_gradient_training.py")
WORKFLOW = ROOT / "20261024_lm1b_word_gradient_training_workflow_v1"
PILOT = ROOT / "20261024_lm1b_word_gradient_training_pilot_v1"
TRAIN = ROOT / "20261024_lm1b_word_gradient_training_train_v1"
BASE_TRAIN = engine.TRAIN
BASE_WORKFLOW = engine.WORKFLOW
BASE_SHA = "bb239a482db67d67d2d7f410588fada7b6782c42474eb9e8d1dffdd8bb7d1490"
BASE_ATTENTION = "38af3e583bde80a09981556860a282e7133cb8d20a7c1f9faffdd15f31c6307d"
SYSTEMS = ROOT / "20261024_lm1b_word_gradient_cuda_pilot_v1"
SYSTEMS_RESULTS = ROOT / "20261024_lm1b_word_gradient_cuda_pilot_results_v1"
SYSTEMS_HYPOTHESIS = "ncm-lm1b-word-gradient-lifecycle-v1"
SYSTEMS_CAMPAIGN = "18bdccb38ce68e0df79886a58a455917c49bc6d5924cdac2d70a639f5e34cc24"
_producing, _train = engine.producing, engine.train
_from_domain = engine.head.from_domain
_original_objective = engine.objective


@dataclass(frozen=True)
class Config:
    updates: int = 1024
    microbatch: int = 4
    accumulation: int = 4
    warmup: int = 64
    max_lr: float = 1e-4
    min_lr: float = 2e-5
    sampler_seed: int = 106142
    neighbor_seed: int = 107142
    dropout_seed: int = 108142
    self_history_start: int = 0
    self_history_interval: int = 1


def rate(step):
    c = Config()
    if not 1 <= step <= c.updates:
        raise ValueError("step outside fixed continuation budget")
    if step <= c.warmup:
        return c.max_lr * step / c.warmup
    fraction = (step - c.warmup) / (c.updates - c.warmup)
    return c.min_lr + (c.max_lr - c.min_lr) * .5 * (1 + math.cos(math.pi * fraction))


def require_systems():
    native.vendor.terminal(SYSTEMS)
    result = q.read(SYSTEMS / "result.json")
    campaign = q.read(SYSTEMS_RESULTS / "campaign_results.json")
    receipt = q.read(SYSTEMS_RESULTS / (SYSTEMS_HYPOTHESIS + "_registry_readback.json"))
    if (file_hash(SYSTEMS_RESULTS / "campaign_results.json") != SYSTEMS_CAMPAIGN
            or receipt["record"]["campaign_sha256"] != SYSTEMS_CAMPAIGN or receipt["exact"]["campaign_sha256"] != SYSTEMS_CAMPAIGN
            or campaign["classification"] != "systems_hard_forward_exact_word_gradient_restored"
            or result["status"] != "systems_word_gradient_hard_forward_and_reload_passed_not_task_quality"
            or not result["bridged_word_loss_concept_gradient_nonzero"]
            or not result["hard_forward_word_logits_bitwise_equal"]
            or not result["exact_hard_generation_optimizer_and_allRNG_reload"]):
        raise ValueError("actual audited registered word-gradient CUDA lifecycle required")


def warm_model(device="cpu", **kwargs):
    if file_hash(BASE_TRAIN / "model.pt") != BASE_SHA:
        raise ValueError("fixed original4096 checkpoint changed")
    model = _from_domain(device=device, **kwargs)
    saved = torch.load(BASE_TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    if saved["step"] != 4096:
        raise ValueError("warm start must be original final4096, never a two-update pilot")
    model.load_attention_state(saved["attention"])
    if engine.attention_hash(model) != BASE_ATTENTION:
        raise ValueError("warm start attention differs")
    return model


def backward_microbatch(model, features, mask, codes, words, neighbor, step, *, accumulation=4, self_history=False):
    schedule = 4096 + step
    if self_history:
        result = objective.backward_microbatch(model, features, mask, codes, words, neighbor, schedule,
            accumulation=accumulation, temperature=1.)
    else:
        result = _original_objective.backward_microbatch(model, features, mask, codes, words, neighbor, schedule,
            accumulation=accumulation, self_history=False)
    result.update(corruption_schedule_step=schedule, word_gradient_auxiliary_executed=self_history)
    return result


def configure():
    engine.Config, engine.rate = Config, rate
    engine.HYPOTHESIS, engine.MODULE, engine.PLAN = HYPOTHESIS, MODULE, PLAN
    engine.WORKFLOW, engine.PILOT, engine.TRAIN = WORKFLOW, PILOT, TRAIN
    engine.producing = producing
    engine.head.from_domain = warm_model
    engine.objective = SimpleNamespace(__file__=objective.__file__, backward_microbatch=backward_microbatch,
        corrupt_history=_original_objective.corrupt_history)


def producing(root, stage, inputs=(), config=None):
    require_systems()
    files = [Path(__file__), PLAN, TEST, Path(objective.__file__), Path(_original_objective.__file__)]
    extra = [BASE_TRAIN / n for n in ("model.pt", "result.json", "reload_probes.json", "worker_terminal_receipt.json")]
    extra += [BASE_WORKFLOW / "tool_terminal_receipt.json", SYSTEMS / "result.json", SYSTEMS / "worker_terminal_receipt.json",
        SYSTEMS_RESULTS / "campaign_results.json", SYSTEMS_RESULTS / (SYSTEMS_HYPOTHESIS + "_registry_readback.json"), *files]
    p = _producing(root, stage, [*inputs, *extra], config)
    p["hypothesis_id"] = HYPOTHESIS
    p["stage"] = "word_gradient_" + stage.replace("fixed4096", "fixed1024")
    p["config"].update(hypothesis_id=HYPOTHESIS, schema="nal.ncm-lm1b-full690-word-gradient-training.v1",
        conformance="EXPLORATORY continuation after two full128 semantic failures; no quality qualification",
        warm_start_checkpoint_sha256=BASE_SHA, warm_start_attention_sha256=BASE_ATTENTION,
        reset_optimizer=True, pilot_checkpoint_reused=False, new_random_trainable_weights=False,
        checkpoint_selection="fixed final1024 only", word_gradient_into_concept_attention=True,
        loss="prior clean.5concept+.7word/noisy.5concept+.3word;eachupdate firstmicro .25selfconcept+1selfword through hard-forward codebook surrogate",
        self_history_frequency="firstmicrobatch every update", auxiliary_word_weight=1., surrogate_temperature=1.,
        corruption="constantmax.25 after inherited4096 schedule;per-case uniform0to.25,rootprotected",
        full_task_accuracy_measured=False)
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"word_gradient_training_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def train(pilot=False):
    configure()
    require_systems()
    _train(pilot)
    root = PILOT if pilot else TRAIN
    result = q.read(root / "result.json")
    if result["initial_attention_sha256"] != BASE_ATTENTION:
        raise ValueError("continuation did not start from fixed4096 attention")
    result.update(warm_start_checkpoint_sha256=BASE_SHA, word_gradient_objective=True,
        final_fixed_updates=2 if pilot else 1024, original_body_updated=False)
    write_json(root / "result.json", result)


def workflow():
    configure()
    require_systems()
    if any(p.exists() for p in (WORKFLOW, PILOT, TRAIN)):
        raise ValueError("fresh continuation roots required")
    p = producing(WORKFLOW, "word_gradient_fixed1024_continuation")
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [], "producer_fingerprint": p["fingerprint"]}
    try:
        for stage, root in (("pilot", PILOT), ("train", TRAIN)):
            native.vendor.verify(p)
            command = [sys.executable, "-u", "-m", MODULE, stage]
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
            journal.update(active_stage=stage, child_pid=child.pid)
            write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            receipt = {"stage": stage, "exit_code": code, "child_pid": child.pid, "observer_pid": os.getpid(),
                "observer": "subprocess.wait", "command": command}
            write_json(WORKFLOW / "stage_receipts" / (stage + ".json"), receipt)
            if root.exists():
                write_json(root / "worker_terminal_receipt.json", receipt)
            journal["stages"].append(receipt)
            if code:
                raise RuntimeError("word-gradient continuation stage failed:" + stage)
        native.vendor.verify(p)
        journal.update(status="word_gradient_fixed1024_completed_task_accuracy_pending", active_stage=None, child_pid=None)
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("pilot", "train", "workflow"))
    args = parser.parse_args()
    {"pilot": lambda: train(True), "train": train, "workflow": workflow}[args.stage]()
