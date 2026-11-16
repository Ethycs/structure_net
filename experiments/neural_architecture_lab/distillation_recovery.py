"""CPU preparation/shakedown and aggregation for NAL distillation recovery.

No scheduler or implicit GPU launcher is provided. Real workers are called by
NAL inside an externally acquired inference-ctl lease. This entry point is CPU
only and deliberately never opens a scientific TEST corpus.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "_nal_response_recovery"


def module(name):
    """Reuse NAL files without importing unrelated optional analyzer backends.

    The canonical package remains unchanged; this isolated namespace permits
    the locked minimal torch environment to run its core/ledger implementation.
    """
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(ROOT / "src/neural_architecture_lab")]
        sys.modules[PACKAGE] = package
    return importlib.import_module(PACKAGE + "." + name)


def source_paths():
    paths = [Path(__file__), ROOT / "tests/neural_architecture_lab/test_distillation_recovery.py",
             ROOT / "docs/05 - Standards/NAL-STD-DISTILLATION-v1.md"]
    paths.extend(ROOT / "src/neural_architecture_lab" / (name + ".py") for name in
                 ("core", "local_scheduler", "distillation_recovery", "distillation_data", "distillation_training"))
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "cpu-shakedown", "aggregate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, help="Append-only aggregate destination")
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Set CUDA_VISIBLE_DEVICES='' before starting this CPU entry point")
    recovery = module("distillation_recovery")
    if args.phase == "prepare":
        if args.config is None:
            raise ValueError("prepare requires a frozen study --config")
        study = recovery.RecoveryStudy.from_dict(recovery.read(args.config))
        result = recovery.prepare(args.root, study, [*source_paths(), args.config])
    elif args.phase == "cpu-shakedown":
        result = cpu_shakedown(args.root)
    else:
        if args.output is None:
            raise ValueError("aggregate requires a new --output path; prior evidence is never overwritten")
        result = recovery.aggregate(args.root)
        recovery.write(args.output, result)
    print(recovery.strict_json(result), flush=True)
    return result


def tiny_problem(seed=17):
    """Synthetic systems fixture; its arbitrary tokens support no quality claim."""
    import torch
    from torch import nn
    data = module("distillation_data")
    torch.manual_seed(seed)
    class TinyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(19, 8)
            self.block = nn.Linear(8, 8)
            self.head = nn.Linear(8, 19, bias=False)
            self.register_buffer("geometry", torch.ones(8))

        def hidden(self, batch):
            return self.block(self.embedding(batch["input_ids"])).tanh() * self.geometry

        def forward(self, batch):
            return self.head(self.hidden(batch))
    student = TinyStudent()
    teacher = TinyStudent().eval().requires_grad_(False)
    records = [data.ResponseRecord(example_id=f"tiny-{i}", group_id=f"tiny-group-{i}", split="train",
        task="systems_fixture", prompt_ids=(1, 2+i), response_ids=(5+i, 8+i, 18), eos_token_id=18,
        finish_reason="eos", provenance=dict(source_uri="synthetic://nal-cpu-shakedown", source_revision="v1",
        tokenizer_sha256="0"*64, chat_template_sha256="1"*64, prompt_provenance="synthetic token fixture",
        response_provenance="synthetic token fixture", assistant_boundary_verified=True)) for i in range(4)]
    return student, teacher, records


def cpu_shakedown(root):
    """Exercise the real shared trainer and ledger; never produce quality rows."""
    recovery = module("distillation_recovery")
    training = module("distillation_training")
    data = module("distillation_data")
    student, teacher, records = tiny_problem()
    identities = dict(teacher=training.module_fingerprint(teacher), student_initializer=training.module_fingerprint(student),
        tokenizer="0"*64, chat_template="1"*64, data=recovery.digest([asdict(r) for r in records]))
    study = recovery.RecoveryStudy(study_id="NAL-DISTILLATION-CPU-SHAKEDOWN", teacher_sha256=identities["teacher"],
        initializer_sha256=identities["student_initializer"], tokenizer_sha256=identities["tokenizer"],
        template_sha256=identities["chat_template"], data_sha256=identities["data"], geometry_sha256=recovery.digest([1.]*8),
        arms=(recovery.RecoveryArm("dense_reference"),), seeds=(17,), intended_tasks=("systems_fixture",),
        precision="float32", deployment_hardware="CPU systems fixture; no deployment claim", max_input_tokens=2,
        max_output_tokens=3, max_context_tokens=5, supervised_token_budget=12, max_updates=2)
    if not Path(root).exists():
        recovery.prepare(root, study, source_paths())
    else:
        _, frozen = recovery.verify_plan(root)
        if frozen != study:
            raise ValueError("CPU fixture differs from prepared study")
    worker = recovery.RecoveryWorker(root, tiny_adapter)
    result = worker(recovery.make_experiments(study)[0], -1)
    if result.error:
        raise RuntimeError(result.error)
    summary = recovery.aggregate(root)
    output = Path(root) / "campaign_results.json"
    if not output.exists():
        recovery.write(output, summary)
    elif recovery.read(output) != summary:
        raise ValueError("Prior shakedown summary differs")
    return summary


def tiny_adapter(*, study, arm, seed, device_id, output_dir, progress, guard):
    import time
    import torch
    training = module("distillation_training")
    data = module("distillation_data")
    recovery = module("distillation_recovery")
    if device_id != -1 or study.profile != "systems":
        raise ValueError("Synthetic adapter is CPU systems-only")
    started = time.monotonic()
    student, teacher, records = tiny_problem(seed)
    identities = dict(teacher=training.module_fingerprint(teacher), student_initializer=training.module_fingerprint(student),
        tokenizer=study.tokenizer_sha256, chat_template=study.template_sha256, data=study.data_sha256)
    if (identities["teacher"], identities["student_initializer"]) != (study.teacher_sha256, study.initializer_sha256):
        raise ValueError("Fresh toy initialization changed")
    before = {name: value.detach().clone() for name, value in student.named_parameters()}
    geometry = student.geometry.clone()
    teacher_before = training.module_fingerprint(teacher)
    batches = [data.response_batch(records[i:i+2], pad_token_id=0, max_length=5) for i in (0, 2)]
    provider = training.FrozenTeacherProvider(teacher, teacher.hidden, teacher.head,
        teacher_id=identities["teacher"], temperature=arm.temperature, guard=guard)
    config = training.DistillationConfig(temperature=arm.temperature, ce_weight=arm.ce_weight,
        token_chunk_size=2, accumulation_batches=1, max_supervised_tokens=study.supervised_token_budget,
        max_updates=study.max_updates)
    trainer = training.DistillationTrainer(student, torch.optim.AdamW(student.parameters(), lr=.01),
        student.hidden, student.head, provider, config=config, identities=identities, guard=guard)
    initialization = time.monotonic()-started
    progress(phase="training", initialization_seconds=initialization, optimizer_updates=0, supervised_tokens=0,
             placement="cpu")
    result = trainer.fit(batches, output_dir / "training", progress=lambda event: progress(**event))
    # An independent fresh object reload follows the trainer's own round trip.
    fresh, _, _ = tiny_problem(seed)
    training.load_checkpoint(output_dir / "training/final.pt", fresh,
        expected_sha256=result["final_checkpoint"]["sha256"], identities=identities)
    with torch.inference_mode():
        parity = all(torch.equal(student(batch), fresh(batch)) for batch in batches)
        path_parity = all(torch.equal(student(batch), student.head(student.hidden(batch))) for batch in batches)
    return dict(identities=identities, systems=dict(teacher_unchanged=training.module_fingerprint(teacher)==teacher_before,
        fixed_geometry_unchanged=torch.equal(student.geometry, geometry), fresh_initializer=True,
        all_trainables_updated=all(not torch.equal(before[name], value) for name, value in student.named_parameters()),
        save_reload_exact=parity, full_vocab_path_parity=path_parity), quality=None,
        progress=dict(optimizer_updates=result["updates"], supervised_tokens=result["supervised_tokens"],
            input_tokens=result["input_tokens"], response_tokens=result["response_tokens"],
            initialization_seconds=initialization, training_seconds=result["elapsed_seconds"]),
        checkpoint="training/final.pt", checkpoint_sha256=recovery.sha(output_dir / "training/final.pt"),
        required_artifacts=["training/initial.pt", "training/final.pt", "training/result.json"],
        model_architecture=[19, 8, 8, 19], model_parameters=sum(p.numel() for p in student.parameters()),
        resources=dict(logical_device=-1, gpu_peak_reserved_bytes=None, placement="cpu"))


if __name__ == "__main__":
    main()
