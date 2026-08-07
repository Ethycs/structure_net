#!/usr/bin/env python3
"""Correct the uncalibrated arm to expose a true 2-D equivariant vector."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import nn

import experiments.structure_net.tinyllm_io_correspondence as v1


SCHEMA_VERSION = "nal.tinyllm-io-correspondence.v1.1"
_V1_WORKER = v1.io_correspondence_worker


class EquivariantVectorEncoder(nn.Module):
    """Expose the complete vector representation, not one non-equivariant coordinate."""

    def __init__(self, sensor_steps: int, vector_channels: int):
        super().__init__()
        self.base = v1.LearnedEquivariantEncoder(sensor_steps, vector_channels)

    def forward(self, sensor: torch.Tensor) -> torch.Tensor:
        return self.base.vector(sensor)


class CorrectedUncalibratedTinyLLM(nn.Module):
    """Raw control or a genuine SO(2)-equivariant planar front end."""

    def __init__(
        self,
        model,
        frontend: str,
        task,
        config: v1.IOCorrespondenceConfig,
    ):
        super().__init__()
        if frontend not in {"raw", "equivariant"}:
            raise ValueError("uncalibrated frontend must be raw or equivariant")
        self.model = model
        self.frontend = frontend
        self.encoder = (
            EquivariantVectorEncoder(task.sensor_steps, config.vector_channels)
            if frontend == "equivariant"
            else None
        )
        self.scalar_embedding = (
            nn.Linear(2, model.config.n_embd) if frontend == "equivariant" else None
        )

    def feature(self, sensor: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            return sensor.flatten(1)
        return self.encoder(sensor)

    def forward_cuts(self, input_ids: torch.Tensor, sensor: torch.Tensor):
        feature = self.feature(sensor)
        if self.encoder is None:
            prefix = self.model.transformer["wte"](input_ids[:, :-1])
            query = self.model.transformer["wte"](input_ids[:, -1:])
            value = torch.cat((prefix, torch.zeros_like(query), query), dim=1)
        else:
            if self.scalar_embedding is None:
                raise AssertionError("equivariant vector embedding missing")
            bos = self.model.transformer["wte"](input_ids[:, :1])
            vector = self.scalar_embedding(feature)[:, None, :]
            query = self.model.transformer["wte"](input_ids[:, -1:])
            value = torch.cat((bos, vector, query), dim=1)
        return v1._transformer_cuts(self.model, value, feature)


def _implementation_digest() -> str:
    digest = v1.hashlib.sha256()
    paths = (
        Path(__file__),
        Path(v1.__file__),
        Path(v1.calibrated_source.__file__),
        Path(v1.invariant_source.__file__),
        Path(v1.nuisance_source.__file__),
    )
    for path in sorted(paths, key=str):
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _activate_correction() -> None:
    v1.UncalibratedTinyLLM = CorrectedUncalibratedTinyLLM
    v1._implementation_digest = _implementation_digest
    v1.SCHEMA_VERSION = SCHEMA_VERSION


def io_correspondence_worker(experiment, device_id):
    _activate_correction()
    return _V1_WORKER(experiment, device_id)


async def run_campaign(config, task, output_dir, calibrated_source_root):
    _activate_correction()
    v1.io_correspondence_worker = io_correspondence_worker
    bundle = await v1.run_campaign(config, task, output_dir, calibrated_source_root)
    bundle["correction"] = {
        "kind": "implementation_conformance_correction",
        "original_schema_version": "nal.tinyllm-io-correspondence.v1",
        "original_artifact": (
            "data/experiments/tinyllm_io_correspondence/20260806_d8_preregistered"
        ),
        "change": (
            "Uncalibrated learned arms expose the complete two-dimensional "
            "SO(2)-equivariant vector instead of its non-equivariant x-component."
        ),
        "outcomes_seen_before_correction": True,
        "claim_status": "disclosed_post-outcome_implementation_correction",
    }
    v1._write_json(output_dir / "campaign_results.json", bundle)
    return bundle


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = v1.build_parser().parse_args(argv)
    reuse_calibrated = True
    if args.shakedown:
        args.seeds = "7"
        args.steps = 2
        args.train_samples = 32
        args.batch_size = 8
        args.probe_steps = 20
        args.probe_train_samples = 64
        args.probe_validation_samples = 32
        args.probe_test_samples = 32
        args.allow_underpowered = True
        reuse_calibrated = False
    config = v1.IOCorrespondenceConfig(
        seeds=v1._comma_ints(args.seeds),
        training_steps=args.steps,
        train_samples=args.train_samples,
        batch_size=args.batch_size,
        probe_steps=args.probe_steps,
        probe_train_samples=args.probe_train_samples,
        probe_validation_samples=args.probe_validation_samples,
        probe_test_samples=args.probe_test_samples,
        device_ids=v1._parse_devices(args.gpus),
        gpu_slots_per_device=args.slots_per_gpu,
        max_parallel_experiments=args.max_parallel,
        max_retries=args.retries,
        resume=args.resume,
        reuse_calibrated_controls=reuse_calibrated,
        allow_underpowered=args.allow_underpowered,
    )
    task = v1.CircleTaskConfig(train_samples=config.train_samples)
    output = args.output
    bundle = asyncio.run(run_campaign(config, task, output, args.calibrated_source))
    print(
        v1.json.dumps(
            {
                "summary": bundle["summary"],
                "conclusion": bundle.get("aggregates", {}).get("conclusion"),
                "correction": bundle["correction"]["claim_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(output / "campaign_results.json")
    return 0 if bundle["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
