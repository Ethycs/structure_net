#!/usr/bin/env python3
"""CLI for the paired patched-density replication protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from neural_architecture_lab.experiments.patched_density_replication import (
    ReplicationConfig,
    run_replication,
)


def _csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate patched-density MNIST and transfer to Fashion-MNIST."
    )
    parser.add_argument("--datasets", type=_csv_strings, default=("mnist", "fashion_mnist"))
    parser.add_argument("--seeds", type=_csv_ints, default=(4,))
    parser.add_argument("--scaffold-epochs", type=int, default=20)
    parser.add_argument("--continuation-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=1000)
    parser.add_argument("--connection-density", type=float, default=0.02)
    parser.add_argument("--patch-width", type=int, default=10)
    parser.add_argument("--patch-scale", type=float, default=0.1)
    parser.add_argument("--max-patches", type=int, default=5)
    parser.add_argument("--probe-split", choices=("train", "test"), default="test")
    parser.add_argument("--probe-batches", type=int, default=1)
    parser.add_argument("--subset-fraction", type=float)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-memory-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path(
        f"data/experiments/patched_density_replication/{timestamp}"
    )
    config = ReplicationConfig(
        datasets=args.datasets,
        seeds=args.seeds,
        connection_density=args.connection_density,
        scaffold_epochs=args.scaffold_epochs,
        continuation_epochs=args.continuation_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        patch_width=args.patch_width,
        patch_scale=args.patch_scale,
        max_patches=args.max_patches,
        probe_split=args.probe_split,
        probe_batches=args.probe_batches,
        subset_fraction=args.subset_fraction,
        num_workers=args.num_workers,
        torch_threads=args.torch_threads,
        device=args.device,
        cache_dataset_in_memory=not args.no_memory_cache,
    )
    result, output_path = run_replication(
        config, output_dir=output_dir, verbose=not args.quiet
    )
    print(f"\nresults: {output_path}")
    for dataset_name, summary in result["summary"]["by_dataset"].items():
        print(
            f"{dataset_name}: scaffold={summary['mean_scaffold_accuracy']:.2%}, "
            f"control={summary['mean_control_accuracy']:.2%}, "
            f"patched={summary['mean_patched_accuracy']:.2%}, "
            f"paired_delta={summary['mean_patched_minus_control']:+.2%}"
        )


if __name__ == "__main__":
    main()
