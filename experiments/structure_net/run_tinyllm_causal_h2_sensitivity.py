#!/usr/bin/env python3
"""Run one preregistered, non-rescuing A7 sensitivity campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from experiments.structure_net.tinyllm_causal_h2_attention import (
        CausalH2Config,
        _write_json,
        run_campaign,
    )
except ModuleNotFoundError:  # Direct script execution.
    from tinyllm_causal_h2_attention import (  # type: ignore[no-redef]
        CausalH2Config,
        _write_json,
        run_campaign,
    )


ARM_OVERRIDES = {
    "leaf8": {"leaf_size": 8},
    "leaf32": {"leaf_size": 32},
    "separation0p5": {"separation_ratio": 0.5},
    "separation2": {"separation_ratio": 2.0},
    "rank0p5": {"rank_multiplier": 0.5},
    "rank2": {"rank_multiplier": 2.0},
}
PRIMARY_CAMPAIGN = (
    "data/experiments/tinyllm_causal_h2_attention/"
    "20260830_registered/campaign_results.json"
)
PRIMARY_CAMPAIGN_SHA256 = (
    "4885696dd746a52cb015b51d34733901c2acd50baccfc59f2f76cfe176eeb9b2"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARM_OVERRIDES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(
        "data/experiments/tinyllm_causal_h2_attention/"
        f"sensitivities/{args.arm}"
    )
    config = CausalH2Config(device=args.device, **ARM_OVERRIDES[args.arm])
    campaign = run_campaign(config, output)
    sensitivity = {
        "arm": args.arm,
        "overrides": ARM_OVERRIDES[args.arm],
        "role": "preregistered_sensitivity_non_rescuing",
        "can_rescue_primary": False,
        "primary_campaign": PRIMARY_CAMPAIGN,
        "primary_campaign_sha256": PRIMARY_CAMPAIGN_SHA256,
        "primary_classification": "h2_representation_pass_no_finite_size_compression",
    }
    for item in campaign["results"]:
        path = Path(item["result"])
        record = json.loads(path.read_text(encoding="utf-8"))
        record["evidence_role"] = "preregistered_sensitivity_non_rescuing"
        record["sensitivity"] = sensitivity
        _write_json(path, record)
    campaign["evidence_role"] = "preregistered_sensitivity_non_rescuing"
    campaign["sensitivity"] = sensitivity
    _write_json(output / "campaign_results.json", campaign)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "classification": campaign["aggregates"]["classification"],
                "representation_pass": campaign["aggregates"]["representation_pass"],
                "compression": campaign["aggregates"]["compression"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
