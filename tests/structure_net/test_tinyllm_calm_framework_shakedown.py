"""Lifecycle test for the CALM-style TinyLLM framework exercise."""

import json

from experiments.structure_net.tinyllm_calm_framework_shakedown import (
    CLAIM_SCOPE,
    ShakedownConfig,
    run_shakedown,
)


def test_tiny_calm_shakedown_writes_strict_restorable_artifacts(tmp_path):
    output = tmp_path / "campaign"
    campaign = run_shakedown(
        ShakedownConfig(
            source_training_steps=1,
            autoencoder_training_steps=2,
            energy_training_steps=2,
            batch_size=4,
        ),
        output,
    )

    assert campaign["status"] == "completed"
    assert campaign["claim_scope"] == CLAIM_SCOPE
    assert all(campaign["aggregates"]["lifecycle_gates"].values())
    serialized = json.loads((output / "campaign_results.json").read_text())
    assert serialized["aggregates"]["conclusion"] == "framework_lifecycle_validated"
    run = serialized["results"][0]
    result = json.loads(open(run["result"], encoding="utf-8").read())
    assert result["lifecycle_pass"]
    assert result["artifacts"]["checkpoint_loss_max_abs_delta"] == 0.0
    assert result["artifacts"]["checkpoint_prediction_max_abs_delta"] == 0.0
