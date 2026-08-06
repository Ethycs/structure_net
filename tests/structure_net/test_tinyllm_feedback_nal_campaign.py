"""Integration tests for the TinyLLM canonical NAL campaign worker."""

import asyncio

from experiments.structure_net.tinyllm_feedback_nal_campaign import (
    TinyLLMNALCampaignConfig,
    run_campaign,
)


def test_cpu_campaign_runs_matched_seeds_and_resumes(tmp_path):
    config = TinyLLMNALCampaignConfig(
        seeds=(3, 5),
        training_steps=1,
        batch_size=2,
        sequence_length=5,
        vocab_size=20,
        device_ids=(-1,),
        gpu_slots_per_device=1,
        max_parallel_experiments=1,
        max_retries=1,
        resume=True,
    )

    first = asyncio.run(run_campaign(config, tmp_path))
    second = asyncio.run(run_campaign(config, tmp_path))

    assert first["summary"] == {
        "requested": 2,
        "completed": 2,
        "failed": 0,
        "all_matched": True,
        "all_checkpoints_exact": True,
    }
    assert second["summary"] == first["summary"]
    assert second["scheduler"]["logical_device_slots"] == [-1]
    assert len(list((tmp_path / ".nal_runner" / "completed").glob("*.json"))) == 2
    for seed in config.seeds:
        assert (tmp_path / "runs" / f"seed_{seed}" / "results.json").is_file()


def test_isolated_timing_rejects_parallel_campaign():
    try:
        TinyLLMNALCampaignConfig(isolated_timing=True, max_parallel_experiments=2)
    except ValueError as error:
        assert "isolated timing" in str(error)
    else:
        raise AssertionError("parallel isolated timing should be rejected")
