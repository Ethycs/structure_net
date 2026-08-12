import math
from pathlib import Path

import pytest
import torch

import experiments.structure_net.tinyllm_temporal_language_task as temporal


TOKENIZER = Path("data/corpora/babylm_10M_bpe16k.tokenizer.json")
needs_tokenizer = pytest.mark.skipif(
    not TOKENIZER.is_file(), reason="BabyLM tokenizer not trained"
)


def test_time_phrase_renders_both_sheets_exactly() -> None:
    # 18:55 -> "five minutes to seven in the evening" on the minus sheet.
    assert temporal.time_phrase(18 * 60 + 55, -1) == (
        "five minutes to seven in the evening"
    )
    assert temporal.time_phrase(18 * 60 + 55, 1) == (
        "fifty five minutes past six in the evening"
    )
    assert temporal.time_phrase(7 * 60 + 25, 1) == (
        "twenty five minutes past seven in the morning"
    )
    assert temporal.time_phrase(7 * 60 + 25, -1) == (
        "thirty five minutes to eight in the morning"
    )
    # Midnight wrap: 23:55 on the minus sheet anchors to twelve at night.
    assert temporal.time_phrase(23 * 60 + 55, -1) == (
        "five minutes to twelve at night"
    )
    with pytest.raises(ValueError, match="two-sheet domain"):
        temporal.time_phrase(7 * 60, 1)


def test_offset_phrase_covers_zero_halves_and_directions() -> None:
    assert temporal._offset_phrase(0) == "exactly on"
    assert temporal._offset_phrase(60) == "one hour ahead of"
    assert temporal._offset_phrase(-330) == (
        "five hours and thirty minutes behind"
    )
    assert temporal._offset_phrase(270) == (
        "four hours and thirty minutes ahead of"
    )


def test_config_locks_regime_disjointness() -> None:
    config = temporal.TemporalLanguageTaskConfig()
    assert set(config.train_offset_hours).isdisjoint(
        config.composition_offset_hours
    )
    assert all(m % 60 != 0 for m in config.extrapolation_offset_minutes)
    assert config.answer_token_ids == list(range(32_000, 32_016))
    with pytest.raises(ValueError, match="collide"):
        temporal.TemporalLanguageTaskConfig(text_vocab_size=40_000)


def test_wrapped_targets_match_circle_geometry() -> None:
    import numpy as np

    config = temporal.TemporalLanguageTaskConfig()
    phases = np.array([0.0, math.pi / 2, math.pi])
    posteriors = temporal.wrapped_target_posteriors(phases, config)
    assert posteriors.shape == (3, 16)
    assert posteriors.sum(axis=1) == pytest.approx([1.0, 1.0, 1.0])
    assert posteriors[0].argmax() == 0
    assert posteriors[1].argmax() == 4
    assert posteriors[2].argmax() == 8


@needs_tokenizer
def test_fibers_are_exact_two_sheet_pairs() -> None:
    config = temporal.TemporalLanguageTaskConfig()
    tokenizer = temporal.load_tokenizer(config)
    dataset = temporal.generate_paired_temporal_dataset(
        config, tokenizer, sample_count=64, seed=11
    )
    assert dataset.input_ids.shape == (64, config.sequence_length)
    assert bool((dataset.input_ids[:, -1] == temporal.QUERY_ID).all())
    assert bool((dataset.input_ids[:, 0] == temporal.BOS_ID).all())
    assert int(dataset.input_ids.max()) < config.answer_token_start
    for fiber in range(32):
        first, second = 2 * fiber, 2 * fiber + 1
        assert int(dataset.fiber_id[first]) == int(dataset.fiber_id[second])
        assert int(dataset.branch[first]) == 1
        assert int(dataset.branch[second]) == -1
        assert torch.equal(
            dataset.target_posteriors[first], dataset.target_posteriors[second]
        )
        assert dataset.texts[first] != dataset.texts[second]
        assert int(dataset.offset_minutes_true[first]) == int(
            dataset.offset_minutes_true[second]
        )


@needs_tokenizer
def test_generation_is_deterministic_per_seed() -> None:
    config = temporal.TemporalLanguageTaskConfig()
    tokenizer = temporal.load_tokenizer(config)
    first = temporal.generate_paired_temporal_dataset(
        config, tokenizer, sample_count=32, seed=5
    )
    second = temporal.generate_paired_temporal_dataset(
        config, tokenizer, sample_count=32, seed=5
    )
    third = temporal.generate_paired_temporal_dataset(
        config, tokenizer, sample_count=32, seed=6
    )
    assert temporal.dataset_digest(first) == temporal.dataset_digest(second)
    assert temporal.dataset_digest(first) != temporal.dataset_digest(third)


@needs_tokenizer
def test_uncalibrated_mode_is_unidentifiable_by_construction() -> None:
    config = temporal.TemporalLanguageTaskConfig()
    tokenizer = temporal.load_tokenizer(config)
    text = temporal.render_example(
        local_minute=9 * 60 + 20,
        branch=1,
        stated_offset_minutes=0,
        template_id=0,
        person="ada",
        event="meeting",
        filler_id=0,
        calibration_template_id=0,
        calibration_first=False,
        mode="uncalibrated",
    )
    # The same uncalibrated text is consistent with different UTC targets under
    # different true offsets, so the target cannot be a function of the text.
    assert "coordinated universal time" not in text
    ids = tokenizer.encode(text).ids
    assert len(ids) < config.sequence_length - 2


@needs_tokenizer
def test_calibration_clause_and_oracle_modes() -> None:
    config = temporal.TemporalLanguageTaskConfig()
    tokenizer = temporal.load_tokenizer(config)
    calibrated = temporal.generate_paired_temporal_dataset(
        config, tokenizer, sample_count=32, seed=3, mode="calibrated"
    )
    oracle = temporal.generate_paired_temporal_dataset(
        config, tokenizer, sample_count=32, seed=3, mode="utc_oracle"
    )
    assert all("coordinated universal time" in text for text in calibrated.texts)
    assert all("coordinated universal time" not in text for text in oracle.texts)
    assert bool((oracle.offset_minutes_true == 0).all())
    # Oracle targets follow the local time directly.
    local = oracle.nuisance["local_minute"].numpy()
    expected = (local % temporal.MINUTES_PER_DAY).astype(float)
    phases = 2.0 * math.pi * expected / temporal.MINUTES_PER_DAY
    assert phases == pytest.approx(oracle.phases.numpy())


@needs_tokenizer
def test_regime_pools_are_disjoint_in_generated_data() -> None:
    config = temporal.TemporalLanguageTaskConfig()
    tokenizer = temporal.load_tokenizer(config)
    train = temporal.generate_paired_temporal_dataset(
        config, tokenizer, sample_count=128, seed=1, regime="train"
    )
    composition = temporal.generate_paired_temporal_dataset(
        config, tokenizer, sample_count=128, seed=2, regime="composition"
    )
    extrapolation = temporal.generate_paired_temporal_dataset(
        config, tokenizer, sample_count=128, seed=3, regime="extrapolation"
    )
    train_offsets = set(train.offset_minutes_true.tolist())
    assert train_offsets.isdisjoint(set(composition.offset_minutes_true.tolist()))
    assert all(m % 60 == 0 for m in train_offsets)
    assert all(
        m % 60 != 0 for m in set(extrapolation.offset_minutes_true.tolist())
    )
    train_templates = set(train.nuisance["template_id"].tolist())
    composition_templates = set(composition.nuisance["template_id"].tolist())
    assert train_templates.isdisjoint(composition_templates)


@needs_tokenizer
def test_stated_offset_noise_perturbs_only_the_clause() -> None:
    config = temporal.TemporalLanguageTaskConfig()
    tokenizer = temporal.load_tokenizer(config)
    clean = temporal.generate_paired_temporal_dataset(
        config, tokenizer, sample_count=64, seed=9
    )
    noisy = temporal.generate_paired_temporal_dataset(
        config, tokenizer, sample_count=64, seed=9, stated_offset_noise_minutes=30.0
    )
    assert torch.equal(clean.offset_minutes_true, noisy.offset_minutes_true)
    assert torch.equal(clean.target_posteriors, noisy.target_posteriors)
    assert not torch.equal(clean.offset_minutes_stated, noisy.offset_minutes_stated)
    # Common random numbers: every nuisance draw is identical across the pair.
    for name, values in clean.nuisance.items():
        assert torch.equal(values, noisy.nuisance[name]), name
    assert temporal._number_word(25) == "twenty five"
    assert temporal._number_word(41) == "forty one"
