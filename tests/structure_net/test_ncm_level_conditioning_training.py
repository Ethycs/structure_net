import pytest
import torch

from experiments.structure_net import ncm_level_conditioning_training as run


def test_budget_streams_and_rate_are_fixed_before_training():
    config = run.Config()
    assert config.microbatch == 1 and config.accumulation == 16 and config.updates == 1024
    assert len({config.sampler_seed, config.neighbor_seed, config.dropout_seed}) == 3
    assert run.rate(1) == config.max_lr / config.warmup
    assert run.rate(64) == config.max_lr and run.rate(1024) == config.min_lr
    with pytest.raises(ValueError):
        run.rate(1025)


def test_raw_cache_is_byte_exact_bounded_and_rejects_forged_prompt(tmp_path):
    rows, manifest, expected = [], {}, {}
    for identity in ('a', 'b', 'c'):
        source = 'complete raw feature unit source ' + identity
        row = {'id': identity, 'source': source, 'source_sha256': run.streaming.text_hash(source),
            'alignment_sha256': 'alignment-' + identity, 'native_prompt_sha256': 'prompt-' + identity}
        value = torch.arange(3072, dtype=torch.float32).reshape(1, 3072) / 128
        path = tmp_path / (identity + '.pt')
        torch.save({**row, 'features': value}, path)
        rows.append(row)
        manifest[identity] = {'path': str(path), 'sha256': run.streaming.file_hash(path), 'source_tokens': 1}
        expected[identity] = value
    cache = run.streaming.RawFeatureCache(rows, manifest, max_bytes=2 * 3072 * 4)
    for identity in ('a', 'b', 'a', 'c'):
        assert torch.equal(cache[identity].view(torch.uint8), expected[identity].view(torch.uint8))
    assert list(cache.values) == ['a', 'c'] and cache.metrics()['peak_resident_bytes'] == 2 * 3072 * 4
    corrupted = [{**r, 'native_prompt_sha256': 'forged'} if r['id'] == 'b' else r for r in rows]
    with pytest.raises(ValueError, match='binding'):
        run.streaming.RawFeatureCache(corrupted, manifest)['b']


def test_source_split_still_rejects_session_overlap_with_distinct_case_names():
    def rows(prefix, count):
        return [{k: f'{prefix}-{k}-{i}' for k in ('id', 'source_id', 'group_id', 'source_sha256')} for i in range(count)]
    train, short, long = rows('train', 655), rows('short', 64), rows('long', 64)
    assert run.split_guard(train, {'short_development': short, 'long_diagnostic': long})['id']['overlap'] == 0
    train[0]['source_id'] = long[0]['source_id']
    with pytest.raises(ValueError):
        run.split_guard(train, {'short_development': short, 'long_diagnostic': long})
