"""Warm checkpoint scope and sampled frame validity before GPU training."""
import torch

from experiments.structure_net import ncm_expanded_codec_training as run


def test_complete_warm_decoder_and_protected_codec_on_CPU():
    run.corpus.component.configure()
    model = run.load_codec('cpu')
    assert not torch.cuda.is_initialized()
    assert run.native.vendor.module_hash(model.decoder) == run.WARM_DECODER_SHA
    assert all(p.requires_grad == name.startswith('decoder.') for name, p in model.named_parameters())
    assert tuple(model.vq.levels) == tuple(run.native.LEVELS)
    assert sum(p.numel() for p in model.decoder.parameters()) == 18_870_656


def test_mixture_and_augmentation_keep_valid_native_frames_and_replay_rng():
    raw = torch.tensor([[101, 300+i, 400+i, 500+i, 101]+[0]*123 for i in range(4)])
    author = torch.tensor([[101, *([600+i]*126), 101] for i in range(4)])
    index = [{'frame_start': 0, 'frame_count': 2}, {'frame_start': 2, 'frame_count': 2}]
    sampler, augmentation = torch.Generator().manual_seed(137142), torch.Generator().manual_seed(138142)
    sam_state, aug_state = sampler.get_state(), augmentation.get_state()
    first = run.sample(raw, author, index, sampler, augmentation)
    sampler.set_state(sam_state)
    augmentation.set_state(aug_state)
    again = run.sample(raw, author, index, sampler, augmentation)
    assert torch.equal(first[0], again[0]) and first[1] == again[1]
    pools, modes = set(), set()
    for _ in range(64):
        values, descriptors = run.sample(raw, author, index, sampler, augmentation)
        assert values.dtype == torch.long and values.shape == (4, 128)
        for frame, descriptor in zip(values.tolist(), descriptors):
            pools.add(descriptor['kind'])
            modes.add(descriptor['mode'])
            assert frame[0] == 101
            end = frame.index(101, 1)
            assert all(v == 0 for v in frame[end+1:])
            assert descriptor['target_sha256'] == run.q.digest(frame)
    assert pools == {'author_TRAIN', 'raw_source'} and modes == {'full', 'prefix'}
