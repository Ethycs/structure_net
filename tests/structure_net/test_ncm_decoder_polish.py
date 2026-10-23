import torch

from experiments.structure_net.train_ncm_decoder_polish import frozen_hash


def test_compatibility_hash_protects_encoder_books_and_buffers_but_allows_decoder_updates():
    codec = torch.nn.Module()
    codec.encoder = torch.nn.Linear(3, 4)
    codec.vq = torch.nn.Embedding(5, 4)
    codec.register_buffer("resampling", torch.ones(2))
    codec.decoder = torch.nn.Linear(4, 3)
    initial = frozen_hash(codec)
    with torch.no_grad():
        codec.decoder.weight.add_(1.)
    assert frozen_hash(codec) == initial
    with torch.no_grad():
        codec.vq.weight[0, 0].add_(1.)
    assert frozen_hash(codec) != initial
    after_book = frozen_hash(codec)
    codec.resampling.add_(1.)
    assert frozen_hash(codec) != after_book
