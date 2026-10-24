import hashlib

import numpy as np
import pytest
import torch

from experiments.structure_net import ncm_fp32_feature_codec as codec


@pytest.mark.parametrize('layout', codec.LAYOUTS)
def test_every_float_bit_pattern_preserved_including_low_bits_and_signed_zero(layout):
    bits = np.arange(2 * 3072, dtype=np.uint32) * np.uint32(1234567)
    bits[:6] = np.array([0, 0x80000000, 0x7fc00001, 0x7f800000, 0xff800000, 0x3f800001], dtype=np.uint32)
    features = torch.from_numpy(bits.view(np.float32).reshape(2, 3072).copy())
    header, payload = codec.encode(features, layout)
    restored = codec.decode(header, payload)
    assert torch.equal(features.view(torch.uint8), restored.view(torch.uint8))
    assert header['raw_sha256'] == hashlib.sha256(bits.tobytes()).hexdigest()


def test_incomplete_payload_wrong_shape_and_changed_bytes_rejected():
    features = torch.arange(3072, dtype=torch.float32).reshape(1, 3072)
    header, payload = codec.encode(features)
    for bad in (payload[:-1], payload + b'extra', b'invalid'):
        with pytest.raises(ValueError):
            codec.decode(header, bad)
    with pytest.raises(ValueError):
        codec.decode({**header, 'shape': [2, 3072]}, payload)
    with pytest.raises(ValueError):
        codec.decode({**header, 'raw_sha256': '0' * 64}, payload)
    with pytest.raises(ValueError):
        codec.encode(features.half())
