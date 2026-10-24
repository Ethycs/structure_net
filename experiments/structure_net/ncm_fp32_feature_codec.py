"""Lossless FP32 feature serialization; no quantization or floating-point math."""
import hashlib
import sys
import zlib

import numpy as np
import torch

SCHEMA = 'ncm.source-feature.fp32-zlib.v1'
LAYOUTS = ('interleaved', 'byte_shuffled')


def shape_for(shape):
    if (not isinstance(shape, (list, tuple)) or len(shape) != 2
            or any(type(v) is not int for v in shape) or not 1 <= shape[0] <= 8192 or shape[1] != 3072):
        raise ValueError('complete1to8192source positions and3072feature channels required')
    return tuple(shape)


def encode(features, layout='byte_shuffled'):
    if sys.byteorder != 'little' or features.dtype != torch.float32 or layout not in LAYOUTS:
        raise ValueError('little-endian FP32 features and a declared lossless layout required')
    shape = shape_for(list(features.shape))
    words = features.detach().cpu().contiguous().view(torch.uint8).numpy().reshape(-1, 4)
    raw = words.tobytes()
    packed = words.T.copy().tobytes() if layout == 'byte_shuffled' else raw
    payload = zlib.compress(packed, level=1)
    return {'schema': SCHEMA, 'dtype': 'float32-le', 'shape': list(shape), 'layout': layout,
        'raw_bytes': len(raw), 'raw_sha256': hashlib.sha256(raw).hexdigest(), 'zlib_level': 1}, payload


def decode(header, payload):
    shape = shape_for(header['shape'])
    expected = shape[0] * shape[1] * 4
    if (sys.byteorder != 'little' or header['schema'] != SCHEMA or header['dtype'] != 'float32-le'
            or header['layout'] not in LAYOUTS or header['raw_bytes'] != expected or header['zlib_level'] != 1):
        raise ValueError('exact declared lossless FP32 cache format required')
    try:
        decoder = zlib.decompressobj()
        packed = decoder.decompress(payload, expected + 1)
        if len(packed) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError('complete single lossless payload of exact raw size required')
    except zlib.error as error:
        raise ValueError('invalid lossless feature payload') from error
    if header['layout'] == 'byte_shuffled':
        raw = np.frombuffer(packed, dtype=np.uint8).reshape(4, -1).T.copy().tobytes()
    else:
        raw = packed
    if hashlib.sha256(raw).hexdigest() != header['raw_sha256']:
        raise ValueError('reconstructed FP32 feature bytes differ')
    return torch.from_numpy(np.frombuffer(raw, dtype='<f4').reshape(shape).copy())
