import json

import numpy as np
import pytest
import torch

from structure_net.components.layers import lossless_tensor_archive as codec


@pytest.mark.parametrize("layout", codec.LAYOUTS)
@pytest.mark.parametrize("width", (1, 2, 4, 8))
def test_all_bit_patterns_and_tail(layout, width):
    rng = np.random.default_rng(109)
    for count in (0, 1, 7, 8, 9, 511, 8193):
        raw = rng.integers(0, 256, size=count * width, dtype=np.uint8).tobytes()
        assert codec.unshuffle(codec.shuffle(raw, width, layout), width, count, layout) == raw
        encoded, compressed = codec.encode_chunk(raw, width, layout)
        assert codec.decode_chunk(encoded, width, count, layout, compressed) == raw


@pytest.mark.parametrize("layout", codec.LAYOUTS)
def test_portable_checkpoint_and_float_payloads(tmp_path, layout):
    # Includes both zeros, infinities and distinct NaN payloads: compare bits.
    raw = np.arange(65536, dtype=np.uint16)
    tensor = torch.from_numpy(raw.view(np.float16).copy())
    value = {"weights": tensor, "alias": tensor, 3: (None, True, "x", [2.25]),
             "empty": torch.empty(0, dtype=torch.int32), "scalar": torch.tensor(-0.)}
    root = tmp_path / "archive"
    result = codec.save_archive(value, root, layout, chunk_bytes=1016)
    assert result["bitwise_equal"]
    restored = codec.load_archive(root)
    assert codec.flatten(restored)[0] == codec.flatten(value)[0]
    assert restored["alias"] is restored["weights"]
    for (_, a), (_, b) in zip(codec.flatten(value)[1], codec.flatten(restored)[1]):
        assert np.array_equal(codec.tensor_bytes(a), codec.tensor_bytes(b))
    assert result["archive_bytes"] == sum(p.stat().st_size for p in root.iterdir())
    payload = root / "payload.bin"
    changed = bytearray(payload.read_bytes()); changed[0] ^= 1; payload.write_bytes(changed)
    with pytest.raises(ValueError, match="hash mismatch"):
        codec.load_archive(root)


def test_decoder_checks_metadata_and_completed_output(tmp_path):
    value = {"x": torch.ones(20, dtype=torch.bfloat16)}
    root = tmp_path / "archive"
    codec.save_archive(value, root, "bit_shuffle")
    assert torch.equal(codec.load_archive(root)["x"].view(torch.int16), value["x"].view(torch.int16))
    with pytest.raises(FileExistsError):
        codec.save_archive(value, root, "bit_shuffle")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["tensors"][0]["source_sha256"] = "0" * 64
    codec.write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="tensor hash"):
        codec.load_archive(root)


def test_strict_json_and_invalid_alignment(tmp_path):
    with pytest.raises(ValueError):
        codec.write_json(tmp_path / "invalid.json", {"bad": float("nan")})
    with pytest.raises(ValueError):
        codec.shuffle(b"a", 2, "raw")


def test_noncontiguous_factor_cores_preserve_logical_bits(tmp_path):
    source = torch.arange(105, dtype=torch.float16).reshape(7, 15).T
    assert not source.is_contiguous()
    root = tmp_path / "factor"
    codec.save_archive({"factor": source}, root, "byte_shuffle")
    restored = codec.load_archive(root)["factor"]
    assert torch.equal(source.view(torch.int16), restored.view(torch.int16))
    assert restored.shape == source.shape
