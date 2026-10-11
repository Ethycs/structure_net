import copy

import pytest
import torch

from experiments.structure_net import run_llama_lossless_compression as study


def test_fingerprint_and_fixed_screen():
    config = copy.deepcopy(study.CONFIG)
    old = study.fingerprint(config)
    config["level"] += 1
    assert study.fingerprint(config) != old
    assert study.selected("model.layers.14.mlp.down_proj.weight")
    assert not study.selected("model.layers.13.mlp.down_proj.weight")
    assert study.selected("model.embed_tokens.weight")
    value = {"residual": torch.ones(200000), "repairs": torch.ones(4), "indices": torch.arange(4)}
    samples = list(study.screen_slices(value, {"name": "model.layers.2.mlp.up_proj.weight"}))
    assert [name for name, _ in samples] == ["root/repairs", "root/indices"]
    samples = list(study.screen_slices(value, {"name": "model.layers.0.mlp.up_proj.weight"}))
    assert [t.numel() for _, t in samples] == [65536, 65536, 65536, 4, 4]


def records():
    names = ["original", "factors"] + [f"residual_{i:03d}" for i in range(197)]
    return [{"id": name, "group": name if i < 2 else "residuals", "raw_bytes": 1000,
             "archive_bytes": 800, "tensor_count": 1, "encode_seconds": 1., "decode_verify_seconds": 1.,
             "bitwise_equal": True} for i, name in enumerate(names)]


def test_full_grid_and_conservative_gate():
    rows = records()
    assert study.summarize(rows)["classification"] == "lossless_storage_reduction"
    assert study.summarize(rows)["compressed_factors_plus_residuals_bytes"] == 198 * 800
    with pytest.raises(ValueError, match="199"):
        study.summarize(rows[:-1])
    rows[0]["bitwise_equal"] = False
    assert study.summarize(rows)["classification"] == "lossless_roundtrip_failed"
    rows = records()
    rows[0]["archive_bytes"] = 1000
    assert study.summarize(rows)["classification"] == "lossless_no_storage_reduction"


def test_family_counts_repairs_separately():
    assert study.family("model.layers.0.self_attn.q_proj.weight/root/indices") == "rounding_repairs"
    assert study.family("root/matrices/model.layers.0.mlp.up_proj.weight/cores/0") == "mlp"
