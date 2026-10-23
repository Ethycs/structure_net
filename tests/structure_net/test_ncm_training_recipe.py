import json

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from experiments.structure_net.ncm_training_recipe import (
    LEGACY, UPDATED, Recipe, corrupt_conditioning, cumulative_latent,
    decoder_loss, generator_loss, hierarchical_loss, sample_cutoff)
from experiments.structure_net.ncm_recipe_control import codec_windows
from experiments.neural_architecture_lab.finalize_ncm_recipe_update import classify
from experiments.structure_net.infer_ncm_recipe_update import paired_codec
from experiments.structure_net.ncm_memory_data import file_hash
from experiments.structure_net.recover_ncm_recipe_test import first_valid_candidate


class ToyQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.levels = [1, 2, 4]
        self.vqvae_vocab_sizes = [3, 4, 5]
        self.embeddings = nn.ModuleList(nn.Embedding(v, 2) for v in self.vqvae_vocab_sizes)

    def _up_to_full(self, x, level, width):
        return F.interpolate(x.transpose(1, 2), size=width, mode="nearest").transpose(1, 2)

    def _apply_quant_resi(self, x, level):
        return x * (level+1)


class ToyCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 2
        self.vq = ToyQuantizer()
        self.decoder = nn.Linear(2, 8)


def codes():
    return torch.tensor([[0, 1, 2, 1, 2, 3, 4]])


def test_real_nested_cutoff_and_full_sum():
    vae = ToyCodec()
    z = codes()
    first = cumulative_latent(vae, z, 1)
    second = cumulative_latent(vae, z, 2)
    full = cumulative_latent(vae, z)
    expected_first = vae.vq._up_to_full(vae.vq.embeddings[0](z[:, :1]), 0, 4)
    expected_second = expected_first + 2*vae.vq._up_to_full(vae.vq.embeddings[1](z[:, 1:3]), 1, 4)
    assert torch.equal(first, expected_first)
    assert torch.equal(second, expected_second)
    assert torch.equal(full, expected_second + 3*vae.vq.embeddings[2](z[:, 3:]))
    assert not torch.equal(first, full)
    assert not first.requires_grad and not second.requires_grad and not full.requires_grad
    assert first.data_ptr() != second.data_ptr()


def test_cutoffs_follow_full_or_strict_prefix_rule():
    rng = torch.Generator().manual_seed(7)
    assert {sample_cutoff(3, 0., rng) for _ in range(20)} == {3}
    assert {sample_cutoff(3, 1., rng) for _ in range(100)} == {1, 2}
    assert {sample_cutoff(3, .5, rng) for _ in range(100)} == {1, 2, 3}
    assert sample_cutoff(1, 1., rng) == 1


def test_corruption_preserves_clean_targets_and_level_vocabularies():
    clean = codes().repeat(64, 1)
    before = clean.clone()
    recipe = Recipe(corruption_first=1., corruption_last=1.)
    corrupted, details = corrupt_conditioning(clean, [1, 2, 4], [3, 4, 5], recipe,
                                            torch.Generator().manual_seed(5))
    assert torch.equal(clean, before) and corrupted.data_ptr() != clean.data_ptr()
    assert details["selected"] == clean.numel() and details["changed"] > 0
    for chunk, vocab in zip(corrupted.split([1, 2, 4], 1), [3, 4, 5]):
        assert chunk.min() >= 0 and chunk.max() < vocab
    assert details["rates"] == [1., 1., 1.]


def test_legacy_has_no_corruption_or_aliasing():
    clean = codes()
    out, info = corrupt_conditioning(clean, [1, 2, 4], [3, 4, 5], LEGACY,
                                    torch.Generator().manual_seed(2))
    assert torch.equal(out, clean) and out.data_ptr() != clean.data_ptr()
    assert info["selected"] == info["changed"] == 0


def test_augmentation_rng_can_resume_exactly():
    rng = torch.Generator().manual_seed(3042)
    state = rng.get_state()
    cutoff = sample_cutoff(3, .5, rng)
    out, info = corrupt_conditioning(codes(), [1, 2, 4], [3, 4, 5], UPDATED, rng)
    rng.set_state(state)
    assert sample_cutoff(3, .5, rng) == cutoff
    again, repeated = corrupt_conditioning(codes(), [1, 2, 4], [3, 4, 5], UPDATED, rng)
    assert torch.equal(out, again) and info == repeated


def test_equal_level_weights_and_legacy_position_weights():
    logits = torch.tensor([[[5., 0.], [5., 0.], [5., 0.]]], requires_grad=True)
    labels = torch.tensor([[0, 1, 1]])
    equal, per_level = hierarchical_loss(logits, labels, [1, 2], 1.)
    legacy, _ = hierarchical_loss(logits, labels, [1, 2], 0.)
    assert float(equal) == pytest.approx(sum(per_level)/2)
    assert float(legacy) == pytest.approx(F.cross_entropy(logits.flatten(0, 1), labels.flatten()).item())
    equal.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_decoder_gradient_does_not_change_codebook():
    vae = ToyCodec()
    loss, cutoff = decoder_loss(vae, codes(), torch.tensor([[0, 1, 2, 50257]]),
                               Recipe(nested_dropout=1.), torch.Generator().manual_seed(4))
    loss.backward()
    assert cutoff in (1, 2) and vae.decoder.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in vae.vq.parameters())


def test_generator_corrupts_inputs_but_supervises_clean_codes(monkeypatch):
    import experiments.structure_net.ncm_frozen_body as external
    vae = ToyCodec()
    clean = codes()
    prediction = torch.randn(1, 7, 5, requires_grad=True)
    seen = {}
    def fake_logits(vae, net, context, mask, conditioning):
        seen["conditioning"] = conditioning.clone()
        return prediction
    monkeypatch.setattr(external, "external_logits", fake_logits)
    recipe = Recipe(corruption_first=1., corruption_last=1.)
    loss, details = generator_loss(vae, object(), clean, recipe, torch.Generator().manual_seed(1),
                                  context=torch.zeros(1, 1, 2), pad_mask=torch.zeros(1, 1, dtype=torch.bool))
    expected, _ = hierarchical_loss(prediction, clean, [1, 2, 4], recipe.level_alpha)
    assert torch.equal(loss, expected)
    assert not torch.equal(seen["conditioning"], clean)
    loss.backward()
    assert prediction.grad is not None and details["corruption"]["selected"] == 7


@pytest.mark.parametrize("kwargs", [{"nested_dropout": -1}, {"corruption_first": 2}, {"level_alpha": 4}])
def test_invalid_recipe_rejected(kwargs):
    with pytest.raises(ValueError):
        Recipe(**kwargs)


def test_invalid_target_code_rejected():
    with pytest.raises(ValueError, match="outside"):
        corrupt_conditioning(torch.tensor([[3, 0, 0, 0, 0, 0, 0]]), [1, 2, 4], [3, 4, 5],
                             UPDATED, torch.Generator())


def test_codec_windows_preserve_all_reference_tokens():
    ids = list(range(113))
    windows = codec_windows(ids)
    assert len(windows) == 3 and all(len(w) == 48 for w in windows)
    assert [token for window in windows for token in window if token != 50257] == ids
    assert windows[-1][17:] == [50257]*31


def test_train_competence_gate_does_not_claim_heldout_success():
    assert classify({"target": {"passes": 32}, "updated": {"passes": 31}}, True) == "recipe_update_train_competence_passed_needs_heldout"
    assert classify({"target": {"passes": 32}, "updated": {"passes": 30}}, True) == "recipe_update_implemented_train_competence_failed"
    assert classify({"target": {"passes": 30}, "updated": {"passes": 32}}, True) == "recipe_update_training_positive_control_failed"
    assert classify({}, False) == "recipe_update_training_judgment_invalid"


def test_inference_requires_the_saved_matching_codec(tmp_path):
    codec = tmp_path / "codec.pt"
    codec.write_bytes(b"original codec")
    state = {"codec_checkpoint": str(codec), "codec_sha256": file_hash(codec),
             "levels": [1, 2], "vocabularies": [4, 4], "frozen_encoder_codebooks_sha256": "bound"}
    assert paired_codec(state) == codec
    codec.write_bytes(b"different codec")
    with pytest.raises(ValueError, match="paired codec"):
        paired_codec(state)
    with pytest.raises(ValueError, match="explicit"):
        paired_codec({})


def test_candidate_recovery_preserves_first_valid_failure(tmp_path):
    packet = {"case_id": "case", "source_lines": [{"line_id": "L0", "text": "A source fact."}],
              "candidates": [{"candidate_id": "ncm", "text": "gibberish"},
                             {"candidate_id": "parent", "text": "A source fact."}]}
    failure = {"candidate_id": "ncm", "faithful": False, "salient": False,
               "readable": False, "claims": [], "reason": "Unreadable."}
    invalid_parent = {"candidate_id": "parent", "faithful": True, "salient": True,
        "readable": True, "claims": [{"claim": "Unsupported.", "verdict": "insufficient evidence", "evidence_ids": []}],
        "reason": "Inconsistent original judgment."}
    def raw(evaluations):
        return {"http_status": 200, "response": {"model": "claude_code/claude-sonnet-4-6",
            "choices": [{"message": {"content": json.dumps({"evaluations": evaluations})}}]}}
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(raw([failure, invalid_parent])))
    later_success = {**failure, "faithful": True, "salient": True, "readable": True,
        "claims": [{"claim": "A source fact.", "verdict": "supported", "evidence_ids": ["L0"]}]}
    second.write_text(json.dumps(raw([later_success, invalid_parent])))
    grade, origin = first_valid_candidate(packet, "ncm", [first, second])
    assert grade["faithful"] is False and origin == str(first)
    assert first_valid_candidate(packet, "parent", [first, second]) == (None, None)
