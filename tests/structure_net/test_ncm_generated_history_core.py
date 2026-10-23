import torch

from experiments.structure_net.ncm_blockwise_codec import BlockwiseCodec
from experiments.structure_net.ncm_frozen_source_head import FrozenSourceConceptHead
from experiments.structure_net.ncm_generated_history_core import (
    independent_word_branch, own_history, conditioning_from_codes, word_logits, word_loss)
from experiments.structure_net.ncm_llama import vendor_classes
from experiments.structure_net.ncm_word_final_head import WordFinalConceptHead


def model():
    torch.manual_seed(19241)
    OmegaConf, VQVAE, Generator = vendor_classes()
    quant = OmegaConf.create({'clss': 'multiscale_residual_vector_quantizer', 'decay': .99,
        'learned_l1_sampling': True, 'learned_l1_cascade': True,
        'levels': {'use_manual_levels': True, 'manual_levels': [1, 2, 4, 8]}})
    codec = VQVAE(in_vocab_size=32, context_length=8, embed_dim=16, vqvae_vocab_size=[16]*4,
        n_layers=4, compression_factor=1, dropout=0., pre_quant_groupnorm=4,
        vector_quantizer_config=quant).eval().requires_grad_(False)
    facade = BlockwiseCodec(codec, 4)
    cfg = OmegaConf.create({'n_layer': 2, 'n_head': 4, 'n_embd': 16, 'bias': True, 'dropout': 0.,
        'prefix_len': 8, 'prefix_pad_token_id': 31, 'vocab_size': [16]*4, 'raw_token_prefix': True,
        'query_level0': True, 'predict_level0': False, 'attn_pattern': 'block_diagonal',
        'use_positional_encoding': True, 'use_level_encoding': True, 'use_rope': True,
        'rope_base': 10000, 'use_qk_norm': True, 'use_flex_attention': False, 'tie_heads_to_codebook': True})
    forecaster = WordFinalConceptHead(FrozenSourceConceptHead(Generator(facade, cfg), 24,
        residual_prefix=True), word_vocab_limit=31).eval().requires_grad_(False)
    features = torch.randn(2, 8, 24, requires_grad=True)
    mask = torch.ones(2, 8, dtype=torch.long)
    mask[0, -2:] = 0
    ids = torch.randint(0, 30, (2, 8))
    return forecaster, codec, features, mask, ids


def test_source_only_rollout_matches_existing_generation_and_complete_forward():
    forecaster, codec, features, mask, ids = model()
    actual = own_history(forecaster, features, mask, ids)
    expected = forecaster.generate(features, mask, ids)
    assert torch.equal(actual['coarse_codes'], torch.cat(expected['coarse_codes'], dim=1))
    facade = forecaster.generator.vqvae_local[0]
    reconstructed = conditioning_from_codes(facade, actual['coarse_codes'])
    assert torch.equal(actual['word_conditioning'], reconstructed)
    branch = independent_word_branch(forecaster).eval()
    logits = word_logits(branch, features, mask, reconstructed, ids)
    assert torch.equal(logits.argmax(-1), expected['raw_word_ids'])
    unused_finest = torch.randint(0, 16, (2, 32))
    complete_codes = torch.cat((actual['coarse_codes'], unused_finest), dim=1)
    full_logits = branch(features, mask, complete_codes, ids)['word_logits'][..., :31]
    torch.testing.assert_close(logits, full_logits, atol=3e-6, rtol=3e-5)
    changed = torch.cat((actual['coarse_codes'], (unused_finest+1) % 16), dim=1)
    assert torch.equal(full_logits, branch(features, mask, changed, ids)['word_logits'][..., :31])


def test_word_training_changes_separate_branch_without_forecaster_or_codec_gradients():
    forecaster, codec, features, mask, ids = model()
    branch = independent_word_branch(forecaster).train()
    fixed = {k: p.detach().clone() for k, p in forecaster.named_parameters()}
    original_codec = {k: v.detach().clone() for k, v in codec.state_dict().items()}
    history = own_history(forecaster, features, mask, ids)
    target = torch.randint(0, 30, (2, 32))
    target[0, 19:] = 31
    optimizer = torch.optim.AdamW([p for p in branch.parameters() if p.requires_grad], lr=1e-3)
    loss = word_loss(word_logits(branch, features, mask, history['word_conditioning'], ids), target, pad=31)
    loss.backward()
    assert features.grad is None
    assert all(p.grad is None for p in forecaster.parameters())
    assert all(p.grad is None for p in codec.parameters())
    assert branch.source_head.source_projection.weight.grad.abs().sum() > 0
    assert branch.word_projection.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.word_decoder.parameters())
    assert all(p.grad is None for p in branch.generator.lm_heads.parameters())
    assert branch.generator.l0_queries.grad is None
    before = branch.word_projection.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, branch.word_projection.weight)
    assert all(torch.equal(v, dict(forecaster.named_parameters())[k]) for k, v in fixed.items())
    assert all(torch.equal(v, codec.state_dict()[k]) for k, v in original_codec.items())
