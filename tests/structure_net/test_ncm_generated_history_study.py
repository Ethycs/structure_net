from copy import deepcopy
import math

import pytest
import torch

from experiments.structure_net import ncm_generated_history_study as study
from experiments.structure_net.ncm_blockwise_codec import BlockwiseCodec
from experiments.structure_net.ncm_frozen_source_head import FrozenSourceConceptHead
from experiments.structure_net.ncm_llama import vendor_classes
from experiments.structure_net.ncm_word_final_head import WordFinalConceptHead


def tiny_pair():
    torch.manual_seed(19242)
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
    branch = study.core.independent_word_branch(forecaster).train()
    batch = {'features': torch.randn(2, 8, 24, requires_grad=True),
        'mask': torch.ones(2, 8, dtype=torch.long), 'source_ids': torch.randint(0, 30, (2, 8)),
        'words': torch.randint(0, 30, (2, 32))}
    batch['words'][0, 19:] = 31
    batch['codes'] = facade.encode(batch['words'])
    own = study.core.own_history(forecaster, batch['features'], batch['mask'], batch['source_ids'])['coarse_codes']
    return forecaster, branch, codec, batch, own


def test_paired_training_runs_both_histories_and_ignores_unused_finest_labels():
    forecaster, branch, codec, batch, own = tiny_pair()
    frozen = study.parameter_hash(forecaster)
    values = study.paired_backward(branch, batch, own, pad=31)
    assert set(values) == {'own_word_CE', 'gold_word_CE'}
    assert all(math.isfinite(v) and v > 0 for v in values.values())
    assert branch.source_head.source_projection.weight.grad.abs().sum() > 0
    assert branch.word_projection.weight.grad.abs().sum() > 0
    assert batch['features'].grad is None and all(p.grad is None for p in forecaster.parameters())
    assert all(p.grad is None for p in codec.parameters())
    assert all(p.grad is None for p in branch.generator.lm_heads.parameters())
    changed = dict(batch)
    changed['codes'] = batch['codes'].clone()
    changed['codes'][:, -32:] = (changed['codes'][:, -32:]+1) % 16
    branch.zero_grad(set_to_none=True)
    assert study.paired_backward(branch, changed, own, pad=31) == values
    assert study.parameter_hash(forecaster) == frozen


def test_cached_history_requires_exact_source_and_complete_unmodified_codes():
    row = {'id': 'case', 'source_sha256': 'source-digest', 'source_frame': 512}
    codes = torch.zeros(1, 608, dtype=torch.long)
    item = {**row, 'coarse_codes': codes[0].tolist(), 'history_sha256': study.base.tensor_digest({'codes': codes})}
    study.validate_history(row, item)
    for key, value in [('source_sha256', 'other'), ('id', 'other'), ('source_frame', 256)]:
        with pytest.raises(ValueError, match='content or identity'):
            study.validate_history(row, {**item, key: value})
    changed = deepcopy(item)
    changed['coarse_codes'][0] = 1
    with pytest.raises(ValueError, match='content or identity'):
        study.validate_history(row, changed)
    with pytest.raises(ValueError, match='target or candidate'):
        study.source_batch({'word_ids': [1]}, {})


def test_fixed_word_schedule_includes_endpoints_and_rejects_extensions():
    config = study.Config()
    assert study.rate(config.warmup) == config.max_lr
    assert study.rate(config.updates) == config.min_lr
    for step in (0, config.updates+1):
        with pytest.raises(ValueError, match='fixed word budget'):
            study.rate(step)
