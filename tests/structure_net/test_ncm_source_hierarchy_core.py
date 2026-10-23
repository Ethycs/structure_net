import torch

from experiments.structure_net.ncm_blockwise_codec import BlockwiseCodec
from experiments.structure_net.ncm_frozen_source_head import FrozenSourceConceptHead
from experiments.structure_net.ncm_llama import vendor_classes
from experiments.structure_net.ncm_word_final_head import WordFinalConceptHead
from experiments.structure_net.ncm_source_hierarchy_core import concept_scope,concept_logits,concept_loss,uniform_history


def test_concept_only_forward_matches_released_masks_and_cannot_use_finest_labels():
    torch.manual_seed(818)
    OmegaConf,VQVAE,Generator=vendor_classes()
    quant=OmegaConf.create({'clss':'multiscale_residual_vector_quantizer','decay':.99,
        'learned_l1_sampling':True,'learned_l1_cascade':True,
        'levels':{'use_manual_levels':True,'manual_levels':[1,2,4,8]}})
    codec=VQVAE(in_vocab_size=32,context_length=8,embed_dim=16,vqvae_vocab_size=[16]*4,
        n_layers=4,compression_factor=1,dropout=0.,pre_quant_groupnorm=4,
        vector_quantizer_config=quant).eval().requires_grad_(False)
    facade=BlockwiseCodec(codec,4)
    cfg=OmegaConf.create({'n_layer':2,'n_head':4,'n_embd':16,'bias':True,'dropout':0.,
        'prefix_len':8,'prefix_pad_token_id':31,'vocab_size':[16]*4,'raw_token_prefix':True,
        'query_level0':True,'predict_level0':False,'attn_pattern':'block_diagonal',
        'use_positional_encoding':True,'use_level_encoding':True,'use_rope':True,
        'rope_base':10000,'use_qk_norm':True,'use_flex_attention':False,'tie_heads_to_codebook':True})
    net=WordFinalConceptHead(FrozenSourceConceptHead(Generator(facade,cfg),24,residual_prefix=True)).eval()
    concept_scope(net)
    features=torch.randn(2,8,24,requires_grad=True);mask=torch.ones(2,8,dtype=torch.long);mask[0,-2:]=0
    ids=torch.randint(0,30,(2,8));words=torch.randint(0,30,(2,32));words[0,16:]=31
    targets=facade.encode(words)
    actual=concept_logits(net,features,mask,targets,ids)
    expected=net(features,mask,targets,ids)['coarse_logits']
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    altered=targets.clone();altered[:,-32:]=(altered[:,-32:]+1)%16
    for a,b in zip(actual,concept_logits(net,features,mask,altered,ids)):assert torch.equal(a,b)
    value,_=concept_loss(actual,targets,words,net.generator.levels,pad=31)
    value.backward()
    assert features.grad is None and all(p.grad is None for p in codec.parameters())
    assert net.source_head.source_projection.weight.grad.abs().sum()>0
    assert all(h.proj.weight.grad.abs().sum()>0 for h in net.generator.lm_heads[:-1])
    assert all(p.grad is None for p in net.word_projection.parameters())
    assert all(p.grad is None for p in net.word_decoder.parameters())


def test_uniform_corruption_is_replayable_and_never_changes_finest_labels():
    levels=[4,8,16,32]
    codes=torch.arange(sum(levels))[None]%16;original=codes.clone()
    a=uniform_history(codes,levels,torch.Generator().manual_seed(931),maximum=1.,vocab=16)
    b=uniform_history(codes,levels,torch.Generator().manual_seed(931),maximum=1.,vocab=16)
    assert torch.equal(a,b) and torch.equal(codes,original)
    assert torch.equal(a[:,-32:],codes[:,-32:]) and not torch.equal(a,codes)
