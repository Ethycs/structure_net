import torch

from experiments.structure_net.ncm_frozen_source_head import FrozenSourceConceptHead
from experiments.structure_net.ncm_prefix_geometry import set_source_frame
from experiments.structure_net.ncm_word_final_head import WordFinalConceptHead
from experiments.structure_net.ncm_llama import vendor_classes
from experiments.structure_net.ncm_blockwise_codec import BlockwiseCodec


def setup():
    OmegaConf,VQVAE,Generator=vendor_classes()
    q=OmegaConf.create({'clss':'multiscale_residual_vector_quantizer','decay':.99,
        'learned_l1_sampling':True,'learned_l1_cascade':True,
        'levels':{'use_manual_levels':True,'manual_levels':[1,2,4,8]}})
    codec=VQVAE(in_vocab_size=32,context_length=8,embed_dim=16,vqvae_vocab_size=[16]*4,
        n_layers=4,compression_factor=1,dropout=0.,pre_quant_groupnorm=4,
        vector_quantizer_config=q).eval().requires_grad_(False)
    facade=BlockwiseCodec(codec,4)
    cfg=OmegaConf.create({'n_layer':2,'n_head':4,'n_embd':16,'bias':True,'dropout':0.,
        'prefix_len':8,'prefix_pad_token_id':31,'vocab_size':[16]*4,'raw_token_prefix':True,
        'query_level0':True,'predict_level0':False,'attn_pattern':'block_diagonal',
        'use_positional_encoding':True,'use_level_encoding':True,'use_rope':True,
        'rope_base':10000,'use_qk_norm':True,'use_flex_attention':False,'tie_heads_to_codebook':True})
    return codec,facade,Generator(facade,cfg)


def test_zero_parent_residual_preserves_raw_prefix_and_receives_gradients():
    torch.manual_seed(111)
    codec,facade,generator=setup()
    head=FrozenSourceConceptHead(generator,24,residual_prefix=True).eval()
    features=torch.randn(2,8,24,requires_grad=True);mask=torch.ones(2,8,dtype=torch.long)
    ids=torch.randint(0,30,(2,8));mask[0,-2:]=0
    latent=head.source_latents(features,mask,ids)
    expected=codec.encoder.wte(ids).masked_fill(~mask.bool()[...,None],0)
    assert torch.equal(latent,expected)
    labels=facade.encode(torch.randint(0,30,(2,32)))
    torch.nn.functional.cross_entropy(head(features,mask,labels,ids).reshape(-1,16),labels.flatten()).backward()
    assert head.source_projection.weight.grad.abs().sum()>0 and features.grad is None
    assert all(p.grad is None for p in codec.parameters())


def test_right_padding_geometry_preserves_teacher_and_source_only_outputs():
    torch.manual_seed(113)
    _,facade,generator=setup()
    head=WordFinalConceptHead(FrozenSourceConceptHead(generator,24,residual_prefix=True)).eval()
    features=torch.randn(2,8,24);ids=torch.randint(0,30,(2,8));mask=torch.ones(2,8,dtype=torch.long);mask[:,-2:]=0
    labels=facade.encode(torch.randint(0,30,(2,32)))
    weights={k:v.clone() for k,v in head.named_parameters()}
    with torch.no_grad():
        before=head(features,mask,labels,ids)
        generated=head.generate(features,mask,ids)
        set_source_frame(generator,12,maximum=16)
        after=head(torch.nn.functional.pad(features,(0,0,0,4)),torch.nn.functional.pad(mask,(0,4)),labels,
            torch.nn.functional.pad(ids,(0,4),value=31))
        expanded=head.generate(torch.nn.functional.pad(features,(0,0,0,4)),torch.nn.functional.pad(mask,(0,4)),
            torch.nn.functional.pad(ids,(0,4),value=31))
    torch.testing.assert_close(before['word_logits'],after['word_logits'],atol=2e-6,rtol=2e-5)
    assert torch.equal(generated['raw_word_ids'],expanded['raw_word_ids'])
    assert all(torch.equal(a,b) for a,b in zip(generated['coarse_codes'],expanded['coarse_codes']))
    assert all(torch.equal(v,weights[k]) for k,v in head.named_parameters())
