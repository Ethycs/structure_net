import torch

from experiments.structure_net.ncm_blockwise_codec import BlockwiseCodec
from experiments.structure_net.ncm_frozen_source_head import FrozenSourceConceptHead
from experiments.structure_net.ncm_llama import vendor_classes
from experiments.structure_net.ncm_word_final_head import WordFinalConceptHead
from experiments.structure_net.ncm_source_root_capacity import root_logits,scope,loss,active_mask


def test_source_only_root_matches_masked_training_and_freezes_word_modules():
    torch.manual_seed(717)
    OmegaConf,VQVAE,Generator=vendor_classes()
    quant=OmegaConf.create({'clss':'multiscale_residual_vector_quantizer','decay':.99,
        'learned_l1_sampling':True,'learned_l1_cascade':True,
        'levels':{'use_manual_levels':True,'manual_levels':[1,2,4,8]}})
    codec=VQVAE(in_vocab_size=32,context_length=8,embed_dim=16,vqvae_vocab_size=[16]*4,
        n_layers=4,compression_factor=1,dropout=0.,pre_quant_groupnorm=4,
        vector_quantizer_config=quant).eval().requires_grad_(False)
    facade=BlockwiseCodec(codec,4)
    config=OmegaConf.create({'n_layer':2,'n_head':4,'n_embd':16,'bias':True,'dropout':0.,
        'prefix_len':8,'prefix_pad_token_id':31,'vocab_size':[16]*4,'raw_token_prefix':True,
        'query_level0':True,'predict_level0':False,'attn_pattern':'block_diagonal',
        'use_positional_encoding':True,'use_level_encoding':True,'use_rope':True,
        'rope_base':10000,'use_qk_norm':True,'use_flex_attention':False,'tie_heads_to_codebook':True})
    net=WordFinalConceptHead(FrozenSourceConceptHead(Generator(facade,config),24,residual_prefix=True))
    scope(net);net.eval()
    features=torch.randn(2,8,24,requires_grad=True)
    mask=torch.ones(2,8,dtype=torch.long);mask[0,-2:]=0
    ids=torch.randint(0,30,(2,8));labels=facade.encode(torch.randint(0,30,(2,32)))
    before={k:v.clone() for k,v in net.named_parameters() if not v.requires_grad}
    logits=root_logits(net,features,mask,ids)
    parallel=net(features,mask,labels,ids)['coarse_logits'][0]
    torch.testing.assert_close(logits,parallel,atol=2e-6,rtol=2e-5)
    value,_=loss(logits,labels[:,:4],torch.ones(2,4,dtype=torch.bool))
    value.backward()
    assert features.grad is None and all(p.grad is None for p in codec.parameters())
    assert net.source_head.source_projection.weight.grad.abs().sum()>0
    assert net.generator.lm_heads[0].proj.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in net.word_projection.parameters())
    assert all(p.grad is None for p in net.word_decoder.parameters())
    assert all(p.grad is None for head in net.generator.lm_heads[1:] for p in head.parameters())
    assert all(torch.equal(v,before[k]) for k,v in net.named_parameters() if not v.requires_grad)


def test_root_active_blocks_include_eos_without_counting_empty_padding():
    words=torch.full((2,192),50257,dtype=torch.long)
    words[0,:20]=17;words[0,20]=50256
    words[1,:48]=19;words[1,48]=50256
    mask=active_mask(words,16)
    assert mask[0].tolist()==[True]*4+[False]*12
    assert mask[1].tolist()==[True]*8+[False]*8
