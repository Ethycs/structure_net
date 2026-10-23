from dataclasses import replace

import torch

from experiments.structure_net.ncm_word_final_training import Config,neighbor_table,paired_backward,perturb
from experiments.structure_net.ncm_frozen_source_head import FrozenSourceConceptHead
from experiments.structure_net.ncm_word_final_head import WordFinalConceptHead
from experiments.structure_net.ncm_llama import vendor_classes
from experiments.structure_net.ncm_blockwise_codec import BlockwiseCodec


def test_bfloat16_paired_training_keeps_sources_and_frozen_codec_out_of_gradients():
    torch.manual_seed(117)
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
    net=WordFinalConceptHead(FrozenSourceConceptHead(Generator(facade,cfg),24,residual_prefix=True),word_vocab_limit=31).train()
    words=torch.randint(0,31,(1,32));words[:,-2:]=31
    codes=facade.encode(words)
    tables=neighbor_table(facade.vq.embeddings[:-1],k=2)
    first=perturb(codes,facade.vq.levels,tables,torch.Generator().manual_seed(6),probability=1)
    again=perturb(codes,facade.vq.levels,tables,torch.Generator().manual_seed(6),probability=1)
    assert torch.equal(first,again) and torch.equal(first[:,-32:],codes[:,-32:])
    assert (first[:,:-32]!=codes[:,:-32]).all()
    features=torch.randn(1,8,24,requires_grad=True)
    batch={'features':features,'mask':torch.ones(1,8,dtype=torch.long),
        'source_ids':torch.randint(0,30,(1,8)),'codes':codes,'words':words}
    frozen={k:v.clone() for k,v in codec.state_dict().items()}
    values,_=paired_backward(net,batch,tables,torch.Generator().manual_seed(7),1,
        replace(Config(),word_vocab=31,pad=31,probability=.5))
    assert all(torch.isfinite(torch.tensor(v)) for v in values.values())
    assert net.word_projection.weight.grad.abs().sum()>0
    assert net.source_head.source_projection.weight.grad.abs().sum()>0
    assert features.grad is None and all(p.grad is None for p in codec.parameters())
    assert all(torch.equal(v,frozen[k]) for k,v in codec.state_dict().items())
