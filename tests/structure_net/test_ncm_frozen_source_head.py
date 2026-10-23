import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.structure_net.ncm_blockwise_codec import BlockwiseCodec
from experiments.structure_net.ncm_frozen_source_head import FrozenSourceConceptHead, frozen_parent_features, greedy_concepts, migrate_generator_state
from experiments.structure_net.ncm_llama import vendor_classes


def modules(blocks=4,prefix=8,codec=None):
    OmegaConf,VQVAE,Generator=vendor_classes()
    q=OmegaConf.create({'clss':'multiscale_residual_vector_quantizer','decay':.99,
        'learned_l1_sampling':True,'learned_l1_cascade':True,
        'levels':{'use_manual_levels':True,'manual_levels':[1,2,4,8]}})
    if codec is None:
        codec=VQVAE(in_vocab_size=32,context_length=8,embed_dim=16,vqvae_vocab_size=[16]*4,
            n_layers=4,compression_factor=1,dropout=0.,pre_quant_groupnorm=4,
            vector_quantizer_config=q).eval().requires_grad_(False)
    facade=BlockwiseCodec(codec,blocks)
    cfg=OmegaConf.create({'n_layer':2,'n_head':4,'n_embd':16,'bias':True,'dropout':0.,
        'prefix_len':prefix,'prefix_pad_token_id':31,'vocab_size':[16]*4,'raw_token_prefix':True,
        'query_level0':True,'predict_level0':False,'attn_pattern':'block_diagonal',
        'use_positional_encoding':True,'use_level_encoding':True,'use_rope':True,
        'rope_base':10000,'use_qk_norm':True,'use_flex_attention':False,'tie_heads_to_codebook':True})
    generator=Generator(facade,cfg).eval()
    return codec,facade,generator


def test_supplied_prefix_eager_generation_matches_actual_reference():
    torch.manual_seed(91)
    codec,facade,generator=modules()
    source=torch.randint(0,30,(2,8));source[0,:2]=31
    with torch.no_grad():
        latent=codec.encoder.wte(source)
        expected=generator.generate(None,source,temperature=0.,top_k=1,top_p=1.,use_kv_cache=False)
        actual=greedy_concepts(generator,latent,source.ne(31))
    assert torch.equal(actual,expected)
    assert facade.decode(actual).shape==(2,32,32)


def test_full_parent_features_masking_detachment_and_hierarchy_conditioning():
    torch.manual_seed(93)
    codec,facade,generator=modules()
    parent=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=24,intermediate_size=48,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=32)).eval().requires_grad_(False)
    parent_before={k:v.clone() for k,v in parent.state_dict().items()}
    source=torch.randint(0,30,(2,8));mask=torch.ones(2,8,dtype=torch.long);mask[0,:2]=0
    features=frozen_parent_features(parent,source,mask)
    assert features.shape==(2,8,24) and not features.requires_grad
    head=FrozenSourceConceptHead(generator,24).eval()
    labels=facade.encode(torch.randint(0,30,(2,32)))
    features.requires_grad_(True)
    logits=head(features,mask,labels)
    padded=features.detach().clone();padded[0,:2]=float('nan')
    assert torch.equal(logits,head(padded,mask,labels))
    changed=labels.clone();changed[:,sum(facade.vq.levels[:2]):]=(changed[:,sum(facade.vq.levels[:2]):]+1)%16
    cutoff=sum(facade.vq.levels[:3])
    assert torch.equal(logits[:,:cutoff],head(features,mask,changed)[:,:cutoff])
    torch.nn.functional.cross_entropy(logits.reshape(-1,16),labels.flatten()).backward()
    assert features.grad is None
    assert head.source_projection.weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in generator.parameters() if p.requires_grad)
    assert all(p.grad is None for p in codec.parameters())
    assert all(p.grad is None and torch.equal(p,parent_before[k]) for k,p in parent.named_parameters())
    with torch.no_grad():
        output=head.generate(features,mask)
    assert output.shape==labels.shape
    with pytest.raises(ValueError,match='full configured prefix'):
        head(features[:,:7],mask[:,:7],labels)
    parent.train()
    with pytest.raises(ValueError,match='frozen'):
        frozen_parent_features(parent,source,mask)


def test_weight_migration_reuses_parameters_and_rejects_changed_code_semantics():
    torch.manual_seed(95)
    codec,_,released=modules(blocks=1,prefix=4)
    _,_,expanded=modules(blocks=4,prefix=8,codec=codec)
    saved={k:v.clone() for k,v in released.state_dict().items()}
    geometry={k:expanded.state_dict()[k].clone() for k in ('attn_mask_for_masking','lvl_1L')}
    result=migrate_generator_state(expanded,saved,codec)
    assert result['all_generator_weights_from_release'] and result['independent_blocks']==4
    for name,value in expanded.named_parameters():
        assert torch.equal(value,saved[name].repeat(1,4,1) if name=='l0_queries' else saved[name])
    assert all(torch.equal(expanded.state_dict()[k],v) for k,v in geometry.items())
    other,_,_=modules()
    with pytest.raises(ValueError,match='reference representation'):
        migrate_generator_state(expanded,saved,other)
