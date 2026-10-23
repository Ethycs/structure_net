import torch

from experiments.structure_net.ncm_frozen_source_head import FrozenSourceConceptHead
from experiments.structure_net.ncm_word_final_head import WordFinalConceptHead
from experiments.structure_net.ncm_blockwise_codec import BlockwiseCodec
from experiments.structure_net.ncm_llama import vendor_classes


def model():
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
    return WordFinalConceptHead(FrozenSourceConceptHead(Generator(facade,cfg),24)).eval(),codec,facade


def test_word_prediction_has_no_finest_label_or_parent_gradient_leakage():
    torch.manual_seed(101)
    net,codec,facade=model()
    features=torch.randn(2,8,24,requires_grad=True);mask=torch.ones(2,8,dtype=torch.long)
    targets=torch.randint(0,31,(2,32));codes=facade.encode(targets)
    before={k:v.clone() for k,v in codec.state_dict().items()}
    output=net(features,mask,codes)
    assert output['word_logits'].shape==(2,32,32)
    changed=codes.clone();changed[:,-32:]=(changed[:,-32:]+1)%16
    other=net(features,mask,changed)
    assert torch.equal(output['word_logits'],other['word_logits'])
    assert all(torch.equal(a,b) for a,b in zip(output['coarse_logits'],other['coarse_logits']))
    torch.nn.functional.cross_entropy(output['word_logits'].reshape(-1,32),targets.flatten()).backward()
    assert features.grad is None
    assert net.word_projection.weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in net.word_decoder.parameters())
    assert net.source_head.source_projection.weight.grad.abs().sum()>0
    assert all(p.grad is None and torch.equal(p,before[k]) for k,p in codec.named_parameters())


def test_source_only_generation_matches_teacher_computation_at_its_own_coarse_codes():
    torch.manual_seed(103)
    net,_,facade=model()
    features=torch.randn(2,8,24);mask=torch.ones(2,8,dtype=torch.long);mask[0,:2]=0
    with torch.no_grad():
        generated=net.generate(features,mask)
        # The final VQ code is unused; fill it with arbitrary valid dictionary IDs.
        full=torch.cat(generated['coarse_codes']+[torch.zeros(2,32,dtype=torch.long)],dim=1)
        teacher=net(features,mask,full)
    assert len(generated['coarse_codes'])==len(facade.vq.levels)-1
    assert torch.equal(generated['raw_word_ids'],teacher['word_logits'].argmax(-1))
    for predicted,logits in zip(generated['coarse_codes'],teacher['coarse_logits']):
        assert torch.equal(predicted,logits.argmax(-1))
