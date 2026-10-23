import pytest
import torch

from experiments.structure_net.ncm_blockwise_codec import BlockwiseCodec,to_block_major
from experiments.structure_net.ncm_llama import vendor_classes


def tiny_codec():
    OmegaConf,VQVAE,_=vendor_classes()
    q=OmegaConf.create({'clss':'multiscale_residual_vector_quantizer','decay':.99,
        'learned_l1_sampling':True,'learned_l1_cascade':True,
        'quant_resi':{'enabled':True,'ratio':.5,'share_mode':0},
        'levels':{'use_manual_levels':True,'manual_levels':[1,2,4,8]}})
    return VQVAE(in_vocab_size=32,context_length=8,embed_dim=16,vqvae_vocab_size=[16]*4,
        n_layers=4,compression_factor=1,dropout=0.,pre_quant_groupnorm=4,
        vector_quantizer_config=q).eval().requires_grad_(False)


def test_actual_codec_blocking_and_sequential_level_inputs_match_exactly():
    torch.manual_seed(19);original=tiny_codec();facade=BlockwiseCodec(original,4)
    words=torch.randint(0,32,(2,32));before={k:v.clone() for k,v in original.state_dict().items()}
    codes=facade.encode(words);folded=to_block_major(codes,(1,2,4,8),4)
    assert torch.equal(folded,original.encode(words.reshape(8,8)))
    assert torch.equal(facade.decode(codes),original.decode(folded).reshape(2,32,32))
    inputs=facade.vq.idx_to_ncm_input(codes)
    latent=torch.zeros(2,32,16);position=0;input_position=0
    for i,width in enumerate(facade.vq.levels):
        labels=codes[:,position:position+width];position+=width
        latent,next_input=facade.vq.get_next_autoregressive_input(i,latent,facade.vq.embeddings[i](labels))
        if i+1<len(facade.vq.levels):
            n=facade.vq.levels[i+1]
            assert torch.equal(next_input,inputs[:,input_position:input_position+n])
            input_position+=n
    assert torch.equal(latent,facade.vq.idx_to_z_q(codes))
    assert all(torch.equal(v,before[k]) for k,v in original.state_dict().items())
    with pytest.raises(ValueError):facade.encode(words[:,:31])


def test_conditioning_excludes_current_future_labels_and_decoder_keeps_input_gradients():
    torch.manual_seed(21);original=tiny_codec();facade=BlockwiseCodec(original,4)
    words=torch.randint(0,32,(2,32));codes=facade.encode(words)
    before=facade.vq.idx_to_ncm_input(codes)
    # Changing level2 and every finer label cannot change inputs for levels1/2.
    cut=sum(facade.vq.levels[:2]);changed=codes.clone();changed[:,cut:]=(changed[:,cut:]+1)%16
    after=facade.vq.idx_to_ncm_input(changed)
    assert torch.equal(before[:,:sum(facade.vq.levels[1:3])],after[:,:sum(facade.vq.levels[1:3])])
    assert not torch.equal(before[:,sum(facade.vq.levels[1:3]):],after[:,sum(facade.vq.levels[1:3]):])
    root_changed=codes.clone();root_changed[:,0]=(root_changed[:,0]+1)%16
    root_inputs=facade.vq.idx_to_ncm_input(root_changed)
    for a,b,width in zip(before.split(facade.vq.levels[1:],1),root_inputs.split(facade.vq.levels[1:],1),(2,4,8)):
        assert torch.equal(a[:,width:],b[:,width:])
    facade.decoder.train(True)
    assert not facade.decoder.training and not original.decoder.training
    latent=facade.vq.idx_to_z_q(codes).detach().requires_grad_(True)
    facade.decoder(latent).square().mean().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all() and torch.count_nonzero(latent.grad)
    assert all(p.grad is None for p in original.parameters())


def test_released_generator_accepts_blockwise_codec_and_has_no_label_leakage():
    torch.manual_seed(31)
    OmegaConf,_,Generator=vendor_classes()
    original=tiny_codec();facade=BlockwiseCodec(original,4)
    config=OmegaConf.create({'n_layer':2,'n_head':4,'n_embd':16,'bias':True,'dropout':0.,
        'prefix_len':4,'prefix_pad_token_id':31,'vocab_size':[16]*4,
        'raw_token_prefix':True,'query_level0':True,'predict_level0':False,
        'attn_pattern':'block_diagonal','use_positional_encoding':True,
        'use_level_encoding':True,'use_rope':True,'rope_base':10000,
        'use_qk_norm':True,'use_flex_attention':False,'tie_heads_to_codebook':True})
    generator=Generator(facade,config).eval()
    prefix=torch.randint(0,30,(2,4));prefix[0,-1]=31
    words=torch.randint(0,30,(2,32));labels=facade.encode(words)
    prefix_latent=original.encoder.wte(prefix)
    def logits(codes):
        current=facade.vq.idx_to_ncm_input(codes)
        conditioning=torch.cat([generator.build_l0_query_input(2),current],dim=1)
        return generator(torch.cat([prefix_latent,conditioning],dim=1),prefix_pad_mask=prefix.eq(31))
    clean=logits(labels)
    assert clean.shape==(2,sum(facade.vq.levels),16)
    altered=labels.clone();altered[:,sum(facade.vq.levels[:2]):]=(altered[:,sum(facade.vq.levels[:2]):]+1)%16
    changed=logits(altered)
    # Neither current nor future gold codes enter the first three predictions.
    cutoff=sum(facade.vq.levels[:3])
    assert torch.equal(clean[:,:cutoff],changed[:,:cutoff])
    assert not torch.equal(clean[:,cutoff:],changed[:,cutoff:])
    torch.nn.functional.cross_entropy(clean.reshape(-1,16),labels.reshape(-1)).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in generator.parameters() if p.requires_grad)
    assert all(p.grad is None for p in original.parameters())
    with torch.no_grad():
        predicted=generator.generate(None,prefix,temperature=0.,top_k=1,top_p=1.,use_kv_cache=False)
    assert predicted.shape==labels.shape
    assert facade.decode(predicted).shape==(2,32,32)


def test_codec_decoder_reads_future_latents_within_its_block():
    torch.manual_seed(71);codec=tiny_codec()
    assert type(codec.decoder.middle_block.attn).__module__=='models.vae'
    z=torch.randn(1,8,16);changed=z.clone();changed[:,-1]+=torch.randn(1,16)*5
    with torch.no_grad():
        a=codec.decoder(z);b=codec.decoder(changed)
    assert not torch.equal(a[:,0],b[:,0])
