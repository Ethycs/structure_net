from dataclasses import replace
import pytest
import torch

from experiments.structure_net.ncm_llama import (
    Config, digest, generator_input, code_loss, label_ids, text_check,
    training_examples, vendor_classes)


def example():
    return {"required":{"project":"Project Birch 1","owner":"Maya Chen","budget":"4600","deadline":"November 18"},
        "superseded":{"owner":"Alex Stone","budget":"6300","deadline":"November 14"}}


def test_fixed_protocol_and_strict_fingerprints():
    assert Config().generator_steps==1024
    with pytest.raises(ValueError):replace(Config(),seed=43)
    assert digest({"source":"A"})!=digest({"source":"B"})
    with pytest.raises(ValueError):digest({"loss":float('nan')})


def test_fact_check_rejects_provisional_and_missing_facts():
    text="Project Birch 1: Maya Chen owns the launch, budget $4,600, deadline November 18."
    assert text_check(text,example())["diagnostic_pass"]
    assert not text_check(text.replace("$4,600","$6,300"),example())["diagnostic_pass"]
    assert not text_check(text+" Alex Stone was approved.",example())["diagnostic_pass"]
    assert not text_check(text.replace("November 18","November 180"),example())["diagnostic_pass"]


class TinyTextTokenizer:
    eos_token_id=1
    pad_token_id=0
    def encode(self,text,add_special_tokens=False):return list(range(2,len(text)+2))


def test_overlong_targets_cannot_be_silently_truncated():
    t=TinyTextTokenizer()
    assert label_ids('abc',t,4)==[2,3,4,1]
    with pytest.raises(ValueError,match="no truncation"):label_ids('abcd',t,4)


def test_training_pairs_are_disjoint_complete_and_quality_filtered():
    ex=[{"id":str(i),"split":"train","source_sha256":str(i)} for i in range(20)]
    ex.append({"id":"heldout","split":"heldout","source_sha256":"heldout"})
    labels=[{"id":e["id"],"source_sha256":e["source_sha256"],"usable_label":e['id']!='0'} for e in ex]
    kept=training_examples(ex,labels)
    assert len(kept)==19 and all(e['split']=='train' for e,_ in kept)
    with pytest.raises(ValueError,match="unique"):training_examples(ex,labels+[labels[0]])
    labels[-1]['source_sha256']='wrong'
    with pytest.raises(ValueError,match="source identity"):training_examples(ex,labels)
    labels[-1]['source_sha256']='heldout';labels[1]['usable_label']=False
    with pytest.raises(ValueError,match="control_failed"):training_examples(ex,labels)


def tiny_models():
    OmegaConf,VQVAE,NextConceptTransformer=vendor_classes()
    vq=OmegaConf.create({"clss":"multiscale_residual_vector_quantizer",
        "levels":{"use_manual_levels":True,"manual_levels":[2,4,8]}})
    vae=VQVAE(in_vocab_size=64,context_length=8,embed_dim=12,n_layers=1,
        vqvae_vocab_size=[16]*3,vector_quantizer_config=vq,compression_factor=1,dropout=0.).eval().requires_grad_(False)
    cfg=OmegaConf.create({"n_embd":12,"n_layer":1,"n_head":3,"dropout":0.,"bias":True,
        "vocab_size":[16]*3,"attn_pattern":"block_diagonal","query_level0":True,
        "prefix_len":4,"use_rope":True,"use_level_encoding":True,"tie_heads_to_codebook":True})
    return vae,NextConceptTransformer(vae,cfg).eval()


def test_current_and_future_codes_cannot_leak_into_level_predictions():
    torch.manual_seed(11);vae,net=tiny_models()
    prefix=torch.tensor([[2,3,4,5]]);codes=torch.randint(16,(1,14));changed=codes.clone()
    changed[:,2:]=(changed[:,2:]+1)%16
    old=generator_input(vae,net,prefix,codes);new=generator_input(vae,net,prefix,changed)
    assert torch.equal(old[:,:10],new[:,:10])
    assert not torch.equal(old[:,10:],new[:,10:])
    assert torch.equal(net(old)[:,:6],net(new)[:,:6])


def test_code_loss_learns_level0_without_changing_frozen_codec():
    torch.manual_seed(12);vae,net=tiny_models()
    prefix=torch.tensor([[2,3,4,5]]);codes=torch.randint(16,(1,14))
    loss=code_loss(vae,net,prefix,codes);loss.backward()
    assert torch.isfinite(loss) and net.l0_queries.grad.abs().sum()>0
    assert all(p.grad is None for p in vae.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in net.lm_heads.parameters())
