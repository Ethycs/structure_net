import torch

from experiments.structure_net.ncm_frozen_representation_decoder import decoder_logits, freeze_representation, representation_hash
from experiments.structure_net.ncm_llama import vendor_classes


def test_actual_released_decoder_can_train_without_changing_representation():
    torch.manual_seed(81)
    OmegaConf,VQVAE,_=vendor_classes()
    cfg=OmegaConf.create({'clss':'multiscale_residual_vector_quantizer','decay':.99,
        'learned_l1_sampling':True,'learned_l1_cascade':True,'aux':{'fine_drop':{'prob':.5,'min_keep':1}},
        'levels':{'use_manual_levels':True,'manual_levels':[1,2,4,8]}})
    codec=VQVAE(in_vocab_size=32,context_length=8,embed_dim=16,vqvae_vocab_size=[16]*4,
        n_layers=4,compression_factor=1,dropout=.1,pre_quant_groupnorm=4,pre_quant_dropout=.25,
        vector_quantizer_config=cfg).eval()
    words=torch.randint(0,30,(4,8))
    with torch.no_grad():
        codes=codec.encode(words); initial=codec.decode(codes)
    freeze_representation(codec)
    frozen=representation_hash(codec)
    assert torch.equal(decoder_logits(codec,words),initial)
    optimizer=torch.optim.AdamW([p for p in codec.parameters() if p.requires_grad],lr=.01)
    assert all(k.startswith('decoder.') for k,p in codec.named_parameters() if p.requires_grad)
    codec.decoder.train()
    torch.nn.functional.cross_entropy(decoder_logits(codec,words).reshape(-1,32),words.flatten()).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in codec.decoder.parameters())
    assert all(p.grad is None for k,p in codec.named_parameters() if not k.startswith('decoder.'))
    optimizer.step();codec.decoder.eval()
    assert representation_hash(codec)==frozen
    assert torch.equal(codec.encode(words),codes)
    assert not torch.equal(decoder_logits(codec,words),initial)
