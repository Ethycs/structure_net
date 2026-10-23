import pytest
import torch
import torch.nn.functional as F

from experiments.structure_net.ncm_broad_learned_codec import Config, rate
from experiments.structure_net.ncm_llama import vendor_classes


def test_codec_schedule_fixed_final_budget():
    cfg = Config()
    assert rate(cfg.warmup) == cfg.max_lr
    assert rate(cfg.updates) == 0.
    assert 0 < rate(1) < rate(cfg.warmup)
    with pytest.raises(ValueError):
        rate(cfg.updates+1)


def test_actual_released_codec_has_encoder_decoder_gradients_and_exact_reload(tmp_path):
    OmegaConf, VQVAE, _ = vendor_classes()
    quant = OmegaConf.create({"clss":"multiscale_residual_vector_quantizer", "decay":.99,
        "commitment_cost":.25, "learned_l1_sampling":True, "learned_l1_cascade":True,
        "quant_resi":{"enabled":True,"ratio":.5,"share_mode":0},
        "aux":{"fine_drop":{"prob":.5,"min_keep":1}},
        "levels":{"use_manual_levels":True,"manual_levels":[1,2,4,8]}})
    def model():
        return VQVAE(in_vocab_size=32,context_length=8,embed_dim=16,vqvae_vocab_size=[16]*4,
            n_layers=4,compression_factor=1,dropout=0.,pre_quant_groupnorm=4,
            vector_quantizer_config=quant)
    torch.manual_seed(12)
    codec=model().train();targets=torch.randint(1,31,(4,8))
    logits,vq,*_=codec(targets)
    loss=F.cross_entropy(logits.flatten(0,1),targets.flatten())+vq
    assert torch.isfinite(loss)
    loss.backward()
    for part in (codec.encoder,codec.decoder):
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in part.parameters())
        assert all(torch.isfinite(p.grad).all() for p in part.parameters() if p.grad is not None)
    assert all(not e.weight.requires_grad and e.weight.grad is None for e in codec.vq.embeddings)
    codec.eval()
    with torch.no_grad():
        codes=codec.encode(targets);before=codec.decode(codes)
    path=tmp_path/"codec.pt"
    torch.save({"model":codec.state_dict(),"fd":codec.vq._fd_step.clone()},path)
    state=torch.load(path,weights_only=True)
    reloaded=model().eval();reloaded.load_state_dict(state["model"],strict=True)
    reloaded.vq._fd_step.copy_(state["fd"])
    with torch.no_grad():
        assert torch.equal(reloaded.encode(targets),codes)
        assert torch.equal(reloaded.decode(codes),before)
    assert torch.equal(reloaded.vq._fd_step,codec.vq._fd_step)
