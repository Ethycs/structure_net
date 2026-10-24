from experiments.structure_net import ncm_lm1b_domain_decoder_eval as run


def test_domain_evaluation_bindings_and_latest_first_cache():
    run.configure()
    assert run.engine.repair.TRAIN == run.domain.TRAIN
    assert run.native.DATA == run.framed.DATA
    assert run.native.PILOT == run.PILOT and run.native.RECON == run.RECON
    assert run.prior.PACKETS == run.PACKETS and run.engine.BASE_PACKETS == run.domain.SEMANTIC_PACKETS
    assert run.engine.BASE_RESULTS == run.domain.PREVIOUS
    assert run.engine.repair.Config().sampler_seed == 84142
    assert run.engine.repair.augment is run.domain.augment
    assert run.engine.base_cache is run.base_cache
    assert run.engine.MODULE == run.MODULE and run.engine.AUDITOR == run.AUDITOR
    assert len(run.base_cache()) == 2266
