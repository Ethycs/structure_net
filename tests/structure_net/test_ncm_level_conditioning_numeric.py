"""Independent numeric path preserves source boundaries and the strict contract."""
from experiments.structure_net import ncm_level_conditioning_numeric as run
from experiments.structure_net import ncm_level_conditioning_development as pending
import torch


def test_independent_contract_keeps_same_model_checks_without_mutating_pending_DEV():
    state=run.q.read(run.training.TRAIN/'result.json')
    saved=torch.load(run.training.TRAIN/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    run.checkpoint.checkpoint_metadata(state,saved)
    pending.checkpoint_metadata(state,saved)
    assert run.checkpoint.training is pending.training
    assert pending.SEMANTIC_LINEAGE_READY is False
    assert run.WORKFLOW != pending.WORKFLOW
    assert 'lineage' not in run.checkpoint.__dict__


def test_all128_target_free_sources_and_complete_train_disjointness():
    run.native.vendor.corpus.quality.dependencies()
    cases,manifest,proof=run.inputs()
    assert [len(cases[c]) for c in run.source.COHORTS]==[64,64]
    assert len(manifest)==128 and proof['group_id']['overlap']==0
    assert all(not (run.source.FORBIDDEN & row.keys()) for c in run.source.COHORTS for row in cases[c])
