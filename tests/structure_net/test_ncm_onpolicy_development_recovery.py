from types import SimpleNamespace

import pytest

from experiments.structure_net import ncm_onpolicy_development_recovery as recovery
from experiments.structure_net import ncm_source_readout_grade as first


@pytest.mark.parametrize('unrelated', [False, True])
def test_sibling_guard_uses_installed_rubric_and_still_rejects_unrelated_firsts(monkeypatch, unrelated):
    profile = recovery.qualified.qualification.profile
    q = first.q
    for module in (q,q.original):
        monkeypatch.setattr(module,'RUBRIC',profile.RUBRIC.replace(profile.NEW_PHRASE,profile.OLD_PHRASE))
        monkeypatch.setattr(module,'JUDGES',dict(module.JUDGES))
    packet = {'case_id':'fixed_case','candidates':[{'candidate_id':'summary','text':'unchanged output'}]}
    expected = recovery.qualified.archived_key(packet,'secondary',recovery.qualified.SECONDARY)
    assert first.packet_key(packet,'secondary') != expected
    inherited = {'historical':{'first':'preserved'}}
    current = {**inherited,expected:{'first':'new_control'}}
    if unrelated:
        current['unrelated'] = {'first':'not_this_packet'}
    observed = []
    def configure_original(arm, *, preflight=False):
        observed.append((arm,preflight))
        allowed = {first.packet_key(packet,'secondary')}
        recovery.preserve_firsts(inherited,current,allowed_new=allowed)
        assert q.RUBRIC == profile.RUBRIC
        assert q.JUDGES['primary'] == recovery.qualified.qualification.MODEL
        return object(),SimpleNamespace(freeze=lambda root,stage,extra:extra),object()
    monkeypatch.setattr(recovery.original,'configure',configure_original)
    monkeypatch.setattr(recovery.original,'ROOTS',dict(recovery.original.ROOTS))
    if unrelated:
        with pytest.raises(ValueError,match='outside'):
            recovery.configure('residual_targets')
    else:
        _,evaluation,_ = recovery.configure('residual_targets')
        assert evaluation.freeze(None,'inference_pilot',['existing']) == ['existing',*recovery.INPUTS]
    assert observed == [('residual_targets',False)]


def test_only_residual_execution_roots_change():
    assert all(str(p).endswith('_v1') for p in recovery.ROOTS['fixed_targets'].values())
    assert all(str(p).endswith('_v2') for p in recovery.ROOTS['residual_targets'].values())
    assert len({p for roots in recovery.ROOTS.values() for p in roots.values()}) == 8


def test_recovered_finalizer_binds_added_inputs_and_keeps_original_replay(monkeypatch):
    from experiments.neural_architecture_lab import finalize_ncm_onpolicy_residual_recovery as finalize
    calls = []
    def replay(arm):
        calls.append(arm)
        return 'result','scores','cache',['original_input'],'training','evaluation'
    monkeypatch.setattr(finalize,'_REPLAY',replay)
    result = finalize.replay_arm('residual_targets')
    assert calls == ['residual_targets']
    assert result[:3] == ('result','scores','cache')
    assert result[3][0] == 'original_input'
    assert finalize.REGISTRAR in result[3]
    assert set(recovery.INPUTS) <= set(result[3])
    assert result[4:] == ('training','evaluation')
