"""Conservative registration of a verified partial model and boundary diagnosis."""
from copy import deepcopy
import json

import pytest


def rows(correct):
    families=('arc','boolq','hellaswag','winogrande');result=[]
    for i in range(64):
        family=families[i//16];count=2 if family in ('boolq','winogrande') else 4
        answer=i%count;prediction=answer if i<correct else (answer+1)%count
        logits=[30.]*count;logits[prediction]=31.
        result.append(dict(identity=f'{family}/{i%16}',family=family,answer=answer,prediction=prediction,logits=logits))
    return result


def metrics(correct,ce=1.,kl=.1):
    return dict(correct=correct,count=64,accuracy=correct/64,kl=kl,ce=ce,loss=2.8*kl+.3*ce)


def storage():
    return dict(cut=25,compressed_layers=11,packed_suffix_and_norm_bytes=981628848,
        frozen_native_prefix_bytes=3853849600,tied_embedding_readout_bytes=622329856,rotary_bytes=256,
        full_model_tensor_bytes=5457808560,native_full_tensor_bytes=6171877632,
        full_model_reduction_fraction=1-5457808560/6171877632)

from neural_architecture_lab.jspace_native_seam_record import (
    NativeBudgetFailureRecord as NativeRecord, SeamDiagnosticRecord as SeamRecord,
    compare_rows, record_for, NATIVE_ID, SEAM_ID,
)


def full_metrics(correct,ce=1.,kl=.1):
    value=metrics(correct,ce,kl);value['original_kl']=.2
    value['by_family']={family:dict(correct=sum(r['prediction']==r['answer'] for r in rows(correct) if r['family']==family),count=16)
        for family in ('arc','boolq','hellaswag','winogrande')}
    return value


def native_evidence():
    reports=[];history=[]
    for epoch in range(13):
        count=45 if epoch==0 else 46 if epoch==10 else 44
        measured=full_metrics(count,ce=.8 if epoch==10 else 1.)
        reports.append(dict(metrics=measured,records=rows(count)))
        history.append(dict(epoch=epoch,metrics=measured,checkpoint_sha256='checkpoint' if epoch==10 else f'epoch{epoch}'))
    gate=dict(passed=True,minimum_correct=45,count=64,maximum_drop=.05,
        native=dict(correct=48,minimum_correct=45,passed=True),
        parent=dict(correct=47,minimum_correct=44,passed=True),policy='original_native_only',parent_gate_required=False)
    train=dict(status='completed',systems_only=False,confirmed=False,updates=384,completed_epochs=12,
        strict_reload_exact=True,fixed_state_unchanged=True,history=history,selected_epoch=10,
        selected=reports[10]['metrics'],initial=reports[0]['metrics'],selected_checkpoint_sha256='checkpoint',
        quality=gate,quality_pass=True)
    fresh=dict(status='completed',passed=True,confirmed=False,fixed_state_unchanged=True,
        selected_checkpoint_sha256='checkpoint',metrics=reports[10]['metrics'],records=reports[10]['records'],
        comparison=compare_rows(reports[10]['records'],reports[10]['records']),quality=gate,storage=storage(),compressed_blocks=11)
    return dict(plan=dict(schema='jspace.qwen3b-native-budget.v1',study_id=NATIVE_ID,confirmed=False,
            seed_cut=26,native_correct=48,fingerprint='root'),
        stage_plan=dict(schema='jspace.qwen3b-native-budget.v1',cut=25,parent_cut=26,
            parent_correct=47,native_correct=48,root_fingerprint='root',layers=list(range(25,36))),
        phases=dict(capture={'status':'completed'},fit={'status':'completed'},pilot={'status':'completed'},train=train,verify=fresh),
        epochs=reports,failure=dict(status='failed',failure_kind='execution_failure',error_type='RuntimeError'),
        failed_stage_plan=dict(cut=24,parent_cut=25,root_fingerprint='root',parent_checkpoint_sha256='checkpoint'),
        failed_phase='capture',failed_cut=24,failed_phase_completed=False)


def seam_evidence():
    reference=rows(46)
    for r in reference:
        r['logits']=[30.+(.25 if i==r['prediction'] else 0.) for i in range(len(r['logits']))]
    modes={}
    for name in ('packed_inference','packed_torch','unpacked_eval','unpacked_grad','native_batch4','native_singleton','native_singleton_padded'):
        actual=deepcopy(reference);kl=0.;fullmax=0.
        if name in ('unpacked_eval','unpacked_grad','native_singleton_padded'):
            for row in actual:row['logits']=[v+.125 for v in row['logits']]
            kl=.0002;fullmax=.625 if name!='native_singleton_padded' else .333
        elif name=='native_batch4':
            row=actual[45];row['prediction']=(row['answer']+1)%len(row['logits'])
            row['logits'][row['prediction']]=30.625;kl=.0003;fullmax=1.40625
        modes[name]=dict(compare_rows(actual,reference),records=actual,full_vocab_kl_t2_by_case=[kl]*64,
            full_vocab_kl_t2=kl,full_vocab_max_abs=fullmax,objective=dict(loss=1.8,kl=.4,ce=2.3))
    boundary={}
    for name in ('batch4','singleton','singleton_padded'):
        values=[dict(exact=True,max_abs=0.,mismatch_fraction=0.,relative_rms=0.,answer_relative_rms=0.) for _ in range(64)]
        if name!='singleton':values[0]=dict(exact=False,max_abs=.5,mismatch_fraction=.125,relative_rms=.002,answer_relative_rms=.001)
        boundary[name]=dict(cases=values,exact=name=='singleton',max_abs=0. if name=='singleton' else .5,
            case_mean_relative_rms=sum(x['relative_rms'] for x in values)/64)
    fingerprint='diagnostic';fixed={'sha256':'unchanged','tensor_bytes':1000}
    return dict(plan=dict(schema='jspace.3b-seam-diagnostic.v1',cut=24,parent_cut=25,confirmed=False,
            fingerprint=fingerprint,identities=[x['identity'] for x in reference],parent_correct=46),
        evaluation=dict(status='completed',confirmed=False,fingerprint=fingerprint,passed=True,
            unchanged_state=True,state_before=deepcopy(fixed),state_after=deepcopy(fixed),
            control=compare_rows(reference,reference),modes=modes,boundary=boundary),
        reference={'records':reference},manifest=dict(status='completed',fingerprint=fingerprint))


def test_native_counts_only_verified_eleven_and_preserves_execution_stop():
    record=NativeRecord.from_evidence(native_evidence())
    assert record.study_id==NATIVE_ID and record.primary_metric==11 and record.accepted_cut==25
    assert record.initial['correct']==45 and record.selected['correct']==46 and record.selected_epoch==10
    assert record.classification=='execution_failure_after_verified_partial_recovery'
    assert record.stop==dict(cut=24,phase='capture',failure_kind='execution_failure',scientific_quality_rejection=False)
    assert record.storage['full_model_reduction_fraction']==pytest.approx(.11569721802287347)
    assert not record.confirmed and not record.full36_recovered and not record.speed_measured
    assert record.independent_confirmations==0 and record_for('native',native_evidence())==record


@pytest.mark.parametrize('mutation',[
    lambda e:e['failure'].update(failure_kind='quality_failure'),
    lambda e:e.update(failed_phase_completed=True),
    lambda e:e['failed_stage_plan'].update(parent_checkpoint_sha256='different-parent'),
    lambda e:e['phases']['train'].update(selected_epoch=0),
    lambda e:e['phases']['train'].update(updates=383),
    lambda e:e['phases']['verify'].update(passed=False),
    lambda e:e['phases']['verify'].update(compressed_blocks=36),
    lambda e:e['phases']['verify']['storage'].update(full_model_tensor_bytes=1),
    lambda e:e['plan'].update(confirmed=True),
    lambda e:e['phases']['train'].update(confirmed=True),
    lambda e:e['phases']['train']['quality'].update(parent_gate_required=True),
])
def test_native_rejects_relabeling_incomplete_training_or_corrupt_selection(mutation):
    value=native_evidence();mutation(value)
    with pytest.raises(ValueError):NativeRecord.from_evidence(value)


def test_native_recomputes_retained_predictions_and_exact_fresh_replay():
    value=native_evidence();value['epochs'][0]['records'][0]['prediction']=1
    with pytest.raises(ValueError):NativeRecord.from_evidence(value)
    value=native_evidence();value['phases']['verify']['records']=deepcopy(value['phases']['verify']['records'])
    value['phases']['verify']['records'][0]['logits'][0]+=.125
    # The tolerance could permit this, but the completed run claims an exact reload.
    with pytest.raises(ValueError,match='Fresh exact'):NativeRecord.from_evidence(value)


def test_native_parent_diagnostic_cannot_override_original_native_gate():
    value=native_evidence();value['stage_plan']['parent_correct']=52
    for phase in ('train','verify'):
        value['phases'][phase]['quality']['parent']=dict(correct=52,minimum_correct=49,passed=False)
    assert NativeRecord.from_evidence(value).quality['passed']
    assert not NativeRecord.from_evidence(value).quality['parent']['passed']


def test_seam_distinguishes_exact_modes_from_tolerant_same_answer_modes():
    record=SeamRecord.from_evidence(seam_evidence())
    assert record.study_id==SEAM_ID and record.primary_metric==1.0
    assert record.classification=='singleton_seam_exact_packed_kernels_exact'
    assert record.modes['native_batch4']['correct']==45 and not record.modes['native_batch4']['passed']
    assert record.modes['native_singleton']['correct']==46 and record.boundary['singleton']['exact']
    assert record.modes['packed_torch']['full_vocab_max_abs']==0
    assert record.modes['unpacked_grad']['predictions_exact'] and record.modes['unpacked_grad']['full_vocab_max_abs']==.625
    assert not record.boundary['singleton_padded']['exact']
    assert not record.confirmed and not record.full36_recovered and not record.speed_measured
    assert record_for('seam',seam_evidence())==record


@pytest.mark.parametrize('mutation',[
    lambda e:e['evaluation']['modes']['packed_torch'].update(full_vocab_max_abs=.001),
    lambda e:e['evaluation']['modes']['native_singleton'].update(full_vocab_kl_t2=.01),
    lambda e:e['evaluation']['modes']['native_batch4'].update(passed=True),
    lambda e:e['evaluation']['modes']['unpacked_grad'].update(correct=47),
    lambda e:e['evaluation']['modes']['unpacked_grad'].update(full_vocab_kl_t2=.4),
    lambda e:e['evaluation']['modes']['unpacked_grad'].update(full_vocab_max_abs=float('nan')),
    lambda e:e['evaluation']['boundary']['singleton'].update(exact=False),
    lambda e:e['evaluation']['boundary']['batch4'].update(case_mean_relative_rms=.5),
    lambda e:e['evaluation']['state_after'].update(sha256='changed'),
    lambda e:e['evaluation']['modes'].pop('unpacked_eval'),
    lambda e:e['manifest'].update(fingerprint='other'),
    lambda e:e['plan']['identities'].reverse(),
    lambda e:e['evaluation'].update(confirmed=True),
])
def test_seam_rejects_fabricated_parity_aggregates_state_or_population(mutation):
    value=seam_evidence();mutation(value)
    with pytest.raises(ValueError):SeamRecord.from_evidence(value)


def test_unknown_kind_cannot_enter_either_study():
    with pytest.raises(ValueError,match='Unknown closeout'):record_for('all36',native_evidence())


def finalized_fixture(tmp_path,monkeypatch,kind='native'):
    from experiments.neural_architecture_lab import finalize_jspace_native_seam as final
    root=tmp_path/'run';root.mkdir()
    source=tmp_path/'source.py';source.write_text('immutable source\n')
    artifact=tmp_path/'result.json';artifact.write_text('immutable artifact\n')
    large=tmp_path/'selected.pt';large.write_bytes(b'immutable selected checkpoint')
    report=tmp_path/'report.md';report.write_text('Existing committed report\n')
    archive=root/final.ARCHIVE;archive.mkdir();(archive/'source.py').write_bytes(source.read_bytes())
    value=native_evidence() if kind=='native' else seam_evidence();record=record_for(kind,value)
    value.update(schema='jspace.native-seam-analysis.v1',kind=kind,confirmed=False,record=record.to_dict(),
        artifact_sha256={str(artifact):final.sha(artifact)},
        authenticated_large_artifacts={str(large):{'sha256':final.sha(large),'identity':final.identity(large)}},
        analysis_source_sha256={str(source):final.sha(source)},report_path=str(report),report_sha256=final.sha(report))
    (root/final.ANALYSIS).write_text(json.dumps(value))
    monkeypatch.setattr(final,'ROOT',tmp_path)
    return final,root,value,record,dict(source=source,artifact=artifact,large=large,report=report,archive=archive/'source.py')


@pytest.mark.parametrize('target',['source','artifact','large','report','archive'])
def test_finalizer_rejects_changed_evidence_analysis_source_archive_and_report(tmp_path,monkeypatch,target):
    final,root,value,record,paths=finalized_fixture(tmp_path,monkeypatch)
    assert final.expected(root)[1]==record
    # Change file size too, avoiding timestamp-resolution assumptions for identity checks.
    paths[target].write_bytes(b'changed different-length payload, deliberately tampered')
    with pytest.raises(ValueError):final.expected(root)


def test_finalizer_recomputes_typed_record_instead_of_trusting_saved_claim(tmp_path,monkeypatch):
    final,root,value,record,_=finalized_fixture(tmp_path,monkeypatch)
    value['record']['accepted_compressed_layers']=36
    (root/final.ANALYSIS).write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Typed record differs'):final.expected(root)


@pytest.mark.parametrize('kind,primary,unit',[('native',11.,'layers'),('seam',1.,'fraction')])
def test_metadata_preserves_distinct_metrics_without_confirmation_or_speed(tmp_path,monkeypatch,kind,primary,unit):
    final,root,_,record,_=finalized_fixture(tmp_path,monkeypatch,kind)
    metadata=final.metadata(root,record,kind)
    assert metadata['hypothesis_id']==record.study_id
    assert metadata['primary_metric']==primary and metadata['primary_metric_unit']==unit
    assert metadata['compressed_layer_count']==11 and metadata['correct']==46 and metadata['count']==64
    assert not metadata['confirmed'] and not metadata['full36_recovered'] and not metadata['speed_measured']
    assert not metadata['scientific_quality_rejection'] and metadata['independent_confirmation_count']==0


def test_fresh_readback_requires_exact_experiment_and_single_hypothesis_link(tmp_path,monkeypatch):
    from types import SimpleNamespace
    final,root,value,record,_=finalized_fixture(tmp_path,monkeypatch)
    key='exact-result';metadata=final.metadata(root,record,'native')
    hypothesis=dict(tested=True,confirmed=False,evidence_count=1,confirmation_status=record.classification,
        direct_result_hashes_json=json.dumps([key]))
    class Collection:
        def __init__(self,identity,values):self.identity,self.values=identity,values
        def get(self,ids,include):
            assert ids==[self.identity] and include==['metadatas']
            return dict(ids=[self.identity],metadatas=[self.values])
    db=SimpleNamespace(experiments_collection=Collection(key,metadata),
        hypotheses_collection=Collection(record.study_id,hypothesis))
    monkeypatch.setattr(final,'logger',lambda:db)
    registry=tmp_path/'registry.json';registry.write_text(json.dumps(dict(
        aggregate_sha256=final.sha(root/final.ANALYSIS),report_sha256=value['report_sha256'],storage={'result_hash':key})))
    receipt=final.readback(root,registry)
    assert receipt['readback_verified'] and receipt['result_hash']==key and not receipt['confirmed']
    metadata['full36_recovered']=True
    with pytest.raises(ValueError,match='Exact experiment readback'):final.readback(root,registry)
    metadata['full36_recovered']=False;hypothesis['direct_result_hashes_json']=json.dumps(['other-result'])
    with pytest.raises(ValueError,match='Exact hypothesis readback'):final.readback(root,registry)
    hypothesis['direct_result_hashes_json']=json.dumps([key]);saved=json.loads(registry.read_text())
    saved['report_sha256']='different-report';registry.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='Registry/report mismatch'):final.readback(root,registry)


@pytest.mark.parametrize('mutation,peak',[
    (lambda m:m['allocation'].update({'GPU-0370fdb0-test':12000}),100),
    (lambda m:m['allocator'].update(allocator_limit_bytes=11264*1024**2),100),
    (lambda m:m.update(lease_id=''),100),
    (lambda m:None,11008*1024**2+1),
])
def test_resources_require_the_actual_reserved_card_budget_and_peak(mutation,peak):
    from experiments.neural_architecture_lab.finalize_jspace_native_seam import check_resources
    uuid='GPU-0370fdb0-test'
    manifest=dict(lease_id='lease',allocation={uuid:11264},allocator=dict(physical_uuid=uuid,
        lease_bytes=11264*1024**2,allocator_limit_bytes=11008*1024**2))
    check_resources(manifest,100)
    mutation(manifest)
    with pytest.raises(ValueError,match='Reserved allocation'):check_resources(manifest,peak)
