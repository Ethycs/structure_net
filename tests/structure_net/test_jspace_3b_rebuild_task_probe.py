"""Task-only sidecar keeps authenticated weights and prior decisions intact."""
from copy import deepcopy
import json

import pytest

from experiments.structure_net import jspace_3b_rebuild_task_probe as study


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))


def stage(tmp_path,monkeypatch):
    root=tmp_path/'stage';run=root/'run';label='update_0046'
    directory=run/'evaluations'/label;export=directory/'export'
    checkpoint=run/'checkpoints/update_0046.pt';checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'fixture checkpoint')
    source={'fingerprint':'source','model':{'revision':'native'}}
    put(root/'plan.json',source)
    export.mkdir(parents=True);(export/'weights.safetensors').write_bytes(b'packed fixture weights')
    put(export/'structure.json',{'schema':study.packed.SCHEMA})
    receipt=dict(schema=study.packed.SCHEMA,packed=True,packed_state_sha256='packed_state',
        files={name:study.sha(export/name) for name in ('structure.json','weights.safetensors')},
        identities=dict(study='source',checkpoint_sha256=study.sha(checkpoint),teacher='native'))
    put(export/'manifest.json',receipt)
    enriched=dict(receipt,manifest_sha256=study.sha(export/'manifest.json'))
    candidate=dict(label=label,packed_artifact_path=str(export),checkpoint={'path':str(checkpoint),'sha256':study.sha(checkpoint)},
        packed_export=enriched,packed_reload_passed=True,eligibility={'passed':False},nll=3.)
    put(directory/'result.json',candidate)
    for name in ('development.json','multiple_choice.json','export_receipt.json','reload_receipt.json'):
        put(directory/name,{})
    put(directory/'reload_parity.json',{'passed':True})
    result=dict(status='completed',fingerprint='source',candidates=[candidate],relative_retention_pass=False)
    put(run/'result.json',result)
    manifest=dict(status='completed',fingerprint='source',
        artifacts={str(p.relative_to(run)):study.sha(p) for p in run.rglob('*.json')},
        checkpoints={str(checkpoint):study.sha(checkpoint)},packed_exports={label:enriched['manifest_sha256']})
    put(run/'manifest.json',manifest)
    monkeypatch.setattr(study.recovery,'verify',lambda *a,**k:(source,{},[]))
    return root,label,export


def test_existing_failed_candidate_can_be_probed_without_mutating_or_repacking(tmp_path,monkeypatch):
    root,label,export=stage(tmp_path,monkeypatch)
    before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    _,_,_,candidate,inputs=study.admit_export(root,label)
    assert not candidate['eligibility']['passed']
    assert candidate['packed_artifact_path']==str(export)
    assert inputs[str(export/'weights.safetensors')]==study.sha(export/'weights.safetensors')
    assert before=={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_changed_export_payload_is_rejected_even_with_completed_stage(tmp_path,monkeypatch):
    root,label,export=stage(tmp_path,monkeypatch)
    (export/'weights.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Packed payload'):study.admit_export(root,label)
    with pytest.raises(ValueError,match='label'):study.admit_export(root,'../../outside')


def controls():
    canary=[dict(id=str(i),input_ids=[1,3],max_new_tokens=128,eos_token_ids=[151645,151643],
                  generation_success=i<2) for i in range(6)]
    corpus=[dict(example_id=f'case{i}') for i in range(16)]
    native={'canaries':{'records':canary},'corpus':corpus};actual=deepcopy(native)
    judgments=[dict(example_id=f'case{i}',status='completed',
        native={'task_success':True,'generation_success':True},
        student={'task_success':True,'generation_success':True}) for i in range(16)]
    semantic={'judgments':judgments,'judge_provenance':dict(judge_id='fixed',judge_version='fixed',
        rubric='pinned rubric',prompt_sha256='prompt',settings={'temperature':0.})}
    executable={'passed':True,'available':True,'gates':{'mbpp':{'passed':True},'gsm8k':{'passed':True}}}
    return native,actual,semantic,executable


def test_task_probe_retention_is_independent_of_prior_nll_gate_but_cannot_promote():
    native,actual,semantic,executable=controls()
    result=study.quality_summary(native,actual,semantic,executable)
    assert result['diagnostic_generation_retention_pass'] is True
    assert result['canary_gate']['student_success_rate']==2/6
    assert result['original_stage_eligibility_unchanged']
    assert not result['automatic_promotion'] and not result['confirmed']


def test_missing_transport_verdict_is_unavailable_not_zero_accuracy():
    native,actual,semantic,executable=controls()
    semantic['judgments'][0]={'example_id':'case0','status':'http_error','http_status':429}
    result=study.quality_summary(native,actual,semantic,executable)
    assert not result['evidence_available']
    assert result['diagnostic_generation_retention_pass'] is None
    observed=result['semantic_observed']['task_success']
    assert observed==dict(native=15,student=15,count=15,complete=False)
    assert result['status']=='incomplete_evidence'


def test_real_task_failure_and_missing_executable_evidence_are_distinct():
    native,actual,semantic,executable=controls()
    semantic['judgments'][0]['student']['task_success']=False
    result=study.quality_summary(native,actual,semantic,executable)
    assert result['evidence_available'] and result['diagnostic_generation_retention_pass'] is False
    assert result['semantic_gate']['rates']['task_success']['student']==15/16
    executable['available']=False;executable['passed']=False
    result=study.quality_summary(native,actual,semantic,executable)
    assert result['diagnostic_generation_retention_pass'] is None


def test_cpu_judge_refuses_visible_cuda_before_reading_inputs(tmp_path,monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    with pytest.raises(ValueError,match='CUDA hidden'):study.judge(tmp_path)


def test_completed_generation_receipt_binds_raw_evidence(tmp_path):
    p={'fingerprint':'probe','generation_example_ids':['case0']}
    directory=tmp_path/'generation';put(directory/'result.json',dict(status='completed',
        fingerprint='probe',corpus=[dict(example_id='case0',response='actual output')]))
    put(directory/'manifest.json',dict(status='completed',fingerprint='probe',
        artifacts={'result.json':study.sha(directory/'result.json')}))
    assert study.generation_result(tmp_path,p)['corpus'][0]['response']=='actual output'
    put(directory/'result.json',dict(status='completed',fingerprint='probe',corpus=[]))
    with pytest.raises(ValueError,match='evidence changed'):study.generation_result(tmp_path,p)
