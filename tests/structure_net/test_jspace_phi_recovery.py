"""CPU lifecycle/provenance tests; the numerical trainer has its own tiny-Phi suite."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch

from experiments.structure_net import jspace_phi_recovery as r


CARD='GPU-0370fdb0-test'


def report(correct=50):
    rows=[]
    for i in range(64):
        pred=0 if i<correct else 1;native=0 if i<50 else 1
        rows.append(dict(identity=f'case{i}',family=r.data.FAMILIES[i//16],answer=0,prediction=pred,
            native_prediction=native,correct=pred==0,choice_logits=[2.,1.] if pred==0 else [1.,2.],
            native_choice_logits=[2.,1.] if native==0 else [1.,2.]))
    return dict(count=64,correct=correct,native_correct=50,ce=1.,records=rows)


def baseline():
    return dict(rows=[dict(identity=x['identity'],family=x['family'],answer=x['answer'],
        prediction=x['native_prediction'],label_logits=x['native_choice_logits'],split='development') for x in report()['records']])


def resource():
    return dict(allocation={CARD:11264},allocator=dict(visible_gpu=CARD,lease_bytes=11264*1024**2,
        allocator_limit_bytes=11008*1024**2),peak_reserved_bytes=1024)


def effective(changed=True):
    return dict(changed=changed,changed_tensor_count=int(changed),total_tensor_count=1,
        changed_by_name={'w':changed},initial_sha256_by_name={'w':'a'*64},
        current_sha256_by_name={'w':('b' if changed else 'a')*64},scope='actual BF16 parameters')


@pytest.mark.parametrize('correct,passed',[(50,True),(47,True),(46,False)])
def test_native_budget_is_three_net_cases_not_parent_or_ce(correct,passed):
    value=report(correct);value['ce']=99.
    assert r.quality(value)['passed'] is passed
    assert not r.quality(value)['confirmed']


def test_native_control_rejects_same_argmax_changed_logits():
    value=report();assert r.native_control(value,baseline())['logits_exact']
    value['records'][0]['native_choice_logits']=[2.1,1.]
    with pytest.raises(ValueError,match='native control'):r.native_control(value,baseline())


@pytest.mark.parametrize('mutation',[
    lambda m:m.update(allocation={'GPU-other':11264}),
    lambda m:m['allocator'].update(allocator_limit_bytes=11264*1024**2),
    lambda m:m.update(peak_reserved_bytes=11009*1024**2),
])
def test_resource_receipt_does_not_accept_wrong_card_or_overrun(mutation):
    value=resource();mutation(value)
    with pytest.raises(ValueError,match='reservation'):r.resources(value)


class FakeData:
    training=list(range(128))
    records=[SimpleNamespace(input_ids=list(range(5+i%5))) for i in range(128)]
    def close(self):self.closed=True


class FakeTrainer:
    fixed={'sha256':'f','tensor_bytes':3};identities={'initializer':'initial','data':'data'}
    normalization={'energy':1.}
    def __init__(self):self.updates=0;self.data=FakeData()
    def development(self):return report()
    def _save(self,path,epoch):Path(path).write_text('raw');return r.sha(path)
    def pilot(self,indices):
        assert len(indices)==4 and self.updates==0;self.updates=1
        return dict(passed=True,discarded=True,systems_only=True,fixed_state_unchanged=True,
            master_before={'sha256':'before'},master_after={'sha256':'after'},
            bf16_parameters_changed=True,effective_parameter_change=effective())
    def fit(self,path,event=None):
        assert self.updates==0;path=Path(path);path.mkdir();history=[]
        for epoch,correct in enumerate((50,51,51,49)):
            cp=path/f'epoch_{epoch:02d}.pt';cp.write_text(f'actual epoch{epoch}')
            rp=path/f'development_{epoch:02d}.json';r.write(rp,report(correct))
            history.append(dict(epoch=epoch,updates=32*epoch,checkpoint=cp.name,checkpoint_sha256=r.sha(cp),
                report=rp.name,report_sha256=r.sha(rp),correct=correct,ce=1.,effective_parameter_change=effective(epoch!=0)))
        winner=min(history,key=r.training.selection_key)
        value=dict(schema=r.training.SCHEMA,status='completed',config=asdict(r.CONFIG),history=history,
            selected=winner,metrics=r.read(path/winner['report']),updates=96,supervised_tokens=768,
            identities=self.identities,fixed=self.fixed,normalization=self.normalization,
            selected_effective_parameter_change=winner['effective_parameter_change'],
            fixed_state_unchanged=True,internal_reload=dict(passed=True,strict_state_load=True,records_and_metrics_exact=True))
        r.write(path/'updates.json',{'rows':[]});r.write(path/'result.json',value)
        r.write(path/'manifest.json',dict(status='completed',schema=r.training.SCHEMA,
            artifacts={f.name:r.sha(f) for f in path.iterdir() if f.is_file()}))
        return value


@pytest.fixture
def lifecycle(tmp_path,monkeypatch):
    root=tmp_path/'study';root.mkdir();n=tmp_path/'native-root/native';n.mkdir(parents=True)
    r.write(n/'baseline.json',baseline());p=dict(fingerprint='study',native_root=str(n.parent))
    monkeypatch.setattr(r,'verify_plan',lambda *a,**k:p)
    proof=resource();lease=dict(id='lease',allocation=proof['allocation'])
    monkeypatch.setattr(r.native,'reservation',lambda:(lease,proof['allocator'],lambda:None))
    monkeypatch.setattr(r.torch.cuda,'set_device',lambda *_:None)
    monkeypatch.setattr(r.torch.cuda,'max_memory_reserved',lambda:proof['peak_reserved_bytes'])
    built=[]
    def build(*a):
        trainer=FakeTrainer();built.append(trainer);return trainer,trainer.data
    monkeypatch.setattr(r,'build_trainer',build)
    def load(trainer,path,expected):
        assert trainer.updates==0 and r.sha(path)==expected and Path(path).read_text()=='raw'
    monkeypatch.setattr(r.training,'load_selected',load)
    def fresh(trainer,path):
        assert trainer.updates==0
        value=r.read(Path(path)/'result.json');winner=value['selected']
        return dict(status='completed',passed=True,strict_state_load=True,records_and_metrics_exact=True,
            checkpoint_sha256=winner['checkpoint_sha256'],selected_epoch=winner['epoch'],metrics=value['metrics'],
            effective_parameter_change=value['selected_effective_parameter_change'],
            fixed_state_unchanged=True,training_result_sha256=r.sha(Path(path)/'result.json'),confirmed=False)
    monkeypatch.setattr(r.training,'verify_checkpoint',fresh)
    return root,p,built


def test_three_distinct_phase_instances_preserve_raw_and_fresh_selected_proof(lifecycle):
    root,p,built=lifecycle
    assert r.run(root,'pilot')['passed']
    assert (root/'pilot/initializer.pt').read_text()=='raw'
    result=r.run(root,'train');assert result['selected']['epoch']==1
    final=r.run(root,'verify');assert final['quality']['passed'] and final['selected_epoch']==1
    assert len(built)==3 and len({id(t) for t in built})==3
    assert built[0].updates==1 and built[1].updates==0 and built[2].updates==0
    assert all(t.data.closed for t in built)
    assert not final['full40'] and not final['generation_recovery'] and not final['confirmed']
    assert r.checked(root,'verify',p)==final
    with pytest.raises(FileExistsError):r.run(root,'verify')


def test_incomplete_or_tampered_training_blocks_fresh_reservation(lifecycle,monkeypatch):
    root,p,_=lifecycle;r.run(root,'pilot');r.run(root,'train')
    (root/'train/recovery/epoch_02.pt').write_text('changed')
    monkeypatch.setattr(r.native,'reservation',lambda:pytest.fail('reservation before completed evidence'))
    with pytest.raises(ValueError,match='artifact'):r.run(root,'verify')
    assert not (root/'verify').exists()


def test_denied_lease_precedes_cuda_and_model_construction(lifecycle,monkeypatch):
    root,p,_=lifecycle
    def denied():raise RuntimeError('denied')
    monkeypatch.setattr(r.native,'reservation',denied)
    monkeypatch.setattr(r.torch.cuda,'set_device',lambda *_:pytest.fail('CUDA before lease'))
    monkeypatch.setattr(r,'build_trainer',lambda *_:pytest.fail('model before lease'))
    with pytest.raises(RuntimeError,match='denied'):r.run(root,'pilot')
    assert 'denied' in r.read(root/'pilot/failure.json')['error']


def test_native_block_is_released_before_optimizer_constructor(monkeypatch):
    dataset=FakeData();ref=[];seen=[]
    monkeypatch.setattr(r.training,'PhiTrainingData',lambda *a,**k:dataset)
    def original(*a,**kw):
        value=torch.nn.Linear(2,2,bias=False);ref.append(weakref.ref(value));return value
    monkeypatch.setattr(r.weights,'load_native_block',original)
    monkeypatch.setattr(r.fitting,'build_block',lambda *_a,**_kw:torch.nn.Linear(2,2,bias=False))
    monkeypatch.setattr(r.weights,'load_readout_norm',lambda *a,**kw:(torch.ones(2,2),torch.nn.Identity(),torch.nn.Identity()))
    def constructor(block,norm,head,rotary,data,**kwargs):
        assert ref[0]() is None and kwargs['native_block_bytes']==16
        seen.append(kwargs);return 'trainer'
    monkeypatch.setattr(r.training,'PhiTrainer',constructor)
    p=dict(native_root='native',data_root='data',native_fingerprint='n',label_ids=[0,1,2,3],
        model={'path':'model'},fit_root='fit',fit_fingerprint='f',fingerprint='study')
    trainer,actual=r.build_trainer(p,'cpu',lambda:None)
    assert trainer=='trainer' and actual is dataset and seen[0]['config']==r.CONFIG


def test_master_only_change_cannot_pass_effective_parameter_proof():
    with pytest.raises(ValueError,match='Effective BF16'):r.effective_change(effective(False),required=True)
    bad=effective();bad['changed_by_name']['w']=False
    with pytest.raises(ValueError,match='Effective BF16'):r.effective_change(bad,required=True)
    assert r.effective_change(effective(False))['changed'] is False  # A selected initializer is legitimate.


def test_prepare_and_live_archive_readback_without_gpu(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    n,f,d,m=(tmp_path/name for name in ('native','fit','data','model'))
    for path in (n/'native',f,d,m):path.mkdir(parents=True)
    model={'path':str(m),'files':{}};dp=dict(model=model,fingerprint='data',label_ids=[32,33,34,35])
    np=dict(model=model,data_fingerprint='data',data_root=str(d),fingerprint='native')
    fp=dict(native_fingerprint='native',data_fingerprint='data',native_root=str(n),data_root=str(d),model=str(m),fingerprint='fit')
    for path,value in ((n/'plan.json',np),(n/'native/manifest.json',{}),(f/'plan.json',fp),
                       (d/'plan.json',dp),(d/'manifest.json',{}),(d/'pilot_responses.jsonl',{})):r.write(path,value)
    (n/'native/prefix.npy').write_bytes(b'fixed native bank')
    nm={'artifacts':{'prefix.npy':r.sha(n/'native/prefix.npy')}}
    monkeypatch.setattr(r.data,'verify',lambda *a,**k:dp)
    monkeypatch.setattr(r.native,'verify_plan',lambda *a,**k:np)
    monkeypatch.setattr(r.native,'checked',lambda *a,**k:nm)
    monkeypatch.setattr(r.fitting,'verify',lambda *a,**k:fp)
    for phase in r.fitting.PHASES:r.write(f/(phase+'_receipt.json'),{'artifacts':{}})
    monkeypatch.setattr(r.fitting,'checked',lambda root,phase,*a:r.read(Path(root)/(phase+'_receipt.json')))
    monkeypatch.setattr(r.fitting,'local_import_closure',lambda paths:{Path(r.__file__).resolve(),r.PROTOCOL})
    out=tmp_path/'out';result=r.prepare(out,n,f,d)
    plan=r.verify_plan(out);assert result['fingerprint']==plan['fingerprint']
    assert set(plan['fit_receipts'])=={'capture','basis','interface','relation'}
    (n/'native/prefix.npy').write_bytes(b'changed source bank')
    with pytest.raises(ValueError,match='input changed'):r.verify_plan(out)


def test_prepare_rejects_different_native_teacher_before_output(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    dp=dict(model={'path':'model'},fingerprint='data')
    monkeypatch.setattr(r.data,'verify',lambda *a,**k:dp)
    monkeypatch.setattr(r.native,'verify_plan',lambda *a,**k:dict(model={'path':'different-model'}))
    monkeypatch.setattr(r.native,'checked',lambda *a,**k:{})
    monkeypatch.setattr(r.fitting,'verify',lambda *a,**k:{})
    with pytest.raises(ValueError,match='identities'):r.prepare(tmp_path/'out',tmp_path/'native',tmp_path/'fit',tmp_path/'data')
    assert not (tmp_path/'out').exists()
