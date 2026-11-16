"""Lifecycle, selected-prefix cache, and owned-storage controls."""
from dataclasses import replace
from pathlib import Path
import json

import pytest
import torch

from experiments.structure_net import jspace_3b_ablation32_distill as study
from test_jspace_3b_ablation32_compile import make_stage,fixture


def test_epochs_cover_inventory_once_and_early_selection_is_explicit():
    indices=list(range(517))
    windows=study.epoch_windows(indices,1)
    assert sorted(sum(windows,[]))==indices and list(map(len,windows))==[256,256,5]
    assert windows==study.epoch_windows(indices,1)
    assert windows!=study.epoch_windows(indices,2)
    candidates=[dict(epoch=2,selection={'objective':1.}),dict(epoch=0,selection={'objective':1.})]
    assert min(candidates,key=study.rank_candidate)['epoch']==0
    with pytest.raises(ValueError):study.rank_candidate(dict(epoch=1,selection={'objective':float('nan')}))


def test_producer_pins_dynamic_nal_and_reservation_dependencies():
    sources=study.source_files()
    for name in ('src/neural_architecture_lab/distillation_training.py','src/neural_architecture_lab/distillation_data.py',
        'src/structure_net_resources/client.py','pixi.lock'):
        assert study.ROOT/name in sources


def test_prefix_cache_replays_actual_student_and_rejects_other_prefix(tmp_path):
    _,stage,dataset=make_stage(layer=1,dtype=torch.bfloat16)
    offset=0;dataset.rows=[]
    for record in dataset.records:
        dataset.rows.append(dict(offset=offset,stop=offset+len(record.input_ids)))
        offset+=len(record.input_ids)
    dataset.plan=dict(total_tokens=offset,records_sha256='fixture')
    cache=study.PrefixCache(tmp_path/'prefix',stage,dataset,[0,1,2])
    try:
        batch=dataset.batch([1]);plain=stage.forward(batch)
        cached=stage.forward(dict(batch,**cache(1)))
        for a,b in zip(plain,cached):torch.testing.assert_close(a,b,atol=0,rtol=0)
        with pytest.raises(ValueError):cache(3)
        old=stage.prefix_fingerprint;stage.prefix_fingerprint='changed'
        with pytest.raises(ValueError,match='bound'):
            study.PrefixCache(tmp_path/'prefix',stage,dataset,[0,1,2],create=False)
        stage.prefix_fingerprint=old
        cache.discard();assert not cache.path.exists()
        assert (tmp_path/'prefix/manifest.json').exists()
    finally:cache.close()


def test_real_epoch_loop_saves_only_block_and_exactly_replays_selection(tmp_path):
    _,stage,dataset=make_stage(dtype=torch.bfloat16)
    cfg=replace(study.compiler.DistillationConfig(),epochs=2,accumulation_examples=2,
        learning_rate=.01,token_chunk_size=1,optimizer_chunk_bytes=64)
    trainer=study.compiler.StageTrainer(stage,dataset,teacher_id='original-native',train_indices=[0,1],
        selection_indices=[2],config=cfg)
    plan=dict(training_indices=[0,1],seed=1001,fingerprint='fixture',arm='learned',layer=0,compiled_root='/fixture')
    progress=[]
    try:
        result=study.train_epochs(plan,trainer,tmp_path,progress=lambda phase,**row:progress.append((phase,row)))
        assert result['epochs']==2 and result['updates']==2 and result['selected_replay_exact']
        assert [row['epoch']for row in result['history']]==[0,1,2]
        assert len(list((tmp_path/'checkpoints').glob('*.pt')))==3
        for row in result['history']:
            state=torch.load(row['checkpoint']['path'],weights_only=True)
            assert 'stack'not in state and 'block'in state and state['norm']is None
        assert [r['response_tokens']for phase,r in progress if phase=='update']==[5,5]
    finally:trainer.close()


def test_pilot_proof_accepts_quantized_no_change_but_requires_real_block_update():
    _,stage,dataset=make_stage(dtype=torch.bfloat16)
    cfg=replace(study.compiler.DistillationConfig(),epochs=1,accumulation_examples=2,
        learning_rate=.01,token_chunk_size=1,optimizer_chunk_bytes=64)
    trainer=study.compiler.StageTrainer(stage,dataset,teacher_id='native',train_indices=[0,1],selection_indices=[2],config=cfg)
    before=[p.detach().clone()for p in trainer.parameters]
    try:
        trainer.begin_epoch(1);receipt=trainer.train_window([0,1])
        # One tensor can be a legitimate zero/sub-ULP update. The proof checks
        # its exact FP32-master cast, rather than requiring all tensors change.
        before[0]=trainer.parameters[0].detach().clone()
        result=study.pilot_update_proof(trainer,before,receipt)
        assert result['all_intended_gradients_received_and_finite']and result['unchanged_tensors']
        with pytest.raises(RuntimeError,match='any actual'):
            study.pilot_update_proof(trainer,[p.detach().clone()for p in trainer.parameters],receipt)
    finally:trainer.close()


def test_retention_only_deletes_owned_provisional_blocks(tmp_path):
    directory=tmp_path/'checkpoints';directory.mkdir();rows=[]
    for epoch,value in enumerate([.1,.2,.3,.9]):
        path=directory/f'epoch_{epoch}.pt';path.write_bytes(bytes([epoch]))
        rows.append(dict(epoch=epoch,selection={'objective':value},
            checkpoint=dict(path=str(path.resolve()),sha256=study.sha(path))))
    removed=study.retain_best_three(tmp_path,rows)
    assert [r['epoch']for r in removed]==[3]
    outside=tmp_path/'someone_else.pt';outside.write_bytes(b'not-ours')
    rows.append(dict(epoch=4,selection={'objective':1.},checkpoint=dict(path=str(outside),sha256=study.sha(outside))))
    with pytest.raises(ValueError,match='nonowned'):study.retain_best_three(tmp_path,rows)
    assert outside.exists()


def test_build_stage_reconstructs_selected_prefix_without_training_future(tmp_path):
    compiler=study.compiler;native,gs,stack=fixture(torch.bfloat16)
    model=tmp_path/'native';native.save_pretrained(model)
    geometry=tmp_path/'geometry';geometry.mkdir();layers={}
    for i,g in enumerate(gs):
        path=geometry/f'layer_{i:02d}.pt';torch.save(g,path)
        layers[str(i)]=dict(path=path.name,sha256=study.sha(path),width=8,retained=4)
    (geometry/'manifest.json').write_text(json.dumps(dict(schema='ablation32.geometry-bank.v1',
        projection_sites=compiler.PROJECTION_SITES,layers=layers)))
    compiled=tmp_path/'compiled';compiler.compile_all(compiled,model,geometry,arm='learned',expected_layers=2)
    p=dict(compiled_root=str(compiled),selected_prefix=[],model={'path':str(model)},layer=0,layers=2,train_final_norm=False)
    stage=study.build_stage(p)
    assert stage.lookahead.start==1 and len(stage.lookahead.blocks)==1
    assert all(x.requires_grad for x in stage.block.parameters())
    assert not any(x.requires_grad for x in stage.stack.blocks[1].parameters())
    assert not any(x.requires_grad for x in stage.lookahead.parameters())
    with torch.no_grad():stage.block.mlp_output.add_(.125)
    checkpoint=tmp_path/'selected.pt'
    torch.save(dict(schema='jspace.ablation32-stage-checkpoint.v1',layer=0,
        block=compiler.native.state_dict_cpu(stage.block),norm=None),checkpoint)
    p.update(layer=1,train_final_norm=True,selected_prefix=[dict(layer=0,checkpoint=str(checkpoint),sha256=study.sha(checkpoint))])
    second=study.build_stage(p)
    assert len(second.lookahead.blocks)==0 and second.train_final_norm
    torch.testing.assert_close(second.stack.blocks[0].mlp_output,stage.block.mlp_output,atol=0,rtol=0)
    assert not any(x.requires_grad for x in second.stack.blocks[0].parameters())


def test_parent_admission_requires_completed_adjacent_same_arm(tmp_path,monkeypatch):
    checkpoint=tmp_path/'block.pt';checkpoint.write_bytes(b'fixture')
    for name in ('plan.json','run/result.json','run/manifest.json'):
        path=tmp_path/name;path.parent.mkdir(exist_ok=True);path.write_text('{}')
    p=dict(layer=0,arm='learned',campaign_fingerprint='campaign',compiled_root='/compiled/learned')
    result=dict(status='completed',selected_blocks=[dict(layer=0,checkpoint=str(checkpoint),sha256=study.sha(checkpoint))])
    monkeypatch.setattr(study,'verify',lambda *a,**kw:(p,None,None))
    monkeypatch.setattr(study,'check',lambda *a:result)
    rows,pins=study._parent(tmp_path,layer=1,arm='learned',campaign_fingerprint='campaign',compiled_root=Path('/compiled/learned'))
    assert rows==result['selected_blocks']and str(checkpoint)in pins
    with pytest.raises(ValueError,match='same-arm'):
        study._parent(tmp_path,layer=1,arm='random',campaign_fingerprint='campaign',compiled_root=Path('/compiled/random'))
    result['status']='failed'
    with pytest.raises(ValueError):study._parent(tmp_path,layer=1,arm='learned',campaign_fingerprint='campaign',compiled_root=Path('/compiled/learned'))
