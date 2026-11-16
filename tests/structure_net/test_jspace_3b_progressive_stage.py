"""One-block expansion: preserved raw candidate, bounded fit, fresh admission."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.structure_net import jspace_3b_progressive_stage as study
from test_jspace_3b_broad_recovery import marker_trainer, rows_for


def trial_fixture(tmp_path, monkeypatch, *, raw_correct=248, cut=20):
    trainer, dev = marker_trainer()
    for row in trainer.dataset.items:
        row.update(content_loss_positions=[0, 0, 1, 2], content_targets=[1, 2, 3, 0])
    cache = tmp_path / "cache"
    cache.mkdir()
    study.base.write(cache / "items.json", {"items": trainer.dataset.items})
    update = trainer.train_window
    def fifteen(indices, learning_rate):
        return dict(update(indices, learning_rate), gradients_present_by_block=[True] * (36-cut))
    trainer.train_window = fifteen
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"accepted fourteen-block parent")
    plan = dict(cut=cut,parent_cut=cut+1,cache_root=str(cache), training_order=list(reversed(trainer.dataset.training)),
                native_records=rows_for(dev, range(256)), parent_checkpoint=str(parent),
                parent_checkpoint_sha256=study.base.sha(parent), fingerprint="fixture", parent_admission={"confirmed": False, "exploratory_parent": True, "parent_audit_passed": False})
    raw = tmp_path / "raw.pt"
    raw_state = trainer.snapshot()
    raw_state["stack"]["mark"].fill_(-1)
    study.old.save(raw, raw_state)
    metrics = dict(count=256, correct=raw_correct, accuracy=raw_correct / 256, loss=4., kl=.5, ce=99.)
    rows = rows_for(dev, range(raw_correct))
    report = tmp_path / "raw.json"
    study.base.write(report, dict(metrics=metrics, records=rows))
    candidate = study.candidate(raw, report, metrics, rows, updates=0, p=plan,
                                native_records=plan["native_records"], origin=f"raw{36-cut}")
    result = study.fit_step(trainer, tmp_path / "train", plan, candidate)
    return result, plan, trainer, candidate


def test_complete_pass_preserves_raw_checkpoint_and_earliest_accuracy_tie(tmp_path, monkeypatch):
    result, plan, trainer, raw = trial_fixture(tmp_path, monkeypatch)
    assert result["updates"] == 288 and trainer.finished
    assert [c["updates"] for c in result["history"]] == [0, 72, 144, 216, 288]
    assert result["candidate"] == raw and result["candidate"]["origin"] == "raw16"
    assert result["development_passed"] and not result["promoted"]
    assert result["history"][1]["metrics"]["correct"] == raw["metrics"]["correct"]
    assert result["history"][1]["metrics"]["ce"] < raw["metrics"]["ce"]
    assert torch.load(raw["checkpoint"], weights_only=True)["stack"]["mark"].item() == -1
    for candidate in result["history"]:
        saved = torch.load(candidate["checkpoint"], weights_only=True)
        assert saved["stack"]["mark"].item() == candidate["updates"]
    assert [i for u in result["training"] for i in u["indices"]] == plan["training_order"]
    assert {u["learning_rate"] for u in result["training"]} == {5e-6}
    assert study.validate_train(result, plan, raw, 2.)["candidate"] == raw


def test_failed_raw_seed_does_not_veto_a_qualifying_trained_checkpoint(tmp_path, monkeypatch):
    result, plan, _, raw = trial_fixture(tmp_path, monkeypatch, raw_correct=230)
    assert not raw["eligibility"]["passed"]
    assert result["candidate"]["updates"] == 72 and result["development_passed"]
    assert result["fallback_parent_checkpoint_sha256"] == plan["parent_checkpoint_sha256"]
    study.validate_train(result, plan, raw, 2.)


@pytest.mark.parametrize("defect", ["partial", "wrong_order", "wrong_gradients", "tie", "raw_metrics", "promotion"])
def test_completed_training_evidence_rejects_budget_or_selection_changes(tmp_path, monkeypatch, defect):
    result, plan, _, raw = trial_fixture(tmp_path, monkeypatch)
    bad = deepcopy(result)
    if defect == "partial":
        bad["training"].pop()
    elif defect == "wrong_order":
        bad["training"][0]["indices"].reverse()
    elif defect == "wrong_gradients":
        bad["training"][0]["gradients_present_by_block"] = [True] * 14
    elif defect == "tie":
        bad["candidate"] = bad["history"][1]
    elif defect == "raw_metrics":
        report = study.read(raw["report"])
        report["metrics"]["correct"] += 1
        Path(raw["report"]).write_text(json.dumps(report))
    else:
        bad["promoted"] = True
    with pytest.raises(ValueError):
        study.validate_train(bad, plan, raw, 2.)


@pytest.mark.parametrize("cut", [20, 0])
def test_build_canonicalizes_parameters_before_snapshot_without_touching_geometry(monkeypatch,cut):
    stack = nn.Module()
    stack.weight = nn.Parameter(torch.tensor([.1234567], dtype=torch.float32))
    stack.polynomial_bias = nn.Parameter(torch.tensor([.2345678], dtype=torch.float32))
    stack.register_buffer("geometry", torch.tensor([.3456789], dtype=torch.float32))
    norm = nn.Linear(1, 1, bias=False)
    norm.register_buffer("fixed", torch.tensor(4.1234567, dtype=torch.float32))
    E, rotary = torch.ones(3, 1, dtype=torch.bfloat16), nn.Identity()
    identities = [id(p) for p in (*stack.parameters(), *norm.parameters())]
    buffers = [v.clone() for m in (stack, norm) for v in m.buffers()]
    calls=[]
    def build(*args,**kwargs):
        calls.append((args,kwargs))
        return stack, norm, E, rotary, {"passed": True}
    monkeypatch.setattr(study.builder, "build_stack", build)
    plan = dict(cut=cut,parent_cut=cut+1,parent_fits={str(i): "fit" for i in range(cut+1, 36)}, fit_root="newfit", model={"path": "model"})
    result = study._build(plan, {"stack": {}}, "cpu", True)
    assert result[0] is stack
    assert [id(p) for p in (*stack.parameters(), *norm.parameters())] == identities
    assert all(p.dtype == torch.bfloat16 for p in (*stack.parameters(), *norm.parameters()))
    assert all(torch.equal(a, b) and a.dtype == b.dtype
               for a, b in zip(buffers, [v for m in (stack, norm) for v in m.buffers()]))
    assert calls[0][0][2]==cut and set(calls[0][0][1])==set(range(cut,36))
    assert calls[0][0][1][cut]==f"newfit/block_{cut:02d}"
    study._build(plan,{"stack":{}},"cpu",False,parent=True)
    assert calls[-1][0][2]==cut+1 and set(calls[-1][0][1])==set(range(cut+1,36))


def test_incomplete_dependencies_block_before_any_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "verify", lambda _: {"fingerprint": "fixture"})
    def incomplete(*_a):
        raise ValueError("prior phase incomplete")
    monkeypatch.setattr(study, "checked", incomplete)
    monkeypatch.setattr(study, "wait_ready", lambda *_: pytest.fail("reserved before dependency admission"))
    with pytest.raises(ValueError, match="incomplete"):
        study.run(tmp_path, "train")
    assert not (tmp_path / "train").exists()


def test_denied_boundary_lease_never_initializes_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "verify", lambda _: {"fingerprint": "fixture"})
    def denied(*_):
        raise RuntimeError("reservation denied")
    monkeypatch.setattr(study, "wait_ready", denied)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: pytest.fail("CUDA before lease"))
    monkeypatch.setattr(study.capture, "run_boundary", lambda *_: pytest.fail("model work before lease"))
    with pytest.raises(RuntimeError, match="reservation denied"):
        study.run(tmp_path, "boundary")
    assert study.read(tmp_path / "boundary/failure.json")["lease"] is None


def resources():
    uuid = "GPU-0370fdb0-fixture"
    return dict(lease_id="fixture", allocation={uuid: 11264},
                allocator={"physical_uuid": uuid, "allocator_limit_bytes": 11008 * 1024**2},
                peak_reserved_bytes=1024)


def verified_fixture(tmp_path, monkeypatch, *, qualified=True,cut=20):
    result, plan, _, _ = trial_fixture(tmp_path, monkeypatch, raw_correct=230,cut=cut)
    if not qualified:
        # Selection remains its actual best checkpoint, but no fifteen-block promotion is authorized.
        result["development_passed"] = False
    (tmp_path / "train/manifest.json").write_text(json.dumps({"fixture": "completed train"}))
    checks = {}
    for candidate in (result["candidate"], result["diagnostic_best"]):
        report = study.read(candidate["report"])
        checks[candidate["checkpoint_sha256"]] = dict(checkpoint_sha256=candidate["checkpoint_sha256"],
            strict_state_load=True, fixed_state=result["fixed_state"], fixed_state_unchanged=True,
            metrics=report["metrics"], records=report["records"])
    winner = result["candidate"]
    receipt = dict(status="completed", passed=True, checks=checks, promoted=False, parent_admission=plan["parent_admission"],
                   review_checkpoint=winner["checkpoint"] if qualified else plan["parent_checkpoint"],
                   review_checkpoint_sha256=winner["checkpoint_sha256"] if qualified else plan["parent_checkpoint_sha256"],
                   compressed_blocks=36-cut if qualified else 35-cut, development_passed=qualified,
                   fallback_parent_checkpoint_sha256=plan["parent_checkpoint_sha256"],
                   confirmed=False, full36=cut==0 and qualified, generation_recovery=False, speed_measured=False)
    directory = tmp_path / "verify"
    directory.mkdir()
    original = study.checked
    def dispatch(root, phase, p=None):
        return result if phase == "train" else original(root, phase, p)
    monkeypatch.setattr(study, "checked", dispatch)
    def publish(value):
        (directory / "result.json").write_text(json.dumps(value))
        manifest = dict(status="completed", phase="verify", fingerprint=plan["fingerprint"],
            dependencies={"train": study.base.sha(tmp_path / "train/manifest.json")},
            artifacts={"result.json": study.base.sha(directory / "result.json")},
            artifact_identity={"result.json": study.old.file_identity(directory / "result.json")}, **resources())
        (directory / "manifest.json").write_text(json.dumps(manifest))
    publish(receipt)
    return plan, receipt, publish


@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize("cut", [20, 0])
def test_fresh_verification_records_only_review_candidate_or_parent_fallback(tmp_path, monkeypatch, qualified,cut):
    plan, receipt, _ = verified_fixture(tmp_path, monkeypatch, qualified=qualified,cut=cut)
    assert study.checked(tmp_path, "verify", plan) == receipt
    assert receipt["compressed_blocks"] == (36-cut if qualified else 35-cut)
    assert receipt["promoted"] is False and receipt["confirmed"] is False
    assert "accepted_checkpoint" not in receipt


@pytest.mark.parametrize("defect", ["missing_proof", "wrong_records", "fixed_state", "wrong_path", "full36", "wrong_depth", "promotion", "confirmed", "accepted_label", "audit_relabel"])
def test_fresh_receipt_rejects_substitution_or_partial_model_overclaim(tmp_path, monkeypatch, defect):
    plan, receipt, publish = verified_fixture(tmp_path, monkeypatch)
    bad = deepcopy(receipt)
    proof = next(iter(bad["checks"].values()))
    if defect == "missing_proof":
        bad["checks"] = {}
    elif defect == "wrong_records":
        proof["records"][0]["prediction"] = 1 - proof["records"][0]["prediction"]
    elif defect == "fixed_state":
        proof["fixed_state"] = {"sha256": "wrong"}
    elif defect == "wrong_path":
        bad["review_checkpoint"] = "different.pt"
    elif defect == "full36":
        bad["full36"] = True
    elif defect == "wrong_depth":
        bad["compressed_blocks"] = 36
    elif defect == "promotion":
        bad["promoted"] = True
    elif defect == "confirmed":
        bad["confirmed"] = True
    elif defect == "accepted_label":
        bad["accepted_checkpoint"] = bad["review_checkpoint"]
    else:
        bad["parent_admission"]["parent_audit_passed"] = True
    publish(bad)
    with pytest.raises(ValueError):
        study.checked(tmp_path, "verify", plan)


@pytest.mark.parametrize("parent_cut", [21,1])
def test_feature_only_parent_capture_matches_original_targets_without_logit_bank(tmp_path,parent_cut):
    import numpy as np
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding
    from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
    from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack
    from test_jspace_3b_parent_feature import source as feature_fixture
    _, _, E, _, data = feature_fixture(interleaved=True)
    cfg=Qwen2Config(hidden_size=8,intermediate_size=12,num_hidden_layers=36,num_attention_heads=2,
        num_key_value_heads=1,attention_dropout=0.,max_position_embeddings=16)
    cfg._attn_implementation='eager'
    torch.manual_seed(218)
    blocks=[AffineQwenBlock(Qwen2DecoderLayer(cfg,i).bfloat16().eval(),torch.eye(8)[:,:4],
        torch.zeros(6,8),trainable=False,parameter_dtype=torch.bfloat16) for i in range(parent_cut,36)]
    stack=CompactQwenStack(blocks,geometry_mode='bf16',pad_to_multiple=0)
    stack.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16,strip_unused=True).eval().requires_grad_(False)
    norm=Qwen2RMSNorm(8).bfloat16().eval().requires_grad_(False)
    rotary=Qwen2RotaryEmbedding(cfg).eval().requires_grad_(False)
    parent='a'*64
    reference=study.feature.capture_packed_parent(tmp_path/'reference',stack,norm,E,rotary,data,parent_checkpoint_sha256=parent)
    actual=study.capture_parent_features(tmp_path/'features',stack,norm,E,rotary,data,parent_sha=parent,parent_cut=parent_cut,guard=lambda:None)
    assert not (tmp_path/'features/packed_teacher_logits.npy').exists()
    assert actual['logits_written'] is False and actual['fixed_state_unchanged']
    assert np.array_equal(np.load(tmp_path/'features/parent_training_final_answers.npy'),
                          np.load(tmp_path/'reference/parent_training_final_answers.npy'))
    assert actual['features']['normalization_energy']==reference['features']['normalization_energy']
    targets,receipt=study.feature.load_targets(tmp_path/'features',data.items,parent)
    assert set(targets.lookup)==set(data.training) and receipt['parent_layers']==list(range(parent_cut,36))
    data.x[data.development]*=17
    changed=study.capture_parent_features(tmp_path/'changed_dev',stack,norm,E,rotary,data,parent_sha=parent,parent_cut=parent_cut,guard=lambda:None)
    assert changed['features']['sha256']==actual['features']['sha256']
    assert changed['features']['normalization_energy']==actual['features']['normalization_energy']


def parent_proof(tmp_path,monkeypatch,*,legacy=True,cut=None):
    cut=(21 if legacy else 7) if cut is None else cut
    plan=dict(schema=study.first_stage.SCHEMA if legacy else study.SCHEMA,cut=cut)
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    candidate=dict(checkpoint='actual_selected.pt',checkpoint_sha256='s'*64,metrics={'correct':190})
    training=dict(candidate=candidate,development_passed=True,fixed_state={'sha256':'fixed','tensor_bytes':24})
    verification=dict(passed=True,development_passed=True,compressed_blocks=36-cut,
        review_checkpoint=candidate['checkpoint'],review_checkpoint_sha256=candidate['checkpoint_sha256'],
        confirmed=False,promoted=False,checks={candidate['checkpoint_sha256']:dict(strict_state_load=True,
            fixed_state_unchanged=True,fixed_state=training['fixed_state'],metrics=candidate['metrics'])})
    owner=study.first_stage if legacy else study
    monkeypatch.setattr(owner,'verify',lambda _:plan)
    monkeypatch.setattr(owner,'checked',lambda root,phase,p:verification if phase=='verify' else training)
    return plan,verification,training


@pytest.mark.parametrize('legacy',[True,False])
def test_parent_admission_dispatches_exact_completed_passing_checkpoint(tmp_path,monkeypatch,legacy):
    expected=parent_proof(tmp_path,monkeypatch,legacy=legacy)
    assert study.admit_parent(tmp_path)==expected


@pytest.mark.parametrize('defect',['failed_dev','fallback','wrong_path','wrong_depth','wrong_fixed','unverified','confirmed','finished_cut0','legacy_wrong_cut'])
def test_parent_fallback_or_incomplete_proof_never_advances(tmp_path,monkeypatch,defect):
    plan,verification,training=parent_proof(tmp_path,monkeypatch)
    if defect=='failed_dev':training['development_passed']=False
    elif defect=='fallback':verification['review_checkpoint_sha256']='parent_fallback'
    elif defect=='wrong_path':verification['review_checkpoint']='other.pt'
    elif defect=='wrong_depth':verification['compressed_blocks']-=1
    elif defect=='wrong_fixed':next(iter(verification['checks'].values()))['fixed_state']={}
    elif defect=='unverified':verification['passed']=False
    elif defect=='confirmed':verification['confirmed']=True
    elif defect=='finished_cut0':plan['cut']=0
    else:plan['cut']=19
    with pytest.raises(ValueError):study.admit_parent(tmp_path)


@pytest.mark.parametrize('parent_cut',[21,1])
def test_prepare_binds_immediate_parent_new_fit_boundary_and_original_native_reference(tmp_path,monkeypatch,parent_cut):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    parent=tmp_path/'parent';parent.mkdir();cache=tmp_path/'cache';cache.mkdir()
    items=[dict(split='train',family=family,content_loss_positions=[0]*4,content_targets=[0]*4)
        for family in study.capture.FAMILIES for _ in range(32)]
    study.base.write(cache/'items.json',dict(items=items))
    fits={str(i):str(tmp_path/f'oldfit{i}') for i in range(parent_cut+1,36)}
    p=dict(schema=study.first_stage.SCHEMA if parent_cut==21 else study.SCHEMA,cut=parent_cut,
        fingerprint='parent-fingerprint',parent_fits=fits,fit_root=str(parent/'fit'),input_sha256={},
        parent_admission=dict(exploratory_parent=True,parent_audit_passed=False,confirmed=False),
        model={'path':'original-native'},data_root=str(tmp_path/'data'),cache_root=str(cache),
        training_order=list(range(128)),label_ids=[32,33,34,35],native_records=[{'fixed_original':195}],native_correct=195)
    for directory in ('fit','boundary','train','verify'):(parent/directory).mkdir()
    newfit=parent/'fit'/f'block_{parent_cut:02d}';newfit.mkdir()
    for name in ('fit.pt','relation.pt'):(newfit/name).write_bytes(name.encode())
    fit_artifacts={f'block_{parent_cut:02d}/{name}':study.base.sha(newfit/name) for name in ('fit.pt','relation.pt')}
    boundary=parent/'boundary'/f'boundary_{parent_cut:02d}.npy';boundary.write_bytes(b'authenticated native states')
    checkpoint=parent/'train/update_072.pt';checkpoint.write_bytes(b'actual selected learned state')
    report=parent/'train/update_072_development.json';study.base.write(report,dict(metrics={'correct':190},records=[]))
    candidate=dict(checkpoint=str(checkpoint),checkpoint_sha256=study.base.sha(checkpoint),report=str(report),report_sha256=study.base.sha(report))
    trained=dict(candidate=candidate,fixed_state={'sha256':'f'*64,'tensor_bytes':1024})
    for name in ('train/manifest.json','train/result.json','verify/manifest.json','verify/result.json',
                 'fit/result.json','boundary/result.json','boundary/step_receipt.json'):
        study.base.write(parent/name,dict(fixture=name))
    study.base.write(parent/'fit/manifest.json',dict(artifacts=fit_artifacts))
    study.base.write(parent/'boundary/manifest.json',dict(artifacts={boundary.name:study.base.sha(boundary)}))
    study.base.write(parent/'plan.json',p)
    monkeypatch.setattr(study,'admit_parent',lambda _: (p,{'passed':True},trained))
    observed=[]
    monkeypatch.setattr(study.capture,'validate_plan_items',lambda plan:observed.append(deepcopy(plan)))
    def closure(paths):
        assert study.PROTOCOL in paths
        return {Path(study.__file__).resolve()}
    monkeypatch.setattr(study.source.inherited,'local_import_closure',closure)
    root=tmp_path/'new';result=study.prepare(root,parent)
    plan=study.verify(root)
    assert result['cut']==plan['cut']==parent_cut-1
    assert plan['parent_checkpoint_sha256']==candidate['checkpoint_sha256']
    assert plan['parent_fits'][str(parent_cut)]==str(newfit.resolve())
    assert plan['parent_boundary_path']==str(boundary.resolve())
    assert plan['parent_boundary_sha256']==study.base.sha(boundary)
    assert plan['parent_boundary_receipt_sha256']==study.base.sha(parent/'boundary/step_receipt.json')
    assert plan['native_records']==p['native_records'] and plan['training_order']==p['training_order']
    assert plan['parent_admission']['parent_audit_passed'] is False
    assert plan['generation_recovery'] is plan['confirmed'] is plan['full36'] is False
    assert observed[0]['cut']==parent_cut-1 and plan['target_compressed_blocks']==37-parent_cut
    (parent/'verify/manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='input changed'):study.verify(root)


@pytest.mark.parametrize('cut',[20,0])
def test_dataset_delegates_dynamic_native_boundary_and_keeps_original_content_teacher(monkeypatch,cut):
    plan={'cut':cut,'parent_cut':cut+1};calls=[]
    def factory(p,*,parent_input):calls.append((p,parent_input));return 'dataset'
    monkeypatch.setattr(study.capture,'recovery_dataset',factory)
    assert study._dataset(plan)=='dataset' and study._dataset(plan,True)=='dataset'
    assert calls==[(plan,False),(plan,True)]
