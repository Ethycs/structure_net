"""CPU contract and real optimizer/adapter tests, never synthetic CUDA evidence."""
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2RotaryEmbedding

from experiments.structure_net import jspace_3b_attention_retry as study
from test_jspace_3b_attention_retry_model import Dataset, fixture, equal_tensors


def budget_fixture():
    # 1136 complete examples: 1136*22+253 =25245 tokens, in71 existing windows.
    records = [SimpleNamespace(split='train', finish_reason='eos', eos_token_id=31,
        response_ids=tuple([6]*(22 if i < 253 else 21)+[31])) for i in range(1136)]
    order = list(range(len(records)))
    dp = dict(training_indices=order, training=dict(order=order,
        windows=[order[i:i+16] for i in range(0, len(order), 16)], seed=20261117))
    return dp, records


def test_first71_exact_response_eos_budget_and_no_new_shuffle():
    dp, records = budget_fixture()
    planned = study.matched_training(dp, records)
    assert planned['actual_supervised_tokens'] == 25245
    assert planned['updates'] == 71 and planned['windows'] == dp['training']['windows']
    assert planned['order'] == dp['training']['order']
    assert not planned['full_corpus_pass'] and not planned['automatic_expansion']


@pytest.mark.parametrize('damage', ['order', 'duplicate', 'dev', 'test', 'eos', 'budget', 'windows', 'admission'])
def test_budget_gates_reject_partial_or_changed_training(damage):
    dp, records = budget_fixture()
    if damage == 'order':
        dp['training']['order'] = list(reversed(dp['training']['order']))
    elif damage == 'duplicate':
        dp['training']['windows'][0][1] = 0
    elif damage in ('dev', 'test'):
        records[0].split = 'development' if damage == 'dev' else 'test'
    elif damage == 'eos':
        records[0].response_ids = (31, 6)
    elif damage == 'budget':
        records[0].response_ids += (31,)
    elif damage == 'windows':
        dp['training']['windows'].pop()
    else:
        dp['training_indices'] = dp['training_indices'][1:]
    with pytest.raises(ValueError):
        study.matched_training(dp, records)


def parent_rows():
    return [dict(label=label, update=int(label[7:]), evidence_available=True, qualified=False,
        useful_completed_tasks=i, reliable_generations=8, nll=5.-i,
        checkpoint=dict(path=label, sha256=label)) for i, label in enumerate(study.PARENT_LABELS)]


def test_parent_selection_is_generation_first_even_when_every_candidate_unqualified():
    rows = parent_rows(); rows[0].update(useful_completed_tasks=10, nll=100.)
    selected = study.select_parent(rows)
    assert selected['selected']['label'] == 'update_0000'
    assert selected['status'] == 'exploratory_unqualified' and selected['qualified'] is False
    rows[1]['qualified'] = True
    assert study.select_parent(rows)['selected']['label'] == 'update_0071'


@pytest.mark.parametrize('damage', ['missing', 'unavailable', 'duplicate', 'reordered'])
def test_parent_missing_judgments_never_fall_back_to_nll(damage):
    rows = parent_rows()
    if damage == 'missing':
        rows.pop()
    elif damage == 'unavailable':
        rows[2]['evidence_available'] = False
    elif damage == 'duplicate':
        rows[2] = rows[1]
    else:
        rows.reverse()
    with pytest.raises(ValueError, match='all five'):
        study.select_parent(rows)


def seal(directory, plan, **fields):
    directory.mkdir(parents=True, exist_ok=True)
    return study.finish(directory, plan, dict(status='completed', fingerprint=plan['fingerprint'], **fields))


def test_phase_receipts_preserve_failure_and_reject_modified_results(tmp_path):
    plan = dict(fingerprint='cpu-receipt-test')
    directory = tmp_path/'phase'; directory.mkdir()
    study.failure(directory, RuntimeError('real failure'))
    assert study.read(directory/'failure.json')['partial_artifacts_preserved'] is True
    seal(directory, plan, evidence_available=False)
    assert study.receipts._completed(directory, plan)['evidence_available'] is False
    (directory/'result.json').write_text(json.dumps(dict(status='completed', fingerprint='different')))
    with pytest.raises(ValueError, match='evidence changed'):
        study.receipts._completed(directory, plan)
    assert (directory/'failure.json').exists()


def test_start_is_append_only_and_never_erases_partial_phase(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'verify', lambda root: dict(fingerprint='cpu-contract'))
    plan, out = study.start(tmp_path, 'quadratic', 'pilot')
    study.failure(out, RuntimeError('kept'))
    original = (out/'failure.json').read_bytes()
    with pytest.raises(FileExistsError):
        study.start(tmp_path, 'quadratic', 'pilot')
    assert (out/'failure.json').read_bytes() == original
    with pytest.raises(ValueError):
        study.start(tmp_path, 'other', 'pilot')
    with pytest.raises(ValueError):
        study.start(tmp_path, 'softmax', 'generation', 'update_0134')


def mc_rows(predictions):
    return [dict(identity=dict(source='exposed-dev', index=i), family='fixture', answer=0, prediction=v)
            for i, v in enumerate(predictions)]


def test_paired_mc_exact_ids_wins_losses_and_five_point_guardrail():
    first, second = mc_rows([0, 1, 0, 1]), mc_rows([1, 0, 0, 1])
    result = study.paired_mc(first, list(reversed(second)))
    assert result['exact_matched_ids'] == [r['identity'] for r in first]
    assert result['quadratic_wins'] == result['quadratic_losses'] == 1
    assert result['both_correct'] == result['both_wrong'] == 1
    assert result['accuracy_delta_pp'] == 0 and result['retention_pass']
    assert not result['independent_population_claim']
    assert not study.paired_mc(mc_rows([0]*4), mc_rows([0, 0, 0, 1]))['retention_pass']
    assert study.paired_mc(mc_rows([0]*20), mc_rows([0]*19+[1]))['retention_pass']


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'answer', 'family'])
def test_paired_mc_rejects_unmatched_cases(damage):
    left, right = mc_rows([0, 1]), mc_rows([1, 0])
    if damage == 'missing':
        right.pop()
    elif damage == 'duplicate':
        right[1] = right[0]
    else:
        right[0][damage] = 'changed'
    with pytest.raises(ValueError):
        study.paired_mc(left, right)


@pytest.mark.parametrize('mode', study.ARMS)
def test_real_cpu_training_pack_export_reload_preserves_attention_and_geometry(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('CPU test initialized CUDA'))
    cfg, original = fixture(torch.bfloat16)
    cfg.save_pretrained(tmp_path/'model')
    norm, E, rotary = Qwen2RMSNorm(16).bfloat16(), torch.randn(32, 16).bfloat16(), Qwen2RotaryEmbedding(cfg)
    dataset = Dataset()
    p = dict(model=dict(path=str(tmp_path/'model')), configuration=dict(max_length=8))
    before = deepcopy(original.state_dict())
    assert study.initial_control(p, (original, norm, E, rotary), dataset.batch([0]), None)
    equal_tensors(before, original.state_dict())
    stack = study.adapter.AttentionRetryStack.from_stack(original, mode=mode, copy=False)
    trainer = study.base.training.ResponseTrainer(stack, norm, E, rotary, dataset, teacher_id='native',
        config=study.base.training.ResponseConfig(learning_rate=2e-5, max_supervised_tokens=4,
            max_length=8, token_chunk_size=1, optimizer_chunk_bytes=64), host_reserve_bytes=0)
    try:
        initial = trainer.parameter_fingerprints()
        updates = study.parent.pilot_updates(trainer, [[0], [1]])
        report = study.campaign._pilot_report(trainer, trainer.change_report(initial), updates['updates'])
        assert report['passed'] and report['norm_fp32_master_changed']
        assert report['fp32_master_changed_by_block'] == [True]*3
        assert study.base.offload.fixed_state(stack, E, rotary) == trainer.fixed
        cp = trainer.save_checkpoint(tmp_path/'checkpoint.pt', identities=dict(cpu_test_only=True, arm=mode))
        packed = study.pack(stack, norm, E, rotary, p)
        assert packed.stack.attention_retry['mode'] == mode and packed.stack.attention_retry['layers'] == [35]
        # Packing is a clone, not a conversion of live optimizer parameters.
        assert not stack.packed and not stack.stripped
        assert tuple(map(id, trainer.parameters)) == trainer.parameter_ids
        exported = study.adapter.export_packed_snapshot(packed, tmp_path/'export', identities=dict(cpu_test_only=True))
        restored, receipt = study.adapter.load_packed_snapshot(tmp_path/'export')
        assert exported['attention_retry'] == receipt['attention_retry']
        with torch.inference_mode():
            left = packed.readout(packed.hidden_forward(dataset.batch([0])).flatten(0, 1))
            right = restored.readout(restored.hidden_forward(dataset.batch([0])).flatten(0, 1))
        assert torch.equal(left, right)
        saved = torch.load(cp['path'], map_location='cpu', weights_only=True)
        description = study.adapter.packed._describe(study.adapter._carrier(packed))
        raw = study.adapter.restore_unpacked(saved['stack'], description)
        equal_tensors(saved['stack'], raw.state_dict())
        assert raw.attention_retry['mode'] == mode
        # The old generic loader must not silently strip execution metadata.
        with pytest.raises(ValueError):
            study.adapter.packed.load_packed_snapshot(tmp_path/'export')
    finally:
        trainer.close()


def test_pilot_gate_needs_both_receipts_and_identical_fresh_initializers(tmp_path):
    p = dict(fingerprint='receipt-contract-only-not-cuda-proof')
    fields = dict(passed=True, discarded=True, fresh_initializer=True, fixed_geometry_preserved=True,
        softmax_control_initial_exact=True, fresh_raw_reload_exact=True, packed_reload_exact=True,
        initializer=dict(learned='same', fixed='same'))
    seal(tmp_path/'arms/softmax/pilot', p, arm='softmax', **fields)
    with pytest.raises(FileNotFoundError):
        study.pilot_receipts(tmp_path, p)
    seal(tmp_path/'arms/quadratic/pilot', p, arm='quadratic', **fields)
    assert set(study.pilot_receipts(tmp_path, p)) == set(study.ARMS)
    other = tmp_path/'arms/quadratic/pilot'
    # Re-sealing synthetic receipts tests the binding gate, not GPU qualification.
    (other/'manifest.json').unlink()
    (other/'result.json').unlink()
    seal(other, p, arm='quadratic', **dict(fields, initializer=dict(learned='different', fixed='same')))
    with pytest.raises(ValueError, match='identical'):
        study.pilot_receipts(tmp_path, p)


def test_check_missing_evidence_keeps_comparison_unknown(tmp_path, monkeypatch):
    p = dict(fingerprint='cpu-summary', parent_selection=dict(status='exploratory_unqualified', qualified=False))
    monkeypatch.setattr(study, 'verify', lambda root, live=False: p)
    result = study.check(tmp_path)
    assert result['comparison']['final_semantic'] is None
    assert result['comparison']['final_mc'] is None
    assert result['selection_final'] is False and result['confirmed'] is False
    assert len(result['generation_or_judgment_pending']) == 8
    assert result['parent_qualified'] is False
    assert all(v['task_selection']['selected'] is None for v in result['arms'].values())


def test_prepare_waits_without_creating_worker_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    root = tmp_path/'new-study'; (root/'controller_v1').mkdir(parents=True)
    def unavailable(parent):
        raise ValueError('Pending blocked: all five available parent judgments required')
    monkeypatch.setattr(study, 'authenticate_parent', unavailable)
    with pytest.raises(ValueError, match='Pending blocked'):
        study.prepare(root, tmp_path/'parent')
    assert [p.name for p in root.iterdir()] == ['controller_v1']
    (root/'pilot').mkdir()
    with pytest.raises(FileExistsError):
        study.prepare(root, tmp_path/'parent')


def authenticated_parent_fixture(tmp_path, monkeypatch):
    """Real local hashes/manifests; no model execution or claimed GPU evidence."""
    root = tmp_path/'parent/stage_a'; root.mkdir(parents=True)
    p = dict(fingerprint='parent-cpu-authentication', data_fingerprint='data',
        model=dict(revision='teacher'), checkpoint_sha256='initializer')
    study.write(root/'plan.json', p)
    preparation = tmp_path/'preparation.json'
    study.write(preparation, dict(parent_worker_root=str(root), parent_worker_fingerprint=p['fingerprint']))
    monkeypatch.setattr(study, 'PREPARATION', preparation)
    monkeypatch.setattr(study.parent, 'verify', lambda root, live=False: (p, {}, []))
    candidates = []; judgments = parent_rows()
    for row in judgments:
        label = row['label']; checkpoint = root/'run/checkpoints'/(label+'.pt')
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(schema='jspace.complete-response-checkpoint.v1', identities=study.parent.identities(p),
            updates=row['update'], stack={'_extra_state': {}}), checkpoint)
        cp = dict(path=str(checkpoint), sha256=study.sha(checkpoint), optimizer=None)
        row['checkpoint'] = cp
        candidate = seal(root/'run/candidates'/label, p, label=label, checkpoint=cp, packed_export={},
            update=row['update'], nll=row['nll'])
        candidates.append(candidate)
        seal(root/'generation'/label, p, label=label, checkpoint=cp,
            canaries=dict(count=6, records=[dict(id=i) for i in range(6)]),
            corpus=[dict(example_id=i) for i in range(16)])
        row.update(seal(root/'judgment'/label, p, **row))
    seal(root/'run', p, candidates=candidates)
    controller = root.parent/'controller_v1'; controller.mkdir()
    cp = dict(worker_fingerprint=p['fingerprint'])
    cp['fingerprint'] = study.data.digest(cp)
    study.write(controller/'plan.json', cp)
    seal(controller, cp, all_generation_phases_completed=True, all_judgments_available=True,
        worker_selection=dict(selection_final=True, task_selection=study.select_parent(judgments)))
    return root, p


def test_authentication_binds_all_five_receipts_and_selected_raw_checkpoint(tmp_path, monkeypatch):
    root, p = authenticated_parent_fixture(tmp_path, monkeypatch)
    _, _, _, selection, paths = study.authenticate_parent(root)
    assert selection['status'] == 'exploratory_unqualified'
    assert selection['selected']['label'] == 'update_0278'
    assert root/'generation/update_0000/manifest.json' in paths
    checkpoint = root/'run/checkpoints/update_0278.pt'
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError, match='evidence changed'):
        study.authenticate_parent(root)


def test_authentication_never_selects_while_any_generation_receipt_is_missing(tmp_path, monkeypatch):
    root, _ = authenticated_parent_fixture(tmp_path, monkeypatch)
    (root/'generation/update_0134/manifest.json').unlink()
    with pytest.raises(FileNotFoundError):
        study.authenticate_parent(root)
