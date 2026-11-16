"""Physical-card admission, immutable inputs, and pilot/science separation."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import jspace_3b_ablation32_campaign as study


SUPER = 'GPU-7b1db3cb-1111-2222-3333-444444444444'
QR = 'GPU-8cf440df-1111-2222-3333-444444444444'


def test_actual_source_closure_pins_dynamic_nal_and_existing_pixi_manifest():
    files = study.source_files()
    assert study.ROOT/'pyproject.toml' in files
    assert study.ROOT/'pixi.lock' in files
    assert study.ROOT/'src/neural_architecture_lab/distillation_training.py' in files
    assert study.ROOT/'src/structure_net_resources/client.py' in files
    assert not any(path.stem in ('jspace_3b_ablation32_controller', 'jspace_3b_ablation32_distill',
                                'jspace_3b_ablation32_evaluate') for path in files)


def records():
    return [SimpleNamespace(split='train', finish_reason='eos',
        provenance={'source_family':f'family{family}'}, group_id=f'group{family}-{length}',
        example_id=f'id{family}-{length}', prompt_ids=list(range(length)), response_ids=[7, 8, 9])
        for family in range(8) for length in (2, 5)]


def test_longest_pilot_covers_eight_distinct_train_families_and_never_dev():
    rows = records()
    indices = study.pilot_indices(rows, list(range(len(rows))))
    assert indices == list(range(1, 16, 2))
    rows[0].split = 'dev'
    with pytest.raises(ValueError, match='TRAIN'):
        study.pilot_indices(rows, list(range(len(rows))))
    with pytest.raises(ValueError, match='eight'):
        study.pilot_indices(records(), list(range(14)))


def test_physical_uuid_mapping_accepts_reversed_ordinals_and_rejects_3060():
    lease = {'allocation':{SUPER:7680, QR:2048}}
    assert study.device_layout(lease, f'{SUPER},{QR}') == {'teacher':0, 'qr':1}
    assert study.device_layout(lease, f'{QR},{SUPER}') == {'teacher':1, 'qr':0}
    wrong = 'GPU-0370fdb0-1111-2222-3333-444444444444'
    with pytest.raises(ValueError, match='cards'):
        study.device_layout({'allocation':{wrong:7680, QR:2048}}, f'{wrong},{QR}')
    with pytest.raises(ValueError):
        study.device_layout({'allocation':{SUPER:7424, QR:2048}}, f'{SUPER},{QR}')
    with pytest.raises(ValueError):
        study.device_layout(lease, f'{SUPER},{SUPER}')


def prepared(tmp_path, monkeypatch):
    rows = records(); data_root = tmp_path/'data'; data_root.mkdir()
    (data_root/'plan.json').write_text('{}')
    source = tmp_path/'input.py'; source.write_text('SOURCE = 1\n')
    dp = dict(fingerprint='data', model={'path':'/native', 'revision':'revision'},
        records_sha256='records', training_indices=list(range(len(rows))),
        response_cache_root='/original', full_data_root='/native_banks', input_identity={})
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    monkeypatch.setattr(study, 'source_files', lambda:{source})
    monkeypatch.setattr(study.data, 'verify', lambda *a, **kw:(dp, rows))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    root = tmp_path/'campaign'
    study.prepare(root, data_root)
    return root, dp, rows, source


def test_preparation_waits_for_anchors_without_changing_data_and_binds_producer(tmp_path, monkeypatch):
    root, dp, rows, source = prepared(tmp_path, monkeypatch)
    p, actual, _ = study.verify(root)
    assert actual == dp and p['pilot_indices'] == list(range(1, 16, 2))
    assert not (Path(p['data_root'])/'anchors').exists()
    source.write_text('SOURCE = 2\n')
    with pytest.raises(ValueError, match='producer'):
        study.verify(root)
    assert study.verify(root, live=False)[0]['fingerprint'] == p['fingerprint']


class FakeNative:
    pass


class FakeDataset:
    anchors = None
    closed = False
    def close(self): self.closed = True


class FakeTrainer:
    instances = []
    def __init__(self, *, loaded_pilot=False):
        self.coordinates = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
        with torch.no_grad(): self.coordinates[0].weight.fill_(2 if loaded_pilot else 1)
        self.updates = 0
        self.optimizer = SimpleNamespace(state={})
        self.closed = False
        self.instances.append(self)
    def pilot(self, indices, *, directory):
        directory.mkdir(parents=True)
        with torch.no_grad(): self.coordinates[0].weight.add_(3)
        self.updates = 1
        self.optimizer.state = {'trained':True}
        result = dict(status='systems_passed', discarded=True, all_interfaces_updated=True,
            interfaces=36, teacher_unchanged=True, fresh_raw_reload_exact=True,
            fresh_projector_replay_exact=True)
        study.write(directory/'result.json', result)
        return result
    def close(self): self.closed = True


def fake_pilot_environment(monkeypatch):
    FakeTrainer.instances = []
    lease = {'id':'paired', 'allocation':{SUPER:7680, QR:2048}}
    monkeypatch.setattr(study, 'reservation', lambda:(lease, {'teacher':0, 'qr':1}, lambda:None))
    monkeypatch.setattr(study.data, 'SearchDataset', lambda *a, **kw:FakeDataset())
    monkeypatch.setattr(study, '_load_native', lambda *a:FakeNative())
    monkeypatch.setattr(study.data.cache.execution, 'arithmetic_policy', nullcontext)
    monkeypatch.setattr(study, '_memory', lambda _:dict(test_fixture=True))
    monkeypatch.setattr(study, '_trainer', lambda *a, **kw:FakeTrainer())


def test_disposable_pilot_requires_fresh_restart_and_authenticates_receipt(tmp_path, monkeypatch):
    root, _, _, _ = prepared(tmp_path, monkeypatch)
    fake_pilot_environment(monkeypatch)
    result = study.pilot(root)
    assert result['passed'] and result['fresh_initializer_proved']
    assert len(FakeTrainer.instances) == 2 and all(t.closed for t in FakeTrainer.instances)
    assert FakeTrainer.instances[0].updates == 1 and FakeTrainer.instances[1].updates == 0
    p, _, _ = study.verify(root)
    assert study.pilot_receipt(root, p)['initializer'] == result['initializer']
    with pytest.raises(FileExistsError, match='preserved'):
        study.pilot(root)
    (root/'pilot/systems/result.json').write_text('{}')
    with pytest.raises(ValueError, match='evidence changed'):
        study.pilot_receipt(root, p)


def test_pilot_contamination_cannot_pass_fresh_initializer_check(tmp_path, monkeypatch):
    root, _, _, _ = prepared(tmp_path, monkeypatch)
    fake_pilot_environment(monkeypatch)
    monkeypatch.setattr(study, '_trainer',
        lambda *a, **kw:FakeTrainer(loaded_pilot=bool(FakeTrainer.instances)))
    with pytest.raises(RuntimeError, match='Fresh initialization differs'):
        study.pilot(root)
    assert (root/'pilot/failure.json').exists()
    assert not (root/'pilot/manifest.json').exists()


def test_search_refuses_missing_pilot_before_any_gpu_admission(tmp_path, monkeypatch):
    root, _, _, _ = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(study, 'reservation', lambda:pytest.fail('GPU admitted before pilot'))
    with pytest.raises(FileNotFoundError):
        study.search_run(root)
    assert not (root/'search_start.json').exists()


def test_search_refuses_missing_scientific_anchors_even_after_pilot(tmp_path, monkeypatch):
    root, _, _, _ = prepared(tmp_path, monkeypatch)
    fake_pilot_environment(monkeypatch)
    study.pilot(root)
    monkeypatch.setattr(study, 'reservation', lambda:pytest.fail('GPU admitted before anchors'))
    with pytest.raises(FileNotFoundError):
        study.search_run(root)


def test_partial_search_is_preserved_and_not_restarted(tmp_path, monkeypatch):
    root, _, _, _ = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(study, 'pilot_receipt', lambda *a:{})
    monkeypatch.setattr(study, 'anchor_receipt', lambda *a:{})
    monkeypatch.setattr(study, 'reservation', lambda:pytest.fail('Partial science was relaunched'))
    (root/'search').mkdir()
    with pytest.raises(FileExistsError, match='preserved'):
        study.search_run(root)
