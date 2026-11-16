"""Reservation ordering and append-only process admission; no real subprocess."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.structure_net import jspace_phi_launch as launch


def prepared(tmp_path):
    root=tmp_path/'prepared model';root.mkdir();(root/'plan.json').write_text(json.dumps({'fingerprint':'f'*64}))
    return root


def test_reservation_precedes_worker_python_and_uses_one_card_argument(tmp_path):
    cmd=launch.command(tmp_path/'has spaces')
    assert cmd[:4]==['/usr/bin/python3','/usr/local/bin/inference-ctl','lease','run']
    assert cmd.count('--allocation')==1 and cmd[cmd.index('--allocation')+1]=='GPU-0370fdb0=11264'
    assert cmd.index('--')<cmd.index('python') and cmd[-1]==str((tmp_path/'has spaces').resolve())
    assert '--locked' in cmd and '--offline' in cmd and '--no-install' in cmd


def test_completed_subprocess_receipt_is_not_scientific_admission(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setenv('IC_LEASE','stale inherited lease')
    monkeypatch.setenv('STRUCTURENET_GPU_LEASE','stale inherited lease')
    root=prepared(tmp_path);calls=[]
    def execute(command,**kw):
        calls.append(command);assert kw['env']['CUDA_VISIBLE_DEVICES']=='' and 'IC_LEASE' not in kw['env']
        assert 'STRUCTURENET_GPU_LEASE' not in kw['env']
        assert kw['check'] is False
        (root/'native').mkdir();(root/'native/manifest.json').write_text('{}')
        kw['stdout'].write(b'worker log\n');return SimpleNamespace(returncode=0)
    result=launch.launch(root,tmp_path/'receipt',executor=execute)
    assert len(calls)==1 and result['returncode']==0 and not result['scientific_artifacts_verified']
    assert result['native_manifest_sha256']==launch.sha(root/'native/manifest.json')
    with pytest.raises(FileExistsError):launch.launch(root,tmp_path/'other',executor=execute)


def test_failure_is_preserved_and_never_retried(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');root=prepared(tmp_path);calls=[]
    def execute(command,**kw):calls.append(command);return SimpleNamespace(returncode=1)
    with pytest.raises(RuntimeError,match='preserved receipt'):launch.launch(root,tmp_path/'receipt',executor=execute)
    assert len(calls)==1 and launch.read(tmp_path/'receipt/exit.json')['returncode']==1
    with pytest.raises(FileExistsError,match='Prior launch'):launch.launch(root,tmp_path/'different',executor=execute)
    assert len(calls)==1


def test_interrupted_attempt_blocks_duplicate_even_without_worker_artifacts(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');root=prepared(tmp_path)
    def interrupted(*a,**kw):raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):launch.launch(root,tmp_path/'receipt',executor=interrupted)
    assert (tmp_path/'receipt/started.json').exists() and not (tmp_path/'receipt/exit.json').exists()
    with pytest.raises(FileExistsError,match='Prior launch'):launch.launch(root,tmp_path/'other',executor=interrupted)


def test_no_cuda_visibility_or_duplicate_lock(tmp_path,monkeypatch):
    root=prepared(tmp_path);monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    with pytest.raises(ValueError,match='CUDA hidden'):launch.launch(root,tmp_path/'receipt')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    with launch.exclusive(root),pytest.raises(RuntimeError,match='Another Phi'):launch.launch(root,tmp_path/'receipt')


def test_zero_exit_without_worker_manifest_does_not_claim_completion(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');root=prepared(tmp_path)
    with pytest.raises(RuntimeError):launch.launch(root,tmp_path/'receipt',executor=lambda *a,**kw:SimpleNamespace(returncode=0))
    assert launch.read(tmp_path/'receipt/exit.json')['status']=='worker_exit_without_manifest'
