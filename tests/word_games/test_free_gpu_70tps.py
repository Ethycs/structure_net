import pytest

from experiments.word_games import free_gpu_70tps as run
from experiments.word_games import gpu_services
from experiments.word_games.common import read


def test_prompt_checkpoint_suppression_keeps_native_speculation_and_rollback():
    from experiments.word_games.mtp_no_context_70tps import CONFIGS
    for c in CONFIGS.values():
        cmd = run.target.command(c, None, 18088, 'test')
        assert cmd[cmd.index('--ctx-checkpoints') + 1] == '0'
        if c.mtp:
            assert cmd[cmd.index('--spec-type') + 1] == 'draft-mtp'
            assert cmd[cmd.index('--spec-draft-n-max') + 1] == str(c.mtp_tokens)
            assert not any('synth' in arg for arg in cmd)
            assert '--override-tensor' not in cmd


def test_expanded_layout_keeps_every_projection_on_gpu_including_mtp():
    for c in run.CONFIGS.values():
        cmd = run.target.command(c, None, 18088, 'test')
        assert cmd[cmd.index('--ctx-size') + 1] == '8192'
        if c.whole:
            assert cmd[cmd.index('--tensor-split') + 1] == '7.5,14,20.5'
            assert '--override-tensor' not in cmd
        if c.mtp:
            assert cmd[cmd.index('--threads') + 1] == '1'
            assert cmd[cmd.index('--spec-draft-n-max') + 1] == str(c.mtp_tokens)
            assert '--ctx-checkpoints' not in cmd


def test_restoration_tries_other_service_after_a_failure(monkeypatch, tmp_path):
    lease = gpu_services.ModelLease(tmp_path)
    lease.before = {s.name: {'name': s.name} for s in lease.services}
    lease.attempted = list(lease.services)
    calls = []
    class Response:
        status_code = 200
        def __init__(self, url): self.url = url
        def raise_for_status(self):
            if self.url.endswith('/qwen3-8b-gguf/load'): raise RuntimeError('simulated load failure')
        def json(self): return {'name': 'glm-ocr'}
    def post(url, **kwargs):
        calls.append(url)
        return Response(url)
    monkeypatch.setattr(gpu_services.requests, 'post', post)
    monkeypatch.setattr(gpu_services.requests, 'get', lambda url, **kwargs: Response(url))
    with pytest.raises(RuntimeError, match='restoration failed'):
        lease.restore()
    assert len(calls) == 2
    result = read(tmp_path / 'restored.json')
    assert not result['all_restored']
    assert result['services']['glm-ocr']['restored']
