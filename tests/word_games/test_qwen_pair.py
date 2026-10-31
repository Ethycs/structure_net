from argparse import Namespace
from pathlib import Path
import runpy

import pytest

from experiments.word_games.check_qwen_tokenizers import compatibility
from experiments.word_games.qwen_pair import MODEL, HYPOTHESIS, PLAN
from experiments.word_games.qwen_runtime import NativeQwen, server_command


def tokenizer():
    return {'tokenizer.ggml.tokens': ['a','b','end'], 'tokenizer.ggml.token_type': [1,1,3],
            'tokenizer.ggml.merges': ['a b'], 'tokenizer.ggml.pre': 'qwen2',
            'tokenizer.ggml.eos_token_id': 2, 'tokenizer.chat_template': 'template'}


def test_shared_id_gate_allows_only_small_nontext_padding():
    a, b = tokenizer(), tokenizer()
    a['tokenizer.ggml.tokens'] += ['[PAD]']
    a['tokenizer.ggml.token_type'] += [5]
    assert compatibility(a,b)['compatible']
    a['tokenizer.ggml.token_type'][-1] = 1
    assert not compatibility(a,b)['compatible']


@pytest.mark.parametrize('key,value', [
    ('tokenizer.ggml.tokens', ['b','a','end']),
    ('tokenizer.ggml.token_type', [1,3,3]),
    ('tokenizer.ggml.merges', ['b a']),
    ('tokenizer.ggml.eos_token_id', 1),
    ('tokenizer.ggml.pre', 'gpt2'),
    ('tokenizer.chat_template', 'other'),
])
def test_shared_id_gate_rejects_semantic_differences(key, value):
    a, b = tokenizer(), tokenizer()
    b[key] = value
    assert not compatibility(a,b)['compatible']


def test_qwen_draft_uses_second_gpu_and_target_always_remains_14b():
    args = Namespace(port=18085, draft=3)
    cmd = server_command(args, 'test')
    assert cmd[cmd.index('--device')+1] == 'CUDA0'
    assert cmd[cmd.index('--spec-draft-device')+1] == 'CUDA1'
    assert '14b' in cmd[cmd.index('-m')+1]
    assert '3b' in cmd[cmd.index('--spec-draft-model')+1]
    assert cmd[cmd.index('--alias')+1] == MODEL
    assert '--no-cache-prompt' in cmd
    args.draft = 0
    assert '--spec-draft-model' not in server_command(args, 'test')


def test_qwen_uses_server_native_template_and_rejects_phi_marker():
    client = NativeQwen(1)
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'prompt': self.prompt}
    response = Response()
    class HTTP:
        def post(self, url, json, timeout):
            assert url.endswith('/apply-template')
            assert json['messages'] == [{'role':'user','content':'test'}]
            return response
    client.http = HTTP()
    response.prompt = ('<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n'
                       '<|im_start|>user\ntest<|im_end|>\n<|im_start|>assistant\n')
    assert client.format_prompt('test') == response.prompt
    response.prompt = '<|im_start|>assistant<|im_sep|>'
    with pytest.raises(ValueError): client.format_prompt('test')


def test_campaign_rejects_qwen_identity_under_phi_plan(tmp_path):
    from experiments.word_games.campaign import run
    from experiments.word_games.common import write
    identity = tmp_path / 'identity.json'
    write(identity, {'model': MODEL, 'ready': True})
    with pytest.raises(ValueError, match='requested model'):
        run(tmp_path/'wrong_model', identity, client=object())
    with pytest.raises(ValueError, match='own hypothesis'):
        run(tmp_path/'wrong_plan', identity, client=object(), expected_model=MODEL)
    with pytest.raises(ValueError, match='explicit transport'):
        run(tmp_path/'wrong_transport', identity, expected_model=MODEL, hypothesis_id=HYPOTHESIS, plan=PLAN)


def test_registry_keeps_pilot_claims_conservative():
    evidence = runpy.run_path(str(Path(__file__).resolve().parents[2] /
        'src/neural_architecture_lab/word_games_qwen_pair_hypothesis.py'))['QwenPairEvidence']
    args = dict(classification='observed_pilot_gain_no_suite_regression', speed_ratio=1.2,
                baseline_successes=10, draft_successes=10, campaign='campaign.json',
                campaign_sha256='a'*64, report='report.md', report_sha256='b'*64)
    record = evidence(**args).record()
    assert not record['confirmed'] and record['paired_pilot_requests'] == 3
    for changes in ({'speed_ratio': .9}, {'draft_successes':9}, {'baseline_successes':13}, {'speed_ratio':float('nan')}):
        with pytest.raises(ValueError): evidence(**{**args, **changes})
