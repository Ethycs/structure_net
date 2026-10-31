import pytest
from pathlib import Path
import runpy

from experiments.word_games.bonsai_runtime import NativeBonsai, NO_THINK_SUFFIX, command, totals


def test_standalone_bonsai_cannot_accidentally_enable_draft_or_thinking():
    cmd=command(18086,'test')
    assert cmd[cmd.index('--spec-type')+1]=='none'
    assert '--spec-draft-model' not in cmd
    assert cmd[cmd.index('--chat-template-kwargs')+1]=='{"enable_thinking":false}'
    assert cmd[cmd.index('--n-gpu-layers')+1]=='99'


def test_template_gate_rejects_open_reasoning_and_missing_input():
    client=NativeBonsai(1)
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'prompt':self.prompt}
    response=Response()
    class HTTP:
        def post(self,url,json,timeout):
            assert json['chat_template_kwargs']=={'enable_thinking':False}
            return response
    client.http=HTTP()
    response.prompt='<|im_start|>user\nuser input<|im_end|>\n'+NO_THINK_SUFFIX
    assert client.format_prompt('user input\n')==response.prompt
    for bad in ('user input<|im_start|>assistant\n<think>\n',NO_THINK_SUFFIX):
        response.prompt=bad
        with pytest.raises(ValueError):client.format_prompt('user input')


def test_throughput_reports_utf8_bytes_independently_of_tokenizer():
    r=totals([{'elapsed_seconds':2,'parsed':{},'response':{'content':'é','tokens_predicted':2,'stop_type':'eos',
              'timings':{'predicted_per_second':1,'predicted_ms':1000}}}])
    assert r['output_tokens']==1 and r['output_bytes']==2
    assert r['output_tokens_per_second']==.5 and r['output_bytes_per_second']==1


def test_registry_cannot_turn_selector_failure_or_small_suite_into_confirmed_accuracy():
    evidence=runpy.run_path(str(Path(__file__).resolve().parents[2]/
        'src/neural_architecture_lab/word_games_bonsai_hypothesis.py'))['BonsaiEvidence']
    args=dict(classification='higher_joint_task_success',successes=12,cases=12,rate=30,
              campaign='result.json',campaign_sha256='a'*64,report='report.md',report_sha256='b'*64)
    assert evidence(**args).record()['confirmed'] is False
    for change in ({'cases':0},{'successes':5},{'rate':float('nan')},{'successes':13}):
        with pytest.raises(ValueError): evidence(**{**args,**change})
