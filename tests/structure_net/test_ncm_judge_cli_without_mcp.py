from copy import deepcopy

import pytest

from experiments.neural_architecture_lab import ncm_judge_cli_without_mcp as cli


def config():
    return {'mcp_servers': {'central_dev': {'url': 'https://example.invalid'},
            'servant-chat': {'command': 'server'}}, 'model_reasoning_effort': 'medium'}


def test_app_server_flags_disable_each_server_without_changing_other_arguments():
    value = config(); before = deepcopy(value)
    original = ['app-server','-c','mcp_servers={}']
    result = cli.command(original,value)
    assert result == [str(cli.CLI),*original,'-c','mcp_servers.central_dev.enabled=false',
        '-c','mcp_servers.servant-chat.enabled=false']
    assert value == before and original == ['app-server','-c','mcp_servers={}']


def test_exec_fallback_keeps_model_prompt_and_output_contract():
    args = ['exec','--model','gpt-5.6-sol','--output-last-message','/tmp/answer.txt','-']
    assert cli.command(args,config())[1:1+len(args)] == args


def test_changed_server_population_and_unexpected_subcommands_are_rejected():
    for change in ('added','removed'):
        value = config()
        if change == 'added':value['mcp_servers']['unreviewed'] = {'command':'server'}
        else:del value['mcp_servers']['central_dev']
        with pytest.raises(ValueError):cli.command(['app-server'],value)
    for args in ([],['login'],['exec;'],['-c','model="other"','app-server']):
        with pytest.raises(ValueError):cli.command(args,config())
