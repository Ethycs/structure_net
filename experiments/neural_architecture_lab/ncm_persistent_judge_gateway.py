"""Isolated persistent gateway with explicit MCP isolation and no exec fallback."""
import argparse
import json
from pathlib import Path

from experiments.neural_architecture_lab import ncm_isolated_judge_gateway_v2 as base
from experiments.neural_architecture_lab import ncm_persistent_judge_launcher as launcher

PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-persistent-judge-mcp-isolation.md')
TEST = Path('tests/structure_net/test_ncm_persistent_judge_gateway.py')
CPU = base.BASE/'20261028_persistent_no_MCP_gateway_CPU_v1.json'
WRAPPER_CPU = base.BASE/'20261028_judge_CLI_without_MCP_CPU_v1.json'
DIRECT = base.BASE/'20261028_codex_no_mcp_host_probe_v1'
SERVICE = base.BASE/'20261028_persistent_no_MCP_gateway_service_v1'
ROOT = base.BASE/'20261028_persistent_no_MCP_gateway_probe_v1'
UNIT = 'structure-net-ncm-persistent-no-mcp-judge-v1.service'
SETTINGS = {**base.SETTINGS, 'LLM_ENDPOINT_CODEX_PERSISTENT': '1',
    'LLM_ENDPOINT_CODEX_MAX_RSS_MB': '1024', 'LLM_ENDPOINT_CODEX_MAX_TURNS': '16'}
INSTALLED = False


def require():
    for path, count in ((CPU, 4), (WRAPPER_CPU, 3)):
        value = json.loads(path.read_text()); base.verify(value['artifacts'])
        if value['actual_tool_exit_code'] != 0 or value['tests_passed'] != count:
            raise ValueError('successful gateway and wrapper contracts required')
    receipt = json.loads((DIRECT/'actual_tool_receipt.json').read_text())
    base.verify(receipt['artifacts'])
    if (receipt['actual_tool_exit_code'] != 0
            or receipt['artifacts'].get(str(DIRECT.resolve()/'result.json'), receipt['artifacts'].get(str(DIRECT/'result.json')))
                != base.sha(DIRECT/'result.json')):
        raise ValueError('actual completed direct-host probe required')
    value = json.loads((DIRECT/'result.json').read_text())
    if not all(value[k] for k in ('both_exact_OK','isolated_host_shutdown','inputs_unchanged')) or value['maximum_MCP_descendants'] != 0:
        raise ValueError('isolated persistent host did not pass its availability probe')


def install():
    global INSTALLED
    if INSTALLED: return
    base.SERVICE,base.ROOT,base.UNIT = SERVICE,ROOT,UNIT
    base.CLI = Path(launcher.__file__).resolve()
    base.PLAN,base.TEST,base.SETTINGS = PLAN,TEST,dict(SETTINGS)
    old_write = base.write
    def write(path,value):
        if Path(path) == SERVICE/'producer.json':
            value = {**value, 'schema':'ncm-persistent-judge-without-MCP-v1', 'exec_fallback_allowed':False}
            value['source_hashes'].update({str(p):base.sha(p) for p in (Path(__file__).resolve(),launcher.WRAPPER)})
            value['input_hashes'] = {str(p):base.sha(p) for p in (CPU,WRAPPER_CPU,DIRECT/'result.json',DIRECT/'actual_tool_receipt.json')}
        return old_write(path,value)
    base.write = write; INSTALLED = True


def credential():
    install(); require()
    proof = json.loads((SERVICE/'producer.json').read_text())
    if proof['schema'] != 'ncm-persistent-judge-without-MCP-v1' or proof['exec_fallback_allowed']:
        raise ValueError('the fixed persistent-only service is required')
    base.verify(proof['input_hashes'])
    return base.credential()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('serve','probe'))
    args=parser.parse_args();require();install()
    base.serve() if args.stage=='serve' else base.probe()
