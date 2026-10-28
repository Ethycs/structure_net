"""Persistent MCP-free judge with an independently initialized SQLite runtime."""
import argparse
import json
from pathlib import Path

from experiments.neural_architecture_lab import ncm_persistent_judge_gateway as previous

base = previous.base
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-judge-sqlite-isolation.md')
TEST = Path('tests/structure_net/test_ncm_persistent_judge_gateway_v2.py')
CPU = base.BASE/'20261028_persistent_SQLite_gateway_CPU_v1.json'
WARMUP = base.BASE/'20261028_persistent_gateway_sqlite_config_diagnostic_v1'
SERVICE = base.BASE/'20261028_persistent_SQLite_gateway_service_v1'
ROOT = base.BASE/'20261028_persistent_SQLite_gateway_probe_v1'
UNIT = 'structure-net-ncm-persistent-sqlite-judge-v1.service'
CLI = Path(__file__).with_name('ncm_persistent_judge_launcher_v2.py').resolve()
INSTALLED = False


def check_warmup(value):
    response=value.get('initialize_response') or {}
    if (response.get('id') != 1 or not response.get('result')
            or value.get('model_requests') != 0 or value.get('process_closed') is not True):
        raise ValueError('completed isolated initialization without model requests required')


def require():
    previous.require()
    checked=json.loads(CPU.read_text());base.verify(checked['artifacts'])
    if checked['actual_tool_exit_code'] != 0 or checked['tests_passed'] != 4:
        raise ValueError('new SQLite isolation contracts required')
    receipt=json.loads((WARMUP/'actual_tool_receipt.json').read_text());base.verify(receipt['artifacts'])
    if receipt['actual_tool_exit_code'] != 0 or receipt['artifacts'].get(str(WARMUP/'result.json')) != base.sha(WARMUP/'result.json'):
        raise ValueError('actual isolated initialization receipt required')
    check_warmup(json.loads((WARMUP/'result.json').read_text()))


def install():
    global INSTALLED
    if INSTALLED:return
    previous.SERVICE,previous.ROOT,previous.UNIT=SERVICE,ROOT,UNIT
    previous.PLAN,previous.TEST=PLAN,TEST
    previous.install()
    base.CLI=CLI
    old_write=base.write
    def write(path,value):
        if Path(path)==SERVICE/'producer.json':
            value=producer_proof(value)
            # The previous writer replaces input_hashes. Write this complete
            # successor proof directly, preserving every prerequisite instead.
            with Path(path).open('x') as stream:
                json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
            return
        return old_write(path,value)
    base.write=write;INSTALLED=True


def producer_proof(value):
    value={**value,'schema':'ncm-persistent-isolated-SQLite-judge-v1','exec_fallback_allowed':False,
           'isolated_SQLite_runtime':'/data2/structure_net/ncm_judge_runtime_20261028_v3/sqlite'}
    value['source_hashes']={**value['source_hashes'],**{str(p):base.sha(p) for p in
        (Path(__file__).resolve(),Path(previous.__file__).resolve(),
         Path(previous.launcher.__file__).resolve(),previous.launcher.WRAPPER)}}
    paths=(previous.CPU,previous.WRAPPER_CPU,previous.DIRECT/'result.json',previous.DIRECT/'actual_tool_receipt.json',
           CPU,WARMUP/'producer.json',WARMUP/'result.json',WARMUP/'actual_tool_receipt.json')
    value['input_hashes']={str(p):base.sha(p) for p in paths}
    return value


def credential():
    install();require()
    proof=json.loads((SERVICE/'producer.json').read_text())
    if (proof.get('schema') != 'ncm-persistent-isolated-SQLite-judge-v1' or proof.get('exec_fallback_allowed') is not False
            or proof.get('isolated_SQLite_runtime') != '/data2/structure_net/ncm_judge_runtime_20261028_v3/sqlite'):
        raise ValueError('fixed isolated SQLite runtime required')
    base.verify(proof['input_hashes'])
    return base.credential()


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('serve','probe'))
    args=parser.parse_args();require();install()
    base.serve() if args.stage=='serve' else base.probe()
