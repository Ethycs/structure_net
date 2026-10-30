"""Restore request-local adapters while retaining only live measured hashes."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_latest_semantic_worker as base
from experiments.structure_net import ncm_latest_generation_recovery as recovery

latest = base.latest
ROOT = latest.BASE/'20261029_latest_semantic_worker_v2'
RUNTIME = Path('/tmp/structure-net-ncm-latest-semantic-20261029-v2')
PLAN = Path('docs/07 - Status Reports/2026-10-29_ncm-scoped-semantic-worker.md')
TEST = Path('tests/structure_net/test_ncm_latest_semantic_worker_v2.py')
CPU = latest.BASE/'20261029_latest_semantic_worker_v2_CPU_v1.json'
ORIGINAL_RUN = base.run_request
ORIGINAL_INSTALL = base.install


def scoped_request(request):
    previous_write = latest.previous.write_proof
    configure, proofs = latest.semantic.configure, latest.semantic.PROOFS
    hash_file, judge_request = latest.data.sha, latest.repaired.qualification.q.request
    try:
        return ORIGINAL_RUN(request)
    finally:
        latest.previous.write_proof = previous_write
        latest.semantic.configure, latest.semantic.PROOFS = configure, proofs
        latest.data.sha, latest.repaired.qualification.q.request = hash_file, judge_request


def install(start):
    checked = json.loads(CPU.read_text())
    if checked['actual_tool_exit_code'] != 0 or checked['tests_passed'] != 10:
        raise ValueError('request isolation plus all nine worker/cache contracts required')
    for path, digest in checked['artifacts'].items():
        if base.sha(path) != digest:
            raise ValueError('qualified scoped-worker source changed')
    start['request_scope_source_sha256'] = base.sha(__file__)
    ORIGINAL_INSTALL(start)
    previous_write = latest.previous.write_proof
    def write_proof(root, proof, files):
        value = {**proof, 'config': {**proof['config'], 'CPU_request_adapters_restored_after_return': True},
                 'input_hashes': dict(proof['input_hashes'])}
        for path in (Path(__file__).resolve(), PLAN, TEST, CPU):
            name, digest = str(path), latest.data.sha(path)
            if name in value['input_hashes'] and value['input_hashes'][name] != digest:
                raise ValueError('request-isolation binding conflicts with existing input')
            value['input_hashes'][name] = digest
        return previous_write(root, recovery.bind_recovery(value), files)
    latest.previous.write_proof = write_proof


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('serve', *base.STAGES))
    parser.add_argument('--arm', choices=latest.semantic.ARMS)
    args = parser.parse_args()
    recovery.install_roots()
    base.ROOT, base.RUNTIME, base.SOCKET = ROOT, RUNTIME, RUNTIME/'worker.sock'
    base.install, base.run_request = install, scoped_request
    if args.action == 'serve':
        if args.arm is not None:
            parser.error('serve has no arm')
        base.serve()
    else:
        start = json.loads((ROOT/'worker_start.json').read_text())
        if start.get('request_scope_source_sha256') != base.sha(__file__):
            raise ValueError('exact scoped worker implementation required')
        base.client(args.action, args.arm)


if __name__ == '__main__':
    main()
