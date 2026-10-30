"""Use the live worker's guarded measurements in the legacy proof verifier."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

from experiments.structure_net import ncm_latest_semantic_worker_v2 as scoped
from experiments.structure_net import ncm_paper_data as paper

base, latest = scoped.base, scoped.latest
ROOT = latest.BASE/'20261029_latest_semantic_worker_v3'
RUNTIME = Path('/tmp/structure-net-ncm-latest-semantic-20261029-v3')
PLAN = Path('docs/07 - Status Reports/2026-10-29_ncm-legacy-verifier-cache.md')
TEST = Path('tests/structure_net/test_ncm_latest_semantic_worker_v3.py')
CPU = latest.BASE/'20261029_latest_semantic_worker_v3_CPU_v1.json'


@contextmanager
def guarded_legacy_verifier():
    original = paper.file_hash
    paper.file_hash = base.WorkerProofs().sha
    try:
        yield
    finally:
        paper.file_hash = original


def run_request(request):
    with guarded_legacy_verifier():
        return scoped.scoped_request(request)


def bind_legacy(proof):
    value = {**proof, 'config': {**proof['config'], 'legacy_producer_hash_validation': {
        'helper': 'experiments.structure_net.ncm_paper_data.file_hash',
        'policy': latest.cache.POLICY,
        'same_live_worker_measurements_only': True,
        'expected_digests_and_verifier_decisions_unchanged': True}},
        'input_hashes': dict(proof['input_hashes'])}
    for path in (Path(__file__).resolve(), PLAN, TEST, CPU):
        name, digest = str(path), latest.data.sha(path)
        if name in value['input_hashes'] and value['input_hashes'][name] != digest:
            raise ValueError('legacy verifier binding conflicts with existing inputs')
        value['input_hashes'][name] = digest
    return value


def install(start):
    checked = json.loads(CPU.read_text())
    if checked['actual_tool_exit_code'] != 0 or checked['tests_passed'] != 7:
        raise ValueError('four legacy verifier and three original hash contracts required')
    for path, digest in checked['artifacts'].items():
        if base.sha(path) != digest:
            raise ValueError('qualified legacy verifier source changed')
    start['legacy_verifier_source_sha256'] = base.sha(__file__)
    scoped.install(start)
    original = latest.previous.write_proof
    latest.previous.write_proof = lambda root, proof, files: original(root, bind_legacy(proof), files)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('serve', *base.STAGES))
    parser.add_argument('--arm', choices=latest.semantic.ARMS)
    args = parser.parse_args()
    scoped.recovery.install_roots()
    base.ROOT, base.RUNTIME, base.SOCKET = ROOT, RUNTIME, RUNTIME/'worker.sock'
    base.install, base.run_request = install, run_request
    if args.action == 'serve':
        if args.arm is not None:
            parser.error('serve has no arm')
        base.serve()
    else:
        start = json.loads((ROOT/'worker_start.json').read_text())
        if (start.get('legacy_verifier_source_sha256') != base.sha(__file__)
                or start.get('request_scope_source_sha256') != base.sha(scoped.__file__)):
            raise ValueError('exact scoped worker and legacy verifier implementations required')
        base.client(args.action, args.arm)


if __name__ == '__main__':
    main()
