"""Requalify the same judge protocol after the user's endpoint provider repair."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from experiments.neural_architecture_lab import ncm_persistent_judge_gateway_v2 as gateway
from experiments.structure_net import ncm_isolated_judge_qualification as qualification

base = gateway.base
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-repaired-judge-route.md')
TEST = Path('tests/structure_net/test_ncm_repaired_judge_route.py')
CPU = base.BASE/'20261028_repaired_judge_route_CPU_v1.json'
OLD_SERVICE = gateway.SERVICE
RETIREMENT = base.BASE/'20261028_paused_semantic_consumer_retirement_receipt_v1.json'
INCIDENT = base.BASE/'20261028_user_resume_endpoint_probe_v1'
SERVICE = base.BASE/'20261028_repaired_SQLite_gateway_service_v1'
PROBE = base.BASE/'20261028_repaired_SQLite_gateway_probe_v1'
UNIT = 'structure-net-ncm-repaired-sqlite-judge-v1.service'
IDENTITY = 'ncm-persistent-isolated-SQLite-judge-transport-v1'
ROOTS = {(role, stage): base.BASE/f'20261028_repaired_SQLite_judge_{role}_{stage}_v1'
         for role in qualification.transport.MODELS for stage in ('qualification', 'audit')}
EXPECTED_CHANGES = {
    '/home/rabbit/llm_endpoint/llm_endpoint/providers/codex_pool.py':
        ('c0ebbf1f58a0c165f95fe64c973258be8d4cce034ea3b16598be0ce88c07b038',
         '6c015da5e286dfc604cccf039cc0a59033c1053ba87919611bde87cc5447b54c'),
    '/home/rabbit/llm_endpoint/llm_endpoint/providers/codex_sdk.py':
        ('5c1a1ff50564cdad7346e1b85a253db38a252685ebe9570e7835a79cdae37e62',
         'd73368e4325ab51e2d9126c241262da8635689f03aac9a621b84b3addbedbabd')}
INSTALLED = False


def check_changes(before, current):
    if set(before) != set(current):
        raise ValueError('same original source inventory required')
    changed = {p: (digest, current[p]) for p, digest in before.items() if current[p] != digest}
    if changed != EXPECTED_CHANGES:
        raise ValueError('only the two reviewed provider repairs are admitted')
    return changed


def require():
    gateway.require()
    checked = json.loads(CPU.read_text())
    base.verify(checked['artifacts'])
    if checked['actual_tool_exit_code'] != 0 or checked['tests_passed'] != 8:
        raise ValueError('four repair and four unchanged semantic contracts required')
    receipt = json.loads(RETIREMENT.read_text())
    base.verify(receipt['artifacts'])
    if (receipt['actual_tool_exit_code'] != 143 or receipt['application_judge_calls'] != 0
            or receipt['quality_failure'] is not False):
        raise ValueError('observed no-request retirement of the paused consumer required')
    old = json.loads((OLD_SERVICE/'producer.json').read_text())
    check_changes(old['source_hashes'], {p: base.sha(p) for p in old['source_hashes']})


def extend_producer(proof):
    value = {**proof, 'source_hashes': dict(proof['source_hashes']),
             'input_hashes': dict(proof['input_hashes']), 'endpoint_repair': {
                 'original_service_producer': str(OLD_SERVICE/'producer.json'),
                 'provider_changes': EXPECTED_CHANGES, 'fresh_service_invocation': True,
                 'requalification_required': True, 'old_judgments_preserved': True}}
    value['source_hashes'][str(Path(__file__).resolve())] = base.sha(Path(__file__).resolve())
    for path in (PLAN, TEST, CPU, RETIREMENT, OLD_SERVICE/'producer.json',
                 INCIDENT/'failure.json', INCIDENT/'actual_tool_receipt.json'):
        value['input_hashes'][str(path)] = base.sha(path)
    return value


def install():
    global INSTALLED
    if INSTALLED:
        return
    require()
    gateway.SERVICE, gateway.ROOT, gateway.UNIT = SERVICE, PROBE, UNIT
    gateway.PLAN, gateway.TEST = PLAN, TEST
    original_producer = gateway.producer_proof
    gateway.producer_proof = lambda proof: extend_producer(original_producer(proof))
    gateway.install()
    qualification.neutral, qualification.ROOTS = base, ROOTS
    qualification.PLAN, qualification.TEST = PLAN, TEST
    # The wire protocol, endpoint and judge aliases are identical. The new
    # service/qualification producers bind the repaired implementation.
    qualification.transport.IDENTITY = IDENTITY
    qualification.transport.credential = gateway.credential
    INSTALLED = True


def consumer_arguments(family, stage, arm):
    if stage not in ('judge_pilot', 'grade', 'audit'):
        raise ValueError('only future semantic stages use this route')
    if family == 'task' and arm in ('existing', 'expanded'):
        return [stage, '--arm', arm]
    if family == 'hierarchy' and arm in ('tree', 'dense'):
        return [family, stage, '--arm', arm]
    if family == 'recovered' and arm in ('new_codec_only', 'parent_with_source'):
        return [family, stage, '--arm', arm]
    raise ValueError('fixed evaluated candidate family and arm required')


def consume(family, stage, arm):
    arguments = consumer_arguments(family, stage, arm)
    from experiments.structure_net import ncm_persistent_task_verified as verified
    verified.route.qualification = SimpleNamespace(install=install)
    verified.route.PLAN = PLAN
    old_write = verified.route.consumer.task.write_proof
    def write_proof(root, proof, files):
        value = {**proof, 'config': {**proof['config'], 'repaired_endpoint_qualified': True},
                 'input_hashes': dict(proof['input_hashes'])}
        for path in (Path(__file__).resolve(), PLAN, TEST, CPU, RETIREMENT):
            name, digest = str(path), verified.data.sha(path)
            if name in value['input_hashes'] and value['input_hashes'][name] != digest:
                raise ValueError('repair proof conflicts with existing producer inputs')
            value['input_hashes'][name] = digest
        return old_write(root, value, files)
    verified.route.consumer.task.write_proof = write_proof
    if family == 'task':
        sys.argv = [verified.__file__, *arguments]
        verified.main()
    else:
        from experiments.structure_net import ncm_persistent_candidate_development as candidates
        sys.argv = [candidates.__file__, *arguments]
        candidates.main()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('serve', 'probe', 'qualify', 'audit', 'consume'))
    parser.add_argument('--role', choices=tuple(qualification.transport.MODELS))
    parser.add_argument('--family', choices=('task', 'hierarchy', 'recovered'))
    parser.add_argument('--stage', choices=('judge_pilot', 'grade', 'audit'))
    parser.add_argument('--arm')
    args = parser.parse_args()
    if args.action in ('qualify', 'audit') and not args.role:
        parser.error('qualification and audit require --role')
    if args.action == 'consume' and not all((args.family, args.stage, args.arm)):
        parser.error('consume requires --family, --stage and --arm')
    install()
    if args.action == 'serve':
        base.serve()
    elif args.action == 'probe':
        base.probe()
    elif args.action == 'consume':
        consume(args.family, args.stage, args.arm)
    else:
        root = ROOTS[args.role, 'qualification' if args.action == 'qualify' else 'audit']
        existed = root.exists()
        try:
            qualification.qualify(args.role) if args.action == 'qualify' else qualification.audit(args.role)
        except Exception as error:
            if root.exists() and not existed and not (root/'failure.json').exists():
                base.write(root/'failure.json', {'type': type(error).__name__, 'message': str(error),
                                                'first_judgments_preserved': True, 'automatic_retry': False})
            raise


if __name__ == '__main__':
    main()
