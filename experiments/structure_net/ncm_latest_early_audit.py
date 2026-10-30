"""Start each independent audit's own verification before grading completes."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from experiments.structure_net import ncm_latest_generation_recovery as recovery

latest = recovery.latest
PLAN = Path('docs/07 - Status Reports/2026-10-29_ncm-overlapped-independent-audits.md')
TEST = Path('tests/structure_net/test_ncm_latest_early_audit.py')
CPU = latest.BASE/'20261029_latest_early_audit_CPU_v1.json'
MEASURED = {}
OriginalProofs = latest.cache.ProcessProofs


class AuditProofs(OriginalProofs):
    def __init__(self):
        super().__init__()
        self.cache = MEASURED


def grade_ready(root, proofs):
    receipt = root/'actual_tool_receipt.json'
    if (root/'failure.json').exists():
        raise ValueError('preserve failed grade; no audit retry')
    if not receipt.exists():
        return False
    value = proofs.read(receipt)
    if value['actual_tool_exit_code'] != 0:
        raise ValueError('successful actual grade completion required')
    result = root/'result.json'
    bindings = {str(Path(p).absolute()): h for p, h in value['artifacts'].items()}
    if bindings.get(str(result.absolute())) != proofs.sha(result):
        raise ValueError('grade receipt must bind exact result bytes')
    return True


def bind_audit(proof, root):
    value = {**proof, 'config': {**proof['config'], 'independent_audit_admission': {
        'started_before_grade': True, 'PID': os.getpid(), 'initial_measured_hashes': 0,
        'hashes_received_from_other_processes': False,
        'only_own_live_stat_guarded_measurements_reused': True}},
        'input_hashes': dict(proof['input_hashes'])}
    for path in (Path(__file__).resolve(), PLAN, TEST, CPU,
                 root/'start.json', root/'warmup_result.json'):
        name, digest = str(path), latest.data.sha(path)
        if name in value['input_hashes'] and value['input_hashes'][name] != digest:
            raise ValueError('independent audit binding conflicts with existing producer')
        value['input_hashes'][name] = digest
    return value


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--arm', required=True, choices=latest.semantic.ARMS)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('PIXI_ENVIRONMENT_NAME') != 'ncm':
        raise ValueError('fresh CPU-only locked Pixi audit process required')
    if MEASURED:
        raise ValueError('independent audit must start with an empty measured cache')
    checked = latest.data.read(CPU); latest.data.verify(checked['artifacts'])
    if checked['actual_tool_exit_code'] != 0 or checked['tests_passed'] != 8:
        raise ValueError('five early-audit and three unchanged hash contracts required')
    root = latest.BASE/f'20261029_latest_recovered_{args.arm}_early_audit_v1'
    if root.exists() or latest.SEMANTIC[args.arm]['audit'].exists():
        raise ValueError('fresh independent audit invocation required')
    latest.cache.ProcessProofs = AuditProofs
    recovery.install_roots(); latest.install()
    root.mkdir()
    latest.data.write(root/'start.json', {'PID': os.getpid(), 'arm': args.arm,
        'initial_measured_hashes': 0, 'cross_process_cache': False,
        'new_model_calls': 0, 'new_judge_requests': 0})
    try:
        _, _, _, _, history, trained, _, cohorts, _, _ = latest.generated.context()
        latest.data.write(root/'warmup_result.json', {
            'status': 'independent_audit_admission_verified', 'PID': os.getpid(),
            'cases': sum(map(len, cohorts.values())), 'canonical_first_judgments': len(history),
            'checkpoint_sha256': trained['checkpoint_sha256'], 'new_model_calls': 0,
            'new_judge_requests': 0, 'task_accuracy': None, 'final_test_opened': False})
        print(json.dumps({'status': 'independent_audit_waiting_for_actual_grade',
                          'arm': args.arm, 'PID': os.getpid()}), flush=True)
        proofs = AuditProofs()
        while not grade_ready(latest.SEMANTIC[args.arm]['grade'], proofs):
            watched = [recovery.ROOTS['outputs'], *[
                latest.SEMANTIC[a][s] for a in latest.semantic.ARMS
                for s in ('inference_pilot', 'outputs', 'judge_pilot', 'grade', 'audit')]]
            if any((p/'failure.json').exists() for p in watched):
                raise ValueError('upstream failure retained; independent audit stopped')
            time.sleep(5)
        original = latest.previous.write_proof
        latest.previous.write_proof = lambda r, proof, files: original(r, bind_audit(proof, root), files)
        sys.argv = [recovery.__file__, 'audit', '--arm', args.arm]
        recovery.main()
        latest.data.write(root/'completion.json', {'status': 'independent_audit_completed',
            'PID': os.getpid(), 'audit_result': str(latest.SEMANTIC[args.arm]['audit']/'result.json'),
            'new_model_calls': 0, 'new_judge_requests': 0})
    except Exception as error:
        latest.data.write(root/'failure.json', {'type': type(error).__name__,
            'message': str(error), 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
