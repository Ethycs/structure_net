"""Use the qualified local transport for new task-arm judgments only."""
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

from experiments.structure_net import ncm_isolated_judge_qualification as qualification
from experiments.structure_net import ncm_task_supervision_development as task
from experiments.structure_net import ncm_recovered_review_lineage as history

transport, q = qualification.transport, qualification.q
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-isolated-task-evaluation.md')
TEST = Path('tests/structure_net/test_ncm_isolated_task_development.py')
CPU = qualification.BASE/'20261028_isolated_task_evaluation_CPU_v1.json'
STAGES = ('judge_pilot', 'grade', 'audit')
REQUEST_LOCK = threading.Lock()


def serialized_request(packet, role, root):
    # Queue before opening HTTP requests so the server's one slot is never overrun.
    with REQUEST_LOCK:
        row = transport.request(packet, role, root)
        transport.replay_row(packet, role, root, row)
        return row


def check_role(role, result, proof):
    if (result.get('status') != 'local_judge_wire_firsts_and_controls_replayed'
            or result.get('judge') != transport.MODELS[role]
            or result.get('qualification_passed') is not True
            or result.get('correct') != 50 or result.get('controls') != 50
            or result.get('cases') != 13 or result.get('valid_requests') != 13
            or proof['config'].get('endpoint') != transport.ENDPOINT
            or proof['config'].get('transport_identity') != transport.IDENTITY
            or proof['config'].get('rubric') != qualification.previous.profile.RUBRIC):
        raise ValueError('both unchanged judges require complete audited local qualification')


def bindings():
    values = {}
    for role in transport.MODELS:
        root = qualification.ROOTS[role, 'audit']
        result = qualification.completed(root); proof = q.read(root/'producer.json')
        qualification.verify(proof['source_hashes']); qualification.verify(proof['input_hashes'])
        check_role(role, result, proof)
        for stage in ('qualification', 'audit'):
            for name in ('producer.json', 'result.json', 'actual_tool_receipt.json'):
                p = qualification.ROOTS[role, stage]/name; values[str(p)] = qualification.previous.file_hash(p)
    for p in (Path(__file__), PLAN, TEST, CPU, qualification.PLAN, Path(qualification.__file__), Path(transport.__file__)):
        values[str(p)] = qualification.previous.file_hash(p)
    return values


def augment_proof(proof, bound):
    if proof['stage'] not in STAGES: raise ValueError('only future semantic stages use the local route')
    if proof['config']['judges'] != transport.MODELS or proof['config']['rubric'] != qualification.previous.profile.RUBRIC:
        raise ValueError('original judge identities and rubric must remain unchanged')
    proof['config'].update(judge_endpoint=transport.ENDPOINT, judge_transport=transport.IDENTITY,
        wire_responses_preserved=True, previous_first_judgments_preserved=True,
        active_judge_requests=1, backend_timeout_seconds=160, client_timeout_seconds=180)
    for name, digest in bound.items():
        if name in proof['input_hashes'] and proof['input_hashes'][name] != digest:
            raise ValueError('consumer proof conflicts with admitted local evidence')
        proof['input_hashes'][name] = digest
    return proof


def audit_local_entries(cache, roots):
    """Check every new local first while leaving all historical entries intact."""
    checked = 0
    for entry in cache.values():
        origin = Path(entry['origin'])
        if any(origin.is_relative_to(root) for root in roots):
            transport.replay_row(entry['packet'], entry['role'], origin, entry['row'])
            checked += 1
    return checked


def install(bound):
    original_configure, original_write = task.configure, task.write_proof
    roots = [stages[s] for stages in task.ROOTS.values() for s in ('judge_pilot', 'grade')]
    def write_proof(root, proof, files):
        qualification.verify(bound)
        return original_write(root, augment_proof(proof, bound), files)
    def configure(arm):
        training, evaluation, trained, environment = original_configure(arm)
        if training.q is not q: raise ValueError('original shared semantic runtime required')
        ancestor = evaluation.lineage
        def require_judges():
            ancestor.require_judges()
            if bindings() != bound: raise ValueError('local route admission changed')
        def audit_cache(cache):
            ancestor.audit_cache(cache); audit_local_entries(cache, roots)
        evaluation.lineage = SimpleNamespace(COUNT=ancestor.COUNT, base_cache=ancestor.base_cache,
            inputs=lambda: [*ancestor.inputs(), *map(Path, bound)], configure=ancestor.configure,
            require_judges=require_judges, uncertain_allowances=ancestor.uncertain_allowances,
            audit_cache=audit_cache)
        q.request = serialized_request
        return training, evaluation, trained, environment
    task.configure, task.write_proof = configure, write_proof


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU/network-only semantic stages required')
    if len(sys.argv) != 4 or sys.argv[1] not in STAGES or sys.argv[2] != '--arm' or sys.argv[3] not in task.study.ARMS:
        raise ValueError('use judge_pilot|grade|audit --arm existing|expanded')
    checked = q.read(CPU); qualification.verify(checked['artifacts'])
    if checked['actual_tool_exit_code'] != 0 or checked['tests_passed'] != 4:
        raise ValueError('local application lineage and admission contracts required')
    history.require_preflight(); history.install()
    install(bindings())
    task.main()


if __name__ == '__main__': main()
