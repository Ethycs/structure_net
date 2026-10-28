"""Qualified semantic transport for fixed hierarchy and recovered candidates."""
import argparse
import fcntl
import os
from pathlib import Path
from types import SimpleNamespace

from experiments.structure_net import ncm_persistent_task_verified as verified
from experiments.structure_net import ncm_source_hierarchy_development as hierarchy
from experiments.structure_net import ncm_recovered_development_semantics as recovered

route=verified.route
consumer=route.consumer
data=verified.data
FAMILIES={'hierarchy':hierarchy,'recovered':recovered}
PLAN=Path('docs/07 - Status Reports/2026-10-28_ncm-qualified-candidate-evaluation.md')
TEST=Path('tests/structure_net/test_ncm_persistent_candidate_development.py')
CPU=data.ROOT/'20261028_persistent_candidate_evaluation_CPU_v1.json'
STAGES=('judge_pilot','grade','audit')


def select(family,arm):
    if family not in FAMILIES or arm not in FAMILIES[family].ROOTS:
        raise ValueError('fixed hierarchy or recovered readout candidate required')
    return FAMILIES[family]


def semantic_roots():
    return [stages[stage] for module in (consumer.task,*FAMILIES.values())
            for stages in module.ROOTS.values() for stage in ('judge_pilot','grade')]


def install_candidate(module,bound):
    original_configure=module.configure
    roots=semantic_roots()
    def configure(arm):
        training,evaluation,trained,environment=original_configure(arm)
        if training.q is not consumer.q:
            raise ValueError('original shared semantic runtime required')
        ancestor=evaluation.lineage
        def require_judges():
            ancestor.require_judges()
            if consumer.bindings()!=bound:
                raise ValueError('qualified persistent route changed')
        def audit_cache(cache):
            ancestor.audit_cache(cache)
            consumer.audit_local_entries(cache,roots)
        evaluation.lineage=SimpleNamespace(COUNT=ancestor.COUNT,base_cache=ancestor.base_cache,
            inputs=lambda:[*ancestor.inputs(),*map(Path,bound)],configure=ancestor.configure,
            require_judges=require_judges,uncertain_allowances=ancestor.uncertain_allowances,
            audit_cache=audit_cache)
        consumer.q.request=consumer.serialized_request
        return training,evaluation,trained,environment
    module.configure=configure


def augment(proof,bound):
    value=verified.bind_policy(consumer.augment_proof(proof,bound))
    for path in (Path(__file__).resolve(),PLAN,TEST,CPU):
        key=str(path);digest=data.sha(path)
        if key in value['input_hashes'] and value['input_hashes'][key]!=digest:
            raise ValueError('candidate transport proof conflicts with existing inputs')
        value['input_hashes'][key]=digest
    return value


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CPU/network-only semantic evaluation required')
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('family',choices=tuple(FAMILIES));parser.add_argument('stage',choices=STAGES)
    parser.add_argument('--arm',required=True);args=parser.parse_args()
    module=select(args.family,args.arm);root=module.ROOTS[args.arm][args.stage]
    if root.exists():raise ValueError('fresh append-only semantic stage required')
    for path in (CPU,verified.CPU,route.CPU):
        checked=data.read(path);data.verify(checked['artifacts'])
        if checked['actual_tool_exit_code']!=0 or checked['tests_passed']!=4:
            raise ValueError('candidate, hash policy and transport contracts must pass')
    # Every independent evaluation and audit starts with no measured hashes.
    proofs=verified.cache.ProcessProofs();data.sha=proofs.sha
    route.install();bound=consumer.bindings()
    install_candidate(module,bound)
    original_write=consumer.task.write_proof
    def write_proof(destination,proof,files):
        consumer.qualification.verify(bound)
        return original_write(destination,augment(proof,bound),files)
    consumer.task.write_proof=write_proof
    try:
        if args.stage=='audit':module.audit(args.arm)
        else:
            with (data.ROOT/'task_supervision_semantic_grade.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                _,evaluation,_,_=module.configure(args.arm)
                evaluation.grade(args.stage=='judge_pilot')
    except Exception as error:
        if root.exists() and not (root/'failure.json').exists():
            data.write(root/'failure.json',{'type':type(error).__name__,'message':str(error),
                'preserve_all_first_responses':True,'automatic_retry':False})
        raise


if __name__=='__main__':main()
