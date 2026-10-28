"""Reuse already measured immutable hashes inside future evaluation processes."""
from pathlib import Path

from experiments.structure_net import ncm_persistent_task_development as route
from experiments.structure_net import ncm_recovered_training_verified_cache as cache

data=route.consumer.task.data
PLAN=Path('docs/07 - Status Reports/2026-10-28_ncm-semantic-validation-cache.md')
TEST=Path('tests/structure_net/test_ncm_persistent_task_verified.py')
CPU=data.ROOT/'20261028_persistent_task_verified_CPU_v1.json'
POLICY={**cache.POLICY,'scope':'shared NCM artifact SHA256 helper in this evaluation process'}


def bind_policy(proof):
    # Keep packet identities, judge configuration, firsts and model/data inputs.
    value={**proof,'config':{**proof['config'],'input_validation_policy':POLICY},
           'input_hashes':dict(proof['input_hashes'])}
    paths=(Path(__file__).resolve(),PLAN,TEST,CPU,Path(cache.__file__).resolve(),cache.PLAN,cache.TEST,cache.CPU)
    for path in paths:
        name=str(path);digest=data.sha(path)
        if name in value['input_hashes'] and value['input_hashes'][name]!=digest:
            raise ValueError('validation policy conflicts with an existing producer input')
        value['input_hashes'][name]=digest
    return value


def main():
    # No cache is restored from disk, receipts or another running consumer.
    checked=data.read(CPU);data.verify(checked['artifacts'])
    if checked['actual_tool_exit_code']!=0 or checked['tests_passed']!=4:
        raise ValueError('guarded hash and unchanged semantic-contract tests required')
    prior=data.read(cache.CPU);data.verify(prior['artifacts'])
    if prior['actual_tool_exit_code']!=0 or prior['tests_passed']!=3:
        raise ValueError('original stat-guarded validation qualification required')
    proofs=cache.ProcessProofs()
    data.sha=proofs.sha
    route.install()
    original_write=route.consumer.task.write_proof
    def write_proof(root,proof,files):
        return original_write(root,bind_policy(proof),files)
    route.consumer.task.write_proof=write_proof
    route.consumer.main()


if __name__=='__main__':main()
