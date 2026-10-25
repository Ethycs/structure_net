"""Unchanged source-disjoint semantic gates for generated-code decoder adaptation."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from experiments.structure_net import ncm_generated_code_decoder as study

ROOTS = {s: study.ROOT / ('20261025_generated_code_decoder_DEV_' + s + '_v1')
         for s in ('inference_pilot','outputs','judge_pilot','grade')}
ANCESTOR = study.ROOT / '20261025_sequential_full732_pixi_results_v1'
ANCESTOR_SHA = 'e54c173035b1f6230cf7ee992dcb6ca0399e8e62b2eceb84eb9ca9657f1a4733'
CACHE = study.ROOT / '20261025_sequential_full732_pixi_DEV_grade_v1/judgment_cache.json'
CACHE_SHA = 'fdf896ed6b38bd1112a757051201b5875b56903a3f0e9413eed4966bbd0bd605'
AUDIT = study.ROOT / '20261025_generated_code_decoder_train_audit_v1'


def configure():
    training, environment = study.runtime.configure()
    from experiments.structure_net import ncm_sequential_development_v2 as evaluation
    original_lineage = evaluation.lineage
    def base_cache():
        old = original_lineage.base_cache()
        if study.file_hash(ANCESTOR/'campaign_results.json')!=ANCESTOR_SHA or study.file_hash(CACHE)!=CACHE_SHA:
            raise ValueError('completed parent semantic evidence changed')
        cache=training.q.read(CACHE)
        if len(cache)!=3667 or any(cache.get(k)!=v for k,v in old.items()):
            raise ValueError('all3667 prior first judgments must survive')
        original_lineage.audit_cache(cache)
        return cache
    def lineage_inputs():
        return [*original_lineage.inputs(),CACHE,ANCESTOR/'campaign_results.json',
            ANCESTOR/'backend_exact_readback.json',ANCESTOR/'backend_actual_tool_receipt.json']
    evaluation.lineage=SimpleNamespace(COUNT=3667,base_cache=base_cache,inputs=lineage_inputs,
        configure=original_lineage.configure,require_judges=original_lineage.require_judges,
        uncertain_allowances=original_lineage.uncertain_allowances,audit_cache=original_lineage.audit_cache)
    original_freeze=evaluation.freeze
    def freeze(root,stage,extra):
        proof=original_freeze(root,stage,[*extra,AUDIT/'campaign_results.json',AUDIT/'actual_tool_receipt.json'])
        proof['schema']='nal.ncm-generated-code-decoder-dev.v1'
        proof['hypothesis_id']=study.HYPOTHESIS
        proof['environment']=environment
        proof['config']['decoder_adaptation_updates']=study.CONFIG['updates']
        proof['config']['frozen_generator']=True
        proof['fingerprint']=training.q.digest({k:v for k,v in proof.items() if k!='fingerprint'})
        study.write_json(root/'producer.json',proof)
        return proof
    training.RUNS,training.PLAN=study.RUNS,study.PLAN
    evaluation.RUNS,evaluation.freeze=ROOTS,freeze
    return training,evaluation,environment


def run(stage):
    training,evaluation,_=configure()
    if stage=='preflight':
        cohorts,_,guard=evaluation.sources()
        cache=evaluation.lineage.base_cache()
        print(training.json.dumps({'source_disjoint':guard,'prior_firsts':len(cache),
            'cases':sum(len(v) for v in cohorts.values()),'final_test_opened':False}))
        return
    # Independent audits contain an aggregate and an observed process receipt,
    # not the producer/result pair used by model execution stages.
    if training.q.read(AUDIT/'actual_tool_receipt.json')['actual_tool_exit_code']!=0:
        raise ValueError('completed independent training audit required')
    audit=training.q.read(AUDIT/'campaign_results.json')
    auditor=Path(__file__).with_name('audit_ncm_generated_decoder_training.py')
    if (audit['status']!='generated_code_decoder_training_independently_replayed'
            or study.file_hash(auditor)!=audit['auditor_sha256']):
        raise ValueError('completed audit or auditor identity differs')
    for path,expected in audit['artifacts'].items():
        if study.file_hash(Path(path))!=expected:raise ValueError('audited training artifact changed')
    if stage in ('inference_pilot','outputs'):evaluation.generate(stage=='inference_pilot')
    else:evaluation.grade(stage=='judge_pilot')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=('preflight',*ROOTS))
    run(p.parse_args().stage)
