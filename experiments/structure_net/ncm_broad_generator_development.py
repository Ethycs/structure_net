"""Bind the existing complete-summary DEV protocol to the broader generator."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from experiments.structure_net import ncm_broad_generator as study

ROOTS = {s:study.runtime.ROOT/('20261025_broad_generator_DEV_'+s+'_v1')
         for s in ('inference_pilot','outputs','judge_pilot','grade')}
ANCESTOR = study.runtime.ROOT/'20261025_generated_code_decoder_results_v1'
ANCESTOR_SHA = 'b8f4d661db3759a14b01d021c333d0a2bff70c3f02a4732d8f4daae494e16047'
CACHE = study.runtime.ROOT/'20261025_generated_code_decoder_DEV_grade_v1/judgment_cache.json'
CACHE_SHA = '4d810ba2795a45ec1b7af958697dc9308ef0e45396b27a7b97953ab67c635579'


def configure():
    # Reuse the last complete first-judgment lineage, including uncertain-call
    # allowances. This configures only the current interpreter.
    from experiments.structure_net import ncm_generated_decoder_development as previous
    training, evaluation, environment = previous.configure()
    old_lineage = evaluation.lineage
    def base_cache():
        old = old_lineage.base_cache()
        if study.file_hash(CACHE) != CACHE_SHA or study.file_hash(ANCESTOR/'campaign_results.json') != ANCESTOR_SHA:
            raise ValueError('completed3795-first semantic lineage changed')
        cache = training.q.read(CACHE)
        if len(cache) != 3795 or any(cache.get(k) != v for k,v in old.items()):
            raise ValueError('all3795 prior first judgments must survive')
        old_lineage.audit_cache(cache)
        return cache
    def inputs():
        return [*old_lineage.inputs(),CACHE,ANCESTOR/'campaign_results.json',
                ANCESTOR/'backend_exact_readback.json',ANCESTOR/'backend_actual_tool_receipt.json']
    evaluation.lineage = SimpleNamespace(COUNT=3795,base_cache=base_cache,inputs=inputs,
        configure=old_lineage.configure,require_judges=old_lineage.require_judges,
        uncertain_allowances=old_lineage.uncertain_allowances,audit_cache=old_lineage.audit_cache)

    def sources():
        cohorts, manifest = evaluation.old.source_inputs()
        guard = evaluation.split_guard(training.q.read(study.catalog.DEST/'cases.json'),cohorts)
        return cohorts, manifest, guard

    def freeze(root,stage,extra):
        from experiments.structure_net import audit_ncm_broad_generator as audit
        files = {Path(__file__), study.PLAN, Path(evaluation.__file__)}
        for module in tuple(study.sys.modules.values()):
            name = getattr(module,'__file__',None)
            if name:
                p = Path(name).resolve()
                if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve())
                        or p.is_relative_to(training.native.vendor.VENDOR.resolve())):
                    files.add(p)
        paths = [evaluation.old.SOURCE/'cases.json',evaluation.old.FEATURES/'manifest.json',
                 study.catalog.DEST/'cases.json',audit.DEST/'campaign_results.json',audit.DEST/'actual_tool_receipt.json']
        paths += [study.RUNS['train']/n for n in ('producer.json','result.json','actual_tool_receipt.json','model.pt')]
        paths += list(extra)
        root.mkdir(); (root/'producer_source').mkdir()
        for i,p in enumerate(sorted(files)): (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
        proof = {'schema':'nal.ncm-broad-generator-dev.v1','stage':stage,'hypothesis_id':study.HYPOTHESIS_ID,
                 'config':{'source_only':True,'target_fields_in_prediction':False,'full_source_features':True,
                     'source_disjoint_DEV_cases':128,'cohorts':evaluation.old.COHORTS,
                     'rubric':training.q.RUBRIC,'judges':training.q.JUDGES,'inherited_judgments':3795,
                     'task_accuracy_metric':'whole_summary_semantic_pass','stored_token_cap':128,
                     'exact_match_task_metric':False,'final_test_opened':False,'speed_tested':False},
                 'environment':environment,'source_hashes':{str(p):study.file_hash(p) for p in sorted(files)},
                 'input_hashes':{str(p):study.file_hash(p) for p in dict.fromkeys(paths)}}
        proof['fingerprint'] = training.q.digest(proof)
        study.write_json(root/'producer.json',proof)
        return proof

    # generate() instantiates the old architecture and then strictly loads the
    # full selected checkpoint. Avoid replacing load_initial (recursive ancestry).
    training.RUNS, training.PLAN = study.RUNS, study.PLAN
    evaluation.RUNS, evaluation.sources, evaluation.freeze = ROOTS, sources, freeze
    return training,evaluation,environment


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('preflight',*ROOTS))
    stage = parser.parse_args().stage
    training,evaluation,_ = configure()
    if stage == 'preflight':
        cohorts,_,guard = evaluation.sources()
        cache = evaluation.lineage.base_cache()
        print(study.json.dumps({'source_disjoint':guard,'prior_firsts':len(cache),
            'DEV_cases':sum(len(v) for v in cohorts.values()),'new_requests':0,'final_test_opened':False}))
        return
    from experiments.structure_net import audit_ncm_broad_generator as audit
    if training.q.read(audit.DEST/'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('independent training audit must complete first')
    proof = training.q.read(audit.DEST/'campaign_results.json')
    if (proof['status'] != 'broad_generator_training_independently_replayed'
            or study.file_hash(audit.__file__) != proof['auditor_sha256']):
        raise ValueError('complete immutable training audit required')
    for path,expected in proof['artifacts'].items():
        if study.file_hash(path) != expected: raise ValueError('audited training artifact changed')
    if stage in ('inference_pilot','outputs'): evaluation.generate(stage == 'inference_pilot')
    else: evaluation.grade(stage == 'judge_pilot')


if __name__ == '__main__': main()
