"""Publish the audited paired semantic result without replacing final accuracy."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_task_supervision_development as dev

data = dev.data
ROOT = data.ROOT/'20261027_task_supervision_results_v1'
REPORT = Path('docs/08 - Analysis/2026-10-27_ncm-task-supervision.md')
TEST = Path('tests/structure_net/test_ncm_task_supervision_results.py')


def classification(comparison):
    if comparison['selected_DEV_candidate'] is not None:return 'DEV_candidate_qualified_final_unmeasured'
    if (comparison['paired_Terra']['unresolved'] or any(v['primary_admitted'] and v['joint_passed'] is None
            for v in comparison['arms'].values())):return 'semantic_comparison_unresolved_invalid_judgments'
    return ('descriptive_supervision_benefit_DEV_failed' if comparison['descriptive_supervision_benefit']
            else 'supervision_benefit_not_established_DEV_failed')


def report_text(value):
    lines = ['# Task-specific supervision: existing versus expanded training','',
        f"Classification: `{value['classification']}`. Final accuracy remains unmeasured; final320 is sealed.",'',
        '| Training arm | Whole-summary Terra passes | Semantic DEV accuracy | Valid Terra judgments | Same-case Terra/Sol passes | DEV qualified |',
        '|---|---:|---:|---:|---:|---|']
    for arm in dev.study.ARMS:
        row = value['arms'][arm];rate = 'unresolved' if row['task_accuracy'] is None else f"{row['task_accuracy']:.2%}"
        joint = 'unmeasured or unresolved' if row['joint_passed'] is None else f"{row['joint_passed']}/128"
        lines.append(f"| {arm} | {row['passed']}/128 | {rate} | {row['valid']}/128 | {joint} | {row['DEV_qualified']} |")
    lines += ['',f"Paired Terra decisions: `{json.dumps(value['paired_Terra'],sort_keys=True)}`.",
        f"Expanded-minus-existing whole-summary passes: `{value['additional_passes']}`. Descriptive seven-pass benefit: `{value['descriptive_supervision_benefit']}`.",
        f"Candidate eligible for the separately guarded final protocol: `{value['selected_DEV_candidate']}`.",
        'Both arms used the same epoch2 model initializer,2,048 updates and65,536 example presentations. The existing arm used all1,376 reviewed pairs; the expanded arm added every admitted new first from the complete1,024-outcome reservation. The Llama body, codec and word-source reconstruction path remained frozen.',
        'Whole-summary semantic judgments accept paraphrases and require faithful, salient, readable, nonempty text within128 stored tokens. These are not code agreement or exact-reference scores. Terra covers both64-case cohorts; Sol is conditional on at least61 valid Terra passes in each cohort. DEV qualification requires at least61 same-case joint passes per cohort.',
        'One paired training seed and reused development sources limit this comparison. A seven-pass descriptive gain does not establish95% accuracy. More data also changes topic and teacher mix. Equal update counts do not establish equal token exposure, FLOPs or elapsed time.',
        'All earlier first judgments and exhausted invalid requests remain preserved. Full reviewed-source and public-data guards were applied. The independent audit re-rendered raw output IDs and replayed semantic first responses. The sealed-final320 protocol remains required before any final95% claim.',
        'Inference speed was not measured; no quality-matched speedup has been established.','',
        f"Detailed component scores, training exposure and evidence hashes: `{ROOT}/campaign_results.json`."]
    return '\n'.join(lines)+'\n'


def report():
    if ROOT.exists() or REPORT.exists():raise ValueError('fresh append-only paired result required')
    dev.completed(dev.PREFLIGHT)
    paths = [Path(__file__),TEST,dev.PLAN,dev.PREFLIGHT/'producer.json',dev.PREFLIGHT/'result.json',dev.PREFLIGHT/'actual_tool_receipt.json']
    summaries,scores,metrics,training = {},{},{},{}
    bindings = data.read(dev.PREFLIGHT/'bindings.json')['bindings']
    prior = data.read(dev.PREFLIGHT/'result.json')['canonical_first_judgments']
    for arm in dev.study.ARMS:
        root = dev.ROOTS[arm]['audit'];value = dev.completed(root)
        if value['status'] != 'task_supervision_semantic_firsts_replayed' or value['inherited_first_judgments'] != prior:
            raise ValueError('both audited arms in preserved first-judgment order required')
        prior = value['canonical_first_judgments'];scores[arm] = data.read(root/'scores.json')
        summaries[arm] = dev.summarize(scores[arm]);metrics[arm] = value['metrics']
        if any(value.get(k) != v for k,v in summaries[arm].items()):raise ValueError('audited semantic summary differs')
        trained_root = Path(bindings['training'][arm+':train']);trained = dev.completed(trained_root)
        if trained['checkpoint_sha256'] != value['checkpoint_sha256']:raise ValueError('evaluated checkpoint differs')
        training[arm] = {k:trained[k] for k in ('cases','updates','presentations','elapsed_seconds','peak_allocated_bytes',
            'measured_exposure','FLOPs','FLOPs_status','checkpoint_sha256','frozen_hashes')}
        paths += [root/n for n in ('producer.json','result.json','actual_tool_receipt.json','scores.json')]
        paths += [trained_root/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
        paths += [dev.ROOTS[arm]['grade']/'judgment_cache.json']
    comparison = dev.compare(scores['existing'],scores['expanded'])
    value = {'schema':'nal.ncm-task-supervision-results.v1','hypothesis_id':dev.study.HYPOTHESIS,
        'classification':classification(comparison),**comparison,'metrics':metrics,'training':training,
        'canonical_first_judgments':prior,'independent_training_seeds':1,'source_scope':'unchanged_source_disjoint128_DEV',
        'metric':'complete_summary_semantic_pass','final_test_opened':False,'report':str(REPORT)}
    hashes = {str(p):data.sha(p) for p in paths}
    text = report_text(value);data.verify(hashes)
    ROOT.mkdir()
    with REPORT.open('x') as stream:stream.write(text)
    data.write(ROOT/'campaign_results.json',{**value,'artifacts':{**hashes,str(REPORT):data.sha(REPORT)}})
    print(json.dumps({'classification':value['classification'],'arms':value['arms'],'report':str(REPORT)}),flush=True)


def registry(stage):
    campaign = ROOT/'campaign_results.json';dev.public_results.receipt(ROOT,('campaign_results.json',))
    value = data.read(campaign);data.verify(value['artifacts'])
    record = {'name':'Additional task-specific supervision for the frozen-body NCM',
        'classification':value['classification'],'application_final_accuracy_measured':False,
        'DEV_cases':128,'existing_Terra_passes':value['arms']['existing']['passed'],
        'expanded_Terra_passes':value['arms']['expanded']['passed'],
        'canonical_first_judgments':value['canonical_first_judgments'],
        'description':'One paired-seed frozen-body NCM comparison of existing versus expanded reviewed summaries. Same epoch2 initializer and fixed2048-update budget. Audited source-only generation and whole-summary semantic judgments on128 source-disjoint development cases, with accepted paraphrases and conditional second-judge same-case gates. This is development evidence; sealed final accuracy and quality-matched speedup remain unmeasured.'}
    import numpy as np
    if not hasattr(np,'NaN'):np.NaN = np.nan
    if stage == 'register':
        from experiments.neural_architecture_lab.ncm_result_registry import register_records
        register_records(campaign,REPORT,ROOT,{dev.study.HYPOTHESIS:record})
        print(json.dumps({'status':'registered_and_read_back'}));return
    import chromadb
    hypothesis = dev.study.HYPOTHESIS;registered = data.read(ROOT/(hypothesis+'_registry_readback.json'))
    stored = registered['record'];collection = chromadb.PersistentClient(path='data/chroma_db').get_collection('hypotheses')
    exact = collection.get(ids=[hypothesis],include=['metadatas'])
    semantic = collection.query(query_texts=[stored['description']],n_results=10)
    if (exact['ids'] != [hypothesis] or any(exact['metadatas'][0].get(k) != v for k,v in stored.items())
            or any(stored.get(k) != v for k,v in record.items()) or hypothesis not in semantic['ids'][0]
            or stored['campaign_sha256'] != data.sha(campaign) or stored['report_sha256'] != data.sha(REPORT)
            or data.read(Path('data/hypothesis_registry')/(hypothesis+'.json')) != registered):
        raise ValueError('independent exact, semantic and local registry readback differs')
    result = {'status':'independent_registry_readback_passed','hypothesis_id':hypothesis,
        'campaign_sha256':data.sha(campaign),'report_sha256':data.sha(REPORT)}
    data.write(ROOT/'backend_exact_readback.json',result);print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('report','register','readback'))
    args = parser.parse_args();report() if args.stage == 'report' else registry(args.stage)
