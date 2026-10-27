"""Replay and report author-rendered semantic scores without replacing history."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_xsum_author_semantics as study
from experiments.neural_architecture_lab.ncm_xsum_semantic_probe_results import paired

data, ROOTS = study.data, study.ROOTS
REPORT = Path('docs/08 - Analysis/2026-10-27_ncm-xsum-author-semantics.md')


def audit():
    study.configure();study.qualified.require_judges();cases,packets = study.prepared()
    pilot,graded = [study.base.require(ROOTS[s]) for s in ('pilot','grade')]
    for s in ('prepare','pilot','grade'):study.base.verify_proof(data.read(ROOTS[s]/'producer.json'))
    inherited = study.prior_cache();cache = data.read(ROOTS['grade']/'judgment_cache.json')
    pilot_cache = data.read(ROOTS['pilot']/'judgment_cache.json')
    study.preserve(inherited,pilot_cache);study.preserve(pilot_cache,cache);study.qualified.audit_cache(cache)
    allowed,metrics,scores,joint = set(),{},{},{}
    for arm in study.ARMS:
        metrics[arm] = {'secondary':None,'primary':None};scores[arm] = {}
        for role in ('secondary','primary'):
            actual = graded['metrics'][arm][role]
            if actual is None:
                if role == 'secondary' or study.qualifies(metrics[arm]['secondary']):
                    raise ValueError('missing admitted judge')
                continue
            if role == 'primary' and not study.qualifies(metrics[arm]['secondary']):
                raise ValueError('primary ran without admission')
            allowed |= {study.first.packet_key(p,role) for p in packets[arm]}
            replay = study.score_rows(cases,packets[arm],cache,arm,role)
            if replay != data.read(ROOTS['grade']/f'{role}_{arm}_scores.json'):
                raise ValueError('raw judgment or token count replay differs')
            metrics[arm][role] = study.metric(replay);scores[arm][role] = replay
            if (any(actual[k] != v for k,v in metrics[arm][role].items())
                    or actual['judge'] != study.q.JUDGES[role]):
                raise ValueError('semantic metric or judge differs')
            if role == 'secondary':
                replay_pilot = study.score_rows(cases[:4],packets[arm][:4],pilot_cache,arm,role)
                if replay_pilot != data.read(ROOTS['pilot']/f'{role}_{arm}_scores.json'):
                    raise ValueError('pilot firsts changed')
        primary = scores[arm].get('primary')
        joint[arm] = (sum(a['passed'] is True and b['passed'] is True for a,b in zip(scores[arm]['secondary'],primary))
                      if primary and all(r['valid'] for r in primary) else None)
    if (set(cache)-set(inherited) != allowed-set(inherited)
            or len(cache) != graded['canonical_first_judgments']
            or any(r['inherited_first_judgments'] != len(inherited) for r in (pilot,graded))):
        raise ValueError('complete original cache and exact corrected additions required')
    for stage,prior in (('pilot',inherited),('grade',pilot_cache)):
        with (ROOTS[stage]/'new_firsts.jsonl').open() as f:journal = list(map(json.loads,f))
        expected = set(data.read(ROOTS[stage]/'judgment_cache.json'))-set(prior)
        if len(journal) != len(expected) or {r['key'] for r in journal} != expected:
            raise ValueError('first-decision journal differs')
        for row in journal:
            if data.sha(row['row_path']) != row['row_sha256'] or data.read(row['row_path']) != cache[row['key']]['row']:
                raise ValueError('journal first changed')
    comparison = paired(scores['distill']['secondary'],scores['mse']['secondary'])
    old = data.read(study.parent.ROOTS['audit']/'result.json')
    changed = data.read(study.rendered.ROOT/'result.json')['changed_ids']
    for arm in study.ARMS:
        old_scores = data.read(study.parent.ROOTS['grade']/f'secondary_{arm}_scores.json')
        for before,after in zip(old_scores,scores[arm]['secondary']):
            if before['id'] != after['id']:raise ValueError('source order differs')
            if before['id'] not in changed[arm] and before != after:
                raise ValueError('unchanged text must retain its entire first decision')
    valid = all(v['secondary']['valid'] == 128 for v in metrics.values())
    classification = ('author_semantics_unresolved_invalid_judgments' if not valid else
        'author_semantic_improvement' if comparison['directional_gate_passed'] else
        'author_semantic_improvement_not_established')
    root = ROOTS['audit'];proof = study.freeze(root,'audit',
        [ROOTS[s]/n for s in ('prepare','pilot','grade') for n in ('producer.json','result.json','actual_tool_receipt.json')]
        +[ROOTS['grade']/'judgment_cache.json'],len(inherited))
    values = {'status':'author_semantic_firsts_replayed','classification':classification,'metrics':metrics,
        'old_render_metrics':old['metrics'],'paired_distill_vs_mse':comparison,'joint_passes':joint,
        'canonical_first_judgments':len(cache),'inherited_first_judgments':len(inherited),
        'new_first_judgments':len(cache)-len(inherited),'new_requests_during_audit':0,
        'public_probe_dual95_arms':[a for a in study.ARMS if joint[a] is not None and joint[a] >= 122],
        'source_scope':'reused public XSum TRAIN probe; adapter-fit disjoint only'}
    data.write(root/'scores.json',values);study.finish(root,proof,values,[root/'scores.json'])


def report():
    study.configure();state = study.base.require(ROOTS['audit'])
    study.base.verify_proof(data.read(ROOTS['audit']/'producer.json'))
    root = ROOTS['results']
    if root.exists() or REPORT.exists():raise ValueError('fresh measured report required')
    lines = ['# Semantic quality with the author text renderer','',
        f"Classification: `{state['classification']}`. Application final accuracy remains unmeasured.",'',
        '| Model | Old renderer passes | Author renderer passes | Author pass rate | Valid | Faithful | Salient | Readable |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,value in state['metrics'].items():
        m = value['secondary'];old = state['old_render_metrics'][arm]['secondary'];c = m['components']
        rate = f"{m['semantic_pass_rate']:.2%}" if m['semantic_pass_rate'] is not None else 'unresolved'
        lines.append(f"| {arm} | {old['passed']}/128 | {m['passed']}/128 | {rate} | {m['valid']}/128 | {c['faithful']} | {c['salient']} | {c['readable']} |")
    lines += ['',
        'Only text rendering changed: stop at EOS, remove PAD tokens, and strip outer whitespace. All raw generated IDs and complete sources stayed fixed. Every unchanged packet reused its first judgment, including failures.',
        'A pass requires faithful, salient, readable, nonempty text within 128 stored tokens. Paraphrases are accepted. Terra judged all cases; Sol was conditional on 122/128 valid Terra passes.',
        f"Retained {state['inherited_first_judgments']} inherited firsts and added {state['new_first_judgments']}. No original result was replaced.",
        f"Paired distill versus MSE: `{json.dumps(state['paired_distill_vs_mse'],sort_keys=True)}`.",
        f"Joint Terra/Sol passes, where admitted and valid: `{json.dumps(state['joint_passes'],sort_keys=True)}`.",
        'These reused public XSum TRAIN sources were excluded from adapter fitting but may have appeared in the released teacher training. They cannot establish application DEV, source-disjoint final accuracy, or the 95% goal. No model was trained, no final320 source was opened and no speed was measured.']
    root.mkdir();REPORT.write_text('\n'.join(lines)+'\n')
    data.write(root/'campaign_results.json',{**state,'hypothesis_id':study.HYPOTHESIS,'report':str(REPORT),
        'artifacts':{str(p):data.sha(p) for p in (ROOTS['audit']/'result.json',ROOTS['audit']/'actual_tool_receipt.json',
            ROOTS['grade']/'judgment_cache.json',REPORT)}})
    print(json.dumps({'status':'author_semantics_reported','classification':state['classification'],'report':str(REPORT)}))


def register():
    study.configure();root = ROOTS['results'];p = root/'campaign_results.json'
    receipt = data.read(root/'actual_tool_receipt.json')
    if receipt['actual_tool_exit_code'] != 0 or receipt['artifacts'].get(str(p)) != data.sha(p):
        raise ValueError('actual report receipt required')
    data.verify(receipt['artifacts']);v = data.read(p)
    record = {'name':'Author-rendered source-grounded semantic XSum comparison','classification':v['classification'],
        'application_final_accuracy_measured':False,'public_probe_cases':128,
        'distill_Terra_passes':v['metrics']['distill']['secondary']['passed'],
        'mse_Terra_passes':v['metrics']['mse']['secondary']['passed'],
        'native_Terra_passes':v['metrics']['native']['secondary']['passed'],
        'canonical_first_judgments':v['canonical_first_judgments'],
        'description':'All 128 reused XSum probe sources graded for four frozen arms with exact author EOS-only text rendering. Original firsts retained; unchanged packets reused. Complete source and anonymous candidate; paraphrases accepted. This is adapter-fit-disjoint development evidence, not application final95 accuracy or speed evidence.'}
    import numpy as np
    if not hasattr(np,'NaN'):np.NaN=np.nan
    from experiments.neural_architecture_lab.ncm_result_registry import register_records
    register_records(p,REPORT,root,{study.HYPOTHESIS:record});print(json.dumps({'status':'registered_and_read_back'}))


def readback():
    import numpy as np
    if not hasattr(np,'NaN'):np.NaN=np.nan
    import chromadb
    root = ROOTS['results'];h = study.HYPOTHESIS
    receipt = data.read(root/(h+'_registry_readback.json'));record = receipt['record']
    collection = chromadb.PersistentClient(path='data/chroma_db').get_collection('hypotheses')
    exact = collection.get(ids=[h],include=['metadatas']);semantic = collection.query(query_texts=[record['description']],n_results=10)
    if (exact['ids'] != [h] or any(exact['metadatas'][0].get(k) != v for k,v in record.items())
            or h not in semantic['ids'][0] or record['campaign_sha256'] != data.sha(root/'campaign_results.json')
            or record['report_sha256'] != data.sha(REPORT) or record['application_final_accuracy_measured']
            or data.read(Path('data/hypothesis_registry')/(h+'.json')) != receipt):
        raise ValueError('independent registry readback differs')
    value = {'status':'independent_registry_readback_passed','hypothesis_id':h,
        'campaign_sha256':data.sha(root/'campaign_results.json'),'report_sha256':data.sha(REPORT)}
    data.write(root/'backend_exact_readback.json',value);print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('audit','report','register','readback'))
    globals()[parser.parse_args().stage]()
