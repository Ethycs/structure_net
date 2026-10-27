"""Replay source-grounded first judgments; report the actual diagnostic score."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_xsum_semantic_probe as study

data, ROOTS = study.data, study.ROOTS
REPORT = Path('docs/08 - Analysis/2026-10-27_ncm-xsum-semantic-probe.md')


def paired(a, b):
    if [r['id'] for r in a] != [r['id'] for r in b]: raise ValueError('identical ordered source population required')
    if not a: raise ValueError('nonempty population required')
    counts = {'both_pass': 0, 'both_fail': 0, 'distill_only': 0, 'mse_only': 0, 'invalid': 0}
    for x, y in zip(a, b):
        if not x['valid'] or not y['valid']: counts['invalid'] += 1
        elif x['passed'] and y['passed']: counts['both_pass'] += 1
        elif not x['passed'] and not y['passed']: counts['both_fail'] += 1
        elif x['passed']: counts['distill_only'] += 1
        else: counts['mse_only'] += 1
    delta = counts['distill_only']-counts['mse_only']
    return {**counts, 'cases': len(a), 'extra_passes': delta,
        'accuracy_difference': delta/len(a) if not counts['invalid'] else None,
        'directional_gate_passed': len(a) == 128 and not counts['invalid'] and delta >= 7}


def audit():
    study.configure(); study.qualified.require_judges(); cases, packets = study.prepared()
    pilot, graded = [study.base.require(ROOTS[s]) for s in ('pilot','grade')]
    for s in ('prepare','pilot','grade'): study.base.verify_proof(data.read(ROOTS[s]/'producer.json'))
    inherited = study.prior_cache(); cache = data.read(ROOTS['grade']/'judgment_cache.json')
    pilot_cache = data.read(ROOTS['pilot']/'judgment_cache.json')
    study.preserve(inherited, pilot_cache); study.preserve(pilot_cache, cache); study.qualified.audit_cache(cache)
    allowed = set(); metrics = {}; scores = {}; joint = {}
    for arm in study.ARMS:
        scores[arm] = {}
        metrics[arm] = {'secondary': None, 'primary': None}
        for role in ('secondary','primary'):
            actual = graded['metrics'][arm][role]
            if actual is None:
                if role == 'secondary' or study.qualifies(metrics[arm]['secondary']): raise ValueError('missing admitted judge')
                continue
            if role == 'primary' and not study.qualifies(metrics[arm]['secondary']): raise ValueError('primary ran without admission')
            keys = {study.first.packet_key(p, role) for p in packets[arm]}; allowed |= keys
            replay = study.score_rows(cases, packets[arm], cache, arm, role)
            if replay != data.read(ROOTS['grade']/f'{role}_{arm}_scores.json'): raise ValueError('raw grade or token-count replay differs')
            metrics[arm][role] = study.metric(replay); scores[arm][role] = replay
            if any(actual[k] != v for k, v in metrics[arm][role].items()): raise ValueError('metric aggregation differs')
            if actual['judge'] != study.q.JUDGES[role]: raise ValueError('judge differs')
            if role == 'secondary':
                pilot_replay = study.score_rows(cases[:4], packets[arm][:4], pilot_cache, arm, role)
                if pilot_replay != data.read(ROOTS['pilot']/f'{role}_{arm}_scores.json'): raise ValueError('pilot firsts differ')
        primary = scores[arm].get('primary')
        joint[arm] = (sum(a['passed'] is True and b['passed'] is True for a,b in zip(scores[arm]['secondary'],primary))
                      if primary and all(r['valid'] for r in primary) else None)
    if set(cache)-set(inherited) != allowed-set(inherited) or len(cache) != graded['canonical_first_judgments']:
        raise ValueError('canonical cache has missing or unrelated new judgments')
    for stage, prior_keys in (('pilot',set(inherited)),('grade',set(pilot_cache))):
        with (ROOTS[stage]/'new_firsts.jsonl').open() as f: journal=list(map(json.loads,f))
        expected = set(data.read(ROOTS[stage]/'judgment_cache.json'))-prior_keys
        if len(journal) != len(expected) or {r['key'] for r in journal} != expected: raise ValueError('first-decision journal differs')
        for row in journal:
            if data.sha(row['row_path']) != row['row_sha256'] or data.read(row['row_path']) != cache[row['key']]['row']:
                raise ValueError('journal first row changed')
    comparison = paired(scores['distill']['secondary'],scores['mse']['secondary'])
    valid = all(v['secondary']['valid'] == 128 for v in metrics.values())
    classification = ('public_probe_semantics_unresolved_invalid_judgments' if not valid else
        'public_probe_semantic_improvement' if comparison['directional_gate_passed'] else
        'public_probe_semantic_improvement_not_established')
    root = ROOTS['audit']; proof = study.freeze(root,'audit',[ROOTS[s]/n for s in ('prepare','pilot','grade')
        for n in ('producer.json','result.json','actual_tool_receipt.json')]+[ROOTS['grade']/'judgment_cache.json'])
    values = {'status':'public_probe_raw_semantic_evidence_replayed','classification':classification,
        'metrics':metrics,'paired_distill_vs_mse':comparison,'joint_passes':joint,
        'canonical_first_judgments':len(cache),'inherited_first_judgments':4691,
        'public_probe_dual95_arms':[a for a in study.ARMS if joint[a] is not None and joint[a]>=122],
        'new_requests_during_audit':0,'source_scope':'reused public XSum TRAIN probe; adapter-fit disjoint only'}
    data.write(root/'scores.json',values);study.finish(root,proof,values,[root/'scores.json'])


def report():
    study.configure(); result=study.base.require(ROOTS['audit']);root=ROOTS['results']
    study.base.verify_proof(data.read(ROOTS['audit']/'producer.json'))
    if root.exists() or REPORT.exists():raise ValueError('fresh measured report required')
    lines=['# Semantic quality of the frozen XSum adapter probe','',
        f"Classification: `{result['classification']}`. These are semantic summary judgments, not exact-token matches.",'',
        '| Model | Terra passes | Semantic pass rate | Valid judgments | Faithful | Salient | Readable |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a,v in result['metrics'].items():
        m=v['secondary'];rate=f"{m['semantic_pass_rate']:.2%}" if m['semantic_pass_rate'] is not None else 'unresolved'
        lines.append(f"| {a} | {m['passed']}/128 | {rate} | {m['valid']}/128 | {m['components']['faithful']} | {m['components']['salient']} | {m['components']['readable']} |")
    lines += ['',f"Paired distill-versus-MSE outcomes: `{json.dumps(result['paired_distill_vs_mse'],sort_keys=True)}`.",
        f"Joint Terra/Sol passes, where admitted and valid: `{json.dumps(result['joint_passes'],sort_keys=True)}`.",
        'A pass requires readable, faithful, salient, nonempty text within128 stored tokens. Paraphrases and minor grammar defects are accepted; all material facts must be supported by the complete source.',
        'Every arm used the same128 sources and frozen saved outputs. Each request contained one anonymous candidate. Existing qualified Terra judged every case; Sol was conditional on at least122 valid Terra passes for an arm. All first valid judgments and exhausted invalid attempts were preserved.',
        f"Canonical first judgments retained: {result['canonical_first_judgments']} (including all4691 inherited firsts).",
        'These public XSum TRAIN sources were excluded from the512-case adapter fit but may have been seen by the released teacher, and this probe was reused in development. These scores are not application final accuracy or full input-range validation. They cannot fulfill the95% goal.',
        'The separate main hybrid still has0/128 on its latest application DEV evaluation, a different population and model. No direct score comparison between these populations is warranted. No new model training, final-test access or speed measurement occurred in this diagnostic.']
    root.mkdir();REPORT.write_text('\n'.join(lines)+'\n')
    data.write(root/'campaign_results.json',{**result,'report':str(REPORT),'hypothesis_id':study.HYPOTHESIS,
        'artifacts':{str(p):data.sha(p) for p in (ROOTS['audit']/'result.json',ROOTS['audit']/'actual_tool_receipt.json',ROOTS['grade']/'judgment_cache.json',REPORT)}})
    print(json.dumps({'classification':result['classification'],'report':str(REPORT)}))


def register():
    study.configure();root=ROOTS['results'];p=root/'campaign_results.json';receipt=data.read(root/'actual_tool_receipt.json')
    if receipt['actual_tool_exit_code']!=0 or receipt['artifacts'].get(str(p))!=data.sha(p):raise ValueError('actual report receipt required')
    data.verify(receipt['artifacts']);v=data.read(p)
    record={'name':'Source-grounded semantic comparison of frozen XSum adapters','classification':v['classification'],
        'application_final_accuracy_measured':False,'public_probe_cases':128,
        'distill_Terra_passes':v['metrics']['distill']['secondary']['passed'],
        'mse_Terra_passes':v['metrics']['mse']['secondary']['passed'],
        'canonical_first_judgments':v['canonical_first_judgments'],
        'description':'Semantically graded all128 reused public XSum TRAIN probe sources for native, initial, MSE-continuation and output-distilled adapters. One anonymous candidate per request; complete source authority; paraphrases accepted. These sources are adapter-fit disjoint but may be in teacher pretraining. This is a development diagnostic, not application final95 qualification or speed evidence.'}
    import numpy as np
    if not hasattr(np,'NaN'):np.NaN=np.nan
    from experiments.neural_architecture_lab.ncm_result_registry import register_records
    register_records(p,REPORT,root,{study.HYPOTHESIS:record});print(json.dumps({'status':'registered_and_read_back'}))


def readback():
    import numpy as np
    if not hasattr(np,'NaN'):np.NaN=np.nan
    import chromadb
    root=ROOTS['results'];h=study.HYPOTHESIS;receipt=data.read(root/(h+'_registry_readback.json'));record=receipt['record']
    collection=chromadb.PersistentClient(path='data/chroma_db').get_collection('hypotheses')
    exact=collection.get(ids=[h],include=['metadatas']);semantic=collection.query(query_texts=[record['description']],n_results=10)
    if (exact['ids']!=[h] or any(exact['metadatas'][0].get(k)!=v for k,v in record.items()) or h not in semantic['ids'][0]
        or record['campaign_sha256']!=data.sha(root/'campaign_results.json') or record['report_sha256']!=data.sha(REPORT)
        or record['application_final_accuracy_measured'] or data.read(Path('data/hypothesis_registry')/(h+'.json'))!=receipt):
        raise ValueError('independent registry readback differs')
    value={'status':'independent_registry_readback_passed','hypothesis_id':h,'campaign_sha256':data.sha(root/'campaign_results.json'),'report_sha256':data.sha(REPORT)}
    data.write(root/'backend_exact_readback.json',value);print(json.dumps(value))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('audit','report','register','readback'))
    globals()[parser.parse_args().stage]()
