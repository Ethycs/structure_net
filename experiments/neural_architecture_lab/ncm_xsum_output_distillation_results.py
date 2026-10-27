"""Independent raw-count replay and conservative registration for the bridge."""
import argparse
import json
import math
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_xsum_output_distillation as study
from experiments.neural_architecture_lab.ncm_xsum_prefix_bridge_results import integers

data, ROOTS = study.data, study.ROOTS
REPORT = Path('docs/08 - Analysis/2026-10-27_ncm-xsum-output-distillation.md')


def counts(rows):
    if not rows or len({r['id'] for r in rows}) != len(rows): raise ValueError('unique complete population required')
    arms = ('initial', 'mse', 'distill')
    totals = {a: dict(code_matches=0, word_matches=0, nonpad_matches=0, content_matches=0) for a in arms}
    content = nonpad = empty = 0; paired = []
    for row in rows:
        for a in ('native', *arms):
            integers(row[a]['codes'], 200, 4096); integers(row[a]['words'], 48, 50304)
        gold = row['native']['words']; end = gold.index(50256) if 50256 in gold else 48
        active = [i for i, v in enumerate(gold) if v != 50257]
        meaningful = [i for i in range(end) if gold[i] != 50257]
        nonpad += len(active); content += len(meaningful); empty += not meaningful
        per_case = {}
        for a in arms:
            words = row[a]['words']
            totals[a]['code_matches'] += sum(x == y for x, y in zip(row[a]['codes'], row['native']['codes']))
            totals[a]['word_matches'] += sum(x == y for x, y in zip(words, gold))
            totals[a]['nonpad_matches'] += sum(words[i] == gold[i] for i in active)
            per_case[a] = sum(words[i] == gold[i] for i in meaningful)
            totals[a]['content_matches'] += per_case[a]
        paired.append({'id': row['id'], 'content_positions': len(meaningful),
                       'distill_minus_mse_content_matches': per_case['distill']-per_case['mse']})
    for value in totals.values():
        value.update(code_agreement=value['code_matches']/(200*len(rows)),
            word_agreement=value['word_matches']/(48*len(rows)),
            nonpad_agreement=value['nonpad_matches']/nonpad if nonpad else 0.,
            content_agreement=value['content_matches']/content if content else 0.)
    delta = totals['distill']['content_agreement']-totals['mse']['content_agreement']
    preservation = not empty and all(totals['distill'][k] >= study.CONFIG.agreement_gate for k in
                                   ('word_agreement', 'nonpad_agreement', 'content_agreement'))
    improvement = delta >= study.CONFIG.content_improvement_gate
    return {'cases': len(rows), 'scores': totals, 'native_content_positions': content,
        'native_nonpad_positions': nonpad, 'empty_native_cases': empty, 'paired_cases': paired,
        'content_improvement_over_mse': delta, 'preservation_gate_passed': preservation,
        'improvement_gate_passed': improvement,
        'classification': 'output_distillation_preservation_and_improvement_passed' if preservation and improvement
            else 'output_distillation_joint_gate_failed'}


def audit():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only audit required')
    cases = study.parent()
    names = ('pilot', 'cache', 'train_mse', 'train_distill', 'evaluate')
    results = {s: study.base.require(ROOTS[s]) for s in names}
    for s in names: study.base.verify_proof(data.read(ROOTS[s]/'producer.json'))
    if not results['pilot']['strict_optimizer_replay'] or results['cache']['status'] != 'teacher_cache_passed':
        raise ValueError('successful mechanism controls required')
    frozen = results['pilot']['frozen_model_hashes']
    if any(r['frozen_model_hashes'] != frozen for r in results.values()): raise ValueError('frozen models differ')
    index = data.read(ROOTS['cache']/'index.json')
    if [r['id'] for r in index] != [r['id'] for r in cases['fit']]: raise ValueError('teacher population differs')
    matched = 0
    for row in index:
        value = study.load_target(row)
        for key in ('codes', 'teacher_decisions'):
            if value[key].dtype != torch.int64: raise ValueError('integer codes required')
            integers(value[key][0].tolist(), 200, 4096)
        if not torch.isfinite(value['hidden']).all(): raise ValueError('nonfinite teacher feature')
        actual = int((value['codes'] == value['teacher_decisions']).sum())
        if actual != row['native_matches'] or actual != value['native_matches']: raise ValueError('teacher cache replay differs')
        matched += actual
    if matched/(512*200) != results['cache']['native_full_forward_agreement']: raise ValueError('cache aggregation differs')
    rng = torch.Generator().manual_seed(study.CONFIG.seed); expected = []
    for epoch in range(10): expected += [(epoch+1, cases['fit'][i]['id']) for i in torch.randperm(512, generator=rng).tolist()]
    for arm in ('mse', 'distill'):
        r = results['train_'+arm]
        if r['updates'] != 5120 or not r['strict_reload']: raise ValueError('complete fixed final checkpoint required')
        with (ROOTS['train_'+arm]/'trace.jsonl').open() as f: trace = list(map(json.loads, f))
        if len(trace) != 5120 or [(v['epoch'], v['id']) for v in trace] != expected: raise ValueError('update population differs')
        for step, row in enumerate(trace, 1):
            if (row['step'] != step or row['arm'] != arm or row['phase'] != study.phase(arm, step)
                    or row['learning_rate'] != study.rate(step) or not math.isfinite(row['loss'])
                    or row['loss'] < -1e-6 or not math.isfinite(row['gradient_norm']) or row['gradient_norm'] < 0):
                raise ValueError('training trace does not replay')
        saved = torch.load(r['checkpoint'], map_location='cpu', weights_only=True)
        if saved['updates'] != 5120 or saved['arm'] != arm or not torch.equal(saved['rng_state'].cpu(), rng.get_state()):
            raise ValueError('checkpoint continuation state differs')
    if results['train_mse']['initial_bridge_sha256'] != results['train_distill']['initial_bridge_sha256']:
        raise ValueError('unequal initial bridge')
    rows = data.read(ROOTS['evaluate']/'outputs.json'); old = data.read(study.base.ROOTS['evaluate']/'outputs.json')
    if [r['id'] for r in rows] != [r['id'] for r in cases['probe']]: raise ValueError('probe changed')
    for a, b in zip(rows, old):
        if a['id'] != b['id'] or a['native'] != b['native'] or a['initial'] != b['bridge']:
            raise ValueError('parent reference changed')
    value = counts(rows); root = ROOTS['audit']
    proof = study.freeze(root, 'audit', [ROOTS[s]/n for s in names for n in
        ('producer.json', 'result.json', 'actual_tool_receipt.json')])
    data.write(root/'scores.json', value)
    study.finish(root, proof, {'status': 'output_distillation_raw_evidence_replayed', **value,
                              'updates_replayed': 10240}, [root/'scores.json'])


def report():
    result = study.base.require(ROOTS['audit']); root = ROOTS['results']
    study.base.verify_proof(data.read(ROOTS['audit']/'producer.json'))
    if root.exists() or REPORT.exists(): raise ValueError('fresh report required')
    lines = ['# Output distillation for the frozen XSum prefix bridge', '',
        f"Classification: `{result['classification']}`. This is not semantic summary accuracy.", '',
        '| Adapter | Code agreement | All decoded IDs | Non-pad IDs | Content IDs |',
        '|---|---:|---:|---:|---:|']
    for a, v in result['scores'].items():
        lines.append(f"| {a} | {v['code_agreement']:.2%} | {v['word_agreement']:.2%} | {v['nonpad_agreement']:.2%} | {v['content_agreement']:.2%} |")
    lines += ['', f"Content improvement over matched MSE continuation: {result['content_improvement_over_mse']:.2%} points.",
        'Both arms started from the same 3.54M-parameter adapter and used 5,120 additional updates on the same 512 fit sources in the same order. All original models stayed frozen.',
        'The distillation arm matched final code features for five epochs, then teacher code distributions for five epochs. The control continued input-representation MSE for ten epochs.',
        'The 128 probe sources were excluded from adapter fitting but reused from the previous diagnostic; the released teacher may have seen them in its public training data. Inputs fit 512 positions and outputs remain 48 positions.',
        'Exact reference agreement tests behavior preservation. Different summaries may be valid paraphrases, so agreement is not semantic accuracy. No application DEV/final cases, new semantic judgments or speed measurements were used. The separate main hybrid remains at 0/128 at its completed epoch1 and epoch2 DEV milestones.']
    root.mkdir(); REPORT.write_text('\n'.join(lines)+'\n')
    data.write(root/'campaign_results.json', {**result, 'hypothesis_id': study.HYPOTHESIS, 'report': str(REPORT),
        'artifacts': {str(p): data.sha(p) for p in (ROOTS['audit']/'result.json', ROOTS['audit']/'actual_tool_receipt.json', REPORT)}})
    print(json.dumps({'classification': result['classification'], 'report': str(REPORT)}))


def register():
    root = ROOTS['results']; receipt = data.read(root/'actual_tool_receipt.json'); p = root/'campaign_results.json'
    if receipt['actual_tool_exit_code'] != 0 or receipt['artifacts'].get(str(p)) != data.sha(p): raise ValueError('report receipt required')
    data.verify(receipt['artifacts']); result = data.read(p)
    record = {'name': 'Final-feature and code-distribution distillation for a frozen XSum adapter',
        'classification': result['classification'], 'task_accuracy_measured': False,
        'native_content_agreement': result['scores']['distill']['content_agreement'],
        'content_improvement_over_mse': result['content_improvement_over_mse'],
        'description': 'Compared equal additional training of a frozen-Llama prefix adapter using input MSE '
            'versus final NCM features then code-distribution KL. Both fit 512 public TRAIN sources and '
            'reused 128 adapter-held-out public TRAIN probes. The teacher may have seen them. This measures '
            'behavior preservation, not semantic accuracy, full application range, final qualification or speed.'}
    import numpy as np
    if not hasattr(np, 'NaN'): np.NaN = np.nan
    from experiments.neural_architecture_lab.ncm_result_registry import register_records
    register_records(p, REPORT, root, {study.HYPOTHESIS: record})
    print(json.dumps({'status': 'registered_and_read_back', 'hypothesis': study.HYPOTHESIS}))


def readback():
    import numpy as np
    if not hasattr(np, 'NaN'): np.NaN = np.nan
    import chromadb
    root = ROOTS['results']; h = study.HYPOTHESIS
    receipt = data.read(root/(h+'_registry_readback.json')); record = receipt['record']
    collection = chromadb.PersistentClient(path='data/chroma_db').get_collection('hypotheses')
    exact = collection.get(ids=[h], include=['metadatas'])
    semantic = collection.query(query_texts=[record['description']], n_results=10)
    if (exact['ids'] != [h] or any(exact['metadatas'][0].get(k) != v for k, v in record.items())
            or h not in semantic['ids'][0] or record['campaign_sha256'] != data.sha(root/'campaign_results.json')
            or record['report_sha256'] != data.sha(REPORT) or record['task_accuracy_measured']
            or data.read(Path('data/hypothesis_registry')/(h+'.json')) != receipt):
        raise ValueError('independent registry readback differs')
    value = {'status': 'independent_registry_readback_passed', 'campaign_sha256': data.sha(root/'campaign_results.json'),
             'report_sha256': data.sha(REPORT), 'hypothesis_id': h}
    data.write(root/'backend_exact_readback.json', value); print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('audit', 'report', 'register', 'readback'))
    globals()[parser.parse_args().stage]()
