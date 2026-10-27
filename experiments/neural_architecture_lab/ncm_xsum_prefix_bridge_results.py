"""Replay source-prefix bridge evidence without claiming summary task accuracy."""
import argparse
import json
import math
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_xsum_prefix_bridge as study

data, ROOTS, HYPOTHESIS, CONFIG = study.data, study.ROOTS, study.HYPOTHESIS, study.CONFIG
REPORT = Path('docs/08 - Analysis/2026-10-27_ncm-xsum-prefix-bridge.md')


def integers(values, count, vocabulary):
    if len(values) != count or any(type(v) is not int or not 0 <= v < vocabulary for v in values):
        raise ValueError('complete in-range integer prediction required')


def replay_counts(rows):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('nonempty unique prediction population required')
    totals = {arm: dict(code_matches=0, word_matches=0, nonpad_matches=0, content_matches=0)
              for arm in ('bridge', 'mean')}
    nonpad = content = empty = elements = 0
    squared = {'bridge': [], 'mean': []}
    for row in rows:
        length = row['source_positions']
        if type(length) is not int or not 0 < length <= 512 or row['projected_elements'] != length*768:
            raise ValueError('complete source prefix geometry required')
        for arm in ('native', 'bridge', 'mean'):
            integers(row[arm]['codes'], 200, 4096); integers(row[arm]['words'], 48, 50304)
        gold = row['native']['words']; end = gold.index(50256) if 50256 in gold else len(gold)
        active = [i for i, v in enumerate(gold) if v != 50257]
        meaningful = [i for i in range(end) if gold[i] != 50257]
        nonpad += len(active); content += len(meaningful); empty += not meaningful
        elements += row['projected_elements']
        for arm in ('bridge', 'mean'):
            words = row[arm]['words']; codes = row[arm]['codes']; target = row['native']['codes']
            totals[arm]['code_matches'] += sum(a == b for a, b in zip(codes, target))
            totals[arm]['word_matches'] += sum(a == b for a, b in zip(words, gold))
            totals[arm]['nonpad_matches'] += sum(words[i] == gold[i] for i in active)
            totals[arm]['content_matches'] += sum(words[i] == gold[i] for i in meaningful)
            error = row[arm+'_squared_error']
            if not math.isfinite(error) or error < 0: raise ValueError('finite nonnegative squared error required')
            squared[arm].append(error)
    for arm, values in totals.items():
        values.update(code_agreement=values['code_matches']/(len(rows)*200),
            word_agreement=values['word_matches']/(len(rows)*48),
            nonpad_agreement=values['nonpad_matches']/nonpad if nonpad else 0.,
            content_agreement=values['content_matches']/content if content else 0.,
            projected_MSE=math.fsum(squared[arm])/elements)
    if totals['mean']['projected_MSE'] <= 0: raise ValueError('nonzero mean-control error required')
    relative = totals['bridge']['projected_MSE']/totals['mean']['projected_MSE']
    passed = (not empty and all(totals['bridge'][k] >= CONFIG.word_agreement_gate for k in
              ('word_agreement', 'nonpad_agreement', 'content_agreement')) and relative <= CONFIG.relative_MSE_gate)
    return {'cases': len(rows), 'scores': totals, 'projected_elements': elements,
        'native_nonpad_positions': nonpad, 'native_content_positions': content,
        'empty_native_cases': empty, 'bridge_MSE_relative_to_mean': relative,
        'preservation_gate_passed': passed,
        'classification': 'prefix_bridge_preservation_passed' if passed else 'prefix_bridge_preservation_failed'}


def audit():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only replay required')
    evaluated = study.require(ROOTS['evaluate']); trained = study.require(ROOTS['train'])
    study.require(ROOTS['prepare']); study.require(ROOTS['pilot'])
    for name in ('prepare', 'pilot', 'train', 'evaluate'):
        study.verify_proof(data.read(ROOTS[name]/'producer.json'))
    cases = data.read(ROOTS['prepare']/'cases.json')
    rows = data.read(ROOTS['evaluate']/'outputs.json')
    if (len(rows) != 128 or [r['id'] for r in rows] != [r['id'] for r in cases['probe']]
            or evaluated['probe_cases'] != 128 or trained['updates'] != 5120):
        raise ValueError('complete fixed bridge probe and training required')
    stats = torch.load(ROOTS['pilot']/'normalizers.pt', map_location='cpu', weights_only=True)
    if stats['fit_ids'] != [r['id'] for r in cases['fit']]: raise ValueError('normalization used a different population')
    rng = torch.Generator().manual_seed(CONFIG.seed); expected = []
    for epoch in range(CONFIG.epochs):
        expected += [(epoch+1, cases['fit'][i]['id']) for i in torch.randperm(512, generator=rng).tolist()]
    with (ROOTS['train']/'trace.jsonl').open() as stream:
        trace = list(map(json.loads, stream))
    if (len(trace) != 5120 or [(r['epoch'], r['id']) for r in trace] != expected
            or [r['step'] for r in trace] != list(range(1, 5121))
            or any(not math.isfinite(r['normalized_MSE']) or r['normalized_MSE'] < 0
                   or not math.isfinite(r['gradient_norm']) or r['gradient_norm'] < 0 for r in trace)):
        raise ValueError('every complete fixed update must replay')
    for row, source in zip(rows, cases['probe']):
        if row['source_positions'] != source['source_positions'] or data.sha(row['prediction_path']) != row['prediction_sha256']:
            raise ValueError('raw projected predictions changed')
        value = torch.load(row['prediction_path'], map_location='cpu', weights_only=True)
        if (set(value) != {'target', 'learned', 'mean'}
                or value['target'].shape != (source['source_positions'], 768)
                or value['learned'].shape != value['target'].shape or value['mean'].shape != (768,)
                or any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in value.values())
                or not torch.equal(value['mean'], stats['projected_mean'])):
            raise ValueError('complete raw FP32 projected inputs and exact mean control required')
        for name, arm in (('learned', 'bridge'), ('mean', 'mean')):
            error = float((value[name].double()-value['target'].double()).square().sum())
            if not math.isclose(error, row[arm+'_squared_error'], rel_tol=CONFIG.MSE_replay_rtol, abs_tol=CONFIG.MSE_replay_atol):
                raise ValueError('raw MSE replay differs')
    result = replay_counts(rows); root = ROOTS['audit']
    proof = study.freeze(root, 'audit', [ROOTS[s]/n for s in ('prepare', 'pilot', 'train', 'evaluate')
        for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]+
        [ROOTS['evaluate']/'outputs.json', ROOTS['train']/'trace.jsonl'])
    data.write(root/'scores.json', result)
    study.finish(root, proof, {'status': 'prefix_bridge_raw_evidence_replayed', **result,
        'updates_replayed': 5120}, [root/'scores.json'])


def report():
    result = study.require(ROOTS['audit']); root = ROOTS['results']
    study.verify_proof(data.read(ROOTS['audit']/'producer.json'))
    if root.exists() or REPORT.exists(): raise ValueError('fresh report and aggregate required')
    rows = ['# Frozen-Llama bridge to the released XSum prefix', '',
        f"Classification: `{result['classification']}`. This is a mechanism-preservation experiment, not semantic summary accuracy.", '',
        '| Prefix | Code agreement | All decoded IDs | Non-pad IDs | Native content IDs | Projected MSE |',
        '|---|---:|---:|---:|---:|---:|']
    for arm, v in result['scores'].items():
        rows.append(f"| {arm} | {v['code_agreement']:.2%} | {v['word_agreement']:.2%} | {v['nonpad_agreement']:.2%} | {v['content_agreement']:.2%} | {v['projected_MSE']:.8g} |")
    rows += ['', f"Bridge MSE / mean-control MSE: {result['bridge_MSE_relative_to_mean']:.6f}. Empty native outputs: {result['empty_native_cases']}.",
        'A 3,540,352-parameter MLP alone was trained for 5,120 updates on 512 public TRAIN sources. The 128 probe sources were excluded from bridge fitting; the released XSum teacher may have seen them during pretraining.',
        'Every source fits the native 512-position prefix without truncation. The native output frame remains 48 positions. The Llama body, NCM generator and codec stayed fixed. No application DEV/final case, new semantic judgment, or speed measurement was used.',
        'Passing this gate would motivate further range and task-quality testing. It cannot establish 95% complete-summary accuracy. The separate current hybrid still has 0/128 at both completed application DEV milestones.']
    root.mkdir(); REPORT.write_text('\n'.join(rows)+'\n')
    campaign = {**result, 'hypothesis_id': HYPOTHESIS, 'report': str(REPORT),
        'artifacts': {str(p): data.sha(p) for p in (ROOTS['audit']/'result.json', ROOTS['audit']/'actual_tool_receipt.json', REPORT)}}
    data.write(root/'campaign_results.json', campaign)
    print(json.dumps({'classification': result['classification'], 'report': str(REPORT)}))


def register():
    receipt = data.read(ROOTS['results']/'actual_tool_receipt.json')
    campaign = ROOTS['results']/'campaign_results.json'
    if receipt['actual_tool_exit_code'] != 0 or receipt['artifacts'].get(str(campaign)) != data.sha(campaign):
        raise ValueError('actual report receipt required')
    data.verify(receipt['artifacts'])
    result = data.read(campaign)
    record = {'name': 'Frozen Llama feature bridge into the released conditional XSum NCM',
        'classification': result['classification'], 'task_accuracy_measured': False,
        'bridge_fit_cases': 512, 'bridge_probe_cases': 128,
        'native_content_agreement': result['scores']['bridge']['content_agreement'],
        'relative_prefix_MSE': result['bridge_MSE_relative_to_mean'],
        'description': 'Trained only a small source-prefix bridge on 512 public TRAIN articles and '
            'compared 128 bridge-held-out public TRAIN articles to the unchanged released XSum generator. '
            'Inputs fit 512 positions and outputs remain 48 positions. This measures preservation of '
            'teacher codes, token IDs and source-prefix representations; it does not measure semantic '
            'summary accuracy, full-range generalization or speed. Application final cases remain sealed.'}
    import numpy as np
    if not hasattr(np, 'NaN'): np.NaN = np.nan
    from experiments.neural_architecture_lab.ncm_result_registry import register_records
    register_records(campaign, REPORT, ROOTS['results'], {HYPOTHESIS: record})
    print(json.dumps({'status': 'registered_and_read_back', 'hypothesis': HYPOTHESIS}))


def readback():
    import numpy as np
    if not hasattr(np, 'NaN'): np.NaN = np.nan
    import chromadb
    root = ROOTS['results']; receipt = data.read(root/(HYPOTHESIS+'_registry_readback.json'))
    record = receipt['record']; campaign = root/'campaign_results.json'
    collection = chromadb.PersistentClient(path='data/chroma_db').get_collection('hypotheses')
    exact = collection.get(ids=[HYPOTHESIS], include=['metadatas'])
    semantic = collection.query(query_texts=[record['description']], n_results=10)
    if (exact['ids'] != [HYPOTHESIS] or any(exact['metadatas'][0].get(k) != v for k, v in record.items())
            or HYPOTHESIS not in semantic['ids'][0] or record['campaign_sha256'] != data.sha(campaign)
            or record['report_sha256'] != data.sha(REPORT) or record['task_accuracy_measured']
            or data.read(Path('data/hypothesis_registry')/(HYPOTHESIS+'.json')) != receipt):
        raise ValueError('independent registry readback differs')
    value = {'status': 'independent_registry_readback_passed', 'campaign_sha256': data.sha(campaign),
             'report_sha256': data.sha(REPORT), 'hypothesis_id': HYPOTHESIS}
    data.write(root/'backend_exact_readback.json', value); print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('audit', 'report', 'register', 'readback'))
    globals()[parser.parse_args().stage]()
