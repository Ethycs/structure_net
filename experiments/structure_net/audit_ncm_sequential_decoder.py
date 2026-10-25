"""Replay decoder TRAIN counts from preserved raw outputs and training traces."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_sequential_decoder as run
from experiments.structure_net.ncm_memory_data import file_hash, write_json


def audit(stage):
    run.native.vendor.corpus.quality.dependencies()
    root = run.RUNS[stage]
    run.preparation.completed(root)
    state = run.q.read(root / 'result.json')
    proof = run.q.read(root / 'producer.json')
    rows = {r['id']: r for r in run.q.read(run.base.DATA / 'cases.json')}
    controls = run.q.read(run.base.DATA / 'selection.json')['diagnostic_ids']
    expected = list(rows) if stage == 'train' else controls
    tokenizer = run.native.tokenizer()
    artifacts = [root / n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json', 'trace.jsonl',
        'initial_oracle_outputs.json', 'initial_source_outputs.json', 'oracle_outputs.json', 'source_outputs.json', 'model.pt')]
    outputs = run.q.read(root / 'oracle_outputs.json')
    source = run.q.read(root / 'source_outputs.json')
    before = run.q.read(root / 'initial_source_outputs.json')
    if [r['id'] for r in outputs] != expected or [r['id'] for r in source] != controls:
        raise ValueError('complete ordered evaluation populations required')
    if [r['codes'] for r in before] != [r['codes'] for r in source]:
        raise ValueError('source-only codes changed')
    counts = []
    for values in (outputs, source):
        matched = active = exact = empty = 0
        for value in values:
            row = rows[value['id']]
            raw, gold = value['raw_word_ids'], row['native_word_ids']
            decoded = run.preparation.rendering.decode(raw, tokenizer)
            reference = run.preparation.rendering.decode(gold, tokenizer)
            correct = sum(a == b for a, b in zip(raw, gold) if b != 0)
            positions = sum(b != 0 for b in gold)
            if (len(raw) != 128 or value['active_word_matches'] != correct or value['active_words'] != positions
                    or value['text'] != decoded['text'] or value['reference_text'] != reference['text']
                    or value['target_sha256'] != row['target_sha256']
                    or value['source_sha256'] != row['source_sha256']
                    or value['rendered_exact'] != (decoded['text'] == reference['text'])):
                raise ValueError('raw reconstruction counts or source/target binding differs')
            matched += correct; active += positions
            exact += decoded['text'] == reference['text']; empty += not decoded['text']
        counts.append(dict(cases=len(values), active_word_matches=matched, active_words=active,
            active_word_agreement=matched/active, rendered_exact_cases=exact, empty_outputs=empty))
    if counts != [state['final_oracle'], state['source_only16']]:
        raise ValueError('reported reconstruction metrics differ from raw IDs')
    trace = [json.loads(line) for line in (root / 'trace.jsonl').read_text().splitlines()]
    updates = {'pilot': 2, 'capacity': 256, 'train': 1024}[stage]
    if [r['step'] for r in trace] != list(range(1, updates+1)):
        raise ValueError('complete fixed update sequence required')
    exposures = {identity: 0 for identity in expected}
    for record in trace:
        if len(record['examples']) != 4 or record['lr'] != run.rate(record['step'], 256 if stage == 'pilot' else updates):
            raise ValueError('fixed schedule or accumulation differs')
        for example in record['examples']:
            exposures[example['id']] += 1
    if stage != 'pilot' and max(exposures.values()) - min(exposures.values()) > 1:
        raise ValueError('uniform complete-epoch sampling differs')
    gate = (counts[0]['cases'] == counts[1]['cases'] == 16 and counts[0]['active_word_agreement'] == 1
        and counts[0]['rendered_exact_cases'] == counts[1]['rendered_exact_cases'] == 16 and not counts[1]['empty_outputs'])
    full_gate = stage == 'train' and counts[0]['active_word_agreement'] >= .99 and counts[0]['rendered_exact_cases']/len(expected) >= .95 and not counts[0]['empty_outputs']
    if (gate != state['capacity_pass'] or full_gate != state['full_decoder_reconstruction_pass']
            or state['checkpoint_sha256'] != file_hash(root / 'model.pt')
            or state['producer_fingerprint'] != proof['fingerprint'] or not state['strict_reload_passed']
            or state['task_accuracy'] is not None or state['final_test_opened'] or state['speed_tested']):
        raise ValueError('scientific gate or artifact identity differs')
    destination = run.ROOT / ('20261025_sequential_decoder_' + stage + '_audit_v1')
    destination.mkdir()
    result = dict(status='raw_decoder_TRAIN_counts_and_trace_replayed', stage=stage,
        oracle=counts[0], source_only16=counts[1], updates=updates, presentations=sum(exposures.values()),
        min_exposures=min(exposures.values()), max_exposures=max(exposures.values()),
        capacity_pass=gate, full_decoder_reconstruction_pass=full_gate,
        first16_mean_CE=sum(r['word_CE'] for r in trace[:16])/len(trace[:16]),
        last16_mean_CE=sum(r['word_CE'] for r in trace[-16:])/len(trace[-16:]),
        artifacts={str(p): file_hash(p) for p in artifacts},
        auditor_sha256=file_hash(Path(__file__)), task_accuracy=None, semantic_quality_measured=False,
        final_test_opened=False, speed_tested=False)
    write_json(destination / 'campaign_results.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=tuple(run.RUNS))
    audit(parser.parse_args().stage)
