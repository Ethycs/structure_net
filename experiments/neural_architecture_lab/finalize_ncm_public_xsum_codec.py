"""Replay fixed selection, raw decoding metrics, and codec identities on CPU."""
import argparse
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path

from experiments.structure_net import ncm_public_xsum_codec_recovery as probe

REPORT = Path('docs/08 - Analysis/2026-10-26_ncm-public-xsum-codec.md')
REGISTRAR = Path('experiments/neural_architecture_lab/register_ncm_public_xsum_codec.py')


def main():
    training, environment = probe.prior.runtime()
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU replay only')
    states = {}
    for stage in ('prepare', 'pilot', 'eval'):
        root = probe.RUNS[stage]
        states[stage] = probe.public.completed(root)
        probe.prior.verify(probe.prior.read(root/'producer.json'))
    rows = probe.prior.read(probe.RUNS['prepare']/'cases.json')
    by_id = {r['id']: r for r in rows}
    if len(rows) != 336 or len(by_id) != 336:
        raise ValueError('all public cases and controls required')
    # Replay the predeclared selection from the entire pinned population.
    candidates = []
    with (probe.EXPORT/'rows.jsonl').open() as sources, (probe.INSPECT/'row_metadata.jsonl').open() as metadata:
        for number, pair in enumerate(itertools.zip_longest(sources, metadata), 1):
            if None in pair: raise ValueError('source/metadata length differs')
            row, meta = map(json.loads, pair)
            if row['id'] != meta['article_id'] or number != meta['row_number']:
                raise ValueError('row identity differs')
            if meta['screen_eligible_not_admitted']:
                candidates.append({'id': row['id'], 'source_sha256': meta['source_sha256'],
                    'target_sha256': meta['summary_sha256'], 'target_text': row['summary'],
                    'native_lexical_tokens': meta['native_lexical_tokens'], 'original_row_number': number})
                if row['id'] in by_id and probe.public.text_sha(row['document']) != by_id[row['id']]['source_sha256']:
                    raise ValueError('selected original source hash differs')
    if len(candidates) != 163051:
        raise ValueError('complete eligible population required')
    selected = probe.select(candidates)
    if [r['id'] for r in selected] != [r['id'] for r in rows if r['split'] != 'TRAIN_control']:
        raise ValueError('fixed sample selection differs')
    for row in selected:
        if any(by_id[row['id']].get(k) != v for k, v in row.items()):
            raise ValueError('selected original target/provenance differs')
    del candidates, selected
    gc.collect()
    tokenizer = training.native.tokenizer()
    for row in rows:
        lexical = tokenizer.encode(row['target_text'], add_special_tokens=False, truncation=False)
        if (probe.frame(lexical, tokenizer.all_special_ids) != row['native_word_ids']
                or probe.public.text_sha(row['target_text']) != row['target_sha256']
                or training.preparation.rendering.decode(row['native_word_ids'], tokenizer)['text'] != row['native_reference']):
            raise ValueError('original complete target framing differs')
    outputs = probe.prior.read(probe.RUNS['eval']/'outputs.json')
    mapping = {(r['id'], r['arm']): r for r in outputs}
    if len(mapping) != len(outputs) or set(mapping) != {(i, arm) for i in by_id for arm in probe.ARMS}:
        raise ValueError('all672 unique matched outputs required')
    for output in outputs:
        row = by_id[output['id']]
        if output['split'] != row['split'] or output['target_sha256'] != row['target_sha256']:
            raise ValueError('output target binding differs')
        if (len(output['raw_word_ids']) != 128 or any(type(v) is not int or not 0 <= v < 30528 for v in output['raw_word_ids'])
                or len(output['codes']) != 641 or any(type(v) is not int or not 0 <= v < 8192 for v in output['codes'])):
            raise ValueError('complete raw native geometry required')
        decoded = training.preparation.rendering.decode(output['raw_word_ids'], tokenizer)['text']
        replayed = probe.prior.metrics(row['native_word_ids'], output['raw_word_ids'], row['native_reference'], decoded)
        replayed.update(text=decoded, invalid_outputs=int(any(v >= len(tokenizer) for v in output['raw_word_ids'])),
                        output_native_tokens=len(tokenizer.encode(decoded, add_special_tokens=False)))
        if any(output[k] != v for k, v in replayed.items()): raise ValueError('raw output metric differs')
        if output['codes'] != mapping[row['id'], probe.ARMS[0]]['codes']: raise ValueError('encoder codes differ by arm')
    for output in probe.prior.read(probe.RUNS['pilot']/'outputs.json'):
        if output != mapping[output['id'], output['arm']]: raise ValueError('pilot/full mismatch')
    for old in probe.prior.read(probe.RUNS['prepare']/'control_outputs.json'):
        now = mapping[old['id'], old['arm']]
        if any(old[k] != now[k] for k in ('codes', 'raw_word_ids', 'text')): raise ValueError('prior TRAIN control mismatch')
    # Independently hash checkpoint tensor dictionaries without loading a CUDA model.
    def state_hash(state):
        h = hashlib.sha256()
        for name, value in sorted(state.items()):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    released_path = training.native.RELEASE/'files/lm1b/tokenizer.pt'
    released = torch.load(released_path, map_location='cpu', weights_only=True, mmap=True)['model']
    repaired = torch.load(probe.BROAD/'model.pt', map_location='cpu', weights_only=True, mmap=True)['model']
    expected = {'released': state_hash(released),
                'repaired': state_hash(probe.codec_state(repaired, released))}
    if expected['repaired'] != probe.CODEC_SHA or states['eval']['codec_hashes'] != expected or states['pilot']['codec_hashes'] != expected:
        raise ValueError('codec tensor hashes differ')
    metrics = probe.aggregate(outputs)
    if metrics != states['eval']['metrics']: raise ValueError('aggregate replay differs')
    if any(not states[s]['models_unchanged'] or not states[s]['all_control_outputs_identical'] for s in ('pilot', 'eval')):
        raise ValueError('unchanged codec/control checks required')
    primary = metrics['representative']['repaired']
    classification = ('repaired_public_TRAIN_codec_gate_passed_task_unproven' if primary['descriptive_reconstruction_gate']
                      else 'repaired_public_TRAIN_codec_reconstruction_gate_failed')
    root = probe.RUNS['results']
    inputs = [p for stage in ('prepare', 'pilot', 'eval') for p in probe.RUNS[stage].glob('*.json')]
    inputs += [Path(__file__), REGISTRAR, released_path, probe.BROAD/'model.pt']
    proof = probe.freeze(root, 'results', training, environment, inputs)
    probe.prior.verify(proof)
    campaign = {'hypothesis_id': probe.HYPOTHESIS, 'classification': classification,
        'conformance': 'EXPLORATORY_UNDERPOWERED', 'counts': probe.COUNTS,
        'outputs': len(outputs), 'metrics': metrics, 'codec_hashes': expected,
        'selection_replayed_from_163051_candidates': True, 'all_raw_metrics_replayed': True,
        'all_prior_TRAIN_controls_equal': True, 'pilot_full_equal': True,
        'independent_model_forwards_reexecuted': False, 'peak_allocated_bytes': states['eval']['peak_allocated_bytes'],
        'no_source_attention_or_generator': True, **probe.BOUNDARIES,
        'artifacts': {str(p): probe.prior.sha(p) for p in [*inputs, root/'producer.json']}}
    probe.public.write(root/'campaign_results.json', campaign)
    print(json.dumps({k: v for k, v in campaign.items() if k not in ('artifacts', 'metrics')}), flush=True)
    print(json.dumps({'representative': metrics['representative']}), flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(__doc__).parse_args()
    main()
