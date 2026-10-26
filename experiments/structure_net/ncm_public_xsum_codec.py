"""Frozen standalone codec reconstruction on screened public TRAIN summaries."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sys

from experiments.structure_net import ncm_codec_roundtrip as prior
from experiments.structure_net import ncm_public_xsum_data as public
from experiments.structure_net.ncm_public_xsum_data_v2 import EXPORT, INSPECT

HYPOTHESIS = 'ncm-public-xsum-codec-v1'
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-public-xsum-codec-plan.md')
TEST = Path('tests/structure_net/test_ncm_public_xsum_codec.py')
RUNS = {s: prior.ROOT/f'20261026_public_xsum_codec_{s}_v1' for s in ('prepare', 'pilot', 'eval', 'results')}
ARMS = ('released', 'repaired')
COUNTS = {'representative': 256, 'long_stress': 32, 'digit_stress': 32, 'TRAIN_control': 16}
BROAD = prior.ROOT/'20261025_broad_generator_train_v1'
CHECKPOINT_SHA = '74cdce0eb14bb99c801b6ec0a025c092f95249663becc05561af8dfbad6040ff'
CODEC_SHA = 'eb962706f10c693871f9c6df007e9fd3e23004c91c6258f435cdb09abf7fb366'
AUDIT = public.ROOT/'20261026_public_xsum_train_inspection_audit_v1'
BOUNDARIES = {'model_training': False, 'new_judge_requests': 0, 'source_only_task_accuracy': None,
    'final_task_goal_achieved': False, 'final_test_opened': False, 'speed_tested': False,
    'target_aware_diagnostic': True, 'training_data_admitted': False}


def rank(identity):
    return hashlib.sha256((HYPOTHESIS+'\0'+identity).encode()).hexdigest()


def select(rows):
    if len({r['id'] for r in rows}) != len(rows) or len({r['source_sha256'] for r in rows}) != len(rows):
        raise ValueError('unique original article IDs and source texts required')
    representative = sorted(rows, key=lambda r: rank(r['id']))[:256]
    used = {r['id'] for r in representative}
    long = sorted((r for r in rows if r['id'] not in used),
        key=lambda r: (-r['native_lexical_tokens'], rank(r['id'])))[:32]
    used.update(r['id'] for r in long)
    digits = sorted((r for r in rows if r['id'] not in used and re.search(r'\d', r['target_text'])),
        key=lambda r: rank(r['id']))[:32]
    groups = {'representative': representative, 'long_stress': long, 'digit_stress': digits}
    if any(len(values) != COUNTS[group] for group, values in groups.items()):
        raise ValueError('all prospective sample quotas must be filled')
    return [{**r, 'split': group, 'selection_rank': rank(r['id'])} for group, values in groups.items() for r in values]


def frame(ids, specials):
    if not 0 < len(ids) <= 126 or set(ids) & set(specials):
        raise ValueError('complete native target without specials or unknowns required')
    return [101, *ids, 101]+[0]*(126-len(ids))


def gate(metric):
    if metric['cases'] != 256 or metric['has_digits'] < 1:
        raise ValueError('complete representative sample and nonempty digit denominator required')
    return (metric['exact_rendered'] >= 244 and metric['content_matches']*100 >= metric['content_tokens']*99
            and metric['digit_cases_preserved']*100 >= metric['has_digits']*95
            and metric['invalid_outputs'] == 0 and metric['empty'] == 0)


def aggregate(outputs):
    result = {}
    if len({(r['id'], r['arm']) for r in outputs}) != len(outputs):
        raise ValueError('duplicate output identity')
    for group in COUNTS:
        result[group] = {}
        ids_by_arm = []
        for arm in ARMS:
            rows = [r for r in outputs if r['split'] == group and r['arm'] == arm]
            if len(rows) != COUNTS[group]:
                raise ValueError('complete fixed population required for aggregation')
            ids_by_arm.append({r['id'] for r in rows})
            keys = ('active_matches', 'active_tokens', 'content_matches', 'content_tokens',
                'exact_active_sequence', 'exact_rendered', 'has_digits', 'empty', 'invalid_outputs')
            m = {k: sum(r[k] for r in rows) for k in keys}
            m.update(cases=len(rows), digit_cases_preserved=sum(r['has_digits'] and r['digit_sequence_preserved'] for r in rows))
            m['content_agreement'] = m['content_matches']/m['content_tokens']
            m['exact_rendered_fraction'] = m['exact_rendered']/m['cases']
            m['descriptive_reconstruction_gate'] = gate(m) if group == 'representative' else None
            result[group][arm] = m
        if ids_by_arm[0] != ids_by_arm[1]:
            raise ValueError('matched arm identities required')
    return result


def freeze(root, stage, training, environment, inputs):
    sources = {Path(__file__), PLAN, TEST, Path(prior.__file__), Path(public.__file__)}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve())
                    or p.is_relative_to(training.native.vendor.VENDOR.resolve())):
                sources.add(p)
    sources.update(training.native.vendor.VENDOR.rglob('*.py'))
    root.mkdir(); (root/'producer_source').mkdir()
    for i, path in enumerate(sorted(sources)):
        (root/'producer_source'/f'{i}_{path.name}').write_bytes(path.read_bytes())
    proof = {'schema': 'nal.ncm-public-xsum-codec.v1', 'hypothesis_id': HYPOTHESIS,
        'stage': stage, 'conformance': 'EXPLORATORY_UNDERPOWERED', 'environment': environment,
        'config': {'counts': COUNTS, 'arms': ARMS, 'FP32': True, 'TF32': False,
            'standalone_codec': True, 'source_attention_present': False, 'peak_limit_bytes': 4900000000,
            'systems_lifecycle_only_not_quality_evidence': stage == 'pilot', **BOUNDARIES},
        'source_hashes': {str(p): prior.sha(p) for p in sorted(sources)},
        'input_hashes': {str(p): prior.sha(p) for p in dict.fromkeys([
            Path('pyproject.toml'), Path('pixi.lock'), *inputs])}}
    proof['fingerprint'] = training.q.digest(proof)
    public.write(root/'producer.json', proof)
    return proof


def prepare():
    training, environment = prior.runtime()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or training.torch.cuda.is_initialized():
        raise ValueError('CPU preparation required')
    for root in (EXPORT, INSPECT, AUDIT):
        public.completed(root)
    old_prepared = prior.read(prior.RUNS['prepare']/'result.json')
    for name, expected in old_prepared['artifacts'].items():
        if prior.sha(prior.RUNS['prepare']/name) != expected:
            raise ValueError('previous codec-control preparation changed')
    if (prior.read(prior.RUNS['eval']/'actual_tool_receipt.json')['actual_tool_exit_code'] != 0
            or prior.sha(prior.RUNS['eval']/'outputs.json') != prior.read(prior.RUNS['eval']/'result.json')['outputs_sha256']):
        raise ValueError('actual completed codec-control outputs required')
    eligible = []
    with (EXPORT/'rows.jsonl').open() as sources, (INSPECT/'row_metadata.jsonl').open() as metadata:
        for number, pair in enumerate(itertools.zip_longest(sources, metadata), 1):
            if None in pair: raise ValueError('complete row pairing required')
            row, meta = map(json.loads, pair)
            if row['id'] != meta['article_id'] or meta['row_number'] != number:
                raise ValueError('public source/target row binding differs')
            if meta['screen_eligible_not_admitted']:
                if meta['reasons'] or public.text_sha(row['summary']) != meta['summary_sha256']:
                    raise ValueError('unchanged eligible target required')
                eligible.append({'id': row['id'], 'source_sha256': meta['source_sha256'],
                    'target_sha256': meta['summary_sha256'], 'target_text': row['summary'],
                    'native_lexical_tokens': meta['native_lexical_tokens'], 'original_row_number': number})
    if len(eligible) != 163051: raise ValueError('full screened population required')
    rows = select(eligible)
    tokenizer = training.native.tokenizer()
    for row in rows:
        ids = tokenizer.encode(row['target_text'], add_special_tokens=False, truncation=False)
        if len(ids) != row['native_lexical_tokens']: raise ValueError('native token count differs')
        row['native_word_ids'] = frame(ids, tokenizer.all_special_ids)
        row['native_reference'] = training.preparation.rendering.decode(row['native_word_ids'], tokenizer)['text']
    old_rows = prior.read(prior.RUNS['prepare']/'cases.json')
    controls = [r for r in old_rows if r['split'] == 'TRAIN_control']
    if len(controls) != 16: raise ValueError('all previous TRAIN controls required')
    control_keys = ('id', 'split', 'source_sha256', 'target_sha256', 'target_text', 'native_word_ids', 'native_reference', 'native_lexical_tokens')
    rows += [{k: r[k] for k in control_keys} for r in controls]
    if len({r['id'] for r in rows}) != 336: raise ValueError('distinct public and TRAIN-control IDs required')
    old_outputs = prior.read(prior.RUNS['eval']/'outputs.json')
    reference = [{**r, 'arm': 'repaired' if r['arm'] == 'parent_codec_only' else 'released'}
        for r in old_outputs if r['split'] == 'TRAIN_control' and r['arm'] in ('released_codec_only', 'parent_codec_only')]
    if len(reference) != 32: raise ValueError('all saved codec-only control outputs required')
    public_rows = [r for r in rows if r['split'] != 'TRAIN_control']
    pilots = [public_rows[0]['id'], min(public_rows, key=lambda r: (-r['native_lexical_tokens'], rank(r['id'])))['id']]
    if len(set(pilots)) != 2: raise ValueError('distinct first and longest pilot required')
    inputs = [EXPORT/'rows.jsonl', INSPECT/'row_metadata.jsonl']
    inputs += [r/n for r in (EXPORT, INSPECT, AUDIT) for n in ('result.json', 'actual_tool_receipt.json')]
    inputs += [prior.RUNS[s]/n for s, n in [('prepare', 'cases.json'), ('prepare', 'result.json'),
        ('eval', 'outputs.json'), ('eval', 'result.json'), ('eval', 'actual_tool_receipt.json')]]
    inputs += [p for p in (training.native.INVENTORY/'bert-tokenizer').iterdir() if p.is_file()]
    proof = freeze(RUNS['prepare'], 'prepare', training, environment, inputs)
    for name, value in [('cases.json', rows), ('control_outputs.json', reference), ('selection.json', {'pilot_ids': pilots})]:
        public.write(RUNS['prepare']/name, value)
    prior.verify(proof)
    public.write(RUNS['prepare']/'result.json', {'status': 'fixed_public_TRAIN_probe_prepared',
        'counts': dict(Counter(r['split'] for r in rows)), 'eligible_population': len(eligible), **BOUNDARIES,
        'artifacts': {str(p): prior.sha(p) for p in RUNS['prepare'].glob('*.json')}})
    print(json.dumps({'prepared_cases': len(rows), 'pilot_ids': pilots}), flush=True)


def measure(stage):
    training, environment = prior.runtime()
    import torch
    from experiments.structure_net.ncm_cpu_adamw import require_rtx2060
    require_rtx2060()
    public.completed(RUNS['prepare'])
    training.preparation.completed(BROAD)
    if prior.sha(BROAD/'model.pt') != CHECKPOINT_SHA: raise ValueError('fixed BROAD checkpoint required')
    all_rows = prior.read(RUNS['prepare']/'cases.json')
    pilots = prior.read(RUNS['prepare']/'selection.json')['pilot_ids']
    rows = [r for r in all_rows if r['id'] in pilots or r['split'] == 'TRAIN_control'] if stage == 'pilot' else all_rows
    control_outputs = {(r['id'], r['arm']): r for r in prior.read(RUNS['prepare']/'control_outputs.json')}
    pilot = {}
    if stage == 'eval':
        public.completed(RUNS['pilot'])
        pilot = {(r['id'], r['arm']): r for r in prior.read(RUNS['pilot']/'outputs.json')}
    codec = training.native.load_codec()
    tokenizer = training.native.tokenizer()
    released_hash = training.native.vendor.module_hash(codec)
    released_state = {k: v.detach().cpu().clone() for k, v in codec.state_dict().items()}
    saved = torch.load(BROAD/'model.pt', map_location='cpu', weights_only=True, mmap=True)
    repaired_state = {k[len('codec.'):]: v for k, v in saved['model'].items() if k.startswith('codec.')}
    if set(repaired_state) != set(released_state): raise ValueError('strict full codec state required')
    inputs = [*RUNS['prepare'].glob('*.json'), *[BROAD/n for n in ('model.pt', 'result.json', 'actual_tool_receipt.json')],
        training.native.RELEASE/'files/lm1b/tokenizer.pt', training.native.RELEASE/'files/lm1b/small/config.yaml']
    if stage == 'eval': inputs += [RUNS['pilot']/n for n in ('outputs.json', 'result.json', 'actual_tool_receipt.json')]
    proof = freeze(RUNS[stage], stage, training, environment, inputs)
    torch.cuda.reset_peak_memory_stats()
    outputs, previous_codes, model_hashes = [], {}, {}
    with torch.no_grad():
        for arm in ARMS:
            codec.load_state_dict(released_state if arm == 'released' else repaired_state, strict=True)
            codec.float().eval().requires_grad_(False)
            before = training.native.vendor.module_hash(codec)
            if before != (released_hash if arm == 'released' else CODEC_SHA): raise ValueError('selected codec identity differs')
            model_hashes[arm] = before
            for row in rows:
                gold = torch.tensor([row['native_word_ids']], device='cuda', dtype=torch.long)
                codes = codec.encode(gold)
                as_list = codes[0].cpu().tolist()
                if len(as_list) != 641 or any(not 0 <= c < 8192 for c in as_list): raise ValueError('complete code geometry required')
                if row['id'] in previous_codes and previous_codes[row['id']] != as_list: raise ValueError('encoder/VQ differs between arms')
                previous_codes[row['id']] = as_list
                def predict():
                    logits = codec.decode(codes)
                    if logits.shape != (1, 128, 30528) or not torch.isfinite(logits).all(): raise ValueError('finite full logits required')
                    return logits.argmax(-1)[0].cpu().tolist()
                raw = predict()
                if stage == 'pilot' and (raw != predict() or not torch.equal(codes, codec.encode(gold))): raise ValueError('repeated pilot differs')
                reference = control_outputs.get((row['id'], arm))
                if reference is not None and (raw != reference['raw_word_ids'] or as_list != reference['codes']): raise ValueError('standalone codec differs from prior control')
                decoded = training.preparation.rendering.decode(raw, tokenizer)
                value = {'id': row['id'], 'split': row['split'], 'arm': arm, 'target_sha256': row['target_sha256'],
                    'codes': as_list, 'raw_word_ids': raw, 'text': decoded['text'],
                    'invalid_outputs': int(any(v >= len(tokenizer) for v in raw)),
                    'output_native_tokens': len(tokenizer.encode(decoded['text'], add_special_tokens=False)),
                    **prior.metrics(row['native_word_ids'], raw, row['native_reference'], decoded['text'])}
                if (row['id'], arm) in pilot and pilot[row['id'], arm] != value: raise ValueError('pilot/full output differs')
                outputs.append(value)
            if training.native.vendor.module_hash(codec) != before: raise ValueError('inference changed codec tensors')
            print(json.dumps({'arm': arm, 'cases': len(rows), 'outputs': len(outputs)}), flush=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    if peak > 4900000000: raise ValueError('CUDA allocation cap exceeded')
    public.write(RUNS[stage]/'outputs.json', outputs)
    prior.verify(proof)
    result = {'status': 'frozen_public_TRAIN_codec_probe_complete', 'cases': len(rows), 'outputs': len(outputs),
        'codec_hashes': model_hashes, 'models_unchanged': True, 'all_control_outputs_identical': True,
        'pilot_repeat_exact': stage == 'pilot', 'peak_allocated_bytes': peak, **BOUNDARIES,
        'metrics': aggregate(outputs) if stage == 'eval' else None,
        'artifacts': {str(p): prior.sha(p) for p in RUNS[stage].glob('*.json')}}
    public.write(RUNS[stage]/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('artifacts', 'metrics')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'pilot', 'eval'))
    stage = parser.parse_args().stage
    root = RUNS[stage]; existed = root.exists()
    try:
        prepare() if stage == 'prepare' else measure(stage)
    except Exception as error:
        if not existed and root.exists():
            public.write(root/'failure.json', {'error_type': type(error).__name__, 'error': str(error), **BOUNDARIES})
        raise
