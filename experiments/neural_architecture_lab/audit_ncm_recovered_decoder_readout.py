"""Replay the four-way decoder diagnostic from native IDs, without model calls."""
from collections import Counter, defaultdict
import html
import json
import os
from pathlib import Path
import re

from experiments.neural_architecture_lab.audit_ncm_codec_roundtrip import read, sha

BASE = Path('data/experiments/ncm_paper_regime')
RUN = BASE/'20261028_recovered_decoder_readout_eval_v1'
DEST = BASE/'20261028_recovered_decoder_readout_replay_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-recovered-decoder-readout.md')
TEST = Path('tests/structure_net/test_ncm_recovered_decoder_readout_audit.py')
ARMS = ('parent_codec_only', 'parent_with_source', 'new_codec_only', 'new_with_source')


def replay(cases, outputs, decode):
    rows = {r['id']: r for r in cases}
    expected = {(r, a) for r in rows for a in ARMS}
    if len(rows) != len(cases) or len(outputs) != len(expected) or {(r['id'], r['arm']) for r in outputs} != expected:
        raise ValueError('each case must have exactly one output per arm')
    counts, codes, exact = defaultdict(Counter), {}, {}
    for value in outputs:
        row = rows[value['id']]
        if any(value[k] != row[v] for k, v in (('split', 'split'), ('target_sha256', 'target_sha256'),
                ('reference', 'native_reference'), ('target_text', 'target_text'))):
            raise ValueError('case/reference binding changed')
        raw, gold, latent = value['raw_word_ids'], row['native_word_ids'], value['codes']
        if (len(raw) != 128 or len(gold) != 128 or len(latent) != 641
                or any(type(v) is not int or not 0 <= v < 30528 for v in raw + gold)
                or any(type(v) is not int or not 0 <= v < 8192 for v in latent)):
            raise ValueError('native ID geometry or vocabulary differs')
        if value['id'] in codes and codes[value['id']] != latent:
            raise ValueError('reference codes differ across arms')
        codes[value['id']] = latent
        if decode(raw) != value['text'] or decode(gold) != value['reference']:
            raise ValueError('independent native rendering differs')
        active = [i for i, v in enumerate(gold) if v != 0]
        content = [i for i in active if gold[i] != 101]
        digits, predicted = (re.findall(r'\d+', v) for v in (value['reference'], value['text']))
        metrics = dict(active_matches=sum(gold[i] == raw[i] for i in active), active_tokens=len(active),
            content_matches=sum(gold[i] == raw[i] for i in content), content_tokens=len(content),
            exact_active_sequence=all(gold[i] == raw[i] for i in active),
            exact_rendered=value['reference'] == value['text'], has_digits=bool(digits),
            digit_sequence_preserved=digits == predicted, empty=not value['text'])
        if any(value[k] != v for k, v in metrics.items()):
            raise ValueError('saved per-case arithmetic differs')
        counter = counts[value['split'], value['arm']]
        counter.update({k: int(v) for k, v in metrics.items() if k != 'digit_sequence_preserved'})
        counter['cases'] += 1
        counter['digit_cases_preserved'] += int(bool(digits) and digits == predicted)
        exact[value['id'], value['arm']] = metrics['exact_rendered']
    summary, paired = {}, {}
    for (split, arm), counter in counts.items():
        value = dict(counter)
        value.update(content_agreement=value['content_matches']/value['content_tokens'],
            exact_rendered_fraction=value['exact_rendered']/value['cases'],
            reconstruction_gate=(value['exact_rendered'] >= 61 and value['content_matches']/value['content_tokens'] >= .99
                if split == 'reserved_component' and value['cases'] == 64 else None))
        summary.setdefault(split, {})[arm] = value
    for split in summary:
        ids = [r['id'] for r in cases if r['split'] == split]
        paired[split] = {}
        for name, a, b in (('decoder_alone', ARMS[0], ARMS[2]), ('decoder_with_source', ARMS[1], ARMS[3]),
                ('new_source_effect', ARMS[2], ARMS[3]), ('parent_source_effect', ARMS[0], ARMS[1])):
            truth = Counter((exact[i, a], exact[i, b]) for i in ids)
            paired[split][name] = dict(cases=len(ids), exact_fixes=truth[False, True],
                exact_regressions=truth[True, False], both_exact=truth[True, True], neither_exact=truth[False, False])
    return summary, paired


def run():
    if DEST.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only replay required')
    receipt, result, producer = (read(RUN/n) for n in ('actual_tool_receipt.json', 'result.json', 'producer.json'))
    if receipt['actual_tool_exit_code'] != 0:
        raise ValueError('actual completed evaluation receipt required')
    expected = {}
    for hashes in (receipt['artifacts'], result['artifacts'], producer['source_hashes'], producer['input_hashes']):
        for p, h in hashes.items():
            key = str(Path(p).resolve())
            if key in expected and expected[key] != h:
                raise ValueError('inconsistent artifact binding')
            expected[key] = h
    for p, h in expected.items():
        if sha(Path(p)) != h:
            raise ValueError('evaluation provenance changed: ' + p)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE/'20261024_long_context_release_inventory_v1/bert-tokenizer', local_files_only=True)
    def decode(raw):
        start = int(raw[0] == 101)
        end = next((i for i in range(start, len(raw)) if raw[i] == 101), len(raw))
        return tok.decode(raw[start:end], skip_special_tokens=True).strip()
    cases, outputs = read(RUN/'cases.json'), read(RUN/'outputs.json')
    if len(cases) != 80 or Counter(r['split'] for r in cases) != {'reserved_component': 64, 'TRAIN_control': 16}:
        raise ValueError('complete fixed component population required')
    metrics, paired = replay(cases, outputs, decode)
    if metrics != result['metrics'] or paired != result['paired_comparisons']:
        raise ValueError('aggregate or paired arithmetic differs')
    a, b = (metrics['reserved_component'][arm] for arm in ('parent_with_source', 'new_with_source'))
    gate = all(b[k] >= a[k] for k in ('exact_rendered', 'content_agreement', 'digit_cases_preserved'))
    if gate != result['descriptive_integration_nonregression']:
        raise ValueError('descriptive integration decision differs')
    proof = {'schema': 'nal.ncm-recovered-decoder-readout-replay.v1', 'model_calls': 0,
        'source_hashes': {str(p): sha(p) for p in (Path(__file__), PLAN, TEST,
            Path('experiments/neural_architecture_lab/audit_ncm_codec_roundtrip.py'))},
        'input_hashes': {str(p): sha(p) for p in (RUN/'actual_tool_receipt.json', RUN/'result.json',
            RUN/'producer.json', RUN/'cases.json', RUN/'outputs.json', Path('pixi.lock'), Path('pyproject.toml'))}}
    DEST.mkdir(); (DEST/'producer_source').mkdir()
    for i, p in enumerate(map(Path, proof['source_hashes'])):
        (DEST/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    def write(name, value):
        with (DEST/name).open('x') as f:
            json.dump(value, f, indent=2); f.write('\n')
    write('producer.json', proof)
    esc = html.escape
    pages = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Decoder readout comparison</title>',
        '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:16px}table{border-collapse:collapse;width:100%}',
        'td,th{border:1px solid #ccc;padding:12px;text-align:left}details{margin:20px 0}.ok{background:#e8f5ed}.bad{background:#fff0ed}</style>',
        '<h1>Decoder reconstruction with and without source attention</h1>',
        '<p>Correct target codes are supplied. This measures reconstruction, not source-only summarization accuracy. ',
        '64 reserved component cases and 16 TRAIN controls. References use the uncased native tokenizer.</p>']
    for row in cases:
        pages += [f'<details><summary>{esc(row["split"])} · {esc(row["id"][:16])}</summary>',
            f'<p>Original summary: {esc(row["target_text"])}</p><p>Native reference: {esc(row["native_reference"])}</p>',
            '<table><tr><th>Configuration</th><th>Decoded summary</th><th>Content tokens</th></tr>']
        for value in (v for v in outputs if v['id'] == row['id']):
            cls = 'ok' if value['exact_rendered'] else 'bad'
            pages += [f'<tr class="{cls}"><td>{esc(value["arm"])}</td><td>{esc(value["text"])}</td>'
                f'<td>{value["content_matches"]}/{value["content_tokens"]}</td></tr>']
        pages += ['</table></details>']
    (DEST/'review.html').write_text('\n'.join(pages + ['</html>']))
    for hashes in (proof['source_hashes'], proof['input_hashes']):
        for p, h in hashes.items():
            if sha(Path(p)) != h: raise ValueError('audit input changed during replay')
    record = {'schema': proof['schema'], 'status': 'all320_outputs_independently_replayed', 'cases': 80,
        'outputs': 320, 'metrics': metrics, 'paired_comparisons': paired,
        'descriptive_integration_nonregression': gate, 'classification': 'target_aware_component_diagnostic',
        'task_accuracy': None, 'new_judge_calls': 0, 'final_test_opened': False, 'speed_tested': False,
        'artifacts': {str(p): sha(p) for p in DEST.iterdir() if p.is_file()}}
    write('result.json', record)
    print(json.dumps({k: v for k, v in record.items() if k not in ('metrics', 'artifacts')}), flush=True)


if __name__ == '__main__':
    run()
