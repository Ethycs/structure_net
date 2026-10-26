"""Independently replay saved codec outputs and export a readable comparison."""
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import re

ROOT = Path('data/experiments/ncm_paper_regime')
DATA = ROOT / '20261025_codec_roundtrip_prepare_v1'
RUN = ROOT / '20261025_codec_roundtrip_eval_v1'
DEST = ROOT / '20261025_codec_roundtrip_results_v1'
ARMS = {'released_codec_only', 'parent_codec_only', 'parent_with_source',
        'adapted_codec_only', 'adapted_with_source'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if DEST.exists():
        raise ValueError('fresh result root required')
    if read(RUN/'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('observed successful process completion required')
    result, producer = read(RUN/'result.json'), read(RUN/'producer.json')
    for group in ('source_hashes', 'input_hashes'):
        for path, expected in producer[group].items():
            if sha(Path(path)) != expected:
                raise ValueError('producing artifact changed: ' + path)
    if sha(RUN/'outputs.json') != result['outputs_sha256']:
        raise ValueError('saved outputs differ')
    rows = {r['id']: r for r in read(DATA/'cases.json')}
    outputs = read(RUN/'outputs.json')
    if len(rows) != 80 or len(outputs) != 400:
        raise ValueError('complete fixed comparison required')
    if {(r['id'],r['arm']) for r in outputs} != {(k,a) for k in rows for a in ARMS}:
        raise ValueError('missing/duplicate/extra comparison cell')
    counters = defaultdict(Counter)
    codes = {}
    for value in outputs:
        row = rows[value['id']]
        if (value['split'] != row['split'] or value['target_sha256'] != row['target_sha256']
                or value['reference'] != row['native_reference'] or value['target_text'] != row['target_text']):
            raise ValueError('source/target/arm binding differs')
        gold, raw = row['native_word_ids'], value['raw_word_ids']
        if len(raw) != 128 or len(value['codes']) != 641:
            raise ValueError('native shape differs')
        if value['id'] in codes and codes[value['id']] != value['codes']:
            raise ValueError('encoder/VQ changed across checkpoints')
        codes[value['id']] = value['codes']
        active = [i for i,v in enumerate(gold) if v != 0]
        content = [i for i in active if gold[i] != 101]
        digit_reference, digit_output = [re.findall(r'\d+',v) for v in (value['reference'], value['text'])]
        expected = dict(active_matches=sum(gold[i]==raw[i] for i in active), active_tokens=len(active),
                        content_matches=sum(gold[i]==raw[i] for i in content), content_tokens=len(content),
                        exact_active_sequence=all(gold[i]==raw[i] for i in active),
                        exact_rendered=value['text']==value['reference'], has_digits=bool(digit_reference),
                        digit_sequence_preserved=digit_reference==digit_output, empty=not value['text'])
        if any(value[k] != v for k,v in expected.items()):
            raise ValueError('case arithmetic differs')
        c = counters[value['split'],value['arm']]
        c.update({k:int(v) for k,v in expected.items() if k != 'digit_sequence_preserved'})
        c['cases'] += 1
        c['digit_cases_preserved'] += int(bool(digit_reference) and digit_reference==digit_output)
    metrics = {}
    for (split,arm),counter in counters.items():
        v = dict(counter)
        v['content_agreement'] = v['content_matches']/v['content_tokens']
        v['exact_rendered_fraction'] = v['exact_rendered']/v['cases']
        v['reconstruction_gate'] = (v['exact_rendered']>=61 and v['content_agreement']>=.99
                                    if split=='reserved_component' and v['cases']==64 else None)
        if v != result['metrics'][split][arm]:
            raise ValueError('aggregate differs')
        metrics.setdefault(split,{})[arm] = v
    # Independent rendering check uses the existing tokenizer, not a model.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ROOT/'20261024_long_context_release_inventory_v1/bert-tokenizer',local_files_only=True)
    for value in outputs:
        raw = value['raw_word_ids']
        start = int(raw[0]==101)
        end = next((i for i in range(start,128) if raw[i]==101),128)
        if tok.decode(raw[start:end],skip_special_tokens=True).strip() != value['text']:
            raise ValueError('independent token rendering differs')
    DEST.mkdir()
    pages = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>NCM codec reconstruction</title>',
             '<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#193047}',
             'table{border-collapse:collapse;width:100%}td,th{border:1px solid #d7dfe7;padding:12px;text-align:left;vertical-align:top}',
             '.ok{background:#e8f5ed}.bad{background:#fff0ed}details{margin:24px 0}p{white-space:pre-wrap}</style>',
             '<h1>NCM codec reconstruction: 64 reserved summaries + 16 TRAIN controls</h1>',
             '<p>Correct summary codes are supplied. These are reconstruction tests, not source-only summary accuracy. ',
             'References use the existing uncased tokenizer; original casing is not preserved by that tokenizer.</p>']
    for row in rows.values():
        esc = html.escape
        pages.extend([f'<details><summary>{esc(row["split"])} · {esc(row["id"][:16])}</summary>',
                      f'<h3>Original summary</h3><p>{esc(row["target_text"])}</p>',
                      f'<h3>Native-tokenizer reference</h3><p>{esc(row["native_reference"])}</p>',
                      '<table><tr><th>Configuration</th><th>Decoded summary</th><th>Content tokens</th></tr>'])
        for value in (v for v in outputs if v['id']==row['id']):
            cls = 'ok' if value['exact_rendered'] else 'bad'
            pages.append(f'<tr class="{cls}"><td>{esc(value["arm"])}</td><td>{esc(value["text"])}</td>'
                         f'<td>{value["content_matches"]}/{value["content_tokens"]}</td></tr>')
        pages.extend(['</table>',f'<details><summary>Source</summary><p>{esc(row["source"])}</p></details></details>'])
    pages.append('</html>')
    (DEST/'review.html').write_text('\n'.join(pages))
    record = {'hypothesis_id':'ncm-summary-codec-roundtrip-v1', 'status':'all400_outputs_independently_replayed',
              'classification':'target_aware_component_diagnostic', 'metrics':metrics, 'cases':80,'outputs':400,
              'encoder_codes_identical_across_three_checkpoints':True,
              'source_only_task_accuracy':None,'final_test_opened':False,'speed_tested':False,
              'model_training':False,'auditor_sha256':sha(Path(__file__)),
              'artifacts':{str(p):sha(p) for p in (RUN/'producer.json',RUN/'result.json',RUN/'outputs.json',
                           RUN/'actual_tool_receipt.json',DATA/'cases.json',DATA/'split_proof.json',DEST/'review.html')}}
    (DEST/'campaign_results.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'metrics':metrics}),flush=True)


if __name__ == '__main__':
    main()
