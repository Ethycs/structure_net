"""Export complete open TRAIN/DEV text and results for the user's inspection."""
import json
from pathlib import Path

ROOT = Path('data/experiments/ncm_paper_regime')
DEST = ROOT / '20261025_pixi_data_review_v1'

HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NCM data and summary review</title>
<style>
:root{font-family:system-ui,sans-serif;color:#17233a;background:#f4f6fa}*{box-sizing:border-box}
body{margin:0}header{padding:22px 28px;background:white;border-bottom:1px solid #dbe2ec}
h1{margin:0 0 8px;font-size:25px}p{line-height:1.55}.sub{color:#52617b;margin:0}
.toolbar{padding:16px 28px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
input,select,button,a.download{font:inherit;padding:9px 12px;border:1px solid #c6d1e1;border-radius:7px;background:white;color:inherit}
input{flex:1;min-width:180px}button{cursor:pointer}button:disabled{opacity:.4}
main{display:grid;grid-template-columns:280px minmax(0,1fr);gap:20px;margin:0 28px 28px}
nav{max-height:76vh;overflow:auto;border:1px solid #dbe2ec;background:white;border-radius:10px}
nav button{display:block;border:0;border-radius:0;border-bottom:1px solid #edf0f5;width:100%;text-align:left;font-size:13px;line-height:1.45}
nav button.active{background:#e4edff;border-left:4px solid #2463bc}.tag{font-weight:650;display:block;margin-bottom:4px}
article{min-width:0}.heading{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:14px}
h2{font-size:19px;margin:0}.muted{font-size:13px;color:#61708a}.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px}
.card{background:white;border:1px solid #dbe2ec;border-radius:10px;padding:18px;margin-bottom:16px}
h3{font-size:15px;margin:0 0 12px}.text{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.65;font-size:14px;max-height:66vh;overflow:auto}
.empty{color:#8f3445;font-style:italic}.fail{color:#963a45;background:#ffe9eb;padding:5px 9px;border-radius:6px;font-size:13px}
.neutral{color:#52617b;background:#edf1f6;padding:5px 9px;border-radius:6px;font-size:13px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.judge{border-left:4px solid #bb5662}
a{color:#245eb1}details{margin-top:14px}summary{cursor:pointer}#counter{font-size:13px;color:#52617b}
@media(max-width:850px){main{grid-template-columns:1fr;margin:0 14px 14px}.grid{grid-template-columns:1fr}nav{max-height:190px}.toolbar,header{padding:16px}.text{max-height:50vh}}
</style>
<header><h1>NCM data and summary review</h1>
<p class="sub">732 training examples · 128 held-out development examples · complete source text</p>
<p class="sub">Held-out semantic score: <strong>0/128 (0%)</strong>, judged by gpt-5.6-terra. Training references are existing stored summaries, not independently verified gold.</p></header>
<div class="toolbar"><select id="split" aria-label="Dataset"><option value="DEV">Held-out DEV (128)</option><option value="DEV short">DEV short (64)</option><option value="DEV long">DEV long (64)</option><option value="TRAIN">Training (732)</option><option value="">All (860)</option></select>
<input id="search" type="search" placeholder="Search sources, summaries, or case IDs" aria-label="Search data"><span id="counter"></span>
<a class="download" href="records.jsonl" download>Download JSONL</a><a class="download" href="records.txt">Plain text</a></div>
<main><nav id="list" aria-label="Cases"></nav><article>
<div class="heading"><button id="prev">← Previous</button><button id="next">Next →</button><h2 id="title"></h2><span id="status"></span></div>
<div class="grid"><section class="card"><h3>Original source</h3><div id="source" class="text"></div></section><div>
<section class="card"><h3>NCM generated summary</h3><div id="model" class="text"></div></section>
<section class="card" id="referenceCard"><h3>Stored training summary</h3><div id="reference" class="text"></div></section>
<section class="card judge" id="judgeCard"><h3>Semantic judge</h3><div id="judge" class="text"></div><details><summary>Claims and source evidence</summary><pre id="claims"></pre></details></section>
</div></div><details class="card"><summary>Case identity and provenance</summary><pre id="meta"></pre></details>
</article></main><script id="data" type="application/json">__DATA__</script>
<script>
const records=JSON.parse(document.getElementById('data').textContent);
const el=id=>document.getElementById(id);let filtered=[],position=0;
function show(){
 const r=filtered[position];el('counter').textContent=`${filtered.length} matches`;
 el('prev').disabled=position<=0;el('next').disabled=position>=filtered.length-1;
 if(!r){el('title').textContent='No matching cases';for(const id of ['source','model','reference','judge','claims','meta','status'])el(id).textContent='';return;}
 el('title').textContent=`${r.split} · ${position+1}/${filtered.length}`;
 el('source').textContent=r.source;el('model').textContent=r.model_summary||'(empty output)';el('model').classList.toggle('empty',!r.model_summary);
 el('referenceCard').hidden=r.stored_summary===null;el('reference').textContent=r.stored_summary||'';
 el('judgeCard').hidden=!r.judge;el('status').className=r.judge?'fail':'neutral';
 el('status').textContent=r.judge?(r.judge.passed?'Passed':'Failed'):'Training: not semantically graded';
 el('judge').textContent=r.judge?`Faithful: ${r.judge.faithful} · Salient: ${r.judge.salient} · Readable: ${r.judge.readable}\n\n${r.judge.reason}`:'';
 el('claims').textContent=r.judge?JSON.stringify(r.judge.claims,null,2):'';
 el('meta').textContent=JSON.stringify({id:r.id,source_sha256:r.source_sha256,source_component:r.source_component,source_file:r.source_file,output_file:r.output_file,provenance:r.provenance},null,2);
 [...el('list').children].forEach((b,i)=>b.classList.toggle('active',i===position));
}
function filter(){const scope=el('split').value,term=el('search').value.toLowerCase();
 filtered=records.filter(r=>(!scope||r.split.startsWith(scope))&&(!term||[r.id,r.source,r.stored_summary||'',r.model_summary].join('\n').toLowerCase().includes(term)));
 position=0;el('list').replaceChildren();filtered.forEach((r,i)=>{const b=document.createElement('button'),tag=document.createElement('span');tag.className='tag';tag.textContent=`${i+1}. ${r.split}`;b.append(tag,document.createTextNode(r.source.slice(0,115)));b.onclick=()=>{position=i;show()};el('list').append(b)});show();}
el('prev').onclick=()=>{if(position>0){position--;show()}};el('next').onclick=()=>{if(position+1<filtered.length){position++;show()}};
el('split').onchange=filter;el('search').oninput=filter;filter();
</script></html>'''


def read(path):
    return json.loads(path.read_text())


def main():
    if DEST.exists():
        raise ValueError('Fresh data-review export required')
    train_path = ROOT / '20261025_engineering_attention_repair732_data_v1/cases.json'
    train_out = ROOT / '20261025_sequential_full732_pixi_train_v1/outputs.json'
    dev_path = ROOT / '20261024_pinned_date_development_sources_v1/cases.json'
    dev_out = ROOT / '20261025_sequential_full732_pixi_DEV_outputs_v1/outputs.json'
    grade_root = ROOT / '20261025_sequential_full732_pixi_DEV_grade_v1'
    result = read(grade_root / 'result.json')
    if result['task_accuracy'] != 0 or result['metrics']['short_development']['secondary']['valid'] != 64 or result['metrics']['long_diagnostic']['secondary']['valid'] != 64:
        raise ValueError('Export headline must match the complete valid observed result')
    train, cohorts = read(train_path), read(dev_path)
    records = []
    train_outputs = {r['id']: r for r in read(train_out)}
    for row in train:
        records.append({'id': row['id'], 'split': 'TRAIN', 'source': row['source'],
            'source_sha256': row['source_sha256'], 'source_component': row['source_component'],
            'stored_summary': row['target_text'], 'model_summary': train_outputs[row['id']]['text'],
            'judge': None, 'provenance': row['provenance'],
            'source_file': str(train_path), 'output_file': str(train_out)})
    outputs = {r['id']: r for r in read(dev_out)}
    for cohort, rows in cohorts.items():
        scores = {r['id']: r for r in read(grade_root / cohort / 'secondary_student_case_scores.json')}
        for row in rows:
            score = scores[row['id']]
            if not score['valid'] or score['text'] != outputs[row['id']]['text']:
                raise ValueError('Exact valid generated candidate required')
            records.append({'id': row['id'], 'split': 'DEV short' if cohort == 'short_development' else 'DEV long',
                'source': row['source'], 'source_sha256': row['source_sha256'],
                'source_component': row['source_component'], 'stored_summary': None,
                'model_summary': score['text'], 'judge': {'model': result['task_accuracy_judge'],
                    'passed': score['passed'], **score['evaluations'][0]},
                'provenance': {'source_id': row['source_id'], 'spans': row['spans']},
                'source_file': str(dev_path), 'output_file': str(dev_out)})
    if len(records) != 860 or len(train) != 732:
        raise ValueError('Complete open732 TRAIN and128 DEV export required')
    DEST.mkdir()
    payload = json.dumps(records, ensure_ascii=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    (DEST / 'index.html').write_text(HTML.replace('__DATA__', payload))
    with (DEST / 'records.jsonl').open('w') as output:
        for row in records:
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (DEST / 'records.txt').open('w') as output:
        for row in records:
            output.write('\n' + '=' * 80 + '\n' + row['split'] + ' · ' + row['id'] + '\n\nSOURCE\n' + row['source'])
            if row['stored_summary'] is not None:
                output.write('\n\nSTORED TRAINING SUMMARY\n' + row['stored_summary'])
            output.write('\n\nNCM GENERATED SUMMARY\n' + (row['model_summary'] or '(empty output)'))
            if row['judge']:
                output.write('\n\nJUDGE: ' + row['judge']['reason'])
            output.write('\n')
    (DEST / 'result.json').write_text(json.dumps({'status': 'complete_open_data_review_export',
        'TRAIN': 732, 'DEV': 128, 'DEV_passed': 0, 'source_text_truncated': False,
        'sealed_final_test_read': False, 'network_resources_required': False}, indent=2))
    print(json.dumps({'viewer': str(DEST / 'index.html'), 'records': len(records)}))


if __name__ == '__main__':
    main()
