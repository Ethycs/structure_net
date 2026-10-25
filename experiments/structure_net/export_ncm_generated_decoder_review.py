"""Readable full-source review of the fixed generated-code decoder experiment."""
import hashlib
import json
from pathlib import Path

from experiments.structure_net.export_ncm_data_review import HTML

ROOT=Path('data/experiments/ncm_paper_regime')
DEST=ROOT/'20261025_generated_code_decoder_data_review_v1'
TRAIN=ROOT/'20261025_generated_code_decoder_train_v1'
OUTPUTS=ROOT/'20261025_generated_code_decoder_DEV_outputs_v1'
GRADE=ROOT/'20261025_generated_code_decoder_DEV_grade_v1'


def main():
    read=lambda p:json.loads(p.read_text())
    if DEST.exists():raise ValueError('fresh export root required')
    for root in (TRAIN,OUTPUTS,GRADE):
        if read(root/'actual_tool_receipt.json')['actual_tool_exit_code']!=0:
            raise ValueError('completed model and semantic stages required')
    state=read(GRADE/'result.json')
    cohorts=('short_development','long_diagnostic')
    valid=sum(state['metrics'][c]['secondary']['valid'] for c in cohorts)
    passed=sum(state['metrics'][c]['secondary']['passed'] for c in cohorts)
    if valid!=128 or state['task_accuracy']!=passed/128:raise ValueError('complete valid semantic score required')
    train_source=ROOT/'20261025_engineering_attention_repair732_data_v1/cases.json'
    dev_source=ROOT/'20261024_pinned_date_development_sources_v1/cases.json'
    predictions={r['id']:r for r in read(TRAIN/'outputs.json')}
    records=[]
    for row in read(train_source):
        records.append({'id':row['id'],'split':'TRAIN','source':row['source'],
            'source_sha256':row['source_sha256'],'source_component':row['source_component'],
            'stored_summary':row['target_text'],'model_summary':predictions[row['id']]['text'],
            'judge':None,'provenance':row['provenance'],'source_file':str(train_source),'output_file':str(TRAIN/'outputs.json')})
    dev_outputs={r['id']:r for r in read(OUTPUTS/'outputs.json')}
    for cohort,rows in read(dev_source).items():
        scores={r['id']:r for r in read(GRADE/cohort/'secondary_student_case_scores.json')}
        for row in rows:
            score=scores[row['id']]
            if not score['valid'] or score['text']!=dev_outputs[row['id']]['text']:
                raise ValueError('judgment must bind to the exact generated output')
            records.append({'id':row['id'],'split':'DEV short' if cohort=='short_development' else 'DEV long',
                'source':row['source'],'source_sha256':row['source_sha256'],'source_component':row['source_component'],
                'stored_summary':None,'model_summary':score['text'],
                'judge':{'model':state['task_accuracy_judge'],'passed':score['passed'],**score['evaluations'][0]},
                'provenance':{'source_id':row['source_id'],'spans':row['spans']},
                'source_file':str(dev_source),'output_file':str(OUTPUTS/'outputs.json')})
    if len(records)!=860:raise ValueError('all732 TRAIN and128 DEV cases required')
    for row in records:
        if hashlib.sha256(row['source'].encode()).hexdigest()!=row['source_sha256']:
            raise ValueError('export source hash mismatch')
    html=HTML.replace('NCM data and summary review','NCM decoder adaptation review')
    html=html.replace('0/128 (0%)',f'{passed}/128 ({100*passed/128:.2f}%)')
    html=html.replace("el('status').className=r.judge?'fail':'neutral';",
        "el('status').className=r.judge?(r.judge.passed?'pass':'fail'):'neutral';")
    html=html.replace('.neutral{','.pass{color:#185a35;background:#e2f5e8;padding:5px 9px;border-radius:6px;font-size:13px}.neutral{')
    payload=json.dumps(records,ensure_ascii=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    DEST.mkdir();(DEST/'index.html').write_text(html.replace('__DATA__',payload))
    with (DEST/'records.jsonl').open('x') as f:
        for row in records:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    with (DEST/'records.md').open('x') as f:
        for row in records:
            f.write('\n'+'='*80+'\n'+row['split']+' '+row['id']+'\n\nSOURCE\n'+row['source'])
            if row['stored_summary'] is not None:f.write('\n\nSTORED SUMMARY\n'+row['stored_summary'])
            f.write('\n\nADAPTED DECODER OUTPUT\n'+(row['model_summary'] or '(empty output)'))
            if row['judge']:f.write('\n\nJUDGE\n'+row['judge']['reason'])
            f.write('\n')
    (DEST/'result.json').write_text(json.dumps({'records':860,'DEV_passed':passed,'DEV_valid':valid,
        'source_hashes_verified':True,'full_text_exported':True,'sealed_final_test_read':False},indent=2)+'\n')
    print(json.dumps({'viewer':str(DEST/'index.html'),'records':860,'DEV_passed':passed}))


if __name__=='__main__':main()
