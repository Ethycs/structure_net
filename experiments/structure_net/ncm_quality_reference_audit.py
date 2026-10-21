"""Post-outcome human-reference diagnostic, separate from the fixed model grade."""
from pathlib import Path
import shutil

from experiments.structure_net import ncm_quality as q
from experiments.structure_net import ncm_quality_evidence as e
from experiments.neural_architecture_lab.finalize_ncm_quality import summarize,verify

PLAN=Path('docs/07 - Status Reports/2026-10-21_ncm-quality-reference-diagnostic.md')


def main():
    from experiments.structure_net.ncm_llama import tokenizers
    _,llama=tokenizers();primary=e.BASE/'20261021_judge_primary_v1';_,pr=verify(primary)
    if pr['status']!='completed':raise ValueError('complete primary grading required')
    execution=q.read(e.BASE/'20261021_execution_v1/execution.json')
    if execution['status']!='completed':raise ValueError('all planned judge jobs must finish first')
    root=e.BASE/'20261021_reference_diagnostic_v1'
    inputs=[q.DATA/'cases.json',primary/'result.json',e.BASE/'20261021_execution_v1/execution.json',PLAN,Path(__file__)]
    p=e.freeze(root,'post_outcome_human_reference_diagnostic',inputs,'primary')
    for path in (PLAN,Path(__file__)):shutil.copyfile(path,root/'producer_source'/path.name)
    cases=sorted([c for c in q.read(q.DATA/'cases.json')['rows'] if c['cohort']=='dialogsum'],
        key=lambda c:q.digest(['reference_audit',c['id']]))[:20]
    packets=[];lookup={}
    for c in cases:
        candidates=[]
        for n in range(2):
            cid=q.digest(['human_reference',c['id'],n])[:20];text=c['references'][n]
            candidates.append({'candidate_id':cid,'text':text});lookup[cid]=(c,n,text)
        packets.append(e.packet(c,candidates,bool(int(q.digest(['reference_order',c['id']]),16)%2)))
    q.write(root/'packets.json',{'packets':packets});rows=e.execute(packets,'primary',root)
    q.write(root/'judgments.json',{'rows':rows});flat=[]
    for row in rows:
        if row['status']!='valid':continue
        for grade in row['evaluations']:
            c,n,text=lookup[grade['candidate_id']];tokens=len(llama.encode(text,add_special_tokens=False))
            flat.append({'id':c['id'],'reference_index':n,'text':text,'source_sha256':c['source_sha256'],
                'text_sha256':q.digest(text),'grade':grade,'common_output_tokens':tokens,'termination':'human_reference',
                'passed':q.passes(grade,text,tokens)})
    q.write(root/'scores.json',{'rows':flat})
    q.seal(root,p,{'status':'completed' if len(flat)==40 else 'invalid_judgments','summary':summarize(flat) if flat else None,
        'requested_pairs':20,'completed_pairs':len(rows),'invalid_pairs':sum(r['status']!='valid' for r in rows),
        'retried_pairs':sum(len(r['attempts'])>1 for r in rows),'judgments_sha256':q.sha256(root/'judgments.json'),
        'scores_sha256':q.sha256(root/'scores.json'),
        'mean_common_tokens':sum(x['common_output_tokens'] for x in flat)/len(flat) if flat else None,
        'boundary':'Post-outcome diagnostic; two correlated references per dialogue, visible in context, may have errors and different lengths. Descriptive only; does not replace the model endpoint.'})
    e.audit(root,packets,'primary')


if __name__=='__main__':main()
