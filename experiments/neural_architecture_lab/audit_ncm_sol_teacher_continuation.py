"""Replay all untouched8631 firsts, including schema and capacity negatives."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_sol_teacher_continuation as run
from experiments.neural_architecture_lab.audit_ncm_evidence_teacher_pilot import replay
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST=run.ROOT/'20261024_sol_teacher8631_raw_audit_v1'
REPORT=Path('docs/08 - Analysis/2026-10-24_ncm-sol-teacher8631-continuation.md')


def main():
    run.previous.configure()
    if DEST.exists() or REPORT.exists() or run.q.read(run.WORKFLOW/'actual_tool_receipt.json')['actual_tool_exit_code']!=0:
        raise ValueError('fresh raw replay after actual complete continuation required')
    run.native.vendor.verify(run.q.read(run.WORKFLOW/'producer.json'))
    for root in (run.PREP,run.GENERATE):
        run.native.vendor.terminal(root)
    rows=run.sources()
    labels=run.q.read(run.GENERATE/'labels.json')
    if (run.q.read(run.PREP/'cases.json')!=rows or len(labels)!=8631
            or [r['id'] for r in labels]!=[r['id'] for r in rows]):
        raise ValueError('all8631 unchanged untouched source firsts required')
    reference=run.previous.qualification.reference
    if run.q.read(run.PREP/'public_proof.json')!={'public_file_sha256':reference.PUBLIC_SHA256,'exact_sources':reference.verify_public(rows)}:
        raise ValueError('exact public source proof differs')
    encoding,tokenizer=run.previous.data.storage_encoding(),run.native.tokenizer()
    attempts=set()
    schema_invalid=[]
    for row,value in zip(rows,labels):
        if any(value[k]!=v for k,v in row.items()):
            raise ValueError('original complete source changed')
        draft=replay(row,'draft',value['draft'],encoding)
        if draft is None and value['review'] is not None:
            raise ValueError('no review is allowed after exhausted draft')
        review=replay(row,'review',value['review'],encoding,draft) if draft is not None else None
        text=review['summary'] if review is not None else ''
        lexical=tokenizer.encode(text,add_special_tokens=False)
        stored=len(encoding.encode(text,disallowed_special=()))
        fit=bool(text.strip()) and 0<stored<=128 and len(lexical)<=126 and not set(lexical)&{0,101}
        expected={'teacher_valid':review is not None,'target_text':text,'capacity_fit':fit,
            'stored_tokens':stored,'native_lexical_tokens':len(lexical),'semantic_qualified':False,
            'native_word_ids':[101,*lexical,101]+[0]*(126-len(lexical)) if fit else None}
        if any(value[k]!=v for k,v in expected.items()) or run.q.read(run.GENERATE/'cases'/(row['id']+'.json'))!=value:
            raise ValueError('literal first target or original negative changed')
        if not value['teacher_valid']:
            if run.fatal_exhaustion(value):
                raise ValueError('a completed continuation cannot contain exhausted service failure')
            schema_invalid.append(row['id'])
        for stage in ('draft','review'):
            observation=value[stage]
            if observation is None:
                continue
            paths=[run.GENERATE/'requests'/row['id']/stage/f'attempt_{i}.json' for i in range(len(observation['attempts']))]
            if ([Path(p) for p in observation['attempts']]!=paths
                    or sorted(paths[0].parent.glob('attempt_*.json'))!=paths
                    or (observation['status']=='invalid' and len(paths)!=3)):
                raise ValueError('first attempt order or allowance changed')
            attempts.update(observation['attempts'])
    actual={str(p) for p in (run.GENERATE/'requests').rglob('attempt_*.json')}
    if actual!=attempts or {p.stem for p in (run.GENERATE/'cases').glob('*.json')}!={r['id'] for r in rows}:
        raise ValueError('all and only first pipelines must be retained')
    state=run.q.read(run.GENERATE/'result.json')
    if (state['cases']!=8631 or state['teacher_valid']!=sum(r['teacher_valid'] for r in labels)
            or state['capacity_fit']!=sum(r['capacity_fit'] for r in labels)
            or state['labels_sha256']!=run.file_hash(run.GENERATE/'labels.json')
            or state['student_training_admitted'] or state['canonical_semantic_cache_changed']):
        raise ValueError('complete population or first validity counts differ')
    artifacts={str(Path(__file__)):run.file_hash(__file__)}
    for root in (run.WORKFLOW,run.PREP,run.GENERATE):
        for path in root.rglob('*'):
            if path.is_file() and 'producer_source' not in path.parts:
                artifacts[str(path)]=run.file_hash(path)
    campaign={'hypothesis_id':run.HYPOTHESIS,'classification':'all8631_untouched_firsts_replayed_with_negatives_preserved',
        'cases':8631,'teacher_valid':state['teacher_valid'],'capacity_fit':state['capacity_fit'],
        'schema_exhausted_firsts':schema_invalid,'prior1369firsts_preserved':True,
        'student_training_admitted':False,'student_task_accuracy':None,'individual_semantic_quality_not_measured':True,
        'canonical_semantic_cache_changed':False,'confirmed':False,'final_test_opened':False,'speed_tested':False,'artifacts':artifacts}
    DEST.mkdir()
    run.write_json(DEST/'campaign_results.json',campaign)
    REPORT.write_text('# Untouched teacher sources after the preserved syntax failure\n\n'
        f"All8631first pipelines completed and were independently replayed: {state['teacher_valid']}valid, "
        f"{state['capacity_fit']}capacity-fit. All{len(schema_invalid)}schema-exhausted firsts remain invalid.\n\n"
        'The earlier1369observed firsts, including the invalid review that stopped the prior collector, '
        'were never rerolled. Only previously untouched exact public benchmark sources were submitted. '
        'The qualified recipe was unchanged; a prospective scheduler amendment distinguished isolated '
        'syntax negatives from global service failure. This is data preparation, not student task accuracy. '
        'All component64source families must remain outside future gradients. No sealed final source was opened or hashed.\n')
    print(json.dumps({k:v for k,v in campaign.items() if k not in ('artifacts','schema_exhausted_firsts')}),flush=True)


def register():
    if run.q.read(DEST/'actual_tool_receipt.json')['actual_tool_exit_code']!=0:
        raise ValueError('actual independent raw replay required')
    path=DEST/'campaign_results.json'
    state=run.q.read(path)
    register_records(path,REPORT,DEST,{run.HYPOTHESIS:{'name':'Previously untouched8631 TRAIN teacher firsts',
        'description':'Same qualified public-source Terra draft/Sol review;1369earlier firsts unchanged, all syntax/capacity negatives retained, bounded8workers and global failure stop. Independent raw replay; no source-only student accuracy claim.',
        'classification':state['classification'],'data_preparation_only':True,'student_task_accuracy_measured':False,'cases':8631}})


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('audit','register'))
    args=parser.parse_args()
    {'audit':main,'register':register}[args.stage]()
