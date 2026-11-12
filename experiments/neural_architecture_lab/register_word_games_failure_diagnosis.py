"""Register audited diagnosis without modifying prior benchmark records."""
import argparse
from pathlib import Path

from experiments.word_games.common import read,sha,write
from neural_architecture_lab.word_games_failure_diagnosis_hypothesis import FailureDiagnosisEvidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','report','output','public-examples','trace-review'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(); audit=read(args.audit)
    if audit['status']!='independently_audited' or audit['programs_reexecuted']!=22:
        raise ValueError('Complete audited diagnosis required')
    for path,expected in audit['artifacts'].items():
        if sha(path)!=expected: raise ValueError('Changed evidence: '+path)
    summary=audit['summary']
    def live(task,condition):
        rows=[r for r in summary['live'] if r['task']==task and r['variant'].endswith('_'+condition)]
        if len(rows)!=2: raise ValueError('Incomplete repeat set')
        return sum(r['official_passed'] and r['targeted_passed'] for r in rows)
    repairs=[r for r in summary['offline'] if r['variant'].endswith('_corrected')]
    if len(repairs)!=5: raise ValueError('Incomplete intervention set')
    record=FailureDiagnosisEvidence(sum(r['official_passed'] and r['targeted_passed'] for r in repairs),
        live(32,'original'),live(32,'corrected'),live(113,'original'),live(113,'corrected')).record()
    record.update(audit=str(args.audit),audit_sha256=sha(args.audit),report=str(args.report),report_sha256=sha(args.report))
    public=read(args.public_examples)
    for row in public['rows']:
        if read(args.public_examples.parent/str(row['task'])/row['variant']/'grade.json')!=row['grade']:
            raise ValueError('Public-example raw grade differs')
    record.update(public_examples=str(args.public_examples),public_examples_sha256=sha(args.public_examples),
        public_example_failures_detected=sum(not r['grade']['passed'] for r in public['rows'] if r['variant']!='canonical'),
        trace_review=str(args.trace_review),trace_review_sha256=sha(args.trace_review))
    local=Path('data/hypothesis_registry')/(record['id']+'.json')
    if local.exists() or args.output.exists(): raise FileExistsError('Existing evidence is append-only')
    from structure_net.logging.standardized_logging import LoggingConfig,StandardizedLogger
    logger=StandardizedLogger(LoggingConfig(project_name='structure_net_completed_hypotheses',
        queue_dir='data/experiment_queue',sent_dir='data/experiment_sent',rejected_dir='data/experiment_rejected',
        enable_wandb=False,auto_upload=False,enable_chromadb=True,chromadb_path='data/chroma_db'))
    logger.log_hypothesis(record); collection=logger.hypotheses_collection
    if collection is None: raise RuntimeError('Registry unavailable')
    collection.update(ids=[record['id']],metadatas=[record])
    exact=collection.get(ids=[record['id']],include=['metadatas'])
    if exact['ids']!=[record['id']]: raise RuntimeError('ID read-back failed')
    observed={k:exact['metadatas'][0].get(k) for k in record}
    if observed!=record: raise RuntimeError('Exact evidence read-back differs')
    semantic=collection.query(query_texts=[record['description']],n_results=10)
    if record['id'] not in semantic['ids'][0]: raise RuntimeError('Semantic read-back missing')
    receipt={'status':'registered_and_read_back','record':record,'exact_readback':observed,'semantic_ids':semantic['ids'][0]}
    write(local,receipt);write(args.output,receipt);print(receipt['status'],record['id'])


if __name__=='__main__': main()
