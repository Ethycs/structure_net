"""Register the audited paired relabeling evaluation with exact read-back."""
import argparse
from pathlib import Path

from experiments.word_games.common import read, sha, write
from neural_architecture_lab.word_games_relabel_coding_hypothesis import RelabelCodingEvidence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','report','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--amend-receipt',type=Path)
    p.add_argument('--previous-report',type=Path)
    args=p.parse_args();audit=read(args.audit)
    if audit['status']!='independently_audited' or not audit['exact_target_outputs_verified']:
        raise ValueError('Complete independent audit required')
    for path,expected in audit['artifacts'].items():
        if sha(path)!=expected:raise ValueError('Artifact changed: '+path)
    native,relabeled=(audit['coding'][arm] for arm in ('native','relabeled'))
    if any(row['tasks']!=8 or row['tests']!=288 for row in (native,relabeled)):
        raise ValueError('Complete coverage required')
    record=RelabelCodingEvidence(native['tasks_passed'],relabeled['tasks_passed'],
        native['tests_passed'],relabeled['tests_passed'],audit['paired_case_disagreements']).record()
    record.update(audit=str(args.audit),audit_sha256=sha(args.audit),report=str(args.report),report_sha256=sha(args.report))
    local=Path('data/hypothesis_registry')/(record['id']+'.json')
    previous=None
    if bool(args.amend_receipt)!=bool(args.previous_report):
        raise ValueError('Report amendments require the previous receipt and archived report')
    if args.amend_receipt:
        previous=read(args.amend_receipt)
        if sha(args.previous_report)!=previous['record']['report_sha256']:
            raise ValueError('Archived report differs from the previous registration')
        evidence=lambda r:{k:v for k,v in r.items() if k not in ('report','report_sha256')}
        if evidence(previous['record'])!=evidence(record):
            raise ValueError('Report amendments cannot change measured evidence')
        local=local.with_name(record['id']+'.'+args.output.stem+'.json')
    if local.exists() or args.output.exists():raise FileExistsError('Use an amendment for existing evidence')
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger=StandardizedLogger(LoggingConfig(project_name='structure_net_completed_hypotheses',
        queue_dir='data/experiment_queue',sent_dir='data/experiment_sent',rejected_dir='data/experiment_rejected',
        enable_wandb=False,auto_upload=False,enable_chromadb=True,chromadb_path='data/chroma_db'))
    logger.log_hypothesis(record);collection=logger.hypotheses_collection
    if collection is None:raise RuntimeError('Registry unavailable')
    collection.update(ids=[record['id']],metadatas=[record])
    exact=collection.get(ids=[record['id']],include=['metadatas'])
    if exact['ids']!=[record['id']]:raise RuntimeError('ID read-back failed')
    observed={key:exact['metadatas'][0].get(key) for key in record}
    if observed!=record:raise RuntimeError('Evidence read-back differs')
    semantic=collection.query(query_texts=[record['description']],n_results=10)
    if record['id'] not in semantic['ids'][0]:raise RuntimeError('Semantic read-back missing')
    receipt={'status':'registered_and_read_back','record':record,'exact_readback':observed,'semantic_ids':semantic['ids'][0]}
    if previous:
        receipt.update(amendment='Report text correction; measured evidence unchanged.',
            previous_receipt=str(args.amend_receipt),previous_receipt_sha256=sha(args.amend_receipt),
            previous_report=str(args.previous_report),previous_report_sha256=sha(args.previous_report))
    write(local,receipt);write(args.output,receipt);print(receipt['status'],record['id'])


if __name__=='__main__':main()
