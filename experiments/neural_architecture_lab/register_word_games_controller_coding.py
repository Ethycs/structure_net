"""Register independently audited controller coding results and verify read-back."""
import argparse
from pathlib import Path

from experiments.word_games.common import read, write, sha
from neural_architecture_lab.word_games_controller_coding_hypothesis import ControllerCodingEvidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','report','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();audit=read(args.audit)
    if audit['status']!='independently_audited':raise ValueError('Independent audit required')
    for path,expected in audit['artifacts'].items():
        if sha(path)!=expected:raise ValueError('Artifact changed: '+path)
    native,translated=(audit['coding'][a] for a in ('native','translated'))
    if native['tasks']!=8 or translated['tasks']!=8 or native['tests']!=32 or translated['tests']!=32:
        raise ValueError('Complete paired task coverage required')
    record=ControllerCodingEvidence(8,native['tasks_passed'],translated['tasks_passed'],
        native['tests_passed'],translated['tests_passed'],translated['clarification_requests']).record()
    record.update(strict_native_correct=audit['strict_completion_coding']['native']['tasks_passed'],
        strict_translated_correct=audit['strict_completion_coding']['translated']['tasks_passed'],
        strict_translated_tests=audit['strict_completion_coding']['translated']['tests_passed'],
        completion_rule_caveat=audit['completion_caveat'])
    record.update(audit=str(args.audit),audit_sha256=sha(args.audit),report=str(args.report),report_sha256=sha(args.report))
    local=Path('data/hypothesis_registry')/(record['id']+'.json')
    if local.exists() or args.output.exists():raise FileExistsError('Use an amendment for existing evidence')
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger=StandardizedLogger(LoggingConfig(project_name='structure_net_completed_hypotheses',
        queue_dir='data/experiment_queue',sent_dir='data/experiment_sent',rejected_dir='data/experiment_rejected',
        enable_wandb=False,auto_upload=False,enable_chromadb=True,chromadb_path='data/chroma_db'))
    logger.log_hypothesis(record);collection=logger.hypotheses_collection
    if collection is None:raise RuntimeError('Registry unavailable')
    collection.update(ids=[record['id']],metadatas=[record])
    exact=collection.get(ids=[record['id']],include=['metadatas'])
    if exact['ids']!=[record['id']]:raise RuntimeError('Exact ID read-back failed')
    observed={key:exact['metadatas'][0].get(key) for key in record}
    if observed!=record:raise RuntimeError('Evidence read-back differs')
    semantic=collection.query(query_texts=[record['description']],n_results=10)
    if record['id'] not in semantic['ids'][0]:raise RuntimeError('Semantic read-back missing')
    receipt={'status':'registered_and_read_back','record':record,'exact_readback':observed,'semantic_ids':semantic['ids'][0]}
    write(local,receipt);write(args.output,receipt)
    print(receipt['status'],record['id'])


if __name__=='__main__':main()
