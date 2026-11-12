"""Register only audited complete results, with exact and semantic read-back."""
import argparse
from pathlib import Path

from experiments.word_games.common import read, write, sha
from neural_architecture_lab.word_games_structured_coding_hypothesis import StructuredCodingEvidence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','report','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit=read(a.audit)
    if audit['status']!='independently_audited':raise ValueError('Audited evidence required')
    for path, expected in audit['artifacts'].items():
        if sha(path)!=expected:raise ValueError('Artifact changed: '+path)
    record=StructuredCodingEvidence(audit['proof']['cases'],8,
        audit['coding']['native']['tasks_passed'],audit['coding']['translated']['tasks_passed'],
        audit['renamed_key_recovery']['correct'],audit['renamed_key_recovery']['total'],
        audit['recovery']['translated']['canaries_recovered']).record()
    record.update(audit=str(a.audit),audit_sha256=sha(a.audit),report=str(a.report),report_sha256=sha(a.report))
    receipt_path=Path('data/hypothesis_registry')/(record['id']+'.json')
    if a.output.exists() or receipt_path.exists():raise FileExistsError('Use an explicit amendment for existing evidence')
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger=StandardizedLogger(LoggingConfig(project_name='structure_net_completed_hypotheses',
        queue_dir='data/experiment_queue',sent_dir='data/experiment_sent',rejected_dir='data/experiment_rejected',
        enable_wandb=False,auto_upload=False,enable_chromadb=True,chromadb_path='data/chroma_db'))
    logger.log_hypothesis(record); collection=logger.hypotheses_collection
    if collection is None:raise RuntimeError('Registry unavailable')
    collection.update(ids=[record['id']],metadatas=[record])
    exact=collection.get(ids=[record['id']],include=['metadatas'])
    if exact['ids']!=[record['id']]:raise RuntimeError('Exact ID read-back differs')
    # Chroma merges updates; the logger also adds timestamps and empty
    # question/prediction fields. Verify every declared evidence field exactly.
    observed={key:exact['metadatas'][0].get(key) for key in record}
    if observed!=record:raise RuntimeError('Exact evidence read-back differs')
    semantic=collection.query(query_texts=[record['description']],n_results=10)
    if record['id'] not in semantic['ids'][0]:raise RuntimeError('Semantic read-back missing')
    receipt={'record':record,'exact_readback':observed,'backend_metadata':exact['metadatas'][0],
             'semantic_ids':semantic['ids'][0],'status':'registered_and_read_back'}
    write(receipt_path,receipt);write(a.output,receipt)
    print(receipt['status'],record['id'])


if __name__=='__main__':main()
