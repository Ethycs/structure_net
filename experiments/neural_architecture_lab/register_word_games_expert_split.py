"""Register both measured expert studies with exact and semantic read-back."""
import argparse
from pathlib import Path
from experiments.word_games.common import read,sha,write


def register(campaign,report):
    from neural_architecture_lab.word_games_expert_split_hypothesis import ExpertSplitEvidence
    c=read(campaign)
    for path,h in c['evidence_hashes'].items():
        if sha(path)!=h:raise ValueError('evidence changed: '+path)
    for source in c['sources']:
        if sha(Path(source['root'])/'campaign_results.json')!=source['campaign_sha256']:raise ValueError('campaign changed')
    for review in c.get('manual_receiver_reviews',{}).values():
        if sha(review['path'])!=review['sha256']:raise ValueError('semantic review changed')
    from structure_net.logging.standardized_logging import LoggingConfig,StandardizedLogger
    logger=StandardizedLogger(LoggingConfig(project_name='structure_net_completed_hypotheses',
        queue_dir='data/experiment_queue',sent_dir='data/experiment_sent',rejected_dir='data/experiment_rejected',
        enable_wandb=False,auto_upload=False,enable_chromadb=True,chromadb_path='data/chroma_db'))
    receipts=[]
    for h in c['hypotheses']:
        record=ExpertSplitEvidence(h['hypothesis_id'],h['classification'],h['description'],
            h['baseline']['output_tokens_per_second'],h['candidate']['output_tokens_per_second'],
            h['full_passes'],h['completed_requests'],h['service_restored'],str(campaign),sha(campaign),str(report),sha(report)).record()
        logger.log_hypothesis(record)
        if logger.hypotheses_collection is None:raise RuntimeError('registry unavailable')
        logger.hypotheses_collection.update(ids=[record['id']],metadatas=[record])
        exact=logger.hypotheses_collection.get(ids=[record['id']],include=['metadatas'])
        if exact['ids']!=[record['id']]:raise RuntimeError('missing exact readback')
        for k,v in record.items():
            if exact['metadatas'][0][k]!=v:raise RuntimeError('readback mismatch: '+k)
        semantic=logger.hypotheses_collection.query(query_texts=[record['description']],n_results=10)
        if record['id'] not in semantic['ids'][0]:raise RuntimeError('missing semantic readback')
        receipt={'record':record,'exact_readback':exact['metadatas'][0],'semantic_ids':semantic['ids'][0]}
        write(Path('data/hypothesis_registry')/(record['id']+'.json'),receipt);receipts.append(receipt)
    write(campaign.parent/'registry_readback.json',receipts)
    print('registered_and_read_back',len(receipts))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();register(a.campaign,a.report)
