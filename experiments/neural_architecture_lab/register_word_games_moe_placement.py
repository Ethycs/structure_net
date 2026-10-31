"""Hash-check and register stabilized MoE placement results with read-back."""
import argparse
from pathlib import Path

from experiments.word_games.common import read,sha,write


def register(campaign,report):
    from neural_architecture_lab.word_games_moe_placement_hypothesis import MoEPlacementEvidence
    c=read(campaign)
    for path,h in c['runtime_evidence_hashes'].items():
        if sha(path)!=h:raise ValueError('changed runtime evidence '+path)
    for review in c['manual_receiver_reviews'].values():
        if sha(review['path'])!=review['sha256']:raise ValueError('changed semantic review')
    for source in c['sources']:
        if sha(Path(source['root'])/'campaign_results.json')!=source['sha256']:raise ValueError('changed source campaign')
    for run in c['runs']:
        path=Path(run['root'])/'result.json'
        if sha(path)!=run['result_sha256']:raise ValueError('changed replay result')
        for raw,h in read(path)['response_hashes'].items():
            if sha(raw)!=h:raise ValueError('changed response')
    passing=c['passing_conditions']
    candidates=passing or [name for name in c['conditions'] if name!='control'] or ['control']
    best=min(candidates,key=lambda name:c['conditions'][name]['mean_request_seconds'])
    rows=[r for r in c['runs'] if r['config']==best]
    record=MoEPlacementEvidence(c['classification'],best,min(r['speedup_vs_fastest_control'] for r in rows),
        min(r['parsed_equal'] for r in rows),len(c['runs']),str(campaign),sha(campaign),str(report),sha(report)).record()
    from structure_net.logging.standardized_logging import LoggingConfig,StandardizedLogger
    logger=StandardizedLogger(LoggingConfig(project_name='structure_net_completed_hypotheses',
        queue_dir='data/experiment_queue',sent_dir='data/experiment_sent',rejected_dir='data/experiment_rejected',
        enable_wandb=False,auto_upload=False,enable_chromadb=True,chromadb_path='data/chroma_db'))
    logger.log_hypothesis(record)
    if logger.hypotheses_collection is None:raise RuntimeError('registry unavailable')
    logger.hypotheses_collection.update(ids=[record['id']],metadatas=[record])
    exact=logger.hypotheses_collection.get(ids=[record['id']],include=['metadatas'])
    if exact['ids']!=[record['id']]:raise RuntimeError('missing exact readback')
    for k,v in record.items():
        if exact['metadatas'][0][k]!=v:raise RuntimeError('incorrect readback '+k)
    semantic=logger.hypotheses_collection.query(query_texts=[record['description']],n_results=10)
    if record['id'] not in semantic['ids'][0]:raise RuntimeError('missing semantic readback')
    receipt={'record':record,'exact_readback':exact['metadatas'][0],'semantic_ids':semantic['ids'][0]}
    for dest in (campaign.parent/'registry_readback.json',Path('data/hypothesis_registry')/(record['id']+'.json')):write(dest,receipt)
    print('registered_and_read_back',c['classification'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();register(a.campaign,a.report)
