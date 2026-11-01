"""Register the audited exploratory golf/poker result without a broad accuracy claim."""
import argparse
from pathlib import Path

from experiments.word_games.common import read, sha, write


def run(args):
    result, audit = read(args.result), read(args.audit)
    if audit['campaign_sha256'] != sha(args.result) or audit['status'] != 'independent_audit_passed':
        raise ValueError('matching audited result required')
    for path, expected in result['artifacts'].items():
        if sha(args.result.parent/path) != expected:
            raise ValueError('campaign artifact changed: '+path)
    poker = result['summary']['poker']
    classification = ('toy_game_optimal_play_preserved' if result['completed'] and
        poker['failed_holes']==0 and poker['optimal']==poker['decisions'] and
        poker['ledger_correct']==poker['decisions'] else 'toy_game_partial_or_failed')
    record = {'id':result['study'],'name':'WORD_GAMES golf through poker proxy experiment',
        'description':'Model-generated poker vocabulary for a finite stochastic golf game; '
        'fresh target-only proxy players versus direct golf, independently scored against '
        'expected-stroke optimal policies. Speed uses Qwen visible-token equivalents because '
        'the proxy returns zero native usage. No broad firewall or golf competence claim.',
        'project':'WORD_GAMES','category':'inference','tested':True,'confirmed':False,
        'confirmation_status':classification,'direct_experiment_count':1,
        'tested_scope':'small synthetic Markov game; generated vocabulary with host-preserved semantics',
        'requested_model':result['requested_model'],'poker_decisions':poker['decisions'],
        'poker_optimal_decisions':poker['optimal'],'poker_completed_holes':poker['completed_holes'],
        'full_request_reference_tok_s':result['full_request_tok_s'],
        'native_provider_tok_s_available':False,'population_accuracy_established':False,
        'result':str(args.result),'result_sha256':sha(args.result),
        'audit':str(args.audit),'audit_sha256':sha(args.audit),
        'report':str(args.report),'report_sha256':sha(args.report)}
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(project_name='structure_net_completed_hypotheses',
        queue_dir='data/experiment_queue',sent_dir='data/experiment_sent',rejected_dir='data/experiment_rejected',
        enable_wandb=False,auto_upload=False,enable_chromadb=True,chromadb_path='data/chroma_db'))
    logger.log_hypothesis(record)
    collection = logger.hypotheses_collection
    if collection is None:
        raise RuntimeError('registry unavailable')
    collection.update(ids=[record['id']],metadatas=[record])
    exact = collection.get(ids=[record['id']],include=['metadatas'])
    if exact['ids'] != [record['id']] or any(exact['metadatas'][0].get(k)!=v for k,v in record.items()):
        raise RuntimeError('registry read-back mismatch')
    semantic = collection.query(query_texts=[record['description']],n_results=10)
    if record['id'] not in semantic['ids'][0]:
        raise RuntimeError('semantic read-back missing')
    receipt = {'record':record,'exact_readback':exact['metadatas'][0],
               'semantic_ids':semantic['ids'][0],'status':'registered_and_read_back'}
    write(Path('data/hypothesis_registry')/(record['id']+'.json'),receipt)
    write(args.output,receipt)
    print(receipt['status'],record['id'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('result','audit','report','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
