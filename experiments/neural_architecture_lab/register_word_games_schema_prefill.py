"""Register the stabilized prefix-cache result with exact/search read-back."""
import argparse
from pathlib import Path
from experiments.word_games.common import read, sha, write
from neural_architecture_lab.word_games_schema_prefill_hypothesis import SchemaPrefillEvidence


def run(args):
    d = read(args.analysis)
    if sha(d['campaign']) != d['campaign_sha256']:
        raise ValueError('campaign changed')
    for p, h in read(d['campaign'])['response_hashes'].items():
        if sha(p) != h:
            raise ValueError('response changed')
    record = SchemaPrefillEvidence(d['conditions']['control']['tok_s'], d['conditions']['cached']['tok_s'],
        d['matching_paired_outputs'], d['existing_checks_pass'], d['classification'],
        str(args.analysis), sha(args.analysis), str(args.report), sha(args.report)).record()
    args.output.mkdir(parents=True, exist_ok=False)
    from structure_net.logging.standardized_logging import StandardizedLogger, LoggingConfig
    logger = StandardizedLogger(LoggingConfig(project_name='structure_net_completed_hypotheses',
        queue_dir='data/experiment_queue', sent_dir='data/experiment_sent', rejected_dir='data/experiment_rejected',
        enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path='data/chroma_db'))
    logger.log_hypothesis(record)
    collection = logger.hypotheses_collection
    if collection is None:
        raise RuntimeError('registry unavailable')
    collection.update(ids=[record['id']], metadatas=[record])
    exact = collection.get(ids=[record['id']], include=['metadatas'])
    if exact['ids'] != [record['id']] or any(exact['metadatas'][0].get(k) != v for k, v in record.items()):
        raise RuntimeError('exact read-back mismatch')
    semantic = collection.query(query_texts=[record['description']], n_results=10)
    if record['id'] not in semantic['ids'][0]:
        raise RuntimeError('semantic read-back missing')
    receipt = {'record': record, 'exact_readback': exact['metadatas'][0], 'semantic_ids': semantic['ids'][0]}
    write(Path('data/hypothesis_registry') / (record['id'] + '.json'), receipt)
    write(args.output / 'registry_readback.json', receipt)
    print('Registered and read back', record['id'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
