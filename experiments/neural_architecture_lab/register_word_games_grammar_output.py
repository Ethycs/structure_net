"""Hash-check and register completed grammar-output replays with read-back."""
import argparse
from collections import Counter
from pathlib import Path

from experiments.word_games.common import digest, read, sha, write
from neural_architecture_lab.word_games_grammar_output_hypothesis import GrammarOutputEvidence, summarize


def checked_record(campaign_path, report):
    from experiments.word_games.run_grammar_output import compare

    c = read(campaign_path)
    summary = summarize(c)
    root = campaign_path.parent
    producer = read(root / 'producer.json')
    if producer['fingerprint'] != digest({k: v for k, v in producer.items() if k != 'fingerprint'}):
        raise ValueError('producer fingerprint mismatch')
    if Counter(sha(p) for p in (root / 'producer_source').iterdir()) != Counter(producer['source_hashes'].values()):
        raise ValueError('producer snapshots changed')
    for cell in c['results']:
        if not cell['completed'] or cell['error'] is not None:
            raise ValueError('incomplete cell')
        for path, expected in cell['response_hashes'].items():
            if sha(path) != expected:
                raise ValueError('response changed: ' + path)
        if read(root / f"{cell['replay']}_{cell['arm']}" / 'downloads.json') != cell['downloads']:
            raise ValueError('download counter mismatch')
    for label, expected in c['parity'].items():
        if compare(root / (label + '_control'), root / (label + '_candidate')) != expected:
            raise ValueError('output parity no longer matches')
    restored = read(root / 'service_restored.json')
    if not restored['ready_and_config_equal'] or not restored['ocr_ready']:
        raise ValueError('service restoration not verified')
    return GrammarOutputEvidence(
        hypothesis_id=c['hypothesis_id'], **summary,
        campaign=str(campaign_path), campaign_sha256=sha(campaign_path),
        report=str(report), report_sha256=sha(report),
    ).record()


def run(args):
    # Validate every campaign before touching the registry.
    records = [checked_record(p, args.report) for p in args.campaign]
    args.output.mkdir(parents=True, exist_ok=False)
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger

    logger = StandardizedLogger(LoggingConfig(
        project_name='structure_net_completed_hypotheses',
        queue_dir='data/experiment_queue', sent_dir='data/experiment_sent',
        rejected_dir='data/experiment_rejected', enable_wandb=False, auto_upload=False,
        enable_chromadb=True, chromadb_path='data/chroma_db',
    ))
    receipts = []
    for record in records:
        logger.log_hypothesis(record)
        collection = logger.hypotheses_collection
        if collection is None:
            raise RuntimeError('registry unavailable')
        collection.update(ids=[record['id']], metadatas=[record])
        exact = collection.get(ids=[record['id']], include=['metadatas'])
        if exact['ids'] != [record['id']] or any(
            exact['metadatas'][0].get(key) != value for key, value in record.items()
        ):
            raise RuntimeError('exact registry read-back mismatch')
        semantic = collection.query(query_texts=[record['description']], n_results=10)
        if record['id'] not in semantic['ids'][0]:
            raise RuntimeError('searchable registry read-back missing')
        receipt = {'record': record, 'exact_readback': exact['metadatas'][0],
                   'semantic_ids': semantic['ids'][0]}
        write(Path('data/hypothesis_registry') / (record['id'] + '.json'), receipt)
        receipts.append(receipt)
    write(args.output / 'registry_readback.json', receipts)
    print('Registered and read back', len(receipts), 'completed experiments')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, action='append', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
