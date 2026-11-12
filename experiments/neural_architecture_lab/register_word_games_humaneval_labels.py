"""Hash-checked registration of the independently audited HumanEval sample."""
import argparse
from pathlib import Path

from experiments.word_games.common import read, sha, write
from neural_architecture_lab.word_games_humaneval_labels_hypothesis import HumanEvalLabelsEvidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audit', 'report', 'output', 'posthoc'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    audit = read(args.audit)
    if audit['status'] != 'independently_audited' or not audit['program_reexecution_verified']:
        raise ValueError('Complete independent audit required')
    for path, expected in audit['artifacts'].items():
        if sha(path) != expected: raise ValueError('Artifact changed: ' + path)
    metrics = audit['coding']
    if any(metrics[arm]['tasks'] != 32 or metrics[arm]['unattempted'] for arm in ('native', 'relabeled')):
        raise ValueError('Complete paired sample required')
    record = HumanEvalLabelsEvidence(metrics['native']['tasks_passed'], metrics['relabeled']['tasks_passed'],
                                     metrics['paired']['both_pass']).record()
    record.update(audit=str(args.audit), audit_sha256=sha(args.audit),
                  report=str(args.report), report_sha256=sha(args.report))
    diagnostic = read(args.posthoc)
    if read(args.posthoc.parent / 'plan.json')['primary_result_sha256'] != audit['result_sha256']:
        raise ValueError('Post-hoc diagnostic belongs to different primary evidence')
    for arm, score in diagnostic['results'].items():
        if read(args.posthoc.parent / arm / 'grade.json') != score:
            raise ValueError('Post-hoc raw grade differs')
    if not diagnostic['results']['canonical']['passed']:
        raise ValueError('Post-hoc reference control failed')
    record.update(posthoc_diagnostic=str(args.posthoc), posthoc_sha256=sha(args.posthoc),
        posthoc_relabeled_associativity_passed=diagnostic['results']['relabeled']['passed'],
        posthoc_native_associativity_passed=diagnostic['results']['native']['passed'])
    record['description'] += ' A separately recorded post-hoc associativity case exposes another relabeled-code defect beyond the official suite.'
    local = Path('data/hypothesis_registry') / (record['id'] + '.json')
    if local.exists() or args.output.exists(): raise FileExistsError('Use a new amendment for existing evidence')
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(project_name='structure_net_completed_hypotheses',
        queue_dir='data/experiment_queue', sent_dir='data/experiment_sent', rejected_dir='data/experiment_rejected',
        enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path='data/chroma_db'))
    logger.log_hypothesis(record)
    collection = logger.hypotheses_collection
    if collection is None: raise RuntimeError('Registry unavailable')
    collection.update(ids=[record['id']], metadatas=[record])
    exact = collection.get(ids=[record['id']], include=['metadatas'])
    if exact['ids'] != [record['id']]: raise RuntimeError('ID read-back failed')
    observed = {key: exact['metadatas'][0].get(key) for key in record}
    if observed != record: raise RuntimeError('Exact metadata read-back differs')
    semantic = collection.query(query_texts=[record['description']], n_results=10)
    if record['id'] not in semantic['ids'][0]: raise RuntimeError('Semantic read-back missing')
    receipt = {'status': 'registered_and_read_back', 'record': record,
               'exact_readback': observed, 'semantic_ids': semantic['ids'][0]}
    write(local, receipt); write(args.output, receipt)
    print(receipt['status'], record['id'])


if __name__ == '__main__':
    main()
