"""Matched exploratory timing of restart-per-turn and persistent worker execution."""
import argparse
import json
from pathlib import Path
import time
import uuid

from .claude_research_bridge import QUERIES, SYSTEM, SCHEMA
from .common import read, write, sha
from .golf_poker_proxy import reference_counts
from .research_profile import build_catalog, menu, authorize, translate, validate_mapping
from .worker_environment import ClaudeWorker, ReferenceTokenizer

PLAN = Path('docs/07 - Status Reports/word-games/2026-11-01_word-games-rlm-overhead-plan.md')


def freeze(out):
    out.mkdir(parents=True, exist_ok=False)
    paths = [Path('experiments/word_games')/name for name in ('worker_environment.py',
        'benchmark_worker_environment.py', 'claude_research_bridge.py', 'research_profile.py',
        'common.py', 'golf_poker_proxy.py')]+[PLAN, Path('pixi.lock')]
    hashes = {}
    for path in paths:
        dest = out/'frozen_source'/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        hashes[str(path)] = sha(path)
    write(out/'producer.json', {'study': 'word-games-rlm-overhead-v1', 'source_hashes': hashes,
        'model': 'sonnet', 'same_prompts_and_authorization': True, 'exploratory': True})


def run(out, stage):
    freeze(out)
    catalog = build_catalog()
    write(out/'catalog.json', catalog)
    mapping = validate_mapping(read('data/experiments/word_games/20261101_claude_research_bridge_v1/public/mapping.json'))
    write(out/'mapping.json', mapping)
    marker = 'WG_PRIVATE_RLM_'+uuid.uuid4().hex
    write(out/'private_control.json', {'canary': marker})
    arms = ('persistent',) if stage == 'pilot' else ('baseline', 'persistent')
    workers = {arm: ClaudeWorker('sonnet', SYSTEM+marker, SCHEMA, str(uuid.uuid4()),
        persistent=arm == 'persistent', stderr_path=out/arm/'worker_stderr.txt') for arm in arms}
    counter = ReferenceTokenizer(out/'tokenizer_stderr.txt')
    rows = {arm: [] for arm in arms}
    api_previous = {arm: 0 for arm in arms}
    questions = [QUERIES[2], QUERIES[13]] if stage == 'pilot' else QUERIES
    try:
        for index, (topic, question) in enumerate(questions, 1):
            scoped = {k: v for k, v in catalog.items() if v['topic'] == topic}
            prompt = question+'\nFact menu (questions and source locations only):\n'+json.dumps(menu(scoped))
            for arm in (arms if index % 2 else tuple(reversed(arms))):
                turn = out/arm/f'Q{index:02d}'
                write(turn/'request.json', {'prompt': prompt, 'topic': topic, 'session_id': workers[arm].session_id})
                started = time.perf_counter()
                final, events, worker_seconds = workers[arm].complete(prompt, turn)
                validate_started = time.perf_counter()
                proposal = final['structured_output']
                facts, receipts = authorize(proposal, catalog, {topic})
                plain = {'facts': [{**f, 'id': 'finding:'+f['id'][1:]} for f in facts],
                    'revision': index, 'topic': topic, 'status': 'verified' if len(facts) == len(scoped) else 'partial',
                    'unknown': ['finding:'+k[1:] for k in sorted(set(scoped)-{f['id'] for f in facts})]}
                packet = translate(plain, mapping)
                assert translate(packet, mapping, reverse=True) == plain
                assert marker not in json.dumps(packet)
                validation_seconds = time.perf_counter() - validate_started
                tokenize_started = time.perf_counter()
                text = json.dumps(packet, separators=(',', ':'))
                count = (counter.counts if arm == 'persistent' else reference_counts)([text])[0]
                tokenization_seconds = time.perf_counter() - tokenize_started
                write(turn/'proposal.json', proposal)
                write(turn/'receipts.json', receipts)
                write(turn/'packet.json', packet)
                elapsed = time.perf_counter() - started
                tools = [b['name'] for e in events if e.get('type') == 'assistant'
                    for b in e.get('message', {}).get('content', []) if b.get('type') == 'tool_use']
                assert set(tools) <= {'Read', 'Grep', 'Glob', 'StructuredOutput'}
                cumulative_api = final['duration_api_ms']/1000
                row = {'query': index, 'session_id': final['session_id'], 'topic': topic,
                    'seconds': elapsed, 'worker_seconds': worker_seconds,
                    'cli_duration_seconds': final['duration_ms']/1000,
                    'cumulative_api_seconds': cumulative_api,
                    'api_delta_seconds': cumulative_api-api_previous[arm],
                    'validation_seconds': validation_seconds, 'tokenization_seconds': tokenization_seconds,
                    'native_output_tokens': final['usage']['output_tokens'], 'public_reference_tokens': count,
                    'accepted_ids': [f['id'] for f in facts], 'rejected': sum(not r['accepted'] for r in receipts),
                    'tools': tools, 'model_names': sorted(final['modelUsage']),
                    'worker_pid': workers[arm].stream.process.pid if arm == 'persistent' else None}
                api_previous[arm] = cumulative_api
                write(turn/'metrics.json', row)
                rows[arm].append(row)
                print(json.dumps({'arm': arm, 'query': index, 'seconds': round(elapsed, 3),
                    'worker_seconds': round(worker_seconds, 3), 'accepted': len(facts), 'tools': tools}), flush=True)
        totals = {}
        for arm, values in rows.items():
            totals[arm] = {key: sum(r[key] for r in values) for key in ('seconds', 'worker_seconds',
                'api_delta_seconds', 'validation_seconds', 'tokenization_seconds', 'native_output_tokens',
                'public_reference_tokens', 'rejected')}
            totals[arm]['unique_facts'] = len({f for r in values for f in r['accepted_ids']})
            totals[arm]['recall_facts'] = sum(len(r['accepted_ids']) for r in values[-3:]) if stage == 'full' else len(values[-1]['accepted_ids'])
            totals[arm]['recall_reads'] = sum(t in ('Read', 'Grep', 'Glob') for r in values[-3:] for t in r['tools']) if stage == 'full' else None
            totals[arm]['non_api_seconds'] = totals[arm]['seconds']-totals[arm]['api_delta_seconds']
        passed = True if stage == 'pilot' else (totals['persistent']['unique_facts'] >= totals['baseline']['unique_facts']
            and totals['persistent']['recall_facts'] >= totals['baseline']['recall_facts'])
        # Close live streams before binding their logs into an immutable result.
        for worker in workers.values():
            worker.close()
        counter.close()
        result = {'study': 'word-games-rlm-overhead-v1', 'stage': stage, 'completed': True,
            'rows': rows, 'totals': totals, 'coverage_gate_passed': passed,
            'timing_causal_claim': False, 'population_accuracy_established': False,
            'artifacts': {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}}
        write(out/'result.json', result)
        print(json.dumps({'completed': True, 'totals': totals, 'coverage_gate_passed': passed}), flush=True)
    except Exception as exc:
        write(out/'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'rows': rows})
        raise
    finally:
        for worker in workers.values():
            worker.close()
        counter.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('pilot', 'full'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
