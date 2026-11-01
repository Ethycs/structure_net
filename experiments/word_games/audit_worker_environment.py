"""Audit matched worker results from native logs and canonical public projections."""
import argparse
import json
from pathlib import Path

from .common import read, write, sha
from .worker_environment import ReferenceTokenizer


def run(root, output):
    output.mkdir(parents=True, exist_ok=False)
    result = read(root/'result.json')
    assert result['completed'] and result['stage'] == 'full'
    for name, expected in result['artifacts'].items():
        assert sha(root/name) == expected, name
    producer = read(root/'producer.json')
    for name, expected in producer['source_hashes'].items():
        assert sha(root/'frozen_source'/name) == expected, name
    catalog = read(root/'catalog.json')
    for fact in catalog.values():
        assert sha(fact['path']) == fact['source_sha256']
    mapping = read(root/'mapping.json')['key_map']
    reverse = {v: k for k, v in mapping.items()}
    def decode(value):
        if isinstance(value, dict):
            return {reverse.get(k, k): decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value
    def encode(value):
        if isinstance(value, dict):
            return {mapping.get(k, k): encode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [encode(v) for v in value]
        return value
    marker = read(root/'private_control.json')['canary']
    texts, counts, sessions, models = [], [], {}, set()
    for arm, rows in result['rows'].items():
        seen = set()
        sessions[arm] = set()
        api_previous = 0
        output_tokens = 0
        for row in rows:
            turn = root/arm/f'Q{row["query"]:02d}'
            events = [json.loads(s) for s in (turn/'worker.jsonl').read_text().splitlines() if s.strip()]
            final = [e for e in events if e.get('type') == 'result']
            assert len(final) == 1 and not final[0]['is_error']
            final = final[0]
            sessions[arm].add(final['session_id'])
            assert final['session_id'] == row['session_id']
            models.update(final['modelUsage'])
            assert final['usage']['output_tokens'] == row['native_output_tokens']
            output_tokens += final['usage']['output_tokens']
            cumulative_api = final['duration_api_ms']/1000
            assert abs(cumulative_api-api_previous-row['api_delta_seconds']) < 1e-9
            api_previous = cumulative_api
            proposal = read(turn/'proposal.json')
            assert proposal == final['structured_output']
            expected = []
            for receipt, citation in zip(read(turn/'receipts.json'), proposal['citations'], strict=True):
                if not receipt['accepted']:
                    continue
                fact = catalog[receipt['fact_id']]
                norm = lambda s: ' '.join(s.split())
                assert fact['topic'] == row['topic']
                assert citation['source_path'] == fact['path']
                assert norm(citation['quote']) in norm(Path(fact['path']).read_text())
                assert norm(fact['anchor']) in norm(citation['quote'])
                seen.add(fact['id'])
                expected.append({'id': 'finding:'+fact['id'][1:], 'subject': fact['subject'],
                    'property': fact['property'], 'value': fact['value'],
                    'evidence': 'record:'+fact['source_sha256'][:12], 'status': 'verified'})
            packet = read(turn/'packet.json')
            plain = decode(packet)
            assert plain['facts'] == expected
            scoped = {k for k, v in catalog.items() if v['topic'] == row['topic']}
            assert plain['unknown'] == ['finding:'+k[1:] for k in sorted(scoped-set(row['accepted_ids']))]
            assert marker not in json.dumps(packet)
            original = {'facts': expected, 'revision': row['query'], 'topic': row['topic'],
                'status': 'verified' if len(expected) == len(scoped) else 'partial', 'unknown': plain['unknown']}
            assert encode(original) == packet
            texts.append(json.dumps(encode(original), separators=(',', ':')))
            counts.append(row['public_reference_tokens'])
            actual_tools = [b['name'] for e in events if e.get('type') == 'assistant'
                for b in e.get('message', {}).get('content', []) if b.get('type') == 'tool_use']
            assert actual_tools == row['tools']
            assert set(actual_tools) <= {'Read', 'Grep', 'Glob', 'StructuredOutput'}
        assert len(seen) == result['totals'][arm]['unique_facts']
        assert len(sessions[arm]) == 1
        assert output_tokens == result['totals'][arm]['native_output_tokens']
        assert abs(api_previous-result['totals'][arm]['api_delta_seconds']) < 1e-9
    assert sessions['baseline'].isdisjoint(sessions['persistent'])
    assert len({r['worker_pid'] for r in result['rows']['persistent']}) == 1
    for index in range(1, 17):
        assert read(root/'baseline'/f'Q{index:02d}'/'request.json')['prompt'] == read(root/'persistent'/f'Q{index:02d}'/'request.json')['prompt']
    tokenizer = ReferenceTokenizer(output/'tokenizer_stderr.txt')
    try:
        assert tokenizer.counts(texts) == counts
    finally:
        tokenizer.close()
    evidence = {'status': 'independent_audit_passed', 'result_sha256': sha(root/'result.json'),
        'source_sha256': sha(__file__), 'models': sorted(models), 'same_prompts': True,
        'separate_native_sessions': True, 'persistent_worker_pid_count': 1,
        'canonical_projections_match': True, 'private_marker_released': False,
        'all_public_token_counts_recomputed': True, 'coverage_gate_passed': result['coverage_gate_passed'],
        'totals': result['totals'], 'causal_cloud_speedup_established': False}
    write(output/'audit.json', evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
