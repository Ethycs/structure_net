"""Independently check completed continuation checkpoints and receiver packets."""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request

from .common import read, write, sha


def run(roots, clients, url, output):
    if output.exists():
        raise FileExistsError(output)
    client_by_query = {read(p/'request.json')['query_id']: p for p in clients}
    rows = []
    session = None
    checkpoints = {}
    repo = Path(__file__).resolve().parents[2]
    previous = None
    for root in roots:
        producer = read(root/'producer.json')
        parent_path = Path(producer.get('parent_checkpoint', str(Path(producer['parent_root'])/'result.json')))
        assert sha(parent_path) == producer['parent_result_sha256']
        parent = read(parent_path)
        assert parent['completed']
        base = parent.get('base_revision', 0) + parent['turns']
        assert producer['base_revision'] == base
        if previous is not None:
            assert base == previous
        if session is None:
            session = parent['session_id']
        assert producer['session_id'] == session == parent['session_id']
        for name, expected in producer['source_hashes'].items():
            relative = Path(name)
            if relative.is_absolute():
                relative = relative.resolve().relative_to(repo)
            assert sha(root/'frozen_source'/relative) == expected, name
        for name, expected in producer['catalog_source_hashes'].items():
            assert sha(name) == expected, name
        catalog = read(root/'private/catalog.json')
        assert sha(root/'private/catalog.json') == producer['catalog_sha256']
        assert catalog == read(Path(producer['parent_root'])/'private/catalog.json')
        mapping = read(root/'public/mapping.json')
        inverse = {v: k for k, v in mapping['key_map'].items()}
        def decode(value):
            if isinstance(value, dict):
                return {inverse.get(k, k): decode(v) for k, v in value.items()}
            if isinstance(value, list):
                return [decode(v) for v in value]
            return value
        marker = read(root/'private/control.json')['canary']
        paths = sorted((root/'checkpoints').glob('Q*.json'), key=lambda p: int(p.stem[1:]))
        assert paths
        for checkpoint in paths:
            result = read(checkpoint)
            assert result['completed'] and result['session_id'] == session
            for name, expected in result['artifacts'].items():
                assert sha(root/name) == expected, name
            checkpoints[str(checkpoint)] = sha(checkpoint)
            row = result['rows'][-1]
            query = row['query_id']
            assert int(query[1:]) == base + result['turns']
            private = root/'private/turns'/query
            events = [json.loads(s) for s in (private/'worker.jsonl').read_text().splitlines()]
            finals = [e for e in events if e.get('type') == 'result']
            assert len(finals) == 1 and not finals[0]['is_error']
            assert finals[0]['session_id'] == session
            tools = [b['name'] for e in events if e.get('type') == 'assistant'
                     for b in e.get('message', {}).get('content', []) if b.get('type') == 'tool_use']
            assert tools == row['tool_calls']
            assert set(tools) <= {'Read', 'Grep', 'Glob', 'StructuredOutput'}
            proposal = read(private/'proposal.json')
            receipts = read(private/'receipts.json')
            expected_facts = []
            for citation, receipt in zip(proposal['citations'], receipts, strict=True):
                if not receipt['accepted']:
                    continue
                fact = catalog[receipt['fact_id']]
                norm = lambda s: ' '.join(s.split())
                assert row['topic'] in {fact['topic'], 'all'}
                assert citation['source_path'] == fact['path']
                assert norm(citation['quote']) in norm(Path(fact['path']).read_text())
                assert norm(fact['anchor']) in norm(citation['quote'])
                expected_facts.append({'id': 'finding:'+fact['id'][1:], 'subject': fact['subject'],
                    'property': fact['property'], 'value': fact['value'],
                    'evidence': 'record:'+fact['source_sha256'][:12], 'status': 'verified'})
            packet = read(root/'public/turns'/(query+'.json'))
            response = read(client_by_query[query]/'response.json')
            assert response['status'] == 200 and response['body'] == packet
            plain = decode(packet)
            assert set(plain) == {'facts', 'revision', 'topic', 'status', 'unknown'}
            assert plain['facts'] == expected_facts and plain['revision'] == int(query[1:])
            assert set(plain['unknown']) <= {'finding:'+k[1:] for k in catalog}
            assert marker not in json.dumps(packet)
            rows.append({'query_id': query, 'released_facts': len(expected_facts),
                'rejected_citations': row['rejected_citations'], 'tools': tools,
                'unknown': plain['unknown'], 'client_seconds': response['seconds'],
                'native_output_tokens': finals[0]['usage']['output_tokens']})
        previous = base + read(paths[-1])['turns']
    assert len(rows) == len(clients)
    def health():
        with urllib.request.urlopen(url+'/health', timeout=10) as response:
            return json.load(response)
    before = health()
    assert before['ready'] and before['interactive_questions'] and before['turns'] == previous
    assert before['session_id'] == session
    request = urllib.request.Request(url+'/bridge/query', data=json.dumps({
        'query_id': rows[-1]['query_id'], 'topic': 'all', 'question': 'Duplicate request control.'}).encode(),
        headers={'Content-Type': 'application/json'}, method='POST')
    try:
        urllib.request.urlopen(request, timeout=10)
        raise AssertionError('duplicate request accepted')
    except urllib.error.HTTPError as error:
        assert error.code == 400 and json.load(error) == {'error': 'request_blocked'}
    assert health() == before
    evidence = {'status': 'continuation_audit_passed', 'session_id': session, 'total_turns': previous,
        'continuation_rows': rows, 'checkpoint_hashes': checkpoints, 'duplicate_call_blocked': True,
        'public_packets_equal_canonical_projections': True, 'private_canary_released': False,
        'health': before, 'audit_source_sha256': sha(__file__)}
    write(output, evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', type=Path, required=True)
    parser.add_argument('--clients', nargs='+', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18091')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.roots, args.clients, args.url, args.output)
