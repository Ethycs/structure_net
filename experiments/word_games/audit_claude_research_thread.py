"""Audit source provenance, actual Claude sessions and receiver-only packets."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .common import read,write,sha


def run(root,client,out):
    out.mkdir(parents=True,exist_ok=False)
    result=read(root/'result.json'); received=read(client/'result.json')
    assert result['completed'] and received['completed']
    for name,expected in result['artifacts'].items():
        assert sha(root/name)==expected,name
    for name,expected in received['response_hashes'].items():
        assert sha(client/name)==expected,name
    producer=read(root/'producer.json')
    for path,expected in producer['source_hashes'].items():
        relative=Path(path)
        if relative.is_absolute():relative=relative.resolve().relative_to(Path(__file__).resolve().parents[2])
        assert sha(root/'frozen_source'/relative)==expected,path
    for path,expected in producer['catalog_source_hashes'].items():
        assert sha(path)==expected,path
        copy=out/'evidence_sources'/path
        copy.parent.mkdir(parents=True,exist_ok=True)
        copy.write_bytes(Path(path).read_bytes())
        assert sha(copy)==expected
    mapping=read(root/'public'/'mapping.json')
    inverse={v:k for k,v in mapping['key_map'].items()}
    def decode(x):
        if isinstance(x,dict):return {inverse.get(k,k):decode(v) for k,v in x.items()}
        if isinstance(x,list):return [decode(v) for v in x]
        return x
    catalog=read(root/'private'/'catalog.json')
    marker=read(root/'private'/'control.json')['canary']
    sessions=set();models=set();tools=Counter();public_tokens=[];public_texts=[]
    accepted, rejected, seen = [], [], set()
    for row in result['rows']:
        query=row['query_id'];private=root/'private'/'turns'/query
        events=[json.loads(s) for s in (private/'worker.jsonl').read_text().splitlines() if s.strip()]
        final=[e for e in events if e.get('type')=='result']
        assert len(final)==1 and not final[0]['is_error']
        sessions.add(final[0]['session_id'])
        models.update(final[0]['modelUsage'])
        observed_tools=[]
        for event in events:
            if event.get('type')=='assistant':
                observed_tools.extend(b['name'] for b in event.get('message',{}).get('content',[]) if b.get('type')=='tool_use')
        assert observed_tools==row['tool_calls']
        assert set(observed_tools)<={'Read','Grep','Glob','StructuredOutput'}
        tools.update(observed_tools)
        proposal=read(private/'proposal.json')
        receipts=read(private/'receipts.json')
        expected=[]
        for receipt,citation in zip(receipts,proposal['citations'],strict=True):
            if not receipt['accepted']:
                rejected.append({'query_id':query,**receipt})
                continue
            fact=catalog[receipt['fact_id']]
            norm=lambda s:' '.join(s.split())
            assert fact['topic']==row['topic']
            assert citation['source_path']==fact['path']
            assert norm(citation['quote']) in norm(Path(fact['path']).read_text())
            assert norm(fact['anchor']) in norm(citation['quote'])
            expected.append({'id':'finding:'+fact['id'][1:],'subject':fact['subject'],
                'property':fact['property'],'value':fact['value'],
                'evidence':'record:'+fact['source_sha256'][:12],'status':'verified'})
            accepted.append({'query_id':query,'fact_id':fact['id']});seen.add(fact['id'])
        packet=read(root/'public'/'turns'/(query+'.json'))
        assert packet==read(client/query/'response.json')['body']
        plain=decode(packet)
        assert set(plain)=={'facts','revision','topic','status','unknown'}
        assert plain['facts']==expected
        assert marker not in json.dumps(packet)
        assert all('source_path' not in f and 'quote' not in f and 'notes' not in f for f in plain['facts'])
        # Artifact JSON sorts keys, whereas the producer counts its insertion-
        # ordered compact packet. Reconstruct that exact producer serialization;
        # different key order can change tokenization at key/value boundaries.
        original_plain={'facts':expected,'revision':plain['revision'],'topic':plain['topic'],
                        'status':plain['status'],'unknown':plain['unknown']}
        def encode(value):
            if isinstance(value,dict):return {mapping['key_map'].get(k,k):encode(v) for k,v in value.items()}
            if isinstance(value,list):return [encode(v) for v in value]
            return value
        original_packet=encode(original_plain)
        assert original_packet==packet
        public_texts.append(json.dumps(original_packet,separators=(',',':')))
        public_tokens.append(row['public_reference_tokens'])
    from .golf_poker_proxy import reference_counts
    assert reference_counts(public_texts)==public_tokens
    assert sessions=={result['session_id']}
    assert len(seen)==result['unique_facts_released']==received['unique_received_facts']
    recall=result['rows'][-3:]
    assert all(set(r['tool_calls'])<={'StructuredOutput'} for r in recall)
    native=sum(r['worker_usage']['output_tokens'] for r in result['rows'])
    evidence={'study':result['study'],'status':'independently_audited','result_sha256':sha(root/'result.json'),
        'client_result_sha256':sha(client/'result.json'),'audit_source_sha256':sha(__file__),
        'session_id':result['session_id'],'turns':len(result['rows']),'models':sorted(models),
        'unique_facts':len(seen),'authorized_catalog_facts':len(catalog),
        'accepted_citations':len(accepted),'rejected_citations':rejected,
        'missing_unique_fact_ids':sorted(set(catalog)-seen),'tools':dict(tools),
        'recall_facts':sum(r['released_facts'] for r in recall),
        'recall_expected_facts':sum(r['requested_facts'] for r in recall),'recall_source_reads':0,
        'client_seconds':received['request_seconds'],'worker_seconds':result['total_worker_seconds'],
        'native_worker_output_tokens':native,
        'native_worker_output_tokens_per_client_second':native/received['request_seconds'],
        'public_reference_tokens':sum(public_tokens),
        'public_reference_tokens_per_client_second':sum(public_tokens)/received['request_seconds'],
        'input_tokens':sum(r['worker_usage'].get('input_tokens',0) for r in result['rows']),
        'cache_creation_input_tokens':sum(r['worker_usage'].get('cache_creation_input_tokens',0) for r in result['rows']),
        'cache_read_input_tokens':sum(r['worker_usage'].get('cache_read_input_tokens',0) for r in result['rows']),
        'private_canary_present_in_public_packets':False,
        'raw_worker_prose_present_in_public_packets':False,'all_public_token_counts_recomputed':True,
        'classification':'completed_bounded_research_channel_partial_coverage',
        'population_accuracy_established':False,'general_free_text_firewall':False}
    write(out/'audit.json',evidence)
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('root','client','output'):p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();run(a.root,a.client,a.output)
