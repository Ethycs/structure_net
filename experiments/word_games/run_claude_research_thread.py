"""Receiving-side client: sees only the WORD_GAMES proxy's translated packets."""
import argparse
from pathlib import Path
import time

import requests

from .common import read, write, sha
from .research_profile import translate, validate_mapping


def run(url, out):
    out.mkdir(parents=True,exist_ok=False)
    http=requests.Session()
    health=http.get(url+'/health',timeout=5);health.raise_for_status()
    if health.json()['turns']!=0:
        raise ValueError('fresh server thread required; no hidden replay')
    mapping_response=http.get(url+'/mapping',timeout=5);mapping_response.raise_for_status()
    mapping=validate_mapping(mapping_response.json())
    write(out/'mapping.json',mapping)
    write(out/'producer.json',{'source_sha256':sha(__file__),'server':url,'initial_health':health.json(),
        'receiver_reads_private_worker_logs':False,'thread_replay':False})
    received, rows = {}, []
    for i in range(1,17):
        query_id=f'Q{i:02d}'
        request={'query_id':query_id}
        write(out/query_id/'request.json',request)
        start=time.perf_counter()
        response=http.post(url+'/bridge/query',json=request,timeout=(5,330))
        elapsed=time.perf_counter()-start
        body=response.json()
        write(out/query_id/'response.json',{'status':response.status_code,'seconds':elapsed,'body':body})
        response.raise_for_status()
        packet=translate(body,mapping,reverse=True)
        for fact in packet['facts']:
            received[fact['id']]=fact
        row={'query_id':query_id,'seconds':elapsed,'received_facts':len(packet['facts']),
             'missing_facts':len(packet['unknown']),'status':packet['status']}
        rows.append(row)
        print(query_id, 'received',row['received_facts'],'missing',row['missing_facts'],
              'seconds',round(elapsed,2),flush=True)
    write(out/'result.json',{'completed':True,'turns':len(rows),'unique_received_facts':len(received),
        'authorized_knowledge':received,'rows':rows,'request_seconds':sum(r['seconds'] for r in rows),
        'response_hashes':{str(p.relative_to(out)):sha(p) for p in out.glob('Q*/response.json')}})
    print('Completed',len(rows),'turns;',len(received),'unique authorized facts received.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:18091')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.url,args.output)
