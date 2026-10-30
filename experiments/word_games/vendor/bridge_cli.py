"""Read-only selective retrieval CLI for a trusted-host-created JSON session.
Interpreter commands: read, search, context. Mutations are returned as proposals
and committed by the trusted host, not by an unrestricted child shell.
"""
import argparse,json
from bridge import Engine

def save_session(engine,context,path):
    data={'state':engine.snapshot(),'context':context,'permissions':{k:sorted(v) for k,v in engine.permissions.items()}}
    with open(path,'w') as f: json.dump(data,f,sort_keys=True,indent=2)

def load_session(path):
    with open(path) as f: data=json.load(f)
    e=Engine.fixture(); s=data['state']
    for k in ('revision','loans','galleries','rules','approvals','sequence','audit'): setattr(e,k,s[k])
    e.permissions={k:set(v) for k,v in data['permissions'].items()}
    c=data['context']; e.contexts={c['context_id']:{k:v for k,v in c.items() if k!='context_id'}}
    e.rebuild_indexes()
    return e,c

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--session',required=True); p.add_argument('command',choices=['context','read','search','occupancy','query']); p.add_argument('args',nargs='*'); a=p.parse_args()
    e,c=load_session(a.session)
    if a.command=='context': out=e.context_summary(context_id=c['context_id'])
    elif a.command=='read': out=e.read(a.args,context_id=c['context_id'])
    elif a.command=='occupancy': out=e.occupancy(*a.args,context_id=c['context_id'])
    elif a.command=='query': out=e.query_records(context_id=c['context_id'],**(json.loads(a.args[0]) if a.args else {}))
    else: out=e.search(' '.join(a.args),context_id=c['context_id'])
    print(json.dumps(out,sort_keys=True,indent=2))
