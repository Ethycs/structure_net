"""Long-lived state workload: valid-heavy reserves/transfers/cancels, no resets."""
import random,time,json
from collections import Counter
from bridge import Engine,Rejected,proposal

def run(n=10000,seed=99817):
    rng=random.Random(seed); e=Engine.fixture(); counts=Counter(); begin=time.perf_counter()
    # Trusted contexts are ephemeral host event snapshots. Drop after a successful event
    # only in this workload to avoid representing them as durable conversation memory.
    for i in range(n):
        lid=rng.choice(['loan:bronze','loan:clock','loan:mask']); loan=e.loans[lid]
        c=e.issue_context(scope=list(e.galleries),loan_ids=list(e.loans),allowed_ops=list(e.OP_FIELDS),event_id='valid-'+str(i))
        if loan['reservation'] and rng.random()<.4: op={'op':'cancel','loan_id':lid}
        else:
            gs=['gal:north','gal:archive'] if loan['medium']=='metal' else list(e.galleries)
            g=rng.choice(gs); op={'op':'transfer' if loan['reservation'] else 'reserve','loan_id':lid,('to_gallery_id' if loan['reservation'] else 'gallery_id'):g,'slot':rng.choice(e.galleries[g]['slots']),'due':'2027-01-10','approval_id':None}
        # State-only invariant digest omits growing audit for O(n) total workload.
        old=json.dumps(e.loans,sort_keys=True); rev=e.revision; audit=len(e.audit); receipt=len(e.receipts)
        try:
            r=e.apply(proposal(c,[op],'valid-'+str(i)),expected_context_id=c['context_id']); counts[op['op']]+=1
            assert e.revision==rev+1 and len(e.audit)==audit+1 and len(e.receipts)==receipt+1
            assert r['status']=='committed'
        except Rejected as ex:
            assert str(ex)=='slot_occupied',str(ex); counts['slot_occupied']+=1
            assert json.dumps(e.loans,sort_keys=True)==old and e.revision==rev and len(e.audit)==audit and len(e.receipts)==receipt
        beds=[(v['reservation']['gallery_id'],v['reservation']['slot']) for v in e.loans.values() if v['reservation']]
        assert len(beds)==len(set(beds))
        assert e.project()['seeds']['seed:mask']['source_id']=='loan:mask'
    return {'seed':seed,'attempts':n,'counts':dict(counts),'final_revision':e.revision,'final_audit_events':len(e.audit),'assertion_failures':0,'elapsed_seconds':round(time.perf_counter()-begin,3),'resets':0,'scope':'Backend synthetic long-lived state; no language model.'}
if __name__=='__main__':
    report=run(); print(json.dumps(report,indent=2)); json.dump(report,open('valid_stress_report.json','w'),indent=2)
