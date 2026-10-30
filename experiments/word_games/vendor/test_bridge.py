import copy,json,random,time,unittest
from collections import Counter
from bridge import Engine,Rejected,proposal,digest

class BridgeTests(unittest.TestCase):
    def setUp(self): self.e=Engine.fixture(); self.n=0
    def ctx(self,**kw):
        d=dict(scope=list(self.e.galleries),loan_ids=list(self.e.loans),allowed_ops=list(self.e.OP_FIELDS),event_id='unit'); d.update(kw); return self.e.issue_context(**d)
    def op(self,l='bronze',g='north',slot='N1',due='2027-01-10',approval=None,typ='reserve'):
        return {'op':typ,'loan_id':'loan:'+l,('gallery_id' if typ=='reserve' else 'to_gallery_id'):'gal:'+g,'slot':slot,'due':due,'approval_id':approval}
    def apply(self,ops,c=None):
        self.n+=1; c=c or self.ctx(); return self.e.apply(proposal(c,ops,'unit-'+str(self.n)),expected_context_id=c['context_id'])
    def rejected(self,ops,code,c=None):
        before=self.e.state_hash(); receipts=copy.deepcopy(self.e.receipts)
        with self.assertRaisesRegex(Rejected,code): self.apply(ops,c)
        self.assertEqual(before,self.e.state_hash()); self.assertEqual(receipts,self.e.receipts)
    def test_hierarchy_and_approval(self):
        self.rejected([self.op(due='2027-02-16')],'inherited_due_limit')
        self.rejected([self.op('orchid','south','S1')],'approval_required')
        self.apply([self.op('orchid','north','N1',approval='approval:orchid:north:exception')])
        self.assertEqual(self.e.loans['loan:orchid']['reservation']['gallery_id'],'gal:north')
    def test_scope_does_not_leak_from_referent(self):
        c=self.ctx(scope=['gal:north'],referents={'it':'loan:bronze'})
        self.rejected([self.op('mask','south','S1')],'neighborhood_not_authorized',c)
        self.assertEqual(self.e.contexts[c['context_id']]['scope'],['gal:north'])
    def test_permission_not_approval(self):
        c=self.ctx(actor='north-curator'); self.rejected([self.op('mask','south','S1')],'actor_permission_denied',c)
        self.rejected([self.op('atlas','north','N1')],'approval_required')
    def test_atomic_rollback(self):
        self.rejected([self.op(),self.op('clock')],'slot_occupied')
        self.assertIsNone(self.e.loans['loan:bronze']['reservation'])
    def test_atomic_cross_block_swap(self):
        self.apply([self.op('mask','north','N1'),self.op('jade','south','S1',approval='approval:jade:south')])
        # Jade lacks North approval: entire cross-block transaction rolls back.
        self.rejected([self.op('mask','south','S1',typ='transfer'),self.op('jade','north','N1',typ='transfer')],'approval_required')
        self.apply([{'op':'cancel','loan_id':'loan:jade'},self.op('mask','south','S1',typ='transfer')])
        self.assertEqual(self.e.project()['seeds']['seed:mask']['stewardship_plot'],'plot:south')
    def test_revision_and_rules(self):
        c=self.ctx(); self.apply([{'op':'annotate','loan_id':'loan:bronze','text':'x'}]); self.rejected([self.op()],'stale_revision',c)
        c=self.ctx(); self.e.update_rule('rule:north',{'latest_due':'2027-02-14'}); self.rejected([self.op()],'stale_rule_version',c)
    def test_idempotency(self):
        p=proposal(self.ctx(),[self.op()],'same'); a=self.e.apply(p,expected_context_id=p['context_id']); h=self.e.state_hash(); self.assertEqual(a,self.e.apply(p,expected_context_id=p['context_id'])); self.assertEqual(h,self.e.state_hash())
        p['operations'][0]['due']='2027-01-11'
        with self.assertRaisesRegex(Rejected,'idempotency_conflict'): self.e.apply(p,expected_context_id=p['context_id'])
        self.assertEqual(h,self.e.state_hash())
    def test_typed_minimal_ops_no_global_patch(self):
        self.rejected([{'op':'set_scope','scope':['gal:south']}],'unknown_operation')
        op=self.op(); op['global_scope']='gal:south'; self.rejected([op],'invalid_operation_fields')
        c=self.ctx(); p=proposal(c,[self.op()],'x'); p['scope']=['gal:south']
        with self.assertRaisesRegex(Rejected,'invalid_proposal_fields'): self.e.apply(p,expected_context_id=p['context_id'])
    def test_creation_order_not_update_order(self):
        c=self.ctx(loan_ids=['loan:new-a','loan:new-b'])
        ops=[dict(op='create',loan_id='loan:'+i,title=i,home='gal:north',medium='paper',restricted=False) for i in ['new-a','new-b']]
        self.apply(ops,c); self.apply([dict(op='annotate',loan_id='loan:new-a',text='last touched')])
        self.assertLess(self.e.loans['loan:new-a']['creation']['sequence'],self.e.loans['loan:new-b']['creation']['sequence'])
    def test_clarification_no_mutation(self):
        h=self.e.state_hash(); r=self.e.apply({'kind':'clarify','question':'Which Jade object?'}); self.assertFalse(r['mutated']); self.assertEqual(h,self.e.state_hash())
    def test_search_is_not_binding(self):
        c=self.ctx(); r=self.e.search('Jade',context_id=c['context_id']); self.assertEqual({v['id'] for v in r['candidates']},{'loan:jade','loan:mask'}); self.assertEqual(r['binding'],'candidate_only')
        self.assertEqual(self.e.read(['loan:jdae'],context_id=c['context_id'])['missing_ids'],['loan:jdae'])
    def test_stable_identity_and_read_copy(self):
        c=self.ctx(); r=self.e.read(['loan:bronze'],context_id=c['context_id']); r['records']['loan:bronze']['title']='hack'; self.assertEqual(self.e.loans['loan:bronze']['title'],'Bronze Heron')
        initial=self.e.project()['seeds']['seed:bronze']['source_id']; self.apply([self.op()]); self.assertEqual(initial,self.e.project()['seeds']['seed:bronze']['source_id'])
    def test_trusted_context_binding(self):
        broad=self.ctx(); narrow=self.ctx(actor='viewer')
        p=proposal(broad,[self.op()],'substitution'); h=self.e.state_hash()
        with self.assertRaisesRegex(Rejected,'context_binding_mismatch'):
            self.e.apply(p,expected_context_id=narrow['context_id'])
        self.assertEqual(h,self.e.state_hash())
    def test_projection_detached(self):
        h=self.e.state_hash(); p=self.e.project(); p['plots']['plot:north']['beds'].append('HACK'); p['plots']['plot:north']['ancestor_rules'].clear()
        self.assertEqual(h,self.e.state_hash())
    def test_concurrent_same_revision_only_one_commit(self):
        from concurrent.futures import ThreadPoolExecutor
        c=self.ctx()
        def submit(i):
            try:
                return self.e.apply(proposal(c,[dict(op='annotate',loan_id='loan:bronze',text=str(i))],'concurrent-'+str(i)),expected_context_id=c['context_id'])['status']
            except Rejected as ex: return str(ex)
        with ThreadPoolExecutor(max_workers=8) as pool: outcomes=list(pool.map(submit,range(8)))
        self.assertEqual(outcomes.count('committed'),1); self.assertEqual(outcomes.count('stale_revision'),7)
    def test_admin_invalid_change_rolls_back(self):
        self.apply([self.op()]); h=self.e.state_hash()
        with self.assertRaises(Rejected): self.e.update_rule('rule:north',{'latest_due':'2026-12-01'})
        self.assertEqual(h,self.e.state_hash())

def stress(n=10000,seed=73129):
    rng=random.Random(seed); e=Engine.fixture(); counts=Counter(); started=time.perf_counter(); failures=[]
    stable={i:v['target_id'] for i,v in e.loans.items()}; origins={i:copy.deepcopy(v['creation']) for i,v in e.loans.items()}
    for step in range(n):
        # Periodic reset bounds audit cost; all random draws and operations deterministic.
        if step and step%200==0: e=Engine.fixture()
        scope=rng.choice([list(e.galleries),['gal:north'],['gal:south'],['gal:archive']]); c=e.issue_context(scope=scope,loan_ids=list(e.loans),allowed_ops=['reserve','transfer','cancel','annotate'],event_id='random-'+str(step),actor=rng.choice(['curator','north-curator','viewer']))
        lid=rng.choice(list(e.loans)); l=e.loans[lid]; g=rng.choice(list(e.galleries)); typ=rng.choice(['reserve','transfer','cancel','annotate','bad_patch'])
        if typ in ('reserve','transfer'):
            op={'op':typ,'loan_id':lid,('gallery_id' if typ=='reserve' else 'to_gallery_id'):g,'slot':rng.choice(e.galleries[g]['slots']+['BAD']),'due':rng.choice(['2027-01-10','2027-03-01','2027-02-14','2026-01-01','not-date']),'approval_id':rng.choice([None,*e.approvals])}
        elif typ=='annotate': op={'op':typ,'loan_id':lid,'text':'event '+str(step)}
        else: op={'op':typ,'loan_id':lid}
        ops=[op]
        if rng.random()<.15: ops.append({'op':'annotate','loan_id':rng.choice(list(e.loans)),'text':'atomic tail'})
        p=proposal(c,ops,'random-'+str(step))
        if rng.random()<.03: p['expected_revision']-=1
        before=e.state_hash(); before_seq=e.sequence; old_loans=copy.deepcopy(e.loans); audit_len=len(e.audit); receipts=copy.deepcopy(e.receipts)
        try:
            result=e.apply(p,expected_context_id=c['context_id']); counts['committed']+=1
            assert e.revision==c['revision']+1 and len(e.audit)==audit_len+1
            assert set(result['touched_neighborhoods'])<=set(scope)
            assert set(result['touched_neighborhoods'])<=e.permissions[c['actor']]
            changed={i for i in e.loans if e.loans[i]!=old_loans[i]}
            assert changed<={x['loan_id'] for x in ops}
            h=e.state_hash(); assert e.apply(p,expected_context_id=c['context_id'])==result and e.state_hash()==h
        except Rejected as ex:
            counts['rejected']+=1; counts['reason:'+str(ex)]+=1
            assert e.state_hash()==before and e.sequence==before_seq and e.receipts==receipts
        except Exception as ex:
            failures.append({'step':step,'proposal':p,'error':repr(ex)}); raise
        # Independent checks: no shared beds; exact mapping and creation immutable;
        # source→target view agrees with source state, untouched loans identical.
        beds=[]; view=e.project()
        for i,v in e.loans.items():
            assert v['target_id']==stable[i] and v['creation']==origins[i]
            assert view['seeds'][v['target_id']]['source_id']==i
            assert view['seeds'][v['target_id']]['stewardship_plot']==e.galleries[e.location(v)]['target_id']
            if v['reservation']: beds.append((v['reservation']['gallery_id'],v['reservation']['slot']))
        assert len(beds)==len(set(beds))
    return {'seed':seed,'attempts':n,'counts':dict(counts),'assertion_failures':failures,'elapsed_seconds':round(time.perf_counter()-started,3),'reset_every':200,'limitations':'Synthetic backend tests, not language understanding or long-horizon model memory evidence.'}

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(BridgeTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'unit_tests':result.testsRun,'unit_failures':len(result.failures),'unit_errors':len(result.errors)}
    if result.wasSuccessful(): report['randomized']=stress()
    with open('backend_report.json','w') as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
    raise SystemExit(not result.wasSuccessful())
