"""V2 domain consistency regression; no external effects."""
import copy,json,unittest
from bridge import Engine,Rejected,proposal
from test_bridge import BridgeTests

class IndexedTests(BridgeTests):
    def assert_indexes(self):
        old=copy.deepcopy((self.e._locations,self.e._occupancy,self.e._tokens))
        self.e.rebuild_indexes()
        self.assertEqual(old,(self.e._locations,self.e._occupancy,self.e._tokens))
    def test_index_transfer_cancel_create_swap(self):
        self.apply([self.op('mask','north','N1'),self.op('clock','north','N2')]);self.assert_indexes()
        self.apply([self.op('mask','north','N2',typ='transfer'),self.op('clock','north','N1',typ='transfer')]);self.assert_indexes()
        self.apply([self.op('mask','south','S1',typ='transfer')]);self.assert_indexes()
        self.apply([{'op':'cancel','loan_id':'loan:mask'}]);self.assert_indexes()
        self.apply([dict(op='create',loan_id='loan:new',title='New Jade',home='gal:south',medium='stone',restricted=False)],self.ctx(loan_ids=['loan:new']));self.assert_indexes()
        c=self.ctx(); self.assertIn('loan:new',[x['id'] for x in self.e.search('Jade',context_id=c['context_id'])['candidates']])
    def test_occupancy_empty_visible_restricted(self):
        self.apply([self.op()]);c=self.ctx(loan_ids=['loan:clock'])
        r=self.e.occupancy('gal:north',context_id=c['context_id']);self.assertEqual(r['slots'][0],{'slot':'N1','occupied':True,'loan_id':None,'visibility':'restricted'})
        self.assertEqual(r['slots'][1]['visibility'],'empty')
        c=self.ctx();self.assertEqual(self.e.occupancy('gal:north','N1',context_id=c['context_id'])['slots'][0]['loan_id'],'loan:bronze')
        with self.assertRaisesRegex(Rejected,'invalid_slot'):self.e.occupancy('gal:north','S1',context_id=c['context_id'])
    def test_summary_cannot_change_host_authority(self):
        c=self.ctx(scope=['gal:north'],loan_ids=['loan:bronze']);s=self.e.context_summary(context_id=c['context_id'])
        self.assertNotIn('loan_ids',s);self.assertEqual(s['authorized_loan_count'],1);s['scope'].append('gal:south')
        self.assertEqual(self.e.contexts[c['context_id']]['scope'],['gal:north'])
    def test_scoped_pagination_and_provenance(self):
        c=self.ctx();seen=[];cursor=None
        while True:
            r=self.e.query_records(context_id=c['context_id'],limit=2,after_sequence=cursor);seen += [x['id'] for x in r['records']];cursor=r['next_after_sequence']
            if cursor is None:break
        self.assertEqual(seen,['loan:jade','loan:clock','loan:mask','loan:atlas','loan:orchid','loan:bronze'])
        c=self.ctx(scope=['gal:north']);self.assertEqual({x['id'] for x in self.e.query_records(context_id=c['context_id'])['records']},{'loan:bronze','loan:clock'})
        with self.assertRaisesRegex(Rejected,'read_outside_capability'):self.e.query_records(context_id=c['context_id'],gallery_id='gal:south')
        self.assertEqual(self.e.query_records(context_id=c['context_id'],event_id='none')['records'],[])
    def test_flag_only_context_write_neighborhood(self):
        self.e=Engine.fixture(private_test_disable_neighborhood_writes=True)
        c=self.ctx(scope=['gal:north']);self.apply([self.op('mask','south','S1')],c)
        c=self.ctx(scope=['gal:north'])
        with self.assertRaisesRegex(Rejected,'read_outside_capability'):self.e.occupancy('gal:south',context_id=c['context_id'])
        with self.assertRaisesRegex(Rejected,'read_outside_capability'):self.e.read(['loan:mask'],context_id=c['context_id'])
        self.rejected([{'op':'annotate','loan_id':'loan:mask','text':'x'}],'actor_permission_denied',self.ctx(actor='north-curator',scope=['gal:north']))
        self.rejected([self.op('orchid','south','S2')],'approval_required',self.ctx(scope=['gal:north']))
        self.rejected([{'op':'annotate','loan_id':'loan:mask','text':'x'}],'loan_not_authorized',self.ctx(scope=['gal:north'],loan_ids=['loan:clock']))
    def test_index_rollback(self):
        old=copy.deepcopy((self.e._locations,self.e._occupancy,self.e._tokens));self.rejected([self.op(),self.op('clock')],'slot_occupied')
        self.assertEqual(old,(self.e._locations,self.e._occupancy,self.e._tokens))
    def test_cli_roundtrip(self):
        import tempfile
        from bridge_cli import save_session,load_session
        self.apply([self.op()]);c=self.ctx()
        with tempfile.NamedTemporaryFile() as f:
            save_session(self.e,c,f.name);e,c2=load_session(f.name)
        self.assertEqual(e.context_summary(context_id=c2['context_id']),self.e.context_summary(context_id=c['context_id']))
        self.assertEqual(e.occupancy('gal:north',context_id=c2['context_id']),self.e.occupancy('gal:north',context_id=c['context_id']))

if __name__=='__main__': unittest.main(verbosity=2)
