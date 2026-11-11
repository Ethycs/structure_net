"""Neighborhood Bridge: symbolic, deterministic museum→garden memory runtime.
No language model or oracle in this module. Context issuance is a TRUSTED HOST API.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, threading, secrets
from collections import Counter
from datetime import date

class Rejected(ValueError):
    pass

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def require(ok, code):
    if not ok: raise Rejected(code)

class Engine:
    OP_FIELDS = {
        'reserve': {'op','loan_id','gallery_id','slot','due','approval_id'},
        'transfer': {'op','loan_id','to_gallery_id','slot','due','approval_id'},
        'cancel': {'op','loan_id'},
        'annotate': {'op','loan_id','text'},
        'create': {'op','loan_id','title','home','medium','restricted'},
    }
    def __init__(self, *, private_test_disable_neighborhood_writes=False):
        # PRIVATE SYNTHETIC EXPERIMENT ONLY; production/default enforcement stays ON.
        require(type(private_test_disable_neighborhood_writes) is bool, "invalid_test_flag")
        self.private_test_disable_neighborhood_writes = private_test_disable_neighborhood_writes
        self.lock = threading.RLock()
        self.revision = 0
        self.rules = {
            'rule:root': {'version':1, 'latest_due':'2027-03-31'},
            'rule:public': {'version':1, 'latest_due':'2027-02-28'},
            'rule:north': {'version':1, 'latest_due':'2027-02-15', 'allowed_media':['metal','paper','stone']},
            'rule:south': {'version':1, 'latest_due':'2027-02-28', 'allowed_media':['textile','paper','stone']},
            'rule:archive': {'version':1, 'latest_due':'2027-03-31', 'allowed_media':['metal','paper','stone','textile']},
        }
        self.galleries = {
            'gal:north': {'id':'gal:north','name':'North Gallery','target_id':'plot:north','department':'dept:public','slots':['N1','N2'], 'rules':['rule:root','rule:public','rule:north']},
            'gal:south': {'id':'gal:south','name':'South Gallery','target_id':'plot:south','department':'dept:public','slots':['S1','S2'], 'rules':['rule:root','rule:public','rule:south']},
            'gal:archive': {'id':'gal:archive','name':'Archive Vault','target_id':'plot:archive','department':'dept:research','slots':['A1','A2','A3'], 'rules':['rule:root','rule:archive']},
        }
        self.loans = {}
        entries = [('bronze','Bronze Heron','gal:north','metal',False),('orchid','Orchid Embroidery','gal:south','textile',True),('atlas','Celestial Atlas','gal:archive','paper',True),('mask','Jade Mask','gal:south','stone',False),('clock','Bronze Clock','gal:north','metal',False),('jade','Jade Tablet','gal:archive','stone',True)]
        for seq,(key,title,home,medium,restricted) in enumerate(entries,1):
            lid='loan:'+key
            self.loans[lid]={'id':lid,'target_id':'seed:'+key,'title':title,'home':home,'medium':medium,'restricted':restricted,'reservation':None,'notes':[], 'creation':{'sequence':seq,'event_id':'fixture','actor':'fixture'}}
        self.sequence=len(entries)
        # Approval is narrow, pre-issued, and NOT implied by permission to write.
        self.approvals={
            'approval:orchid:south': {'loan_id':'loan:orchid','gallery_id':'gal:south','until':'2027-02-20','exception':None},
            'approval:atlas:north': {'loan_id':'loan:atlas','gallery_id':'gal:north','until':'2027-02-10','exception':None},
            'approval:atlas:archive': {'loan_id':'loan:atlas','gallery_id':'gal:archive','until':'2027-03-20','exception':None},
            'approval:jade:south': {'loan_id':'loan:jade','gallery_id':'gal:south','until':'2027-02-28','exception':None},
            'approval:orchid:north:exception': {'loan_id':'loan:orchid','gallery_id':'gal:north','until':'2027-02-12','exception':'medium'},
        }
        self.permissions={'curator':set(self.galleries),'north-curator':{'gal:north'},'viewer':set()}
        self.contexts={}; self.receipts={}; self.audit=[]
        self.rebuild_indexes()

    @classmethod
    def fixture(cls, **kwargs): return cls(**kwargs)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy({'revision':self.revision,'loans':self.loans,'galleries':self.galleries,'rules':self.rules,'approvals':self.approvals,'sequence':self.sequence,'audit':self.audit})

    def state_hash(self): return digest(self.snapshot())

    def issue_context(self, *, actor='curator', scope, loan_ids, allowed_ops, event_id, referents=None):
        """Trusted host only. This is NOT offered as an interpreter tool.
        Authorizations must come from user/host policy, never from an LM proposal.
        Read dependencies cover all ancestor rules in scoped neighborhoods.
        """
        with self.lock:
            require(actor in self.permissions, 'unknown_actor')
            require(set(scope)<=set(self.galleries), 'unknown_scope')
            require(set(allowed_ops)<=set(self.OP_FIELDS), 'unknown_operation')
            require(all(isinstance(x,str) and re.fullmatch(r'loan:[a-z0-9-]+',x) for x in loan_ids),'invalid_id')
            refs=referents or {}
            require(all(v in loan_ids for v in refs.values()),'referent_outside_capability')
            versions={r:self.rules[r]['version'] for g in scope for r in self.galleries[g]['rules']}
            c={'actor':actor,'scope':sorted(set(scope)),'loan_ids':sorted(set(loan_ids)),'allowed_ops':sorted(set(allowed_ops)), 'event_id':event_id, 'referents':copy.deepcopy(refs), 'revision':self.revision,'rule_versions':versions}
            cid='ctx:'+secrets.token_urlsafe(24)
            self.contexts[cid]=c
            self._context_ids[cid]=set(c["loan_ids"])
            return {'context_id':cid,**copy.deepcopy(c)}

    def read(self, ids, *, context_id):
        """Exact ID lookup. No guessed/fuzzy binding. Returns only scoped records."""
        with self.lock:
            c=self.contexts[context_id]; records={}; missing=[]
            authorized=self._context_ids[context_id]
            for i in ids:
                if i in self.loans:
                    loan=self.loans[i]; loc=self.location(loan)
                    require(loc in c['scope'] and i in authorized,'read_outside_capability')
                    records[i]=copy.deepcopy(loan)
                elif i in self.galleries:
                    require(i in c['scope'],'read_outside_capability'); records[i]=copy.deepcopy(self.galleries[i])
                elif i in self.rules:
                    require(i in c['rule_versions'],'read_outside_capability'); records[i]=copy.deepcopy(self.rules[i])
                elif i in self.approvals:
                    a=self.approvals[i]
                    require(a['loan_id'] in authorized and a['gallery_id'] in c['scope'],'read_outside_capability'); records[i]=copy.deepcopy(a)
                else: missing.append(i)
            return {'revision':self.revision,'rule_versions':copy.deepcopy(c['rule_versions']),'records':records,'missing_ids':missing}

    def rebuild_indexes(self):
        """Trusted host import/setup hook, called after loading a snapshot.
        Runtime mutations maintain indexes; direct fixture edits require this hook.
        """
        self._locations={g:set() for g in self.galleries}
        self._occupancy={}
        self._tokens={}
        self._context_ids={cid:set(c['loan_ids']) for cid,c in self.contexts.items()}
        for lid,loan in self.loans.items(): self._index_add(lid,loan)

    def _index_add(self,lid,loan):
        self._locations[self.location(loan)].add(lid)
        if loan['reservation']:
            r=loan['reservation']; self._occupancy[(r['gallery_id'],r['slot'])]=lid
        self._tokens[lid]=re.findall(r'\w+', (lid+' '+loan['title']).lower())

    def _index_remove(self,lid,loan):
        self._locations[self.location(loan)].discard(lid)
        if loan['reservation']:
            r=loan['reservation']; self._occupancy.pop((r['gallery_id'],r['slot']),None)
        self._tokens.pop(lid,None)

    def context_summary(self, *, context_id):
        """Compact interpreter description, never an authorization input."""
        with self.lock:
            c=self.contexts[context_id]
            return {'context_id':context_id, **copy.deepcopy({k:v for k,v in c.items() if k!='loan_ids'}),
                    'authorized_loan_count':len(self._context_ids[context_id]),
                    'authority':'host_stored_only'}

    def occupancy(self, gallery_id, slot=None, *, context_id):
        """Indexed slot view. Hidden occupants are not incorrectly reported as free."""
        with self.lock:
            c=self.contexts[context_id]
            require(gallery_id in c['scope'],'read_outside_capability')
            require(slot is None or slot in self.galleries[gallery_id]['slots'],'invalid_slot')
            slots=self.galleries[gallery_id]['slots'] if slot is None else [slot]
            out=[]
            for name in slots:
                lid=self._occupancy.get((gallery_id,name))
                visible=lid in self._context_ids[context_id]
                out.append({'slot':name,'occupied':lid is not None,'loan_id':lid if visible else None,
                            'visibility':'empty' if lid is None else ('visible' if visible else 'restricted')})
            return {'revision':self.revision,'gallery_id':gallery_id,'slots':out}

    def _scoped_ids(self,context_id,gallery_id=None):
        c=self.contexts[context_id]
        require(gallery_id is None or gallery_id in c['scope'],'read_outside_capability')
        locations=c['scope'] if gallery_id is None else [gallery_id]
        ids=set().union(*(self._locations[g] for g in locations))
        return ids & self._context_ids[context_id]

    def query_records(self, *, context_id, gallery_id=None, order='creation_desc', limit=20,
                      after_sequence=None, event_id=None):
        """Stable creation-order cursor; full records include immutable provenance.
        Cursor is a sequence bound, not a frozen multi-call snapshot. Revision is explicit.
        """
        with self.lock:
            require(order in ('creation_desc','creation_asc'),'invalid_order')
            require(type(limit) is int and 1<=limit<=20,'invalid_limit')
            require(after_sequence is None or type(after_sequence) is int,'invalid_cursor')
            require(event_id is None or isinstance(event_id,str),'invalid_event_id')
            records=[self.loans[i] for i in self._scoped_ids(context_id,gallery_id)]
            records=[r for r in records if (event_id is None or r['creation']['event_id']==event_id)
                     and (after_sequence is None or (r['creation']['sequence']<after_sequence if order=='creation_desc'
                          else r['creation']['sequence']>after_sequence))]
            records.sort(key=lambda r:r['creation']['sequence'],reverse=order=='creation_desc')
            page=records[:limit]
            return {'revision':self.revision,'records':copy.deepcopy(page),
                    'next_after_sequence':page[-1]['creation']['sequence'] if len(records)>limit else None,
                    'order':order}

    def search(self, query, *, context_id, k=5):
        """Cached token index + scoped BM25 scoring; NOT constant-time retrieval."""
        with self.lock:
            docs={i:self._tokens[i] for i in self._scoped_ids(context_id)}
            terms=re.findall(r'\w+',query.lower()); avg=sum(map(len,docs.values()))/max(len(docs),1)
            dfs=Counter(t for ts in docs.values() for t in set(ts))
            scored=[]
            for i,tokens in docs.items():
                score=0; counts=Counter(tokens)
                for term in terms:
                    df=dfs[term]; f=counts[term]
                    idf=math.log(1+(len(docs)-df+.5)/(df+.5))
                    if f: score+=idf*f*2.2/(f+1.2*(.25+.75*len(tokens)/max(avg,1)))
                if score: scored.append({'id':i,'title':self.loans[i]['title'],'score':round(score,6)})
            return {'method':'stdlib BM25 k1=1.2 b=0.75','candidates':sorted(scored,key=lambda x:(-x['score'],x['id']))[:max(0,min(k,20))], 'binding':'candidate_only'}

    @staticmethod
    def location(loan): return loan['reservation']['gallery_id'] if loan['reservation'] else loan['home']

    def project(self):
        """Total deterministic source→target view; target never independently writable."""
        return {'plots':{g['target_id']:{'source_id':gid,'beds':list(g['slots']),'ancestor_rules':list(g['rules'])} for gid,g in sorted(self.galleries.items())}, 'seeds':{v['target_id']:{'source_id':i,'label':v['title'],'stewardship_plot':self.galleries[self.location(v)]['target_id'], 'planting':None if not v['reservation'] else {'bed':v['reservation']['slot'],'return_by':v['reservation']['due']},'creation':copy.deepcopy(v['creation']),'notes':list(v['notes'])} for i,v in sorted(self.loans.items())}}

    def apply(self, proposal, *, expected_context_id=None):
        """All failures preserve committed state, audit, sequence and idempotency receipts."""
        with self.lock:
            require(isinstance(proposal,dict),'invalid_proposal')
            if proposal.get('kind')=='clarify':
                require(set(proposal)=={'kind','question'} and isinstance(proposal['question'],str),'invalid_clarification')
                return {'status':'clarify','question':proposal['question'],'revision':self.revision,'mutated':False}
            require(set(proposal)=={'kind','context_id','tx_id','expected_revision','rule_versions','operations'},'invalid_proposal_fields')
            require(proposal['kind']=='transaction','invalid_kind')
            require(isinstance(expected_context_id,str) and proposal['context_id']==expected_context_id,'context_binding_mismatch')
            require(isinstance(proposal['tx_id'],str) and 0<len(proposal['tx_id'])<=200,'invalid_tx_id')
            require(isinstance(proposal['context_id'],str) and proposal['context_id'] in self.contexts,'unknown_context')
            c=self.contexts[proposal['context_id']]; key=(c['actor'],proposal['tx_id']); ph=digest(proposal)
            if key in self.receipts:
                old=self.receipts[key]; require(old['proposal_hash']==ph,'idempotency_conflict')
                return copy.deepcopy(old['result'])
            require(type(proposal['expected_revision']) is int and proposal['expected_revision']==self.revision==c['revision'],'stale_revision')
            require(proposal['rule_versions']==c['rule_versions'],'rule_dependency_mismatch')
            require(all(self.rules[r]['version']==v for r,v in c['rule_versions'].items()),'stale_rule_version')
            ops=proposal['operations']; require(isinstance(ops,list) and 0<len(ops)<=20,'invalid_operations')
            draft=copy.deepcopy(self.loans); seq=self.sequence; touched=set()
            for op in ops:
                require(isinstance(op,dict) and isinstance(op.get('op'),str) and op.get('op') in self.OP_FIELDS,'unknown_operation')
                typ=op['op']; require(set(op)==self.OP_FIELDS[typ],'invalid_operation_fields')
                require(typ in c['allowed_ops'],'operation_not_authorized')
                lid=op['loan_id']; require(isinstance(lid,str) and lid in self._context_ids[proposal['context_id']],'loan_not_authorized')
                if typ=='create':
                    require(lid not in draft,'duplicate_id'); home=op['home']; self._scope(home,c)
                    require(isinstance(op['title'],str) and 0<len(op['title'])<=200,'invalid_title')
                    require(op['medium'] in ['metal','paper','stone','textile'] and type(op['restricted']) is bool,'invalid_attributes')
                    seq+=1; draft[lid]={'id':lid,'target_id':'seed:'+lid.split(':',1)[1],'title':op['title'],'home':home,'medium':op['medium'],'restricted':op['restricted'],'reservation':None,'notes':[], 'creation':{'sequence':seq,'event_id':c['event_id'],'actor':c['actor']}}
                    touched.add(home); continue
                require(lid in draft,'unknown_loan'); loan=draft[lid]; old=self.location(loan); self._scope(old,c); touched.add(old)
                if typ in ('reserve','transfer'):
                    require((loan['reservation'] is None)==(typ=='reserve'),'wrong_reservation_state')
                    dest=op['gallery_id'] if typ=='reserve' else op['to_gallery_id']; self._scope(dest,c); touched.add(dest)
                    self._placement(loan,dest,op['slot'],op['due'],op['approval_id'])
                    loan['reservation']={'gallery_id':dest,'slot':op['slot'],'due':op['due'],'approval_id':op['approval_id']}
                elif typ=='cancel':
                    require(loan['reservation'] is not None,'not_reserved'); self._scope(loan['home'],c); touched.add(loan['home']); loan['reservation']=None
                elif typ=='annotate':
                    require(isinstance(op['text'],str) and 0<len(op['text'])<=500,'invalid_note'); loan['notes'].append(op['text'])
            # Validate END state, enabling atomic swaps without transient slot conflicts.
            self._invariants(draft)
            result={'status':'committed','revision':self.revision+1,'mutated':True,'touched_neighborhoods':sorted(touched),'operation_count':len(ops)}
            changed={op['loan_id'] for op in ops}
            # Remove all old entries first so atomic swaps cannot erase new occupancy.
            for lid in changed:
                if lid in self.loans: self._index_remove(lid,self.loans[lid])
            for lid in changed: self._index_add(lid,draft[lid])
            self.loans=draft; self.sequence=seq; self.revision+=1
            self.audit.append({'tx_id':proposal['tx_id'],'event_id':c['event_id'],'actor':c['actor'],'revision':self.revision,'operations':copy.deepcopy(ops),'scope':c['scope'],'referents':copy.deepcopy(c['referents']),'rule_versions':copy.deepcopy(c['rule_versions'])})
            self.receipts[key]={'proposal_hash':ph,'result':copy.deepcopy(result)}
            return result

    def _scope(self,g,c):
        require(isinstance(g,str) and g in self.galleries,'unknown_gallery'); require(self.private_test_disable_neighborhood_writes or g in c['scope'],'neighborhood_not_authorized'); require(g in self.permissions[c['actor']],'actor_permission_denied')

    def _placement(self,loan,g,slot,due,approval_id):
        require(isinstance(g,str) and g in self.galleries,'unknown_gallery'); require(slot in self.galleries[g]['slots'],'invalid_slot')
        try: parsed=date.fromisoformat(due); require(parsed.isoformat()==due,'invalid_date')
        except (TypeError,ValueError): raise Rejected('invalid_date')
        require(due>='2026-10-30','due_in_past')
        a=None
        if approval_id is not None:
            require(isinstance(approval_id,str) and approval_id in self.approvals,'unknown_approval'); a=self.approvals[approval_id]
            require(a['loan_id']==loan['id'] and a['gallery_id']==g and due<=a['until'],'approval_mismatch')
        require(not loan['restricted'] or a is not None,'approval_required')
        for rid in self.galleries[g]['rules']:
            rule=self.rules[rid]; require(due<=rule['latest_due'],'inherited_due_limit')
            if 'allowed_media' in rule:
                require(loan['medium'] in rule['allowed_media'] or (a and a['exception']=='medium'),'medium_not_allowed')

    def _invariants(self,loans):
        slots=set(); targets=set(); sequences=set()
        for lid,l in loans.items():
            require(l['id']==lid and l['target_id'] not in targets,'identity_invariant'); targets.add(l['target_id'])
            require(l['creation']['sequence'] not in sequences,'creation_invariant'); sequences.add(l['creation']['sequence'])
            if l['reservation']:
                r=l['reservation']; self._placement(l,r['gallery_id'],r['slot'],r['due'],r['approval_id']); key=(r['gallery_id'],r['slot']); require(key not in slots,'slot_occupied'); slots.add(key)

    def update_rule(self,rule_id,patch):
        """Trusted administrator API, deliberately absent from LM operation schema.
        Existing reservations must remain valid or the administrative change rolls back.
        """
        with self.lock:
            require(rule_id in self.rules,'unknown_rule'); require(set(patch)<=set(self.rules[rule_id])-{'version'},'invalid_rule_patch')
            old=copy.deepcopy(self.rules[rule_id]); self.rules[rule_id].update(copy.deepcopy(patch)); self.rules[rule_id]['version']+=1
            try: self._invariants(self.loans)
            except Exception: self.rules[rule_id]=old; raise


def proposal(context,ops,tx_id):
    return {'kind':'transaction','context_id':context['context_id'],'tx_id':tx_id,'expected_revision':context['revision'],'rule_versions':context['rule_versions'],'operations':ops}

if __name__=='__main__':
    e=Engine.fixture(); c=e.issue_context(scope=['gal:north'],loan_ids=['loan:bronze'],allowed_ops=['reserve'],event_id='demo')
    print(json.dumps(e.apply(proposal(c,[{'op':'reserve','loan_id':'loan:bronze','gallery_id':'gal:north','slot':'N1','due':'2027-01-10','approval_id':None}],'demo-1'),expected_context_id=c['context_id']),indent=2))
    print(json.dumps(e.project(),indent=2))
