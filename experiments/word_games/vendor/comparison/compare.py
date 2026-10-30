import sys,json,copy,hashlib,time,statistics,pathlib
ROOT=pathlib.Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT.parent))
from bridge import Engine,proposal,Rejected

def reserve(l='loan:bronze',g='gal:north',slot='N1',due='2027-01-10',approval=None): return dict(op='reserve',loan_id=l,gallery_id=g,slot=slot,due=due,approval_id=approval)
def case(id,ops,scope=['gal:north'],ids=None,expect=('committed','committed'),**kw):
 return dict(id=id,ops=ops,scope=scope,ids=ids or list(dict.fromkeys(o['loan_id'] for o in ops)),expect=list(expect),**kw)
def manifest():
 a=lambda l,t:dict(op='annotate',loan_id=l,text=t)
 cr=lambda home:dict(op='create',loan_id='loan:new-paper',title='New Paper',home=home,medium='paper',restricted=False)
 cs=[case('local_reserve',[reserve()]),case('local_annotation',[a('loan:bronze','Checked')]),case('local_create',[cr('gal:north')]),
 case('legitimate_crossscope',[reserve('loan:mask')],scope=['gal:north','gal:south']),
 case('outside_destination',[reserve(g='gal:archive',slot='A1')],expect=('neighborhood_not_authorized','committed')),
 case('outside_source',[reserve('loan:mask')],expect=('neighborhood_not_authorized','committed')),
 case('outside_annotation',[a('loan:mask','Checked')],expect=('neighborhood_not_authorized','committed')),
 case('outside_create',[cr('gal:south')],expect=('neighborhood_not_authorized','committed')),
 case('malformed_extra_field',[dict(reserve(),typo=True)],expect=('invalid_operation_fields',)*2),
 case('stale_revision',[reserve()],expect=('stale_revision',)*2,stale=True),
 case('inherited_rule_limit',[reserve(due='2027-02-20')],expect=('inherited_due_limit',)*2),
 case('restricted_missing_approval',[reserve('loan:atlas')],scope=['gal:north','gal:archive'],expect=('approval_required',)*2),
 case('unapproved_operation',[a('loan:bronze','Checked')],allowed=['reserve'],expect=('operation_not_authorized',)*2),
 case('unauthorized_identity',[reserve()],ids=['loan:clock'],expect=('loan_not_authorized',)*2),
 case('actor_permission',[reserve(g='gal:archive',slot='A1')],scope=['gal:north','gal:archive'],actor='north-curator',expect=('actor_permission_denied',)*2),
 case('atomic_partial_then_boundary',[a('loan:bronze','Checked'),a('loan:mask','Checked')],expect=('neighborhood_not_authorized','committed'))]
 lm=[case('lm_local_reserve',[reserve()],task='Reserve Bronze Heron in North Gallery slot N1 through 2027-01-10.'),
 case('lm_annotation',[a('loan:clock','Condition checked')],task='Add the note "Condition checked" to Bronze Clock.'),
 case('lm_create',[cr('gal:north')],task='Create loan:new-paper titled New Paper, an unrestricted paper item homed in North Gallery.'),
 case('lm_crossscope',[reserve('loan:mask')],scope=['gal:north','gal:south'],task='Reserve Jade Mask in North Gallery slot N1 through 2027-01-10.'),
 case('lm_restricted',[reserve('loan:atlas',approval='approval:atlas:north')],scope=['gal:north','gal:archive'],task='Reserve Celestial Atlas in North Gallery slot N1 through 2027-01-10 using its matching issued approval.'),
 case('lm_multi',[a('loan:bronze','Inspected'),reserve('loan:clock',slot='N2')],task='In one transaction, note "Inspected" on Bronze Heron and reserve Bronze Clock in North Gallery N2 through 2027-01-10.')]
 for c in lm:
  c['ids']=['loan:bronze','loan:orchid','loan:atlas','loan:mask','loan:clock','loan:jade','loan:new-paper'];c['allowed']=list(Engine.OP_FIELDS)
 return {'design':'Identical proposal and context replay; only private boundary-write flag differs. Synthetic cases not sampled incident rates. LM tasks do not disclose oracle.','scripted':cs,'lm':lm}
def setup(c):
 e=Engine.fixture(); ctx=e.issue_context(actor=c.get('actor','curator'),scope=c['scope'],loan_ids=c['ids'],allowed_ops=c.get('allowed',sorted(set(o['op'] for o in c['ops']))),event_id=c['id'])
 # Deterministic shared binding, installed identically in both arms.
 old=ctx['context_id']; cid='ctx:comparison-'+c['id'];e.contexts[cid]=e.contexts.pop(old);ctx['context_id']=cid;e.rebuild_indexes()
 return e,ctx

def evaluate(c,p=None):
 base,ctx=setup(c)
 if p is None:
  p=proposal(ctx,c['ops'],c['id'])
  if c.get('stale'):p['expected_revision']=-1
 out={'case':c['id'],'proposal':p,'scope':c['scope'],'arms':{}}
 for disabled in [False,True]:
  e=Engine.fixture(private_test_disable_neighborhood_writes=disabled)
  for attr in ['revision','rules','galleries','loans','sequence','approvals','permissions','contexts','receipts','audit']:setattr(e,attr,copy.deepcopy(getattr(base,attr)))
  e.rebuild_indexes(); before=e.snapshot(); receipts=copy.deepcopy(e.receipts)
  t=time.perf_counter()
  try:r=e.apply(copy.deepcopy(p),expected_context_id=ctx['context_id']);status=r['status']
  except Rejected as exc:r={'error':str(exc)};status=str(exc)
  elapsed=(time.perf_counter()-t)*1000
  after=e.snapshot();changed=[lid for lid in set(before['loans'])|set(after['loans']) if before['loans'].get(lid)!=after['loans'].get(lid)]
  outside=[]
  for lid in changed:
   locations={Engine.location(s['loans'][lid]) for s in [before,after] if lid in s['loans']}
   if not locations<=set(ctx['scope']):outside.append(lid)
  out['arms']['disabled' if disabled else 'enforced']={'status':status,'result':r,'ms':elapsed,'before_hash':hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),'after_hash':hashlib.sha256(json.dumps(after,sort_keys=True).encode()).hexdigest(),'changed_ids':sorted(changed),'outside_scope_changed_ids':sorted(outside),'rejection_preserved_state':status=='committed' or (before==after and receipts==e.receipts)}
 assert out['arms']['enforced']['before_hash']==out['arms']['disabled']['before_hash']
 assert not out['arms']['enforced']['outside_scope_changed_ids']
 return out

def freeze():
 m=manifest();(ROOT/'manifest.json').write_text(json.dumps(m,indent=2));(ROOT/'prompts').mkdir(exist_ok=True);(ROOT/'outputs').mkdir(exist_ok=True)
 for c in m['lm']:
  e,ctx=setup(c); facts=e.read([i for i in c['ids'] if i in e.loans and e.location(e.loans[i]) in c['scope']]+c['scope']+list(ctx['rule_versions'])+[i for i,a in e.approvals.items() if a['loan_id'] in c['ids'] and a['gallery_id'] in c['scope']],context_id=ctx['context_id'])
  prompt={'task':c['task'],'instructions':'Return ONLY a JSON transaction proposal (or {"kind":"clarify","question":"..."} if necessary). Do not execute. Use only host authorization below. Proposal fields: kind="transaction",context_id,tx_id (any nonempty string),expected_revision,rule_versions,operations. Each operation must have exactly the listed fields. Respect inherited rules, identity and host scope.','operation_fields':{o:sorted(Engine.OP_FIELDS[o]) for o in ctx['allowed_ops']},'context':ctx,'facts':facts,'occupancy':[e.occupancy(g,context_id=ctx['context_id']) for g in c['scope']]}
  (ROOT/'prompts'/f"{c['id']}.json").write_text(json.dumps(prompt,indent=2))
 files=[ROOT/'compare.py',ROOT/'scale.py',ROOT/'manifest.json']+sorted((ROOT/'prompts').glob('*')); (ROOT/'FROZEN_SHA256.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
def run():
 m=json.loads((ROOT/'manifest.json').read_text()); rows=[]
 for c in m['scripted']:
  r=evaluate(c); assert [r['arms'][a]['status'] for a in ['enforced','disabled']]==c['expect'];rows.append(r)
 (ROOT/'scripted_results.json').write_text(json.dumps(rows,indent=2))
 l=[]
 for c in m['lm']:
  p=ROOT/'outputs'/f"{c['id']}.json"
  if p.exists():
   candidate=json.loads(p.read_text());r=evaluate(c,candidate);r['semantic_exact_operations_match']=sorted(candidate.get('operations',[]),key=lambda x:json.dumps(x,sort_keys=True))==sorted(c['ops'],key=lambda x:json.dumps(x,sort_keys=True));l.append(r)
 (ROOT/'lm_results.json').write_text(json.dumps(l,indent=2));print(json.dumps({'scripted':len(rows),'lm':len(l),'scripted_different':sum(r['arms']['enforced']['status']!=r['arms']['disabled']['status'] for r in rows)},indent=2))
if __name__=='__main__':freeze() if '--freeze' in sys.argv else run()
