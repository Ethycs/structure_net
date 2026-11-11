"""Data-driven autonomous-map bridge over byte-identical V2 indexed Engine.
No map repair, target-specific branches, oracle access, or language model here.
"""
import copy,json,re,hashlib
from pathlib import Path
from .source_runtime import Engine, Rejected, proposal
KEYS='id title home medium restricted reservation notes creation sequence event_id actor gallery_id slot due approval_id name department slots rules version latest_due allowed_media loan_id until exception op to_gallery_id text kind context_id tx_id expected_revision rule_versions operations question scope loan_ids allowed_ops referents revision loans galleries approvals records occupancy occupied visibility authorized_loan_count authority missing_ids next_after_sequence order'.split()
ENUMS='reserve transfer cancel annotate create transaction clarify metal paper stone textile medium empty visible restricted host_stored_only creation_desc creation_asc'.split()
PREFIXES=['loan:','gal:','rule:','approval:','dept:']
class Mapping:
 def __init__(self,path):
  self.raw=Path(path).read_bytes();self.data=json.loads(self.raw);m=self.data
  if m.get('schema_version')!=1: raise ValueError('schema_version')
  for k,keys in [('key_map',KEYS),('enum_map',ENUMS),('id_prefixes',PREFIXES)]:
   d=m[k]
   if set(d)!=set(keys) or len(set(d.values()))!=len(d) or not all(isinstance(v,str) and v for v in d.values()):raise ValueError('invalid_bijection:'+k)
  if not all(re.fullmatch('[a-z][a-z0-9_]*',v) for v in m['key_map'].values()):raise ValueError('invalid_key')
  if not all(v.endswith(':') for v in m['id_prefixes'].values()):raise ValueError('invalid_prefix')
  for k in ['name','identity','objective','rules']:
   if not isinstance(m['target'].get(k),str) or not m['target'][k]:raise ValueError('target:'+k)
  self.hash=hashlib.sha256(self.raw).hexdigest()
 def transform(self,x,reverse=False):
  km=self.data['key_map'];em=self.data['enum_map'];pm=self.data['id_prefixes']
  if reverse:km={v:k for k,v in km.items()};em={v:k for k,v in em.items()};pm={v:k for k,v in pm.items()}
  def scalar(v):
   if not isinstance(v,str):return v
   if v in em:return em[v]
   for a,b in pm.items():
    if v.startswith(a):return b+v[len(a):]
   return v
  def walk(v,field=None):
   if field in ('title','text','notes','name','actor','event_id','question','context_id','tx_id'):return copy.deepcopy(v)
   if isinstance(v,list):return [walk(a,field) for a in v]
   if isinstance(v,dict):
    out={}
    for k,a in v.items():
     if not reverse and k=='target_id':continue
     dest=km.get(k,scalar(k)); source_field=dest if reverse else k
     out[dest]=walk(a,source_field)
    return out
   return scalar(v)
  return walk(x)
 def encode(self,x):return self.transform(x)
 def decode(self,x):return self.transform(x,True)
 def public_manual(self):return copy.deepcopy(self.data['target'])
class Bridge:
 def __init__(self,map_path,engine=None):self.mapping=Mapping(map_path);self.engine=engine or Engine.fixture()
 def view(self,context):
  e=self.engine;cid=context['context_id'];scope=context['scope'];ids=context['loan_ids']
  visible=[i for i in ids if i in e.loans and e.location(e.loans[i]) in scope]
  records=e.read(visible+scope+list(context['rule_versions'])+[i for i,a in e.approvals.items() if a['loan_id'] in ids and a['gallery_id'] in scope],context_id=cid)
  # Actual indexed reads and occupancy; host does not resolve ambiguous referents.
  source={'records':records['records'],'occupancy':[e.occupancy(g,context_id=cid) for g in scope],**copy.deepcopy(context)}
  # Solver-view minimization: display names are unnecessary structural source
  # language; identities remain stable. Free-text title/note payloads are retained.
  actors=sorted(set(e.permissions)|{l['creation']['actor'] for l in e.loans.values()})
  actor_alias={a:'principal_'+str(i) for i,a in enumerate(actors)}
  def minimize(x):
   if isinstance(x,dict):return {k:(actor_alias[v] if k=='actor' else minimize(v)) for k,v in x.items() if k!='name'}
   if isinstance(x,list):return [minimize(v) for v in x]
   return x
  return self.mapping.encode(minimize(source))
 def apply(self,target_response,context,txid):
  m=self.mapping.data; k=m['key_map']; enums=m['enum_map']
  if not isinstance(target_response,dict):raise Rejected('target_response_type')
  if target_response.get(k['kind'])==enums['clarify']:
   if set(target_response)!={k['kind'],k['question']}:raise Rejected('target_clarification_fields')
  else:
   if set(target_response)!={k['operations']} or not isinstance(target_response[k['operations']],list):raise Rejected('target_response_fields')
   inverse_ops={enums[o]:o for o in self.engine.OP_FIELDS}
   for operation in target_response[k['operations']]:
    if not isinstance(operation,dict) or operation.get(k['op']) not in inverse_ops:raise Rejected('target_operation')
    typ=inverse_ops[operation[k['op']]]
    if set(operation)!={k[f] for f in self.engine.OP_FIELDS[typ]}:raise Rejected('target_operation_fields')
    for field,prefix in [('loan_id','loan:'),('gallery_id','gal:'),('to_gallery_id','gal:'),('home','gal:'),('approval_id','approval:')]:
     if k[field] in operation:
      value=operation[k[field]]
      if field=='approval_id' and value is None:continue
      if not isinstance(value,str) or not value.startswith(m['id_prefixes'][prefix]):raise Rejected('target_identifier')
    if typ=='create' and operation[k['medium']] not in {enums[v] for v in ('metal','paper','stone','textile')}:raise Rejected('target_enum')
  src=self.mapping.decode(target_response)
  if src.get('kind')=='clarify':p=src
  else:
   if set(src)!={'operations'}:raise Rejected('target_response_fields')
   p=proposal(context,src['operations'],txid)
  before=self.engine.snapshot()
  try:r=self.engine.apply(p,expected_context_id=context['context_id'])
  except Rejected as ex:r={'status':'rejected','code':str(ex),'mutated':False,'revision':self.engine.revision}
  return {'decoded':p,'result':r,'before':before,'after':self.engine.snapshot()}
