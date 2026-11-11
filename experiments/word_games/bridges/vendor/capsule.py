"""Strict capsule ABI and bounded checkpoint validator; no model or oracle here."""
import copy, hashlib, json, tempfile
from pathlib import Path
from .runtime import Mapping
CONCEPTS='bridge source_spec target_spec correspondence invariants neighborhood checkpoint translate self_translate awaiting_schema committed validate_then_install'.split()
TOP={'schema_version','generation','deployment','task_mapping','bridge_state','carried_state','limitations'}
def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def sha(x): return hashlib.sha256(canonical(x).encode()).hexdigest()
def mapping(data):
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'mapping.json';p.write_text(json.dumps(data));return Mapping(p)
def validate(c,parent=None):
    if not isinstance(c,dict) or set(c)!=TOP: raise ValueError('capsule_fields')
    if c['schema_version']!=4 or type(c['generation']) is not int: raise ValueError('capsule_version')
    d=c['deployment']
    if set(d)!={'identity','operating_instructions','self_translation_instructions','concepts'}:raise ValueError('deployment_fields')
    for k in ('identity','operating_instructions','self_translation_instructions'):
        if not isinstance(d[k],str) or not d[k].strip():raise ValueError('deployment:'+k)
    v=d['concepts']
    if set(v)!=set(CONCEPTS) or not all(isinstance(t,str) and t for t in v.values()) or len(set(v.values()))!=len(CONCEPTS):raise ValueError('concepts')
    m=mapping(c['task_mapping'])
    if not isinstance(c['limitations'],str) or not c['limitations']:raise ValueError('limitations')
    s=c['bridge_state']
    if s['active_domain']!=c['task_mapping']['target']['name']:raise ValueError('active_mapping_pointer')
    if s['pending_job']['phase']!=v['awaiting_schema'] or s['mode']!=v['translate'] or s['retained_receipt']['status']!=v['committed'] or s['install_policy']!=v['validate_then_install']:raise ValueError('protocol_concept_binding')
    if parent is not None:
        if c['generation']!=parent['generation']+1:raise ValueError('generation')
        ps=copy.deepcopy(parent['bridge_state'])
        ps['active_domain']=s['active_domain'];ps['mode']=s['mode'];ps['install_policy']=s['install_policy'];ps['pending_job']['phase']=s['pending_job']['phase'];ps['retained_receipt']['status']=s['retained_receipt']['status']
        if ps!=s:raise ValueError('bridge_state_changed')
        before=mapping(parent['task_mapping']).decode(parent['carried_state'])
        after=m.decode(c['carried_state'])
        if before!=after:raise ValueError('carried_state_changed')
        if d['operating_instructions']==parent['deployment']['operating_instructions']:raise ValueError('instructions_not_translated')
        if c['task_mapping']['target']['name']==parent['task_mapping']['target']['name']:raise ValueError('domain_not_changed')
    return {'valid':True,'generation':c['generation'],'capsule_sha256':sha(c),'source_checkpoint_sha256':sha(m.decode(c['carried_state'])),'active_domain':s['active_domain']}
