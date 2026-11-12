"""Typed relabeling and fixed semantic checks for the eight coding tasks.

Free text is deliberately preserved: this is a coding-equivalence test, not an
information firewall. ID prefixes change while suffixes/order remain intact.
"""
import copy
import random

from .coding_recovery_tasks import inputs, reference, SPECS
from .common import canonical

FIELDS = {'loans':'jobs','galleries':'pools','rules':'limits','approvals':'permits',
    'id':'identifier','title':'label','medium':'hardware_class','restricted':'requires_permit',
    'reservation':'allocation','home':'home_pool','gallery_id':'pool_id','loan_id':'job_id',
    'approval_id':'permit_id','slot':'lane','slots':'lanes','due':'deadline',
    'latest_due':'cutoff','until':'valid_until','creation':'origin','event_id':'batch_id',
    'notes':'messages','text':'query','department':'group','allowed_media':'allowed_hardware'}
CATEGORIES = {'metal':'cpu','paper':'gpu','stone':'tpu','textile':'fpga'}
PREFIXES = {'loan:':'job:','gal:':'pool:','rule:':'limit:','approval:':'permit:','dept:':'group:'}
TEXT_FIELDS = {'title','notes','text','name','actor','event_id'}
GENERATED = 32
SEED = 731112


class Relabel:
    def __init__(self):
        for table in (FIELDS,CATEGORIES,PREFIXES):
            if len(table)!=len(set(table.values())):raise ValueError('noninjective_table')

    def transform(self, value, reverse=False):
        fields={v:k for k,v in FIELDS.items()} if reverse else FIELDS
        categories={v:k for k,v in CATEGORIES.items()} if reverse else CATEGORIES
        prefixes={v:k for k,v in PREFIXES.items()} if reverse else PREFIXES
        def identity(text):
            for old,new in prefixes.items():
                if text.startswith(old):return new+text[len(old):]
            return text
        def walk(item, field=None):
            if field in TEXT_FIELDS:return copy.deepcopy(item)
            if isinstance(item,list):return [walk(x,field) for x in item]
            if isinstance(item,dict):
                result={}
                for key,child in item.items():
                    target=fields.get(key,identity(key))
                    if target in result:raise ValueError('dictionary_key_collision')
                    result[target]=walk(child,target if reverse else key)
                return result
            if not isinstance(item,str):return item
            if field in ('medium','allowed_media'):return categories.get(item,item)
            if field=='exception':
                return ({'hardware_class':'medium'} if reverse else {'medium':'hardware_class'}).get(item,item)
            return identity(item)
        return walk(value)

    def encode(self, value):return self.transform(value)
    def decode(self, value):return self.transform(value,True)

    def output(self, task, value, reverse=False):
        # Category-count dictionary keys are enum values, not schema fields.
        if task=='c03' and isinstance(value,dict):
            table={v:k for k,v in CATEGORIES.items()} if reverse else CATEGORIES
            return {table.get(key,key):count for key,count in value.items()}
        return self.transform(value,reverse)


def cases(task):
    base=inputs(task); result=copy.deepcopy(base)
    rng=random.Random(SEED+int(task[1:]))
    dates=['2027-01-01','2027-01-10','2027-02-01','2027-02-20']
    for _ in range(GENERATED):
        data=copy.deepcopy(base[1]); items=list(data['loans'].items());rng.shuffle(items)
        data['loans']=dict(items[:rng.randrange(len(items)+1)])
        cells=[(g,s) for g,record in data['galleries'].items() for s in record['slots']]
        rng.shuffle(cells)
        data['text']=rng.choice(['urgent','ROUTINE','', 'missing','metal','cpu'])
        data['event_id']=rng.choice(['event-alpha','event-beta','unknown-event'])
        for sequence,(identifier,record) in enumerate(data['loans'].items()):
            record['title']=rng.choice(['Alpha','alpha','Zulu','metal','paper','cpu'])
            record['medium']=rng.choice(list(CATEGORIES))
            record['notes']=rng.choice([[],['Urgent inspection'],['routine'],['metal cpu'],['URGENT','Routine']])
            record['restricted']=rng.choice([True,False])
            record['creation']['sequence']=sequence+1
            record['creation']['event_id']=rng.choice(['event-alpha','event-beta'])
            record['reservation']=None
            if cells and rng.random()<.7:
                gallery,slot=cells.pop()
                record['reservation']={'gallery_id':gallery,'slot':slot,'due':rng.choice(dates),
                    'approval_id':rng.choice([None,'approval:missing',*data['approvals']])}
        for rule in data['rules'].values():rule['latest_due']=rng.choice(dates)
        for permit in data['approvals'].values():permit['until']=rng.choice(dates)
        result.append(data)
    return result


def target_reference(task, data):
    """Independent target-field implementation, never sent to either model."""
    jobs,pools=data['jobs'],data['pools']
    if task=='c01':
        taken={(x['allocation']['pool_id'],x['allocation']['lane']) for x in jobs.values() if x['allocation'] is not None}
        return [{'pool_id':p,'lane':lane} for p in sorted(pools) for lane in sorted(pools[p]['lanes']) if (p,lane) not in taken]
    if task=='c02':
        allocated=[key for key in jobs if jobs[key]['allocation'] is not None]
        return min(allocated,key=lambda key:(jobs[key]['allocation']['deadline'],key)) if allocated else None
    if task=='c03':
        result={}
        for job in jobs.values():result[job['hardware_class']]=result.get(job['hardware_class'],0)+1
        return result
    if task=='c04':
        return {p:min(data['limits'][key]['cutoff'] for key in record['limits']) for p,record in pools.items()}
    if task=='c05':
        return sorted(key for key,job in jobs.items() if any(data['query'].lower() in text.lower() for text in job['messages']))
    if task=='c06':return sorted(jobs,key=lambda key:(jobs[key]['label'].lower(),key))
    if task=='c07':
        invalid=[]
        for key,job in jobs.items():
            allocation=job['allocation']
            if not job['requires_permit'] or allocation is None:continue
            permit=data['permits'].get(allocation['permit_id'])
            if not permit or permit['job_id']!=key or permit['pool_id']!=allocation['pool_id'] or permit['valid_until']<allocation['deadline']:
                invalid.append(key)
        return sorted(invalid)
    if task=='c08':
        matching=[key for key,job in jobs.items() if job['origin']['batch_id']==data['batch_id']]
        return max(matching,key=lambda key:(jobs[key]['origin']['sequence'],key)) if matching else None
    raise ValueError(task)


def suite():
    codec=Relabel(); result={}
    for task,goal in SPECS:
        data=cases(task); expected=[reference(task,x) for x in data]
        for source,want in zip(data,expected):
            encoded=codec.encode(source)
            assert codec.decode(encoded)==source
            actual=target_reference(task,encoded)
            assert canonical(codec.output(task,actual,True))==canonical(want)
            assert canonical(codec.output(task,want))==canonical(actual)
        result[task]={'goal':goal,'inputs':data,'expected':expected}
    return result
