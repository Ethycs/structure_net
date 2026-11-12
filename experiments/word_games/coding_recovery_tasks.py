"""New synthetic, shareable coding tasks. No external/private task material.

Reference functions and hidden cases stay host-side; actors get only a prose
contract and one example input, without its expected answer.
"""
import copy
from .bridges.vendor.source_runtime import Engine


SPECS = (
 ('c01', 'Return a list of all unoccupied gallery slots as objects with gallery_id and slot. '
  'Read galleries and loans. A loan with reservation null occupies no slot. '
  'Sort the output by gallery_id, then slot.'),
 ('c02', 'Return the id of the reserved loan with the earliest reservation due date. '
  'Break equal-date ties by id in lexicographic order. Return null if none are reserved.'),
 ('c03', 'Return a dictionary counting loans for each medium present. '
  'Do not include categories with zero loans. Count each loan exactly once.'),
 ('c04', 'Return a dictionary from each gallery id to the minimum latest_due among '
  'all rules listed by that gallery. The rules dictionary holds those rule records.'),
 ('c05', 'Return loan ids whose notes contain the query in data.text as a case-insensitive '
  'substring of at least one note. Ignore the top-level notes field. Sort the ids.'),
 ('c06', 'Return the ids of all loans ordered by their title, case-insensitively. '
  'Break equal-title ties by id in lexicographic order.'),
 ('c07', 'Return sorted ids of restricted loans with a non-null reservation that lacks '
  'a valid approval_id. An approval is valid exactly when it exists, matches loan_id '
  'and gallery_id, and its until date is at least the reservation due date. '
  'Ignore unreserved and unrestricted loans.'),
 ('c08', 'Return the id of the loan with the greatest creation sequence among those '
  'whose creation event_id equals data.event_id. Return null if none match.'),
)


def reference(task, data):
    loans, galleries = data['loans'], data['galleries']
    if task=='c01':
        occupied={(x['reservation']['gallery_id'],x['reservation']['slot']) for x in loans.values() if x['reservation']}
        return [{'gallery_id':g,'slot':s} for g in sorted(galleries)
                for s in sorted(galleries[g]['slots']) if (g,s) not in occupied]
    if task=='c02':
        rows=[(x['reservation']['due'],i) for i,x in loans.items() if x['reservation']]
        return min(rows)[1] if rows else None
    if task=='c03':
        result={}
        for x in loans.values(): result[x['medium']]=result.get(x['medium'],0)+1
        return result
    if task=='c04':
        return {g:min(data['rules'][r]['latest_due'] for r in x['rules']) for g,x in galleries.items()}
    if task=='c05':
        return sorted(i for i,x in loans.items() if any(data['text'].lower() in n.lower() for n in x['notes']))
    if task=='c06':
        return sorted(loans,key=lambda i:(loans[i]['title'].lower(),i))
    if task=='c07':
        bad=[]
        for i,x in loans.items():
            r=x['reservation']
            if not x['restricted'] or not r: continue
            a=data['approvals'].get(r['approval_id'])
            if not a or a['loan_id']!=i or a['gallery_id']!=r['gallery_id'] or a['until']<r['due']: bad.append(i)
        return sorted(bad)
    if task=='c08':
        rows=[(x['creation']['sequence'],i) for i,x in loans.items() if x['creation']['event_id']==data['event_id']]
        return max(rows)[1] if rows else None
    raise ValueError(task)


def inputs(task):
    e=Engine.fixture()
    d={k:copy.deepcopy(v) for k,v in e.snapshot().items() if k in ('loans','galleries','rules','approvals')}
    for x in list(d['loans'].values())+list(d['galleries'].values()): x.pop('target_id',None)
    d.update(text='urgent',event_id='event-alpha',notes=['WG_SYNTHETIC_PRIVATE_'+task+'_46a17d90b5'])
    for i,x in enumerate(d['loans'].values()):
        x['notes']=['Urgent inspection needed' if i%2==0 else 'Routine check only']
        x['creation']['event_id']='event-alpha' if i%2==0 else 'event-beta'
    variants=[copy.deepcopy(d) for _ in range(4)]
    for v in variants[1:]:
        v['loans']['loan:bronze']['reservation']={'gallery_id':'gal:north','slot':'N1','due':'2027-01-10','approval_id':None}
        v['loans']['loan:clock']['reservation']={'gallery_id':'gal:north','slot':'N2','due':'2027-01-10','approval_id':None}
        v['loans']['loan:atlas']['reservation']={'gallery_id':'gal:archive','slot':'A1','due':'2027-02-20','approval_id':'approval:atlas:archive'}
    variants[2]['loans']['loan:atlas']['reservation']['approval_id']=None
    variants[2]['text']='ROUTINE'
    variants[2]['event_id']='event-beta'
    variants[2]['loans']['loan:bronze']['title']='ZZZ Synthetic Title'
    variants[2]['loans']['loan:clock']['title']='AAA Synthetic Title'
    variants[2]['rules']['rule:root']['latest_due']='2027-01-31'
    variants[3]['loans']={}
    variants[3]['text']='absent'
    variants[3]['event_id']='unknown-event'
    return variants
