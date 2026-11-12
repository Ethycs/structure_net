"""Exhaust the declared finite action domain against the actual V4 adapter."""
import argparse
from collections import Counter
import copy
import hashlib
import itertools
import json
from pathlib import Path
import threading
import time

from .bridges.service import export
from .bridges.vendor.runtime import Bridge, Mapping, KEYS, ENUMS, PREFIXES
from .bridges.vendor.source_runtime import Engine, Rejected, proposal
from .common import canonical, digest, sha, write

PHI = Path('experiments/word_games/fixtures/verified_reservation_phi.jsonc')
REGIMES = ('full','north_only','bronze_only','reserve_only','viewer',
           'north_curator','stale_revision','stale_rule')


def alphabet():
    def place(op, loan='bronze', gallery='north', slot='N1', due='2027-01-10', approval=None):
        return {'op':op,'loan_id':'loan:'+loan,
                'gallery_id' if op=='reserve' else 'to_gallery_id':'gal:'+gallery,
                'slot':slot,'due':due,'approval_id':approval}
    events = [place('reserve'),place('reserve',slot='N2'),
        place('reserve',gallery='archive',slot='A1'),
        place('reserve','atlas',approval='approval:atlas:north'),
        place('reserve','atlas'),
        place('reserve','atlas','south','S1',approval='approval:jade:south'),
        place('reserve',due='2026-10-29'),place('reserve',due='2027-02-16'),
        place('reserve',due='not-a-date'),place('reserve',slot='missing'),
        place('transfer',slot='N2'),place('transfer',gallery='archive',slot='A1'),
        place('transfer','atlas',slot='N2',approval='approval:atlas:north'),
        {'op':'cancel','loan_id':'loan:bronze'}, {'op':'cancel','loan_id':'loan:atlas'}]
    events += [{'op':'annotate','loan_id':'loan:bronze','text':x}
               for x in ('', 'metal', 'Synthetic note.')]
    events += [{'op':'create','loan_id':lid,'title':'Synthetic item','home':home,
                'medium':medium,'restricted':False}
               for lid,home,medium in [('loan:new','gal:north','metal'),
                 ('loan:new','gal:north','textile'),('loan:bronze','gal:north','metal'),
                 ('loan:new','gal:south','paper')]]
    return events


def states():
    cells = [None,('gal:north','N1'),('gal:north','N2'),('gal:archive','A1'),('gal:archive','A2')]
    for bronze, atlas in itertools.product(cells, repeat=2):
        if bronze is not None and bronze == atlas:
            continue
        e = Engine.fixture()
        for lid, cell in [('loan:bronze',bronze),('loan:atlas',atlas)]:
            if cell:
                g, slot = cell
                e.loans[lid]['reservation'] = {'gallery_id':g,'slot':slot,'due':'2027-01-10',
                    'approval_id':None if lid=='loan:bronze' else 'approval:atlas:'+g.split(':')[1]}
        e.rebuild_indexes(); e._invariants(e.loans)
        yield e


def clone(e):
    out = Engine.__new__(Engine)
    out.__dict__.update(copy.deepcopy({k:v for k,v in e.__dict__.items() if k!='lock'}))
    out.lock = threading.RLock()
    return out


def context(e, regime):
    c = e.issue_context(actor={'viewer':'viewer','north_curator':'north-curator'}.get(regime,'curator'),
        scope=['gal:north'] if regime=='north_only' else list(e.galleries),
        loan_ids=['loan:bronze'] if regime=='bronze_only' else ['loan:bronze','loan:atlas','loan:new'],
        allowed_ops=['reserve'] if regime=='reserve_only' else list(e.OP_FIELDS),event_id='synthetic-proof')
    old = c['context_id']; fixed = 'ctx:bounded-proof'
    e.contexts[fixed] = e.contexts.pop(old); e._context_ids[fixed] = e._context_ids.pop(old)
    c['context_id'] = fixed
    if regime=='stale_revision': e.revision += 1
    if regime=='stale_rule': e.rules['rule:north']['version'] += 1
    return c


def project(value):
    if isinstance(value,dict): return {k:project(v) for k,v in value.items() if k!='target_id'}
    if isinstance(value,list): return [project(v) for v in value]
    return value


def native(e, c, ops, txid):
    try:
        return e.apply(proposal(c,ops,txid), expected_context_id=c['context_id'])
    except Rejected as exc:
        return {'status':'rejected','code':str(exc),'mutated':False,'revision':e.revision}


def adapter(e, mapping):
    b = Bridge.__new__(Bridge); b.engine=e; b.mapping=mapping
    return b


def compare(base, c, ops, mapping, replay=True):
    left, right = clone(base), clone(base)
    before = export(base)
    b = adapter(right,mapping)
    expected = native(left,c,ops,'tx:proof')
    observed = b.apply(mapping.encode({'operations':ops}),c,'tx:proof')['result']
    checks = {
        'result':expected==observed,
        'complete_source_state':export(left)==export(right),
        'commutation':mapping.encode(left.snapshot())==mapping.encode(right.snapshot()),
        'typed_roundtrip':mapping.decode(mapping.encode(left.snapshot()))==project(left.snapshot()),
        'rollback':expected['status']!='rejected' or (export(left)==before and export(right)==before),
    }
    if replay:
        first_left, first_right = export(left), export(right)
        lr = native(left,c,ops,'tx:proof')
        rr = b.apply(mapping.encode({'operations':ops}),c,'tx:proof')['result']
        checks['replay'] = (lr==expected==rr and export(left)==first_left and export(right)==first_right)
    return checks, expected


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    m=Mapping(PHI); events=alphabet()
    state_list=list(states())
    words=[[]]+[[x] for x in events]+[list(x) for x in itertools.product(events,repeat=2)]
    sources=[Path(__file__),PHI,Path('experiments/word_games/bridges/service.py'),
             Path('experiments/word_games/bridges/vendor/runtime.py'),
             Path('experiments/word_games/bridges/vendor/source_runtime.py')]
    plan={'study':'word-games-structured-proof-coding-recovery-v1','kind':'bounded_exhaustive',
          'alphabet':events,'alphabet_size':len(events),'initial_states':len(state_list),
          'regimes':REGIMES,'atomic_lengths':[0,1,2],'replays_per_case':1,
          'expected_cases':len(state_list)*len(REGIMES)*len(words),
          'sources':{str(p):sha(p) for p in sources}}
    write(output/'plan.json',plan); write(output/'phi.json',m.data)
    tables={name:{'size':len(m.data[name]),'total':set(m.data[name])==set(keys),
                  'injective':len(set(m.data[name].values()))==len(keys)}
            for name,keys in [('key_map',KEYS),('enum_map',ENUMS),('id_prefixes',PREFIXES)]}
    prefixes=list(m.data['id_prefixes'].values())
    prefix_free=all(not a.startswith(b) for a,b in itertools.permutations(prefixes,2))
    # Counterexamples intentionally concern the *unrestricted* claim.
    legacy_a={'id':'loan:bronze','target_id':'one'}
    legacy_b={'id':'loan:bronze','target_id':'two'}
    target_token=next(v for k,v in m.data['enum_map'].items() if k!=v and v not in ENUMS)
    source_token=next(k for k,v in m.data['enum_map'].items() if v==target_token)
    collisions=[{'kind':'dropped_legacy_field','a':legacy_a,'b':legacy_b,
                 'same_encoding':m.encode(legacy_a)==m.encode(legacy_b)},
                {'kind':'open_string_enum_collision','a':source_token,'b':target_token,
                 'same_encoding':m.encode(source_token)==m.encode(target_token)}]
    counts=Counter(); failures=[]; executed=0; stream=hashlib.sha256(); start=time.monotonic()
    for si, original in enumerate(state_list):
        for regime in REGIMES:
            base=clone(original); c=context(base,regime)
            for wi, word in enumerate(words):
                checks,result=compare(base,c,word,m)
                executed+=1; counts[result['status']]+=1
                counts['length_'+str(len(word))]+=1
                if result.get('code'): counts['rejection:'+result['code']]+=1
                passed=all(checks.values())
                stream.update(canonical([si,regime,wi,checks,result]).encode())
                if not passed:
                    counts['counterexamples']+=1
                    if len(failures)<20: failures.append({'state':si,'regime':regime,'word':word,'checks':checks,'result':result})
        print(json.dumps({'states_done':si+1,'cases':executed,'counterexamples':counts['counterexamples']}),flush=True)
    bad=Mapping(PHI); decode=bad.decode
    def corrupted(value):
        result=decode(value)
        if isinstance(result,dict) and result.get('operations'):
            result['operations'][0]={'op':'cancel','loan_id':'loan:bronze'}
        return result
    bad.decode=corrupted
    base=Engine.fixture(); c=context(base,'full')
    control,_=compare(base,c,[events[0]],bad,replay=False)
    fault_detected=not all(control.values())
    passed=(executed==plan['expected_cases'] and not failures and counts['counterexamples']==0
            and counts['committed']>0 and counts['rejected']>0 and fault_detected
            and prefix_free and all(t['total'] and t['injective'] for t in tables.values()))
    result={'study':plan['study'],'passed':passed,'classification':'bounded_adapter_commutation_proved' if passed else 'counterexample_or_invalid_control',
        'unrestricted_json_bijection':False,'unrestricted_counterexamples':collisions,
        'table_checks':tables,'prefix_free':prefix_free,'cases':executed,'counts':dict(counts),
        'counterexamples':failures,'fault_control_detected':fault_detected,
        'case_stream_sha256':stream.hexdigest(),'plan_sha256':sha(output/'plan.json'),
        'seconds':time.monotonic()-start,
        'scope':'Actual Bridge adapter over actual Engine; finite declared states/contexts/atomic words only; no independent target Engine, natural-language proof, arbitrary JSON proof or secrecy proof.'}
    write(output/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not run(args.output)['passed']: raise SystemExit(1)
