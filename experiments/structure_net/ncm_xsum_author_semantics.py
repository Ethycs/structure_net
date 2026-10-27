"""Grade author-rendered frozen outputs, retaining every prior first decision."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

from experiments.structure_net import ncm_xsum_semantic_probe as parent
from experiments.structure_net import ncm_xsum_render_parity as rendered
from experiments.structure_net import ncm_public_probe_lineage as ancestry

data, base, q = parent.data, parent.base, parent.q
qualified, first, ARMS = parent.qualified, parent.first, parent.ARMS
preserve, metric, qualifies = parent.preserve, parent.metric, parent.qualifies
score_rows, grade_arm = parent.score_rows, parent.grade_arm
HYPOTHESIS = 'ncm-xsum-author-semantics-v1'
SCHEMA = 'nal.ncm-xsum-author-semantics.v1'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-xsum-author-semantics-plan.md')
TEST = Path('tests/structure_net/test_ncm_xsum_author_semantics.py')
ANALYSIS = Path('experiments/neural_architecture_lab/ncm_xsum_author_semantics_results.py')
SERVICE = Path('experiments/neural_architecture_lab/ncm_xsum_author_semantics_service.py')
ROOTS = {s:data.ROOT/f'20261027_xsum_author_semantics_{s}_v1'
         for s in ('prepare','pilot','grade','audit','results')}


def configure():
    parent.configure()


def prior_cache():
    return ancestry.branch_cache(parent.prior_cache())


def checked_correction(old, corrected, tokenizer):
    if len(old) != 128 or len(corrected) != 128:
        raise ValueError('all128 unchanged sources required')
    for a,b in zip(old,corrected):
        expected = deepcopy(a)
        expected['prior_texts'] = dict(a['texts'])
        expected['rendering_policy'] = 'stop_at_first_EOS_remove_PAD_strip_outer_whitespace'
        expected['texts'] = {arm:rendered.decoding.render_author(a['raw_words'][arm],tokenizer) for arm in ARMS}
        if b != expected:
            raise ValueError('only the exact author rendering may change')
    return corrected


def require_rendering():
    state = base.require(rendered.ROOT)
    data.verify(data.read(rendered.ROOT/'producer.json')['source_and_input_hashes'])
    campaign = rendered.ROOT/'campaign_results.json'
    published = data.read(campaign);data.verify(published['artifacts'])
    backend = data.read(rendered.ROOT/'backend_exact_readback.json')
    receipt = data.read(rendered.ROOT/'backend_actual_tool_receipt.json')
    if (state['status'] != 'all512_author_text_decodes_match'
            or receipt['actual_tool_exit_code'] != 0
            or receipt['artifacts'].get(str(rendered.ROOT/'backend_exact_readback.json')) != data.sha(rendered.ROOT/'backend_exact_readback.json')
            or backend['status'] != 'independent_registry_readback_passed'
            or backend['campaign_sha256'] != data.sha(campaign)
            or backend['report_sha256'] != data.sha(published['report'])):
        raise ValueError('completed registered author rendering proof required')
    data.verify(receipt['artifacts'])
    return state


def freeze(root, stage, inputs, inherited_count):
    if root.exists():
        raise ValueError('fresh append-only stage required')
    files = {Path(__file__).resolve(),PLAN.resolve(),TEST.resolve(),ANALYSIS.resolve(),SERVICE.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()):
                files.add(p)
    inputs = [*inputs,*ancestry.branch_inputs(),*qualified.judge_inputs(),parent.FRONTIER,
        *[rendered.ROOT/n for n in ('producer.json','result.json','actual_tool_receipt.json',
            'campaign_results.json','backend_exact_readback.json','backend_actual_tool_receipt.json',
            'corrected_cases.json','corrected_packets.json')],Path('pixi.lock'),Path('pyproject.toml')]
    root.mkdir();(root/'producer_source').mkdir()
    for i,p in enumerate(sorted(files)):
        (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof = {'schema':SCHEMA,'hypothesis_id':HYPOTHESIS,'stage':stage,
        'config':{'arms':ARMS,'probe_cases':128,'pilot_cases':4,'workers':4,'judges':dict(q.JUDGES),
            'rubric':q.RUBRIC,'inherited_first_judgments':inherited_count,'stored_token_cap':128,
            'primary_gate':122,'directional_minimum_extra_passes':7,
            'rendering':'stop_at_first_EOS_remove_PAD_strip_outer_whitespace',
            'final_test_opened':False,'speed_tested':False,'public_reused_probe':True},
        'source_hashes':{str(p):data.sha(p) for p in sorted(files)},
        'input_hashes':{str(p):data.sha(p) for p in dict.fromkeys(inputs)}}
    proof['fingerprint'] = base.fingerprint(proof);data.write(root/'producer.json',proof)
    return proof


def finish(root, proof, value, files):
    base.verify_proof(proof)
    data.write(root/'result.json',{'schema':SCHEMA,'hypothesis_id':HYPOTHESIS,**value,
        'application_task_accuracy':None,'model_training':False,'final_test_opened':False,
        'speed_tested':False,'NCM_success_established':False,
        'artifacts':{str(p):data.sha(p) for p in [root/'producer.json',*files]}})
    print(json.dumps({'stage':proof['stage'],'status':value['status'],
        'canonical_first_judgments':value.get('canonical_first_judgments')}),flush=True)


def prepare():
    configure();qualified.require_judges();inherited = prior_cache();require_rendering()
    old,_ = parent.prepared()
    cases = checked_correction(old,data.read(rendered.ROOT/'corrected_cases.json'),parent.tokenizer())
    packets = parent.packets_for(cases)
    if packets != data.read(rendered.ROOT/'corrected_packets.json'):
        raise ValueError('saved corrected packets changed')
    uncertain = {r['key'] for r in data.read(parent.FRONTIER) if r['status'] == 'unknown_inflight_or_unstarted'}
    keys = {first.packet_key(p,role) for ps in packets.values() for p in ps for role in ('secondary','primary')}
    if keys & uncertain:
        raise ValueError('uncertain requests cannot receive new attempt budgets')
    root = ROOTS['prepare'];proof = freeze(root,'prepare',[],len(inherited))
    data.write(root/'cases.json',cases);data.write(root/'exact_packets.json',packets)
    needed = {first.packet_key(p,'secondary') for ps in packets.values() for p in ps}-set(inherited)
    finish(root,proof,{'status':'author_semantics_prepared','cases':128,'arm_cases':512,
        'inherited_first_judgments':len(inherited),'new_secondary_packets':len(needed),
        'new_judge_requests':0},[root/'cases.json',root/'exact_packets.json'])


def prepared():
    state = base.require(ROOTS['prepare']);base.verify_proof(data.read(ROOTS['prepare']/'producer.json'))
    cases = data.read(ROOTS['prepare']/'cases.json');packets = data.read(ROOTS['prepare']/'exact_packets.json')
    if (state['status'] != 'author_semantics_prepared' or len(cases) != 128
            or packets != parent.packets_for(cases)
            or cases != data.read(rendered.ROOT/'corrected_cases.json')):
        raise ValueError('complete frozen corrected packets required')
    return cases,packets


def grade(pilot=False):
    configure();qualified.require_judges();cases,packets = prepared()
    inherited = prior_cache();cache = dict(inherited)
    inputs = [ROOTS['prepare']/n for n in ('producer.json','result.json','actual_tool_receipt.json','cases.json','exact_packets.json')]
    if not pilot:
        if base.require(ROOTS['pilot'])['status'] != 'author_semantics_pilot_passed':
            raise ValueError('all pilot decisions must be valid')
        pilot_cache = data.read(ROOTS['pilot']/'judgment_cache.json');preserve(inherited,pilot_cache)
        for key,entry in pilot_cache.items():first.remember(cache,key,entry)
        inputs += [ROOTS['pilot']/n for n in ('producer.json','result.json','actual_tool_receipt.json','judgment_cache.json')]
    stage = 'pilot' if pilot else 'grade';root = ROOTS[stage]
    proof = freeze(root,stage,inputs,len(inherited));count = 4 if pilot else 128;cases = cases[:count]
    metrics = {}
    with (root/'new_firsts.jsonl').open('x') as journal:
        for arm in ARMS:
            metrics[arm] = {'secondary':grade_arm(root,cases,packets[arm][:count],cache,arm,'secondary',journal),'primary':None}
        if not pilot:
            for arm in ARMS:
                if qualifies(metrics[arm]['secondary']):
                    metrics[arm]['primary'] = grade_arm(root,cases,packets[arm],cache,arm,'primary',journal)
    preserve(inherited,cache);data.write(root/'judgment_cache.json',cache)
    status = ('author_semantics_pilot_passed' if all(v['secondary']['valid'] == 4 for v in metrics.values())
              else 'author_semantics_pilot_failed') if pilot else 'author_semantics_grade_completed'
    finish(root,proof,{'status':status,'metrics':metrics,'inherited_first_judgments':len(inherited),
        'canonical_first_judgments':len(cache),'new_judge_requests':sum(v[r]['new_requests'] for v in metrics.values() for r in v if v[r]),
        'pilot':pilot,'cases_per_arm':count},[root/'judgment_cache.json',root/'new_firsts.jsonl',
            *sorted(root.glob('*_scores.json')),*sorted((root/'requests').rglob('*.json'))])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('prepare','pilot','grade'))
    stage = parser.parse_args().stage;root = ROOTS[stage];existed = root.exists()
    try:
        prepare() if stage == 'prepare' else grade(stage == 'pilot')
    except Exception as error:
        if root.exists() and not existed:data.write(root/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
