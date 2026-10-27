"""Compare exact author text rendering on every frozen XSum probe output."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

from experiments.structure_net import ncm_xsum_semantic_probe as probe
from experiments.structure_net import ncm_decode_policy as decoding

data = probe.data
ROOT = data.ROOT/'20261027_xsum_render_parity_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-xsum-render-parity.md')
AUTHOR = Path('data/experiments/ncm_llama/vendor/ncm-20261021/scripts/eval_shard.py')
AUTHOR_TOKENIZER = AUTHOR.parent.parent/'tokenization/hf_tokenizer.py'


def author_decoder(tokenizer):
    source = AUTHOR.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == 'dec')
    scope = {'CUT_AT_EOS':True,'EOS':50256,'PAD':50257,
             'tk':SimpleNamespace(pre_tokenizer=tokenizer)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(AUTHOR),'exec'),scope)
    return scope['dec'],ast.get_source_segment(source,node)


def run():
    import tiktoken
    if ROOT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only rendering diagnostic required')
    probe.configure();cases,packets = probe.prepared();tokenizer = probe.tokenizer()
    if tokenizer.eos_token_id != 50256 or tokenizer.pad_token_id != 50257:
        raise ValueError('released EOS/PAD identities required')
    author,source = author_decoder(tokenizer)
    score_path = probe.ROOTS['grade']/'secondary_native_scores.json'
    native = data.read(score_path)
    if len(native) != 128 or [r['id'] for r in native] != [r['id'] for r in cases]:
        raise ValueError('completed fixed native128 score file required')
    cache, raw_paths = {}, []
    for packet,score in zip(packets['native'],native):
        path = Path(score['first_origin'])/'row.json';row = data.read(path)
        key = probe.first.packet_key(packet,'secondary')
        if score['packet_key'] != key:
            raise ValueError('native first-decision identity differs')
        cache[key] = {'packet':packet,'role':'secondary','row':row,'origin':score['first_origin']}
        raw_paths += [path,*map(Path,row['attempts'])]
    replay = probe.score_rows(cases,packets['native'],cache,'native','secondary')
    if replay != native or not all(r['valid'] for r in replay):
        raise ValueError('complete native first judgments must replay exactly')
    corrected = deepcopy(cases);changes = {a:[] for a in probe.ARMS}
    for old,new in zip(cases,corrected):
        new['prior_texts'] = dict(old['texts'])
        for arm in probe.ARMS:
            expected = author(old['raw_words'][arm]).strip()
            actual = decoding.render_author(old['raw_words'][arm],tokenizer)
            if actual != expected:
                raise ValueError('local author renderer differs from exact pinned script')
            new['texts'][arm] = actual
            if actual != old['texts'][arm]:
                changes[arm].append(old['id'])
        new['rendering_policy'] = 'stop_at_first_EOS_remove_PAD_strip_outer_whitespace'
    corrected_packets = probe.packets_for(corrected)
    changed_native = set(changes['native'])
    unchanged_passes = sum(r['passed'] is True and r['id'] not in changed_native for r in native)
    paths = [Path(__file__),PLAN,Path(decoding.__file__),AUTHOR,AUTHOR_TOKENIZER,score_path,
        *[probe.ROOTS['prepare']/n for n in ('producer.json','result.json','actual_tool_receipt.json',
                                          'cases.json','exact_packets.json')],*raw_paths]
    hashes = {str(p):data.sha(p) for p in dict.fromkeys(paths)}
    ROOT.mkdir()
    proof = {'schema':'nal.ncm-xsum-render-parity.v1','hypothesis_id':'ncm-xsum-render-parity-v1',
        'source_and_input_hashes':hashes,'author_decoder_source':source,
        'configuration':{'cases_per_arm':128,'arms':probe.ARMS,'stored_token_cap':128,
            'outer_whitespace_strip':True,'no_model_generation':True,'judge_requests':0}}
    data.write(ROOT/'producer.json',proof)
    data.write(ROOT/'corrected_cases.json',corrected)
    data.write(ROOT/'corrected_packets.json',corrected_packets)
    detail = [];stored_tokenizer = tiktoken.get_encoding('cl100k_base')
    for i,(old,new) in enumerate(zip(cases,corrected)):
        for arm in probe.ARMS:
            old_key = probe.first.packet_key(packets[arm][i],'secondary')
            new_key = probe.first.packet_key(corrected_packets[arm][i],'secondary')
            if (old_key != new_key) != (old['texts'][arm] != new['texts'][arm]):
                raise ValueError('text/packet change identity differs')
            detail.append({'id':old['id'],'arm':arm,'changed':old_key != new_key,
                'prior_text':old['texts'][arm],'corrected_text':new['texts'][arm],
                'prior_stored_tokens':len(stored_tokenizer.encode(old['texts'][arm],disallowed_special=())),
                'corrected_stored_tokens':len(stored_tokenizer.encode(new['texts'][arm],disallowed_special=())),
                'prior_packet_key':old_key,'corrected_packet_key':new_key})
    data.write(ROOT/'comparison.json',detail)
    data.verify(hashes)
    result = {'status':'all512_author_text_decodes_match','cases_per_arm':128,'arm_cases':512,
        'changed_text_counts':{a:len(v) for a,v in changes.items()},'changed_ids':changes,
        'original_native_judgments_valid':128,'original_native_passes':sum(r['passed'] for r in native),
        'unchanged_native_passes':unchanged_passes,
        'native_render_only_pass_upper_bound':unchanged_passes+len(changed_native),
        'corrected_semantic_accuracy':None,'bound_is_not_measured_corrected_accuracy':True,
        'complete_original_comparison_at_observation':(probe.ROOTS['grade']/'result.json').exists(),
        'judge_requests':0,'model_updates':0,'final_test_opened':False,'speed_tested':False,
        'artifacts':{str(ROOT/n):data.sha(ROOT/n) for n in
            ('producer.json','corrected_cases.json','corrected_packets.json','comparison.json')}}
    data.write(ROOT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('changed_ids','artifacts')}),flush=True)


if __name__ == '__main__':
    existed = ROOT.exists()
    try:
        run()
    except Exception as error:
        if ROOT.exists() and not existed:
            data.write(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
