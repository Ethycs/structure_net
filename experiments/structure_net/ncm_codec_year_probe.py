"""Post-hoc two-year discrimination probe after a concrete codec date error."""
import json
from pathlib import Path

from experiments.structure_net.ncm_codec_roundtrip import ROOT, CHECKPOINTS, read, sha, write, runtime

DEST = ROOT/'20261025_codec_year_discrimination_v1'
CASE = 'baf790eb9c52f1a7f6dd1ceff399e17bbcb927f3a6fbd19e31f127305b67958b'


def main():
    training, environment = runtime()
    import torch
    training.offload.require_rtx2060()
    if DEST.exists():
        raise ValueError('fresh diagnostic root required')
    data = ROOT/'20261025_codec_roundtrip_prepare_v1/cases.json'
    previous = ROOT/'20261025_codec_roundtrip_eval_v1/outputs.json'
    row = next(r for r in read(data) if r['id']==CASE)
    if row['target_text'].count('2021') != 1:
        raise ValueError('exact single-year substitution required')
    original = next(r for r in read(previous) if r['id']==CASE and r['arm']=='released_codec_only')
    DEST.mkdir()
    (DEST/'producer.py').write_bytes(Path(__file__).read_bytes())
    # Fix the two probes before encoding either variant. This follows an observed
    # failure and is explanatory only, never an independent accuracy estimate.
    proof = {'hypothesis_id':'ncm-summary-codec-roundtrip-v1',
             'stage':'post_hoc_specific_year_discrimination', 'case_id':CASE,
             'years':[2019,2021], 'selected_after_observed_error':True,
             'question':'Do distinct years collide in the full code hierarchy or its reconstructed latent?',
             'training':False,'source_attention':False,'FP32':True,'TF32':False,
             'environment':environment,
             'artifacts':{str(p):sha(p) for p in (data,previous,Path(__file__))}}
    write(DEST/'producer.json',proof)
    codec, tokenizer = training.native.load_codec(), training.native.tokenizer()
    base = {k:v.detach().cpu().clone() for k,v in codec.state_dict().items()}
    codes, latents, outputs = {}, {}, []
    before = training.native.vendor.module_hash(codec)
    with torch.no_grad():
        for year in (2019,2021):
            text = row['target_text'].replace('2021',str(year))
            lexical = tokenizer.encode(text,add_special_tokens=False)
            if len(lexical)>126 or set(lexical)&set(tokenizer.all_special_ids):
                raise ValueError('whole supported frame required')
            gold = [101,*lexical,101]+[0]*(126-len(lexical))
            codes[year] = codec.encode(torch.tensor([gold],device='cuda'))
            latents[year] = codec.vq.idx_to_z_q(codes[year])
            if year==2021 and (gold!=row['native_word_ids'] or codes[year][0].cpu().tolist()!=original['codes']):
                raise ValueError('original code replay differs')
        for name in ('released','parent','adapted'):
            if name=='released':
                codec.load_state_dict(base,strict=True)
            else:
                path, expected = CHECKPOINTS[name]
                if sha(path)!=expected:raise ValueError('checkpoint changed')
                saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)['model']
                prefix='concepts.codec.'
                codec.load_state_dict({k[len(prefix):]:v for k,v in saved.items() if k.startswith(prefix)},strict=True)
                proof['artifacts'][str(path)]=expected
            for year in (2019,2021):
                raw=codec.decode(codes[year]).argmax(-1)[0].cpu().tolist()
                text=training.preparation.rendering.decode(raw,tokenizer)['text']
                outputs.append({'checkpoint':name,'input_year':year,'text':text,'raw_word_ids':raw,
                                'codes':codes[year][0].cpu().tolist()})
        codec.load_state_dict(base,strict=True)
        if training.native.vendor.module_hash(codec)!=before:raise ValueError('restored codec differs')
        changed=int((codes[2019]!=codes[2021]).sum())
        delta=float((latents[2019]-latents[2021]).double().norm())
    write(DEST/'outputs.json',outputs)
    write(DEST/'producer.json',proof)
    result={'status':'specific_year_discrimination_measured', 'code_positions_different':changed,
            'code_positions':641,'latent_L2_difference':delta,'code_collision':changed==0,
            'latent_collision':delta==0,'case_id':CASE,'cases':1,'variants':2,
            'outputs_sha256':sha(DEST/'outputs.json'),'source_only_task_accuracy':None,
            'general_accuracy_claim':False,'training':False,'final_test_opened':False}
    write(DEST/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
