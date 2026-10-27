"""Fetch pinned larger NCM assets and inspect compatibility, without training."""
import hashlib
import json
import os
from pathlib import Path

REPO = 'hazyresearch/ncm-checkpoints'
REVISION = 'd342ce8aaea58830eef3b8ad473664673e41f692'
ROOT = Path('data/experiments/ncm_paper_regime/20261027_lm1b_large_assets_v1')
EXPECTED = {'lm1b/large/config.yaml':('4410e3710360cae162f8be4186a44a0c7c473be674a1ac3d1d940bfe45d5e70c',2753),
    'lm1b/large/generator.pt':('ec34d6c56a81f428303b2fa3a8e1615ceee81b7dec7f59380427f46808ec224c',2901184267)}
SMALL = Path('data/experiments/ncm_paper_regime/20261024_lm1b_native_release_v1/files/lm1b')
CODEC_SHA = 'd88aa17b66d730f2cb1fb31c5dbe63888a10da47ed8bea29dcb1f89423c3059c'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or ROOT.exists():
        raise ValueError('fresh CPU-only asset inventory required')
    import torch
    from huggingface_hub import HfApi,hf_hub_download,try_to_load_from_cache
    from omegaconf import OmegaConf
    info = HfApi().model_info(REPO,revision=REVISION,files_metadata=True)
    if info.sha != REVISION:raise ValueError('pinned HF revision differs')
    metadata = {}
    for f in info.siblings:
        if f.rfilename in EXPECTED or f.rfilename == 'lm1b/tokenizer.pt':
            metadata[f.rfilename] = {'size':f.size,'blob_id':f.blob_id,'lfs':dict(f.lfs) if f.lfs else None}
    if (metadata['lm1b/large/generator.pt']['lfs']['sha256'] != EXPECTED['lm1b/large/generator.pt'][0]
            or metadata['lm1b/tokenizer.pt']['lfs']['sha256'] != CODEC_SHA):
        raise ValueError('pinned author weight identities differ')
    ROOT.mkdir();write(ROOT/'producer.json',{'repository':REPO,'revision':REVISION,
        'source_sha256':sha(__file__),'expected':EXPECTED,'author_metadata':metadata,
        'model_training':False,'new_judge_requests':0,'final_test_opened':False})
    files,cache_hits = {},{}
    for name,(expected,size) in EXPECTED.items():
        cached = try_to_load_from_cache(REPO,name,revision=REVISION)
        cache_hits[name] = isinstance(cached,str)
        p = Path(hf_hub_download(REPO,name,revision=REVISION))
        if p.stat().st_size != size or sha(p) != expected:
            raise ValueError('download size or SHA256 differs: '+name)
        files[name] = str(p)
        print(json.dumps({'verified':name,'bytes':size,'cache_hit':cache_hits[name]}),flush=True)
    codec = SMALL/'tokenizer.pt'
    if sha(codec) != CODEC_SHA:raise ValueError('existing shared native codec differs')
    large = OmegaConf.load(files['lm1b/large/config.yaml']);small = OmegaConf.load(SMALL/'small/config.yaml')
    a,b = [OmegaConf.to_container(v.tokenizer.vqvae,resolve=True) for v in (small,large)]
    a.pop('checkpoint_path',None);b.pop('checkpoint_path',None)
    if a != b:raise ValueError('small and large codec geometries differ')
    if (large.model.n_layer,large.model.n_head,large.model.n_embd) != (36,20,1280):
        raise ValueError('expected released large architecture differs')
    saved = torch.load(files['lm1b/large/generator.pt'],map_location='cpu',weights_only=True,mmap=True)
    state = saved['model']
    if not state or not all(isinstance(v,torch.Tensor) for v in state.values()):
        raise ValueError('plain released tensor state required')
    shapes = {k:{'shape':list(v.shape),'dtype':str(v.dtype),'elements':v.numel()} for k,v in state.items()}
    write(ROOT/'state_shapes.json',shapes)
    keys = set(small.model)|set(large.model)
    differences = {k:{'small':OmegaConf.to_container(small.model,resolve=True).get(k),
                      'large':OmegaConf.to_container(large.model,resolve=True).get(k)}
                   for k in keys if small.model.get(k) != large.model.get(k)}
    value = {'status':'pinned_large_NCM_assets_verified','repository':REPO,'revision':REVISION,
        'files':files,'file_sha256':{k:v[0] for k,v in EXPECTED.items()},'cache_hits':cache_hits,
        'reused_codec':str(codec),'codec_sha256':CODEC_SHA,'codec_configuration_identical':True,
        'layers':36,'heads':20,'width':1280,'model_configuration_differences':differences,
        'state_tensors':len(state),'state_elements_including_buffers':sum(v.numel() for v in state.values()),
        'strict_model_load_tested':False,'GPU_fit_tested':False,'model_training':False,
        'new_judge_requests':0,'task_accuracy':None,'final_test_opened':False,'speed_tested':False,
        'artifacts':{str(p):sha(p) for p in (ROOT/'producer.json',ROOT/'state_shapes.json')}}
    write(ROOT/'result.json',value);print(json.dumps({k:v for k,v in value.items() if k not in ('artifacts','files','file_sha256')}),flush=True)


if __name__ == '__main__':
    existed = ROOT.exists()
    try:run()
    except Exception as error:
        if ROOT.exists() and not existed:write(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
