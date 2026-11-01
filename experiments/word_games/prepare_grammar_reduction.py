"""Build the reduced preview using verified prior objects and unchanged control."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
from .common import sha,read,write
from .prepare_sampling_runtime import IMAGE,CHANGED

PATCH=Path('experiments/word_games/native_sampling/grammar-partitioned-preview.patch')


def run(args):
    out,root=args.output,args.root
    out.mkdir(parents=True,exist_ok=False)
    root.mkdir(parents=True,exist_ok=False)
    parent=read(args.parent_manifest)
    previous=Path(parent['runtimes']['candidate']['path']).parent
    for name,h in parent['source_hashes'].items():
        if sha(previous/'source'/name)!=h:raise ValueError('parent source changed: '+name)
    for arm in ('control','candidate'):
        for name,h in parent['runtimes'][arm]['hashes'].items():
            if sha(previous/arm/name)!=h:raise ValueError('parent binary changed: '+name)
    producer={'parent_manifest':str(args.parent_manifest),'parent_sha256':sha(args.parent_manifest),
              'patch_sha256':sha(PATCH),'builder_sha256':sha(Path(__file__)),'build_image':IMAGE,'runtimes':{}}
    write(out/'producer.json',producer)
    (out/PATCH.name).write_bytes(PATCH.read_bytes())
    for name in ('source','build','control','candidate'):
        shutil.copytree(previous/name,root/name,symlinks=True)
    docker=['docker','run','--rm','--user',f'{os.getuid()}:{os.getgid()}',
            '-v',f'{root.resolve()}:/work','-v',f'{PATCH.resolve()}:/delta.patch:ro',IMAGE]
    def execute(command,name):
        write(out/(name+'.command.json'),command)
        with (out/name).open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        execute(docker+['git','-C','/work/source','apply','--check','/delta.patch'],'patch-check.log')
        execute(docker+['git','-C','/work/source','apply','/delta.patch'],'patch.log')
        execute(docker+['cmake','--build','/work/build','--target','llama-common','test-sampling','-j','8'],'build.log')
        execute(docker+['/work/build/bin/test-sampling'],'upstream-tests.log')
        for name in CHANGED:shutil.copy2(root/'build/bin'/name,root/'candidate'/name)
        for arm in ('control','candidate'):
            producer['runtimes'][arm]={'path':str(root/arm),'hashes':{
                p.name:sha(p) for p in (root/arm).iterdir() if p.is_file() and not p.is_symlink()}}
        for name,h in producer['runtimes']['candidate']['hashes'].items():
            if name not in CHANGED and h!=producer['runtimes']['control']['hashes'][name]:
                raise ValueError('unexpected backend/server change')
        producer['source_hashes']={str(p.relative_to(root/'source')):sha(p)
            for p in (root/'source').rglob('*') if p.is_file() and not p.is_symlink()}
        producer['completed']=True
        write(out/'result.json',producer)
        print('Partitioned preview built; upstream sampler tests passed',flush=True)
    except BaseException as exc:
        write(out/'interruption.json',{'error':repr(exc)})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent-manifest',type=Path,default=Path('data/experiments/word_games/20261031_grammar_build_v2/result.json'))
    run(parser.parse_args())
