"""Build patch 2 on verified patch-1 sources, retaining patch 1 as control."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import hashlib
from .common import sha, read, write
from .prepare_sampling_runtime import ARCHIVE, ARCHIVE_SHA, IMAGE, CHANGED

PATCH = Path('experiments/word_games/native_sampling/grammar-on-demand.patch')
PATCH1 = Path('experiments/word_games/native_sampling/sampling-output-once.patch')


def run(args):
    out, root = args.output, args.root
    out.mkdir(parents=True, exist_ok=False)
    root.mkdir(parents=True, exist_ok=False)
    prior = read(args.parent_manifest)
    if sha(ARCHIVE) != ARCHIVE_SHA or sha(PATCH1) != prior['patch_sha256']:
        raise ValueError('parent source archive/patch mismatch')
    parent = Path(prior['runtimes']['candidate']['path'])
    for name, h in prior['runtimes']['candidate']['hashes'].items():
        if sha(parent/name) != h: raise ValueError('parent runtime changed')
    with tarfile.open(ARCHIVE) as archive:
        archive.extractall(root, filter='data')
    (root/'llama.cpp-b11960').rename(root/'source')
    native = PATCH.parent.resolve()
    docker = ['docker','run','--rm','--user',f'{os.getuid()}:{os.getgid()}',
              '-v',f'{root.resolve()}:/work','-v',f'{native}:/patches:ro',IMAGE]
    commands = []
    def execute(command, name):
        commands.append(command)
        write(out/(name+'.command.json'),command)
        with (out/name).open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    producer = {'parent_manifest':str(args.parent_manifest),'parent_sha256':sha(args.parent_manifest),
                'patch_sha256':sha(PATCH),'builder_sha256':sha(Path(__file__)),
                'source_archive_sha256':ARCHIVE_SHA,'build_image':IMAGE,'runtimes':{}}
    write(out/'producer.json',producer)
    (out/PATCH.name).write_bytes(PATCH.read_bytes())
    (out/PATCH1.name).write_bytes(PATCH1.read_bytes())
    try:
        execute(docker+['git','-C','/work/source','apply','/patches/'+PATCH1.name],'patch1.log')
        # Reuse only matching compilation inputs from the earlier build.
        old_source = parent.parent/'source'
        for path in (root/'source').rglob('*'):
            if path.is_file() and not path.is_symlink():
                previous = old_source/path.relative_to(root/'source')
                if not previous.is_file() or sha(previous) != sha(path):
                    raise ValueError('parent build source mismatch: '+str(path))
                shutil.copystat(previous,path)
        shutil.copytree(parent.parent/'build',root/'build',symlinks=True)
        shutil.copytree(parent,root/'control',symlinks=True)
        producer['runtimes']['control'] = {'path':str(root/'control'),
                                          'hashes':prior['runtimes']['candidate']['hashes']}
        execute(docker+['git','-C','/work/source','apply','--check','/patches/'+PATCH.name],'patch2-check.log')
        execute(docker+['git','-C','/work/source','apply','/patches/'+PATCH.name],'patch2.log')
        execute(docker+['cmake','--build','/work/build','--target','llama-common','test-sampling','-j','8'],'build.log')
        execute(docker+['/work/build/bin/test-sampling'],'upstream-tests.log')
        shutil.copytree(parent,root/'candidate',symlinks=True)
        for name in CHANGED: shutil.copy2(root/'build/bin'/name,root/'candidate'/name)
        producer['runtimes']['candidate'] = {'path':str(root/'candidate'),
             'hashes':{p.name:sha(p) for p in (root/'candidate').iterdir() if p.is_file() and not p.is_symlink()}}
        for name,h in producer['runtimes']['candidate']['hashes'].items():
            if name not in CHANGED and h != producer['runtimes']['control']['hashes'][name]:
                raise ValueError('unexpected runtime change')
        producer['completed'] = True
        producer['source_hashes'] = {str(p.relative_to(root/'source')):sha(p)
            for p in (root/'source').rglob('*') if p.is_file() and not p.is_symlink()}
        write(out/'result.json',producer)
        print('Grammar preview candidate built; upstream sampler tests passed',flush=True)
    except BaseException as exc:
        write(out/'interruption.json',{'error':repr(exc)})
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent-manifest',type=Path,default=Path('data/experiments/word_games/20261031_sampling_output_build_v4/result.json'))
    run(parser.parse_args())
