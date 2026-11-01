"""Build matched b11960 runtimes for the sampling-output acquisition patch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path('.pixi/word-games-sampling-v1')
ARCHIVE = Path('.pixi/word-games-hybrid/b11960.tar.gz')
ARCHIVE_SHA = '2b37ffa871595f90b94164f6c510ad9a34bd7f94da21557cc9878587e163af37'
PATCH = Path('experiments/word_games/native_sampling/sampling-output-once.patch')
ORIGINAL = Path('.pixi/word-games-native-b11960/llama-b11960')
IMAGE = 'sha256:be4253918f64f049a1967f498cd3d6351d623189d143d66a05320fc713a54416'
CHANGED = ('libllama.so.0.8.0', 'libllama-common.so.0.8.0')


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def main(output, resume_control=False):
    output.mkdir(parents=True, exist_ok=False)
    if sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('b11960 archive changed')
    if resume_control:
        # Recover packaging-only failures without rebuilding unchanged objects.
        # Every source file must still match the pristine archive.
        with tarfile.open(ARCHIVE) as tar:
            for member in tar.getmembers():
                if member.isfile():
                    path = ROOT/'source'/Path(member.name).relative_to('llama.cpp-b11960')
                    with tar.extractfile(member) as handle:
                        expected = hashlib.file_digest(handle, 'sha256').hexdigest()
                    if sha(path) != expected:
                        raise ValueError('resume requires pristine source: '+str(path))
        if (ROOT/'candidate').exists():
            raise ValueError('candidate already exists')
    else:
        ROOT.mkdir(exist_ok=False)
        with tarfile.open(ARCHIVE) as tar:
            tar.extractall(ROOT, filter='data')
        (ROOT/'llama.cpp-b11960').rename(ROOT/'source')
    docker = ['docker','run','--rm','--user',f'{os.getuid()}:{os.getgid()}',
              '-v',f'{ROOT.resolve()}:/work', '-v',f'{PATCH.resolve()}:/sampling.patch:ro', IMAGE]
    config = ['cmake','-S','/work/source','-B','/work/build','-G','Ninja',
              '-DCMAKE_BUILD_TYPE=Release','-DGGML_BACKEND_DL=ON','-DGGML_NATIVE=OFF',
              '-DGGML_CUDA=OFF','-DLLAMA_BUILD_TESTS=ON','-DLLAMA_BUILD_EXAMPLES=OFF',
              '-DLLAMA_BUILD_TOOLS=OFF','-DLLAMA_BUILD_SERVER=OFF','-DLLAMA_CURL=OFF',
              '-DLLAMA_BUILD_COMMON=ON','-DLLAMA_BUILD_NUMBER=11960']
    build = ['cmake','--build','/work/build','--target','llama-common','test-sampling','-j','8']
    manifest = {'archive':str(ARCHIVE),'archive_sha256':ARCHIVE_SHA,
                'patch':str(PATCH),'patch_sha256':sha(PATCH),'builder_sha256':sha(Path(__file__)),
                'image':IMAGE,'config_command':docker+config,'build_command':docker+build,
                'resumed_pristine_control':resume_control,
                'original_hashes':{p.name:sha(p) for p in ORIGINAL.iterdir() if p.is_file() and not p.is_symlink()},
                'classification':'systems_lifecycle_only_not_quality_evidence','runtimes':{}}
    write(output/'producer.json',manifest)
    (output/PATCH.name).write_bytes(PATCH.read_bytes())
    def run(command, log):
        with (output/log).open('x') as handle:
            subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True)
    run(docker+config,'configure.log')
    for arm in ('control','candidate'):
        if arm == 'candidate':
            run(docker+['git','-C','/work/source','apply','--check','/sampling.patch'],'patch_check.log')
            run(docker+['git','-C','/work/source','apply','/sampling.patch'],'patch.log')
        print('Building '+arm,flush=True)
        run(docker+build,arm+'_build.log')
        runtime = ROOT/arm
        if not (resume_control and arm == 'control'):
            shutil.copytree(ORIGINAL,runtime,symlinks=True)
        for name in CHANGED:
            shutil.copy2(ROOT/'build/bin'/name,runtime/name)
        hashes = {p.name:sha(p) for p in runtime.iterdir() if p.is_file() and not p.is_symlink()}
        if any(h != manifest['original_hashes'][name] for name,h in hashes.items() if name not in CHANGED):
            raise ValueError('unexpected backend or executable change')
        manifest['runtimes'][arm]={'path':str(runtime),'hashes':hashes,
            'source_hashes':{name:sha(ROOT/'source'/name) for name in ['include/llama.h','src/llama-context.cpp','common/sampling.cpp']}}
        write(output/(arm+'_producer.json'),manifest['runtimes'][arm])
        run(docker+['/work/build/bin/test-sampling'],arm+'_test_sampling.log')
        print(arm+' built; upstream sampler tests passed',flush=True)
    manifest['completed']=True
    write(output/'result.json',manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--resume-control',action='store_true')
    args=parser.parse_args()
    ROOT=args.root
    main(args.output,args.resume_control)
