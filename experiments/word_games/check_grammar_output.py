"""Build and run native grammar-output regression checks on the cached 3B."""
import argparse
import os
from pathlib import Path
import subprocess
from .common import sha, write, read
from .native_speed import GPU, IMAGE
from .prepare_sampling_runtime import IMAGE as BUILD_IMAGE
from .qwen_pair import entrypoint, PAIRS


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    root = args.root.resolve()
    native = Path(__file__).with_name('native_sampling').resolve()
    model = entrypoint('draft').resolve()
    if sha(model) != PAIRS['draft']['files'][0][2]:
        raise ValueError('cached model hash mismatch')
    build = read(args.build_manifest)
    for arm in ('control', 'candidate'):
        if any(sha(root/arm/name) != digest for name, digest in build['runtimes'][arm]['hashes'].items()):
            raise ValueError('runtime hash mismatch')
    producer = {'source_hashes': {str(p): sha(p) for p in [Path(__file__), *native.glob('*.cpp')]},
                'build_manifest': str(args.build_manifest), 'build_sha256': sha(args.build_manifest),
                'model': str(model), 'model_sha256': sha(model)}
    write(out/'producer.json', producer)
    def execute(command, name):
        write(out/(name+'.command.json'), command)
        with (out/name).open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    docker = ['docker', 'run', '--rm', '--user', f'{os.getuid()}:{os.getgid()}',
              '-v', f'{root}:/work', '-v', f'{native}:/tests:ro', BUILD_IMAGE]
    try:
        execute(docker+['g++', '-O2', '-shared', '-fPIC', '/tests/count-output.cpp', '-I/work/source/ggml/include',
                        '-o', '/work/count-sync.so', '-ldl'], 'build-counter.log')
        for arm in ('control', 'candidate'):
            compile_cmd = ['g++', '-std=c++17', '-O2', '-rdynamic', '/tests/test-grammar-output.cpp',
                           '-I/work/source/tests', '-I/work/source/include', '-I/work/source/common',
                           '-I/work/source/ggml/include', f'-L/work/{arm}', '-lllama-common', '-lllama',
                           '-lggml', '-lggml-base', '-ldl', '-pthread', '-o', f'/work/test-{arm}']
            if arm == 'control': compile_cmd.append('-DTEST_CONTROL')
            execute(docker+compile_cmd, f'{arm}-compile.log')
            command = ['docker', 'run', '--rm', '--gpus', f'device={GPU}',
                       '-v', f'{root/arm}:/llama:ro', '-v', f'{root}:/work:ro',
                       '-v', f'{model}:/model.gguf:ro',
                       '-v', '/usr/local/cuda/targets/x86_64-linux/lib:/cuda:ro',
                       '-e', 'LD_LIBRARY_PATH=/llama:/cuda', '-e', f'CUDA_VISIBLE_DEVICES={GPU}',
                       '--entrypoint', f'/work/test-{arm}', IMAGE, '/model.gguf']
            execute(command, f'{arm}-test.log')
            print(arm+' native integration tests passed', flush=True)
        # The context/graph changes must also preserve patch 1's ordinary raw,
        # backend logits/probability/token and multi-output accessor paths.
        compatibility = [s.replace('/tests/test-grammar-output.cpp','/tests/test-sampling-output.cpp')
                         for s in compile_cmd]
        compatibility[compatibility.index('-o')+1] = '/work/test-compatibility'
        execute(docker+compatibility, 'compatibility-compile.log')
        command[command.index('--entrypoint')+1] = '/work/test-compatibility'
        execute(command, 'compatibility-test.log')
        print('Patch-1 compatibility integration tests passed', flush=True)
        write(out/'result.json', {'passed': True, 'producer_sha256': sha(out/'producer.json'),
                                 'counter_library_sha256': sha(root/'count-sync.so')})
    except BaseException as exc:
        write(out/'interruption.json', {'error': repr(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--build-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
