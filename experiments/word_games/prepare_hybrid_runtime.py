"""Patch the pinned b11960 tree; compile libllama against its unchanged GGML ABI."""
import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path('.pixi/word-games-hybrid')
SOURCE = ROOT / 'source'
NATIVE = Path('experiments/word_games/native_hybrid')


def replace(path, old, new):
    text = path.read_text()
    if new in text:
        return
    if text.count(old) != 1:
        raise ValueError(f'Pinned source anchor mismatch: {path}: {old[:80]}')
    path.write_text(text.replace(old, new))


def patch():
    for p in NATIVE.iterdir():
        shutil.copy2(p, SOURCE/'src'/p.name)
    replace(SOURCE/'src/CMakeLists.txt', '    llama-moe-cache.cpp\n',
            '    llama-moe-cache.cpp\n    llama-word-games-hybrid.cpp\n')
    for name in ['llama-context.cpp', 'llama-graph.cpp']:
        p = SOURCE/'src'/name
        replace(p, '#include "'+name.replace('.cpp','.h')+'"',
                '#include "'+name.replace('.cpp','.h')+'"\n#include "llama-word-games-hybrid.h"')
    replace(SOURCE/'src/llama-context.cpp',
            '        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);',
            '        word_games_hybrid_setup(model, cparams.n_threads);\n'
            '        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);')
    replace(SOURCE/'src/llama-context.cpp',
            '        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);',
            '        if (word_games_hybrid_enabled()) {\n'
            '            word_games_hybrid_set_user(cparams.cb_eval, cparams.cb_eval_user_data);\n'
            '            ggml_backend_sched_set_eval_callback(sched.get(), word_games_hybrid_callback, nullptr);\n'
            '        } else {\n'
            '            ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);\n'
            '        }')
    replace(SOURCE/'src/llama-graph.cpp',
            '    //call early so that topk-moe can be used\n',
            '    if (word_games_hybrid_active(il, n_tokens)) {\n'
            '        GGML_ASSERT(arch == LLM_ARCH_QWEN35MOE && type_op == LLM_FFN_SILU);\n'
            '        GGML_ASSERT(!gate_up_exps && !up_exps_b && !gate_exps_b && !down_exps_b);\n'
            '        GGML_ASSERT(!up_exps_s && !gate_exps_s && !down_exps_s && !post_exp_norm);\n'
            '        GGML_ASSERT(!moe_cache && !weight_before_ffn);\n'
            '        return word_games_hybrid_build(ctx0, gf, sched, cur, selected_experts, weights, il);\n'
            '    }\n\n    //call early so that topk-moe can be used\n')


def build():
    mount = str(ROOT.resolve())+':/work'
    docker = ['docker','run','--rm','--user',f'{__import__("os").getuid()}:{__import__("os").getgid()}',
              '-v',mount,'word-games-hybrid-build:b11960']
    subprocess.run(docker+['cmake','-S','/work/source','-B','/work/build','-G','Ninja',
        '-DCMAKE_BUILD_TYPE=Release','-DGGML_BACKEND_DL=ON','-DGGML_NATIVE=OFF',
        '-DGGML_CUDA=OFF','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF',
        '-DLLAMA_BUILD_TOOLS=OFF','-DLLAMA_BUILD_SERVER=OFF','-DLLAMA_CURL=OFF'],check=True)
    subprocess.run(docker+['cmake','--build','/work/build','--target','llama','-j','8'],check=True)
    # Reuse the exact original CPU/CUDA backends and server; only libllama changes.
    binary = Path('.pixi/word-games-native-b11960/llama-b11960')
    runtime = ROOT/'runtime'
    shutil.copytree(binary,runtime,symlinks=True,dirs_exist_ok=True)
    for p in (ROOT/'build/bin').glob('libllama.so*'):
        dst = runtime/p.name
        if dst.exists() or dst.is_symlink(): dst.unlink()
        if p.is_symlink(): dst.symlink_to(p.readlink())
        else: shutil.copy2(p,dst)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--patch-only',action='store_true')
    args=parser.parse_args()
    patch()
    if not args.patch_only: build()


if __name__=='__main__': main()
