"""Run the existing NCM experiment with locked Pixi packages and disk streaming."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import sysconfig


PROJECT = Path(__file__).resolve().parents[2]
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-pixi-migration.md')
PACKAGES = {
    'torch': 'torch', 'numpy': 'numpy', 'transformers': 'transformers',
    'tokenizers': 'tokenizers', 'huggingface-hub': 'huggingface_hub',
    'safetensors': 'safetensors', 'einops': 'einops', 'hydra-core': 'hydra',
    'omegaconf': 'omegaconf', 'tiktoken': 'tiktoken', 'pysbd': 'pysbd',
    'requests': 'requests', 'pytest': 'pytest', 'triton': 'triton',
}
ROOT = Path('data/experiments/ncm_paper_regime')
CHECK = ROOT / '20261025_ncm_pixi_runtime_v2'
RUNS = {s: ROOT / ('20261025_sequential_full732_pixi_' + s + '_v1')
        for s in ('pilot', 'train')}


def configure():
    """Reject the legacy interpreter and load all extras from the Pixi prefix."""
    prefix = PROJECT / '.pixi/envs/ncm'
    if (Path.cwd().resolve() != PROJECT or Path(sys.prefix).resolve() != prefix.resolve()
            or os.environ.get('PIXI_ENVIRONMENT_NAME') != 'ncm'):
        raise RuntimeError('Run from the repository root using pixi run --locked -e ncm')
    if os.environ.get('LD_LIBRARY_PATH'):
        raise RuntimeError('NCM uses the CUDA libraries supplied by its locked wheels')
    packages = {}
    for distribution, module_name in PACKAGES.items():
        module = importlib.import_module(module_name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(prefix.resolve()):
            raise RuntimeError('Package imported outside Pixi: ' + str(path))
        packages[distribution] = {'version': importlib.metadata.version(distribution),
                                  'path': str(path)}
    # The immutable historical helper prepends DEPENDENCIES to sys.path.
    # Redirect only this process to Pixi's site-packages, preserving its pinned
    # upstream model-source artifact and every historical source fingerprint.
    from experiments.structure_net import ncm_llama as legacy
    from experiments.structure_net import ncm_sequential_full_training_v2 as training
    legacy.DEPENDENCIES = Path(sysconfig.get_path('purelib'))
    training.native.vendor.corpus.quality.dependencies = lambda: None
    return training, {'manager': 'pixi', 'environment': 'ncm',
        'executable': sys.executable, 'prefix': sys.prefix,
        'python': sys.version, 'packages': packages,
        'cuda': training.torch.version.cuda,
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'legacy_dependency_overlays_used': False}


def check():
    training, runtime = configure()
    if CHECK.exists():
        raise ValueError('Fresh runtime qualification root required')
    if training.torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('The tokenizer and package qualification is CPU only')
    training.preparation.completed(training.base.DATA)
    rows = training.q.read(training.base.DATA / 'cases.json')
    tokenizer = training.native.tokenizer()
    for row in rows:
        training.preparation.frame(row, tokenizer)
    if len(rows) != 732:
        raise ValueError('All 732 unchanged TRAIN token frames must be checked')
    # The dev build of tokenizers is replaced with the released 0.21.4 wheel.
    # Check rendering against saved pre-training outputs as well as input IDs.
    prior_outputs = training.q.read(training.RUNS['pilot'] / 'initial_outputs.json')
    for output in prior_outputs:
        rendered = training.preparation.rendering.decode(output['raw_word_ids'], tokenizer)
        if any(output[k] != value for k, value in rendered.items()):
            raise ValueError('Existing raw word IDs render differently under Pixi')
    paths = [Path('pyproject.toml'), Path('pixi.lock'), Path(__file__), PLAN,
             training.base.DATA / 'cases.json', training.RUNS['pilot'] / 'initial_outputs.json']
    result = {'status': 'runtime_and_tokenizer_qualified', 'runtime': runtime,
        'TRAIN_frames_identical': len(rows), 'saved_renderings_identical': len(prior_outputs),
        'input_hashes': {str(p): training.file_hash(p) for p in paths},
        'systems_lifecycle_only_not_quality_evidence': True, 'task_accuracy': None}
    CHECK.mkdir()
    training.write_json(CHECK / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_hashes', 'runtime')}), flush=True)


def run(stage):
    training, runtime = configure()
    qualified = training.q.read(CHECK / 'result.json')
    if (qualified['status'] != 'runtime_and_tokenizer_qualified'
            or qualified['runtime'] != runtime
            or training.q.read(CHECK / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0):
        raise ValueError('Completed matching Pixi runtime qualification required')
    for path, expected in qualified['input_hashes'].items():
        if training.file_hash(Path(path)) != expected:
            raise ValueError('Qualified runtime input changed: ' + path)
    original_freeze = training.freeze

    def freeze(root, mode):
        proof = original_freeze(root, mode)
        proof['environment']['pixi'] = runtime
        for path in (Path('pyproject.toml'), Path('pixi.lock'), PLAN,
                     CHECK / 'result.json', CHECK / 'actual_tool_receipt.json'):
            proof['input_hashes'][str(path)] = training.file_hash(path)
        proof['fingerprint'] = training.q.digest({k: v for k, v in proof.items() if k != 'fingerprint'})
        training.write_json(root / 'producer.json', proof)
        return proof

    previous = training.CONFIG, training.RUNS, training.PLAN, training.freeze
    training.CONFIG = {**training.CONFIG, 'host_feature_cache_bytes': 256 * 1024**2,
        'feature_loading_variant': 'existing_lossless_disk_LRU_256MiB',
        'runtime_variant': 'locked_pixi_ncm_v1'}
    training.RUNS, training.PLAN, training.freeze = RUNS, PLAN, freeze
    existed = RUNS[stage].exists()
    try:
        training.run(stage)
    except Exception as error:
        if not existed and RUNS[stage].exists():
            training.write_json(RUNS[stage] / 'failure.json',
                {'exception': type(error).__name__, 'message': str(error), 'quality_result': None})
        raise
    finally:
        training.CONFIG, training.RUNS, training.PLAN, training.freeze = previous


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('check', *RUNS))
    stage = parser.parse_args().stage
    check() if stage == 'check' else run(stage)
