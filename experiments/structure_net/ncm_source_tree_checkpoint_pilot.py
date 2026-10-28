"""Real-checkpoint parity and gradient check for the shared source hierarchy."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_source_coverage_pilot as common
from experiments.structure_net import ncm_source_tree as tree

reference, task, data = common.reference, common.task, common.data
ROOT = data.ROOT/'20261027_source_tree_checkpoint_pilot_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-source-tree-checkpoint-pilot.md')
QUALIFICATION = data.ROOT/'20261027_source_tree_CPU_qualification_v2/result.json'
HYPOTHESIS = 'ncm-source-tree-checkpoint-pilot-v1'
LIMIT = 3_500_000_000
UNITS = {
    'structure-net-ncm-public-epoch4-attempt1.service': '4a547d3283d743998a3e2843b17a4992',
    'structure-net-ncm-task-supervision-expanded-train-v3.service': '7bbfbca757aa473f9ef4ff2b48520779',
}


def services():
    states = {}
    for unit, identity in UNITS.items():
        raw = subprocess.check_output(['systemctl', '--user', 'show', unit,
            '--property=MainPID,ExecMainPID,InvocationID,NRestarts,SubState,ExecMainCode,ExecMainStatus'], text=True)
        state = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
        successful = state['SubState'] == 'exited' and (state['ExecMainCode'], state['ExecMainStatus']) == ('1', '0')
        if (state['InvocationID'] != identity or state['NRestarts'] != '0'
                or not (state['SubState'] == 'running' or successful)):
            raise ValueError('existing worker identity or healthy lifecycle changed')
        states[unit] = state
    return states


def wrapped_name(name):
    marker = 'concepts.adapters.'
    if name.startswith(marker):
        layer, suffix = name[len(marker):].split('.', 1)
        return f'{marker}{layer}.leaf.{suffix}'
    return name


def freeze(training, environment, cases):
    files = {Path(__file__).resolve(), PLAN.resolve(), Path(tree.__file__).resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.is_file() and path.suffix == '.py' and (
                path.is_relative_to(Path('experiments').resolve()) or path.is_relative_to(training.native.vendor.VENDOR.resolve())):
                files.add(path)
    paths = [QUALIFICATION, Path('pixi.lock'), Path('pyproject.toml')]
    paths += [task.PARENT/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json', 'model.pt')]
    paths += [reference.ROOTS['run']/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json', 'outputs.json')]
    paths += [reference.broad.RUNS['prepare']/n for n in ('cases.json', 'selection.json')]
    paths += [reference.broad.catalog.DEST/n for n in ('source_index.json', 'feature_manifest.json')]
    ROOT.mkdir(); (ROOT/'producer_source').mkdir(); data.write(ROOT/'cases.json', cases)
    for i, path in enumerate(sorted(files)):
        (ROOT/'producer_source'/f'{i}_{path.name}').write_bytes(path.read_bytes())
    proof = {'schema': 'nal.ncm-source-tree-checkpoint-pilot.v1', 'hypothesis_id': HYPOTHESIS,
        'environment': environment, 'config': {'physical_GPU': 2, 'precision': 'FP32', 'TF32': False,
            'branching': 8, 'maximum_levels': 5, 'peak_limit_bytes': LIMIT, 'cases': 2,
            'model_updates': 0, 'new_judge_requests': 0, 'parent_model_sha256': task.PARENT_MODEL_SHA},
        'source_hashes': {str(p): data.sha(p) for p in sorted(files)},
        'input_hashes': {str(p): data.sha(p) for p in [*paths, ROOT/'cases.json']}}
    proof['fingerprint'] = training.q.digest(proof)
    data.write(ROOT/'producer.json', proof)
    return proof


def run():
    if ROOT.exists():
        raise ValueError('fresh append-only checkpoint pilot required')
    qualified = data.read(QUALIFICATION)
    if qualified['actual_tool_exit_code'] != 0 or qualified['tests_passed'] != 4:
        raise ValueError('actual successful CPU contracts required')
    data.verify(qualified['artifacts'])
    training, environment = reference.broad.runtime.configure()
    reference.study.pipeline.determinism.configure_runtime()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '2' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 2060 SUPER' or torch.cuda.mem_get_info()[0] < LIMIT):
        raise ValueError('physicalGPU2 with sufficient memory required')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    before_services = services()
    reference.receipt(task.PARENT); data.completed(task.PARENT)
    if data.sha(task.PARENT/'model.pt') != task.schedule.PARENT_CHECKPOINT_SHA:
        raise ValueError('fixed epoch2 checkpoint required')
    reference.receipt(reference.ROOTS['run']); data.completed(reference.ROOTS['run'])
    controls, selection = reference.prepared_controls()
    ids = list(dict.fromkeys([controls[0]['id'], selection['longest_source_id'], *[r['id'] for r in controls]]))[:2]
    cases = [next(r for r in controls if r['id'] == i) for i in ids]
    proof = freeze(training, environment, cases)
    index, manifest = [data.read(reference.broad.catalog.DEST/n) for n in ('source_index.json', 'feature_manifest.json')]
    loader = training.preparation.streaming.FeatureCache(index, manifest, max_bytes=256*1024**2)
    torch.cuda.reset_peak_memory_stats(); model = task.load_initial(training)
    old = common.bindings(model); old_parameters = dict(model.named_parameters())
    protected = training.prior.protected(model)
    before, before_logits = common.outputs(model, cases, loader, training)
    archive = {r['id']: r for r in data.read(reference.ROOTS['run']/'outputs.json')['epoch2']['matched']}
    if any(any(r[k] != archive[r['id']][k] for k in ('codes', 'raw_word_ids')) for r in before):
        raise ValueError('original saved epoch2 codes or words differ')
    cpu_rng = torch.get_rng_state().clone(); cuda_rng = torch.cuda.get_rng_state().clone()
    metadata = tree.install(model)
    if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
        raise ValueError('installation consumed RNG')
    reference.broad.mechanism.configure(model, False)
    expanded = common.bindings(model); expanded_parameters = dict(model.named_parameters())
    if any(expanded.get(wrapped_name(k)) != v for k, v in old.items()):
        raise ValueError('original tensor changed during installation')
    if any(expanded_parameters[wrapped_name(k)] is not p for k, p in old_parameters.items()):
        raise ValueError('original parameter object changed during installation')
    after, after_logits = common.outputs(model, cases, loader, training)
    if before != after or any(not torch.equal(a, b) for a, b in zip(before_logits, after_logits)):
        raise ValueError('initial logits, codes and words must match exactly')
    torch.save({'before': before_logits, 'after': after_logits}, ROOT/'teacher_logits.pt')
    data.write(ROOT/'outputs.json', {'before': before, 'after': after})
    gradients = []
    for row in cases:
        model.zero_grad(set_to_none=True)
        features, valid, gold = training.prior.tensors(row, loader)
        logits = model.concepts.teacher_logits(features, valid, gold)
        loss = training.objective.losses(logits, gold, model.concepts.generator.levels).mean()
        if not torch.isfinite(loss):
            raise ValueError('finite hierarchy loss required')
        loss.backward(); added = {}
        for layer in range(12):
            adapter = model.concepts.adapters[str(layer)]
            output, router = adapter.tree_out.grad, adapter.level_weights.grad
            if output is None or not torch.isfinite(output).all() or not output.abs().sum():
                raise ValueError('all tree output gradients must be finite and nonzero')
            if router is None or router.count_nonzero():
                raise ValueError('initial level-router gradients must be zero')
            record = {'tree_output_L2': float(output.double().norm()), 'router_nonzero': int(router.count_nonzero())}
            if layer not in (8, 10, 11):
                leaf, query = adapter.leaf.out.weight.grad, adapter.leaf.q.weight.grad
                if leaf is None or not torch.isfinite(leaf).all() or not leaf.abs().sum() or query is None or query.count_nonzero():
                    raise ValueError('new leaf-output/query gradient contract failed')
                record.update(leaf_output_L2=float(leaf.double().norm()), query_nonzero=int(query.count_nonzero()))
            added[str(layer)] = record
        if any(p.grad is not None and (not p.requires_grad or not torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError('protected or nonfinite gradient')
        gradients.append({'id': row['id'], 'loss': float(loss.detach()), 'layers': added})
        model.zero_grad(set_to_none=True); del features, valid, gold, logits, loss
        if torch.cuda.max_memory_allocated() > LIMIT:
            raise ValueError('pilot own allocation exceeds fixed limit')
    if common.bindings(model) != expanded or training.prior.protected(model) != protected or any(p.grad is not None for p in model.parameters()):
        raise ValueError('gradient-only pilot modified weights or left gradients')
    after_services = services()
    if any(after_services[u]['ExecMainPID'] != before_services[u]['ExecMainPID'] for u in UNITS):
        raise ValueError('existing worker process changed')
    data.write(ROOT/'tree.json', metadata); data.write(ROOT/'gradients.json', gradients)
    data.write(ROOT/'tensor_bindings.json', {'original': old, 'tree': expanded, 'after_backward': common.bindings(model)})
    data.write(ROOT/'services.json', {'before': before_services, 'after': after_services})
    data.verify(proof['source_hashes']); data.verify(proof['input_hashes'])
    value = {'schema': proof['schema'], 'hypothesis_id': HYPOTHESIS, 'status': 'source_tree_checkpoint_parity_and_gradients_passed',
        'cases': 2, 'source_blocks': 12, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'all_initial_teacher_logits_exact': True, 'all_initial_codes_and_words_exact': True,
        'original_parameters_preserved': True, 'installation_RNG_unchanged': True,
        'all_tree_output_gradients_nonzero': True, 'all_new_leaf_output_gradients_nonzero': True,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'model_updates': 0, 'new_judge_requests': 0,
        'task_accuracy': None, 'final_test_opened': False, 'speed_tested': False,
        'full_training_lifecycle_qualified': False, 'artifacts': {str(p): data.sha(p) for p in ROOT.iterdir() if p.is_file()}}
    data.write(ROOT/'result.json', value); model.close()
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(__doc__).parse_args()
    existed = ROOT.exists()
    try:
        run()
    except Exception as error:
        if ROOT.exists() and not existed:
            data.write(ROOT/'failure.json', {'error_type': type(error).__name__, 'error': str(error)})
        raise
