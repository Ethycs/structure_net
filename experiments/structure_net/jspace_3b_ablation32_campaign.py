"""Authenticated pilot, fresh joint search, and upfront paired compilation.

The outer controller owns reservation acquisition/release.  This module never
starts a lease or touches the progressive campaign's RTX3060.  CPU preparation
may precede native-anchor availability; scientific search cannot.  Later
distillation/evaluation drivers consume completed receipts without changing
this producer or promoting a systems pilot to a scientific result.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import time

import torch

from . import jspace_3b_ablation32_data as data
from . import jspace_3b_ablation32_search as search
from . import jspace_3b_ablation32_compile as compiler
from . import jspace_qwen3_train as storage
from .jspace_full_lease import wait_ready

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'jspace.ablation32-campaign.v1'
TEACHER_PREFIX, QR_PREFIX = 'GPU-7b1db3cb-', 'GPU-8cf440df-'
ALLOCATION_MIB = dict(teacher=7680, qr=2048)
ALLOCATOR_MIB = dict(teacher=7424, qr=1792)
read, write, sha, identity = data.read, data.write, data.sha, data.identity
digest = data.cache.data.digest


def pilot_indices(records, training_indices):
    """Longest admitted complete TRAIN sequence in each of eight families."""
    groups = {}
    for index in training_indices:
        record = records[index]
        if record.split != 'train' or record.finish_reason != 'eos':
            raise ValueError('Pilot inventory must contain complete observed-EOS TRAIN examples')
        family = record.provenance['source_family']
        groups.setdefault(family, []).append(index)
    if len(groups) != 8:
        raise ValueError('The production systems pilot must cover all eight admitted task families')
    result = [min(indices, key=lambda i: (
        -(len(records[i].prompt_ids) + len(records[i].response_ids)), records[i].example_id))
        for _, indices in sorted(groups.items())]
    if len({records[i].group_id for i in result}) != len(result):
        raise ValueError('Systems pilot groups must be distinct')
    return result


def device_layout(lease, visible):
    """Authenticate physical cards and derive logical ordinals from visibility."""
    cards = visible.split(',') if isinstance(visible, str) else list(visible)
    allocation = lease.get('allocation', {})
    if len(cards) != 2 or len(set(cards)) != 2 or set(cards) != set(allocation):
        raise ValueError('One paired SUPER/native plus plain2060/QR lease is required')
    result = {}
    for role, prefix in (('teacher', TEACHER_PREFIX), ('qr', QR_PREFIX)):
        matches = [i for i, card in enumerate(cards) if card.startswith(prefix)]
        if len(matches) != 1 or allocation[cards[matches[0]]] != ALLOCATION_MIB[role]:
            raise ValueError('Paired allocation differs from the fixed teacher/QR cards or budgets')
        result[role] = matches[0]
    return result


def reservation():
    lease = wait_ready([0, 1])
    layout = device_layout(lease, os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    if torch.cuda.device_count() != 2:
        raise ValueError('Only the two reserved study GPUs may be visible')
    for role, ordinal in layout.items():
        total = torch.cuda.get_device_properties(ordinal).total_memory
        torch.cuda.set_per_process_memory_fraction(ALLOCATOR_MIB[role] * 1024**2 / total, ordinal)
        torch.cuda.reset_peak_memory_stats(ordinal)
    torch.cuda.set_device(layout['teacher'])
    def guard():
        current = wait_ready([0, 1])
        if current['id'] != lease['id'] or current['allocation'] != lease['allocation']:
            raise RuntimeError('The paired experiment reservation changed')
        if device_layout(current, os.environ.get('CUDA_VISIBLE_DEVICES', '')) != layout:
            raise RuntimeError('Logical device mapping changed')
        for role, ordinal in layout.items():
            if torch.cuda.max_memory_reserved(ordinal) > ALLOCATOR_MIB[role] * 1024**2:
                raise MemoryError('Declared ' + role + ' allocator cap exceeded')
    return lease, layout, guard


def source_files():
    return data.cache.prior.source.inherited.local_import_closure([
        Path(__file__), data.PROTOCOL,
        ROOT/'pyproject.toml', ROOT/'pixi.lock',
        ROOT/'src/structure_net_resources/client.py',
        ROOT/'experiments/neural_architecture_lab/distillation_recovery.py',
        *[ROOT/'src/neural_architecture_lab'/f'{name}.py' for name in
          ('distillation_data', 'distillation_training', 'distillation_recovery', 'core', 'local_scheduler')],
        ROOT/'tests/structure_net/test_jspace_3b_ablation32_campaign.py',
        ROOT/'tests/structure_net/test_jspace_3b_ablation32_search.py',
        ROOT/'tests/structure_net/test_jspace_3b_ablation32_compile.py',
        ROOT/'tests/structure_net/test_jspace_3b_ablation32_geometry.py'])


def prepare(root, data_root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Campaign preparation requires CUDA hidden')
    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    if root.exists():
        raise FileExistsError(root)
    dp, records = data.verify(data_root)
    indices = pilot_indices(records, dp['training_indices'])
    sources = {str(path): sha(path) for path in sorted(source_files())}
    inputs = {str(data_root/'plan.json'): sha(data_root/'plan.json')}
    plan = dict(schema=SCHEMA, data_root=str(data_root), data_fingerprint=dp['fingerprint'],
        model=dp['model'], response_cache_root=dp['response_cache_root'], full_data_root=dp['full_data_root'],
        records_sha256=dp['records_sha256'], search_configuration=asdict(search.SearchConfig()),
        pilot_indices=indices, pilot_example_ids=[records[i].example_id for i in indices],
        pilot_sequence_lengths=[len(records[i].prompt_ids)+len(records[i].response_ids) for i in indices],
        pilot_zero_anchors_allowed=True, scientific_anchors_required=True,
        allocation_mib=ALLOCATION_MIB, allocator_mib=ALLOCATOR_MIB,
        teacher_gpu_prefix=TEACHER_PREFIX, qr_gpu_prefix=QR_PREFIX, qr_backend='auxiliary_gpu_fp64_cpu_saved_graph',
        input_sha256=inputs, input_identity={path: identity(path) for path in inputs},
        dependency_identity=dp['input_identity'], source_sha256=sources,
        progressive_3060_untouched=True, pilot_updates_enter_scientific_search=False,
        selection_scope='500 held-out TRAIN groups; no task accuracy claim', confirmed=False)
    plan['fingerprint'] = digest(plan)
    root.mkdir(parents=True)
    write(root/'plan.json', plan)
    for path, wanted in sources.items():
        target = root/'producer'/Path(path).relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha(target) != wanted:
            raise ValueError('Campaign producer archive differs')
    return dict(status='prepared', fingerprint=plan['fingerprint'], pilot_examples=len(indices),
                scientific_anchors_ready=(data_root/'anchors/result.json').is_file())


def verify(root, *, live=True):
    root = Path(root).resolve()
    p = read(root/'plan.json')
    body = dict(p); fingerprint = body.pop('fingerprint')
    if p.get('schema') != SCHEMA or digest(body) != fingerprint:
        raise ValueError('Campaign plan changed')
    for path, old in {**p['dependency_identity'], **p['input_identity']}.items():
        if identity(path) != old:
            raise ValueError('Campaign dependency changed: ' + path)
    for path, wanted in p['source_sha256'].items():
        if sha(root/'producer'/Path(path).relative_to(ROOT)) != wanted or live and sha(path) != wanted:
            raise ValueError('Campaign producer changed: ' + path)
    dp, records = data.verify(p['data_root'], live=False)
    if (dp['fingerprint'] != p['data_fingerprint'] or dp['model'] != p['model']
            or dp['records_sha256'] != p['records_sha256']
            or asdict(search.SearchConfig()) != p['search_configuration']
            or pilot_indices(records, dp['training_indices']) != p['pilot_indices']):
        raise ValueError('Campaign data, teacher, pilot or search recipe differs')
    return p, dp, records


def _identities(p):
    return dict(campaign=p['fingerprint'], data=p['data_fingerprint'],
                teacher=p['model']['revision'], records=p['records_sha256'])


def _progress(directory, phase, **values):
    row = dict(utc=datetime.now(timezone.utc).isoformat(), phase=phase, **values)
    directory = Path(directory)
    with (directory/'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(row, allow_nan=False), flush=True)


def _memory(layout):
    return dict(host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        host_available_bytes=storage.available_host_bytes(), gpu={role:dict(
            logical_device=ordinal, peak_allocated_bytes=torch.cuda.max_memory_allocated(ordinal),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(ordinal)) for role, ordinal in layout.items()})


def _manifest(directory, p, files):
    directory = Path(directory).resolve()
    artifacts = {}
    for path in files:
        path = Path(path).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('Phase artifact escaped its owned directory')
        artifacts[str(path.relative_to(directory))] = dict(sha256=sha(path), identity=identity(path))
    write(directory/'manifest.json', dict(status='completed', fingerprint=p['fingerprint'], artifacts=artifacts))


def _completed(directory, p, *, hash_payloads=False):
    directory = Path(directory).resolve()
    m = read(directory/'manifest.json')
    if m.get('status') != 'completed' or m.get('fingerprint') != p['fingerprint']:
        raise ValueError('Matching completed phase receipt required')
    for name, item in m['artifacts'].items():
        path = (directory/name).resolve()
        if (not path.is_relative_to(directory) or identity(path) != item['identity']
                or (hash_payloads or path.suffix in ('.json', '.jsonl')) and sha(path) != item['sha256']):
            raise ValueError('Completed phase evidence changed')
    result = read(directory/'result.json')
    if result.get('fingerprint') != p['fingerprint'] or result.get('status') != 'completed':
        raise ValueError('Completed phase result does not match campaign')
    return result


def pilot_receipt(root, p):
    result = _completed(Path(root)/'pilot', p)
    checks = ('passed', 'discarded', 'fresh_initializer_proved', 'teacher_unchanged',
              'all_interfaces_updated', 'fresh_raw_reload_exact', 'fresh_projector_replay_exact')
    if any(result.get(key) is not True for key in checks) or result.get('interfaces') != 36:
        raise ValueError('All36 real disposable systems-update/reload pilot must pass before search')
    return result


def anchor_receipt(data_root, dp):
    root = Path(data_root).resolve(); receipt = read(root/'anchors/result.json')
    native = Path(dp['full_data_root'])/'native'
    if (receipt.get('status') != 'completed' or receipt.get('fingerprint') != dp['fingerprint']
            or receipt.get('fitting_indices') != dp['training_indices']
            or receipt.get('heldout_groups_excluded') is not True
            or receipt.get('source_native_manifest_sha256') != sha(native/'manifest.json')
            or identity(root/'anchors/anchors.pt') != receipt['anchor_identity']
            or sha(root/'anchors/anchors.pt') != receipt['anchors_sha256']):
        raise ValueError('Authenticated fitting-only native position anchors required')
    if any(identity(path) != old for path, old in receipt['source_input_identity'].items()):
        raise ValueError('Native anchor source inventory changed')
    return dict(receipt_sha256=sha(root/'anchors/result.json'), payload_sha256=receipt['anchors_sha256'],
                source_native_manifest_sha256=receipt['source_native_manifest_sha256'])


def _load_native(p, layout, guard):
    # Construction outside inference_mode preserves genuine parameter version counters.
    with torch.inference_mode(False), torch.no_grad():
        model = data.cache.execution.load_native_model(p['model']['path'],
            device=f"cuda:{layout['teacher']}", attention='eager', guard=guard)
    if len(model.model.layers) != 36 or model.config.hidden_size != 2048:
        raise ValueError('The complete pinned36-block native Qwen model is required')
    return model


def _trainer(native, dataset, p, layout, guard, *, systems_only):
    return search.SearchTrainer(native, dataset, teacher_id=p['model']['revision'], identities=_identities(p),
        guard=guard, config=search.SearchConfig(), systems_only=systems_only, qr_device=f"cuda:{layout['qr']}")


def _fresh_fingerprint(trainer):
    if trainer.updates or trainer.optimizer.state:
        raise ValueError('Fresh empty-optimizer search state is required')
    return storage.fingerprint(list(trainer.coordinates.named_parameters()))


def pilot(root):
    root = Path(root).resolve(); p, dp, _ = verify(root)
    out = root/'pilot'
    if out.exists():
        raise FileExistsError('Existing pilot artifacts are preserved; use a new diagnosed attempt')
    out.mkdir(); started = time.monotonic(); trainer = dataset = native = None; lease = None
    try:
        lease, layout, guard = reservation()
        _progress(out, 'native_load_start', lease_id=lease['id'], layout=layout)
        dataset = data.SearchDataset(p['data_root'], require_anchors=False)
        # A pilot cannot accidentally consume partially written scientific anchors.
        if dataset.anchors is not None:
            raise ValueError('The pre-anchor systems pilot must explicitly use zero anchors')
        with data.cache.execution.arithmetic_policy():
            native = _load_native(p, layout, guard)
            trainer = _trainer(native, dataset, p, layout, guard, systems_only=True)
            initializer = _fresh_fingerprint(trainer)
            _progress(out, 'real_all36_update_start', indices=p['pilot_indices'],
                      sequence_lengths=p['pilot_sequence_lengths'], initializer=initializer)
            actual = trainer.pilot(p['pilot_indices'], directory=out/'systems')
            teacher_unchanged = actual.get('teacher_unchanged') is True
            trainer.close(); del trainer; trainer = None; gc.collect()
            fresh = _trainer(native, dataset, p, layout, guard, systems_only=True)
            trainer = fresh
            restarted = _fresh_fingerprint(fresh)
            if restarted != initializer:
                raise RuntimeError('Fresh initialization differs after the disposable systems update')
            guard(); memory = _memory(layout)
        checks = ('all_interfaces_updated', 'fresh_raw_reload_exact', 'fresh_projector_replay_exact')
        passed = (actual.get('status') == 'systems_passed' and teacher_unchanged
                  and actual.get('discarded') is True and all(actual.get(key) is True for key in checks))
        if not passed:
            raise RuntimeError('Actual discarded all36 systems pilot did not pass')
        result = dict(schema=SCHEMA, status='completed', fingerprint=p['fingerprint'], passed=True,
            discarded=True, interfaces=actual['interfaces'], **{key:actual[key] for key in checks},
            fresh_initializer_proved=True, initializer=initializer, teacher_unchanged=teacher_unchanged,
            zero_anchor_systems_only=True, scientific_weights_saved=False,
            lease_id=lease['id'], allocation=lease['allocation'], layout=layout, memory=memory,
            actual_systems_result_sha256=sha(out/'systems/result.json'),
            elapsed_seconds=time.monotonic()-started, confirmed=False)
        write(out/'result.json', result)
        _progress(out, 'pilot_completed', passed=True, elapsed_seconds=result['elapsed_seconds'], memory=memory)
        _manifest(out, p, [out/'result.json', out/'progress.jsonl',
            *sorted(path for path in (out/'systems').rglob('*') if path.is_file())])
        return result
    except BaseException as error:
        write(out/'failure.json', dict(status='failed', fingerprint=p['fingerprint'], lease=lease,
            error_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic()-started,
            partial_artifacts_preserved=True, requires_diagnosis=True))
        raise
    finally:
        if trainer is not None: trainer.close()
        if dataset is not None: dataset.close()
        del trainer, dataset, native
        gc.collect()


def search_run(root):
    root = Path(root).resolve(); p, dp, _ = verify(root)
    pilot_result = pilot_receipt(root, p)
    anchors = anchor_receipt(p['data_root'], dp)
    out = root/'search'
    if out.exists() or (root/'search_failure.json').exists() or (root/'search_start.json').exists():
        raise FileExistsError('Existing scientific search artifacts are preserved; no automatic rerun')
    started = time.monotonic(); trainer = dataset = native = None; lease = None
    try:
        lease, layout, guard = reservation()
        write(root/'search_start.json', dict(fingerprint=p['fingerprint'], anchors=anchors,
            pilot_manifest_sha256=sha(root/'pilot/manifest.json'), lease_id=lease['id'], layout=layout))
        print(json.dumps(dict(phase='scientific_native_load_start', lease_id=lease['id'], layout=layout)), flush=True)
        dataset = data.SearchDataset(p['data_root'], require_anchors=True)
        with data.cache.execution.arithmetic_policy():
            native = _load_native(p, layout, guard)
            trainer = _trainer(native, dataset, p, layout, guard, systems_only=False)
            initializer = _fresh_fingerprint(trainer)
            if initializer != pilot_result['initializer']:
                raise ValueError('Scientific search did not start from the declared fresh geometry')
            actual = trainer.fit(out/'fit', progress=lambda row:_progress(out, **row))
            guard(); memory = _memory(layout)
        result = dict(schema=SCHEMA, status='completed', fingerprint=p['fingerprint'],
            data_fingerprint=p['data_fingerprint'], model=p['model'], anchors=anchors,
            initializer=initializer, fresh_after_pilot=True, pilot_update_reused=False,
            selection_epoch=actual['selected_epoch'], search_result_sha256=sha(out/'fit/result.json'),
            learned_export=actual['learned_export'], initial_random_export=actual['initial_random_export'],
            total_updates=actual['total_updates'], total_response_tokens=actual['total_response_tokens'],
            lease_id=lease['id'], allocation=lease['allocation'], layout=layout, memory=memory,
            elapsed_seconds=time.monotonic()-started, task_accuracy_measured=False, confirmed=False)
        write(out/'result.json', result)
        _progress(out, 'search_completed', selected_epoch=result['selection_epoch'],
                  total_updates=result['total_updates'], elapsed_seconds=result['elapsed_seconds'])
        _manifest(out, p, [out/'result.json', out/'progress.jsonl', out/'fit/result.json',
            Path(actual['learned_export']['directory'])/'manifest.json',
            Path(actual['initial_random_export']['directory'])/'manifest.json',
            Path(actual['selected_checkpoint']['path'])])
        return result
    except BaseException as error:
        write(root/'search_failure.json', dict(status='failed', fingerprint=p['fingerprint'], lease=lease,
            error_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic()-started,
            partial_artifacts_preserved=True, requires_diagnosis=True))
        raise
    finally:
        if trainer is not None: trainer.close()
        if dataset is not None: dataset.close()
        del trainer, dataset, native
        gc.collect()


def compile_run(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Upfront paired compilation is CPU-only')
    root = Path(root).resolve(); p, _, _ = verify(root)
    result = _completed(root/'search', p)
    for key in ('learned_export', 'initial_random_export'):
        item = result[key]
        if sha(Path(item['directory'])/'manifest.json') != item['manifest_sha256']:
            raise ValueError('Selected geometry export changed')
    out = root/'compiled'
    if out.exists(): raise FileExistsError(out)
    compiled = compiler.compile_pair(out, p['model']['path'], result['learned_export']['directory'],
                                     result['initial_random_export']['directory'])
    receipt = dict(schema=SCHEMA, status='completed', fingerprint=p['fingerprint'],
        both_arms_all_layers_compiled=compiled['both_arms_all_layers_compiled'],
        search_manifest_sha256=sha(root/'search/manifest.json'), arms=compiled['arms'],
        distillation_started=False, compressed_quality_measured=False, confirmed=False)
    write(out/'result.json', receipt)
    _manifest(out, p, [out/'result.json', out/'pair.json',
        *[out/arm/'manifest.json' for arm in ('learned', 'random')]])
    return receipt


def check(root):
    root = Path(root).resolve(); p, _, _ = verify(root, live=False)
    result = dict(status='prepared', fingerprint=p['fingerprint'], confirmed=False)
    for phase in ('pilot', 'search', 'compiled'):
        if (root/phase/'manifest.json').is_file():
            result[phase] = _completed(root/phase, p)
        elif (root/phase/'failure.json').is_file():
            result[phase] = read(root/phase/'failure.json')
        elif (root/phase).exists():
            result[phase] = dict(status='partial_or_active', automatic_retry_allowed=False)
    if (root/'search_failure.json').exists(): result['search_failure'] = read(root/'search_failure.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'pilot', 'search', 'compile', 'check'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--data-root', type=Path)
    args = parser.parse_args()
    if args.phase == 'prepare':
        if args.data_root is None: parser.error('--data-root is required for preparation')
        result = prepare(args.root, args.data_root)
    else:
        result = {'pilot':pilot, 'search':search_run, 'compile':compile_run, 'check':check}[args.phase](args.root)
    print(json.dumps({key:result[key] for key in ('status', 'fingerprint', 'passed', 'total_updates',
        'selection_epoch', 'both_arms_all_layers_compiled') if key in result}), flush=True)


if __name__ == '__main__':
    main()
