"""Matched, fresh one-block attention retry; no TEST access or automatic expansion.

Only exposed DEV is scored. A disposable real-GPU pilot for each arm precedes
both scientific starts. Every phase owns an append-only directory; failures are
retained, never resumed. The controller owns leases and the shared judge lock is
acquired here only by CPU judgment. All inference uses the mode-bound adapter.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import time

import torch

from . import jspace_3b_attention_retry_model as adapter
from . import jspace_3b_usable_recovery as parent
from . import jspace_3b_response_campaign as campaign

base, data, receipts = parent.base, parent.data, parent.receipts
read, write, sha, identity = parent.read, parent.write, parent.sha, parent.identity
ROOT = parent.ROOT
PREPARATION = Path(__file__).with_name('jspace_3b_attention_retry_plan.json')
SCHEMA = 'jspace.3b-attention-retry.v1'
ARMS = ('softmax', 'quadratic')
LABELS = ('update_0000', 'update_0071')
PARENT_LABELS = ('update_0000', 'update_0071', 'update_0134', 'update_0199', 'update_0278')
UPDATES, TOKENS = 71, 25245


def require_arm(arm):
    if arm not in ARMS:
        raise ValueError('Explicit softmax or quadratic arm required')
    return arm


def require_label(label):
    if label not in LABELS:
        raise ValueError('Only baseline update_0000 and final update_0071 are admitted')
    return label


def matched_training(dp, records):
    """The existing first71 complete windows, not a newly shuffled token cap."""
    windows = dp['training']['windows'][:UPDATES]
    order = [i for window in windows for i in window]
    if (len(windows) != UPDATES or order != dp['training']['order'][:len(order)]
            or len(set(order)) != len(order) or not order
            or any(not window or len(window) > 16 for window in windows)):
        raise ValueError('Existing first71 no-replacement windows required')
    for i in order:
        record = records[i]
        if (i not in dp['training_indices'] or record.split != 'train'
                or record.finish_reason != 'eos' or not record.response_ids
                or record.response_ids[-1] != record.eos_token_id):
            raise ValueError('Only admitted complete TRAIN responses with actual EOS are allowed')
    tokens = sum(len(records[i].response_ids) for i in order)
    if tokens != TOKENS:
        raise ValueError('Matched budget must be exactly25245 response/EOS tokens')
    return dict(windows=windows, order=order, actual_supervised_tokens=tokens,
        updates=UPDATES, seed=dp['training']['seed'], complete_response_eos=True,
        full_corpus_pass=False, automatic_expansion=False)


def select_parent(rows):
    if ([row.get('label') for row in rows] != list(PARENT_LABELS)
            or any(row.get('evidence_available') is not True for row in rows)):
        raise ValueError('Pending blocked: all five available parent judgments required')
    return parent.select_quality(rows)


def authenticate_parent(parent_root):
    """Require every phase, not the parent's provisional minimum-NLL candidate."""
    parent_root = Path(parent_root).resolve()
    pp, dp, records = parent.verify(parent_root, live=False)
    frozen = read(PREPARATION)
    if (str(parent_root) != frozen['parent_worker_root']
            or pp['fingerprint'] != frozen['parent_worker_fingerprint']):
        raise ValueError('Unexpected parent recovery study')
    trained = receipts._completed(parent_root/'run', pp)
    if [r['label'] for r in trained['candidates']] != list(PARENT_LABELS):
        raise ValueError('All five parent saved candidates required')
    paths = {parent_root/'plan.json', parent_root/'run/manifest.json', parent_root/'run/result.json'}
    judgments = []
    for label in PARENT_LABELS:
        phases = {}
        for name, directory in (('candidate', parent_root/'run/candidates'/label),
                                ('generation', parent_root/'generation'/label),
                                ('judgment', parent_root/'judgment'/label)):
            phases[name] = receipts._completed(directory, pp)
            paths.update((directory/'manifest.json', directory/'result.json'))
        candidate = phases['candidate']
        trained_candidate = next(r for r in trained['candidates'] if r['label'] == label)
        if any(candidate[k] != trained_candidate[k] for k in ('checkpoint', 'packed_export', 'nll', 'update')):
            raise ValueError('Parent candidate differs from training receipt')
        if any(row.get('label') != label or row.get('checkpoint') != candidate['checkpoint']
               for row in phases.values()):
            raise ValueError('Parent phase checkpoint binding differs')
        if (phases['generation']['canaries']['count'] != 6
                or len(phases['generation']['canaries']['records']) != 6 or len(phases['generation']['corpus']) != 16):
            raise ValueError('Complete parent six-canary and16-case generation required')
        judgments.append(phases['judgment'])
    selection = select_parent(judgments)
    controller = parent_root.parent/'controller_v1'
    cp = read(controller/'plan.json'); body = dict(cp); fp = body.pop('fingerprint')
    if data.digest(body) != fp or cp['worker_fingerprint'] != pp['fingerprint']:
        raise ValueError('Parent controller plan differs')
    completed = receipts._completed(controller, cp)
    if (completed.get('all_generation_phases_completed') is not True
            or completed.get('all_judgments_available') is not True
            or completed['worker_selection']['selection_final'] is not True
            or completed['worker_selection']['task_selection'] != selection):
        raise ValueError('Pending blocked: final parent controller generation-first selection required')
    paths.update((controller/'plan.json', controller/'manifest.json', controller/'result.json'))
    selected = selection['selected']; checkpoint = selected['checkpoint']
    if sha(checkpoint['path']) != checkpoint['sha256']:
        raise ValueError('Selected parent raw checkpoint changed')
    paths.add(Path(checkpoint['path']))
    raw = torch.load(checkpoint['path'], map_location='cpu', weights_only=True, mmap=True)
    if (raw.get('schema') != 'jspace.complete-response-checkpoint.v1'
            or raw.get('identities') != parent.identities(pp) or raw.get('updates') != selected['update']
            or 'attention_retry' in raw['stack']['_extra_state']):
        raise ValueError('Selected parent raw checkpoint provenance differs')
    return pp, dp, records, selection, paths


def prepare(root, parent_root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Preparation requires CUDA hidden')
    root = Path(root).resolve()
    if root.exists() and any(path.name != 'controller_v1' for path in root.iterdir()):
        raise FileExistsError('Fresh retry root (apart from controller_v1) required')
    pp, dp, records, selection, paths = authenticate_parent(parent_root)
    training = matched_training(dp, records)
    if len(pp['generation_indices']) != 16 or any(records[i].split != 'development' for i in pp['generation_indices']):
        raise ValueError('Exactly16 existing exposed DEV generation cases required')
    checkpoint = selection['selected']['checkpoint']
    sources = data.cache.prior.source.inherited.local_import_closure([
        Path(__file__), Path(adapter.__file__), ROOT/'tests/structure_net/test_jspace_3b_attention_retry.py',
        ROOT/'tests/structure_net/test_jspace_3b_attention_retry_model.py', ROOT/'pyproject.toml', ROOT/'pixi.lock'])
    # Preparation status may later change; freeze its content, not its live identity.
    p = {key: pp[key] for key in ('data_root', 'data_fingerprint', 'model', 'cut', 'structure_export',
        'corpus', 'corpus_fingerprint', 'response_cache_root', 'full_data_root', 'generation_indices',
        'generation_eos_token_ids', 'max_new_tokens', 'judge', 'judge_lock', 'seed')}
    inputs = dict(pp['input_sha256'])
    inputs.update({str(path): sha(path) for path in paths})
    p.update(schema=SCHEMA, preparation=read(PREPARATION), parent_root=str(Path(parent_root).resolve()),
        parent_fingerprint=pp['fingerprint'], parent_selection=selection,
        checkpoint=checkpoint['path'], checkpoint_sha256=checkpoint['sha256'],
        training=training, configuration=asdict(base.training.ResponseConfig(learning_rate=2e-5, max_supervised_tokens=TOKENS)),
        pilot_windows=campaign.pilot_windows(records, dp['training']['order']),
        arms=list(ARMS), labels=list(LABELS), changed_attention_layers=[35], absolute_layer_indices=[33, 34, 35],
        profile='plain2060', resource_profiles={'plain2060': parent.PROFILES['plain2060']},
        backend='packed_bf16_torch_attention_retry', input_sha256=inputs,
        input_identity={path: identity(path) for path in inputs}, dependency_identity=pp['dependency_identity'],
        source_sha256={str(path): sha(path) for path in sources},
        objective=dict(name='forward_kl', temperature=1., ce_weight=0.),
        selection=pp['selection'], exposed_dev=True, test_access=False, confirmed=False,
        automatic_extra_epochs=False, automatic_three_layer_expansion=False, resume_supported=False)
    p['fingerprint'] = data.digest(p)
    root.mkdir(parents=True, exist_ok=True); write(root/'plan.json', p)
    for path, wanted in p['source_sha256'].items():
        target = root/'producer'/Path(path).relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
        if sha(target) != wanted:
            raise ValueError('Retry source archive differs')
    return dict(status='prepared', fingerprint=p['fingerprint'], training_updates=UPDATES, response_tokens=TOKENS,
                parent_selection_status=selection['status'], confirmed=False)


def verify(root, live=True):
    root = Path(root).resolve(); p = read(root/'plan.json'); body = dict(p); fp = body.pop('fingerprint')
    if p.get('schema') != SCHEMA or data.digest(body) != fp:
        raise ValueError('Retry plan changed')
    for path, old in {**p['dependency_identity'], **p['input_identity']}.items():
        if identity(path) != old:
            raise ValueError('Retry immutable dependency changed: '+path)
    for path, wanted in p['input_sha256'].items():
        if sha(path) != wanted:
            raise ValueError('Retry input changed: '+path)
    for path, wanted in p['source_sha256'].items():
        if sha(root/'producer'/Path(path).relative_to(ROOT)) != wanted or live and sha(path) != wanted:
            raise ValueError('Retry producer changed: '+path)
    dp, records = data.verify(p['data_root'], live=False)
    if dp['fingerprint'] != p['data_fingerprint'] or matched_training(dp, records) != p['training']:
        raise ValueError('Retry corpus or matched budget changed')
    return p


def identities(p, arm):
    return dict(parent.identities(p), arm=require_arm(arm), attention_layers=[35])


def build(p, arm, *, device='cpu', guard=None, checkpoint=None):
    """Parent tensors are adopted once; retry checkpoints use strict adapter restore."""
    require_arm(arm)
    if checkpoint is None:
        stack, norm, E, rotary = parent.build(p, device=device, guard=guard)
        stack = adapter.AttentionRetryStack.from_stack(stack, mode=arm, layers=(35,), copy=False)
    else:
        # Readout/RoPE come from the same authenticated native parent, never refitted.
        stack, norm, E, rotary = parent.build(p, device='cpu')
        del stack
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        stack = adapter.restore_unpacked(saved['stack'], read(Path(p['structure_export'])/'structure.json'))
        norm.load_state_dict(saved['norm'], strict=True)
        if stack.attention_retry['mode'] != arm or base.offload.fixed_state(stack, E, rotary) != saved['fixed_state']:
            raise ValueError('Retry raw mode or fixed geometry differs')
        if guard is not None:
            guard()
        stack, norm, E, rotary = stack.to(device), norm.to(device), E.to(device), rotary.to(device)
    return stack, norm, E, rotary


def pack(stack, norm, E, rotary, p, guard=None):
    return adapter.PackedAttentionRetryModel(base.builder.packed_clone(stack),
        base.cpu_module_clone(norm).to(E.device), E.detach(), base.cpu_module_clone(rotary).to(E.device),
        config=base.builder.native._config(p['model']['path']), max_length=p['configuration']['max_length'], guard=guard)


def finish(out, p, result):
    write(out/'result.json', result)
    receipts._manifest(out, p, [f for f in out.rglob('*') if f.is_file()])
    return result


def failure(out, error):
    write(out/'failure.json', dict(status='failed', error_type=type(error).__name__, error=str(error),
                                  partial_artifacts_preserved=True, retry_supported=False))


def start(root, arm, phase, label=None):
    require_arm(arm)
    if label is not None:
        require_label(label)
    p = verify(root)
    out = Path(root).resolve()/'arms'/arm/phase
    if label is not None:
        out /= label
    out.mkdir(parents=True, exist_ok=False)
    return p, out


@torch.inference_mode()
def initial_control(p, values, batch, guard):
    """Real original-Torch versus wrapped softmax equality on the pilot device."""
    stack, norm, E, rotary = values
    original = base.training.ResponseModel(stack, norm, E, rotary, max_length=p['configuration']['max_length'], guard=guard)
    reference = original.readout(original.hidden_forward(batch)[:, -1]).cpu()
    control = adapter.AttentionRetryStack.from_stack(stack, mode='softmax', copy=False)
    wrapped = base.training.ResponseModel(control, norm, E, rotary, max_length=p['configuration']['max_length'], guard=guard)
    actual = wrapped.readout(wrapped.hidden_forward(batch)[:, -1]).cpu()
    if not torch.equal(reference, actual):
        raise ValueError('Initial softmax control differs from original native-score Torch execution')
    return True


@torch.inference_mode()
def score_candidate(trainer, dataset, mc_dataset, p, out, checkpoint, label, arm):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    model = pack(trainer.stack, trainer.norm, trainer.readout, trainer.rotary, p, trainer.guard)
    export = adapter.export_packed_snapshot(model, out/'export', identities=dict(identities(p, arm), checkpoint=checkpoint['sha256']))
    first = dataset.batch([dataset.development[0]])
    reference = model.readout(model.hidden_forward(first)[:, -1]).cpu()
    del model; gc.collect(); torch.cuda.empty_cache()
    model, reload = adapter.load_packed_snapshot(out/'export', device=trainer.device, guard=trainer.guard)
    actual = model.readout(model.hidden_forward(first)[:, -1]).cpu()
    if not torch.equal(reference, actual) or model.stack.attention_retry['mode'] != arm:
        raise ValueError('Candidate packed reload changed mode or vocabulary output')
    teacher = base.training.CachedHiddenTeacher(model.embedding, teacher_id=p['model']['revision'], guard=trainer.guard)
    by_id = {record.example_id: i for i, record in enumerate(dataset.records)}
    response = base.evaluation.response_losses(model, (dataset.records[i] for i in dataset.development),
        pad_token_id=dataset.pad_token_id, teacher_provider=teacher, token_chunk_size=8,
        prepare_batch=lambda b: dataset.batch([by_id[n] for n in b['example_ids']]))
    mc = base.packed_mc(model, mc_dataset)
    result = dict(status='completed', fingerprint=p['fingerprint'], arm=arm, label=label,
        checkpoint=checkpoint, update=trainer.updates, response_tokens=trainer.supervised_tokens,
        nll=response['nll'], teacher_kl=response['kl'], mc=mc['metrics'], mc_eligibility=mc['eligibility'],
        packed_export=export, packed_reload_exact=True, attention_retry=reload['attention_retry'],
        generation_status='pending', backend=p['backend'], qualified=None, confirmed=False)
    write(out/'development.json', response); write(out/'multiple_choice.json', mc)
    return finish(out, p, result)


def pilot(root, arm):
    p, out = start(root, arm, 'pilot'); dataset = mc_dataset = trainer = None
    try:
        lease, guard = parent.reservation(p['profile']); torch.set_num_threads(2)
        dataset = data.RecoveryDataset(p['data_root']); mc_dataset = data.MCDataset(p['data_root'])
        with base.execution.arithmetic_policy():
            values = parent.build(p, device='cuda:0', guard=guard)
            control_exact = initial_control(p, values, dataset.batch([p['pilot_windows'][0][0]]), guard)
            stack, norm, E, rotary = values
            stack = adapter.AttentionRetryStack.from_stack(stack, mode=arm, copy=False)
            del values
            initial = base.initializer_fingerprint(stack, norm, E, rotary)
            windows = p['pilot_windows']
            cfg = base.training.ResponseConfig(**{**p['configuration'], 'max_supervised_tokens':
                sum(len(dataset.records[i].response_ids) for w in windows for i in w)})
            trainer = base.training.ResponseTrainer(stack, norm, E, rotary, dataset,
                teacher_id=p['model']['revision'], config=cfg, guard=guard)
            before = trainer.parameter_fingerprints()
            updates = parent.pilot_updates(trainer, windows)
            report = campaign._pilot_report(trainer, trainer.change_report(before), updates['updates'])
            if not report['passed']:
                raise ValueError('All blocks and final norm must receive actual FP32 master updates')
            cp = trainer.save_checkpoint(out/'discarded.pt', identities=identities(p, arm))
            evaluation = score_candidate(trainer, dataset, mc_dataset, p, out/'evaluation', cp, 'discarded', arm)
            trainer.close(); trainer = None
            # Raw reload checks exact full-vocabulary output without sharing live parameters.
            batch = dataset.batch([windows[0][0]])
            live = base.training.ResponseModel(stack, norm, E, rotary, guard=guard)
            with torch.inference_mode():
                reference = live.readout(live.hidden_forward(batch)[:, -1]).cpu()
            del live, stack, norm, E, rotary; gc.collect(); torch.cuda.empty_cache()
            restored = build(p, arm, device='cuda:0', guard=guard, checkpoint=cp['path'])
            replay = base.training.ResponseModel(*restored, guard=guard)
            with torch.inference_mode():
                actual = replay.readout(replay.hidden_forward(batch)[:, -1]).cpu()
            if not torch.equal(reference, actual):
                raise ValueError('Fresh raw retry reload changed logits')
            del replay, restored; gc.collect(); torch.cuda.empty_cache()
            fresh = build(p, arm)
            if base.initializer_fingerprint(*fresh) != initial:
                raise ValueError('Scientific initializer inherited disposable pilot mutations')
            result = dict(status='completed', fingerprint=p['fingerprint'], arm=arm, passed=True,
                discarded=True, initializer=initial, fresh_initializer=True, fixed_geometry_preserved=True,
                softmax_control_initial_exact=control_exact, fresh_raw_reload_exact=True, packed_reload_exact=True,
                updates=updates, update_report=report, evaluation=evaluation, checkpoint=cp,
                lease_id=lease['id'], peak_reserved_bytes=torch.cuda.max_memory_reserved(), confirmed=False)
            return finish(out, p, result)
    except BaseException as error:
        failure(out, error); raise
    finally:
        for value in (trainer, dataset, mc_dataset):
            if value is not None:
                value.close()


def pilot_receipts(root, p):
    proofs = {}
    for arm in ARMS:
        proof = receipts._completed(Path(root)/'arms'/arm/'pilot', p)
        if proof.get('arm') != arm or any(proof.get(k) is not True for k in
                ('passed', 'discarded', 'fresh_initializer', 'fixed_geometry_preserved',
                 'softmax_control_initial_exact', 'fresh_raw_reload_exact', 'packed_reload_exact')):
            raise ValueError('Both independent disposable GPU pilots must pass')
        proofs[arm] = proof
    if proofs['softmax']['initializer'] != proofs['quadratic']['initializer']:
        raise ValueError('Paired pilots did not start from identical tensors and geometry')
    return proofs


def run(root, arm):
    require_arm(arm); p = verify(root); proofs = pilot_receipts(root, p)
    p, out = start(root, arm, 'run'); (out/'checkpoints').mkdir()
    dataset = mc_dataset = trainer = None; started = time.monotonic()
    try:
        lease, guard = parent.reservation(p['profile']); torch.set_num_threads(2)
        dataset = data.RecoveryDataset(p['data_root']); mc_dataset = data.MCDataset(p['data_root'])
        with base.execution.arithmetic_policy():
            values = build(p, arm, device='cuda:0', guard=guard)
            if base.initializer_fingerprint(*values) != proofs[arm]['initializer']:
                raise ValueError('Scientific arm is not a fresh matched initializer')
            trainer = base.training.ResponseTrainer(*values, dataset, teacher_id=p['model']['revision'],
                config=base.training.ResponseConfig(**p['configuration']), guard=guard)
            before = trainer.parameter_fingerprints(); seen = []; candidates = []; training_seconds = 0.
            def record():
                label = require_label(f'update_{trainer.updates:04d}')
                cp = trainer.save_checkpoint(out/'checkpoints'/(label+'.pt'), identities=identities(p, arm))
                candidates.append(score_candidate(trainer, dataset, mc_dataset, p, out/'candidates'/label, cp, label, arm))
            record()
            for window in p['training']['windows']:
                row = trainer.train_window(window); seen.extend(row['indices']); training_seconds += row['train_window_seconds']
                if seen != p['training']['order'][:len(seen)]:
                    raise ValueError('Observed training order differs')
                base._event(out, 'matched_attention_recovery', arm=arm, **row)
            if trainer.updates != UPDATES or trainer.supervised_tokens != TOKENS or seen != p['training']['order']:
                raise ValueError('Matched training budget or order differs')
            change = trainer.change_report(before)
            if not all(change['bf16_changed_by_block']):
                raise ValueError('Every compact block must actually update')
            record()
            coverage = data.coverage_report(dataset.plan, seen)
            coverage.update(planned_run_completed=True, full_parent_pass_completed=False,
                            matched_optimizer_windows=UPDATES, matched_response_tokens=TOKENS)
            return finish(out, p, dict(status='completed', fingerprint=p['fingerprint'], arm=arm,
                systems_passed=True, coverage=coverage, candidates=candidates, joint_change=change,
                initializer=proofs[arm]['initializer'], training_seconds=training_seconds,
                elapsed_seconds=time.monotonic()-started, lease_id=lease['id'],
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), selection_status='Generation and judgment pending',
                qualified=None, automatic_extra_epochs=False, resume_supported=False, confirmed=False))
    except BaseException as error:
        failure(out, error); raise
    finally:
        for value in (trainer, dataset, mc_dataset):
            if value is not None:
                value.close()


def candidate(root, p, arm, label):
    require_arm(arm); require_label(label)
    row = receipts._completed(Path(root)/'arms'/arm/'run/candidates'/label, p)
    if row.get('arm') != arm or row.get('label') != label or row['packed_export']['attention_retry']['mode'] != arm:
        raise ValueError('Candidate attention arm or label differs')
    return row


@torch.inference_mode()
def generate(root, arm, label):
    require_arm(arm); require_label(label)
    p = verify(root); row = candidate(root, p, arm, label)
    p, out = start(root, arm, 'generation', label)
    try:
        lease, guard = parent.reservation(p['profile']); torch.set_num_threads(2)
        _, records = data.verify(p['data_root'], live=False)
        with base.execution.arithmetic_policy():
            model, reload = adapter.load_packed_snapshot(row['packed_export']['directory'], device='cuda:0', guard=guard)
            if reload['attention_retry']['mode'] != arm:
                raise ValueError('Generation attention mode differs')
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(p['model']['path'], local_files_only=True, trust_remote_code=False)
            contexts = {r['example_id']: r for r in map(json.loads, (Path(p['corpus'])/'text.jsonl').read_text().splitlines())}
            with adapter.packed.attach_prefix(model, p['model']['path'], mode='cpu') as prefix:
                probe = base.packed_evaluation.packed_probe(model, records[p['generation_indices'][0]].prompt_ids)
                replay = base.packed_evaluation.packed_probe(model, records[p['generation_indices'][0]].prompt_ids,
                    prefixes=[r['input_ids'] for r in probe])
                pathproof = base.packed_evaluation.reload_parity(probe, replay)
                started = time.monotonic()
                canaries = base.evaluation.evaluate_canaries(model, tokenizer, device='cuda:0',
                    eos_token_ids=p['generation_eos_token_ids'], max_new_tokens=p['max_new_tokens'])
                write(out/'canaries.json', canaries); rows = []
                for i in p['generation_indices']:
                    record = records[i]
                    result = base.evaluation.generate_record(model, record, tokenizer, device='cuda:0',
                        source_text=contexts[record.example_id]['context'], eos_token_ids=p['generation_eos_token_ids'],
                        max_new_tokens=p['max_new_tokens'])
                    rows.append(result)
                    with (out/'responses.jsonl').open('a') as stream:
                        stream.write(json.dumps(result)+'\n')
                seconds = time.monotonic()-started
            if canaries['count'] != 6 or len(canaries['records']) != 6 or len(rows) != 16:
                raise ValueError('All six canaries and16 exposed DEV cases required')
            write(out/'contexts.json', {r['example_id']: contexts[r['example_id']] for r in rows})
            return finish(out, p, dict(status='completed', fingerprint=p['fingerprint'], arm=arm, label=label,
                checkpoint=row['checkpoint'], canaries=canaries, corpus=rows, generation_seconds=seconds,
                prefix=prefix, prefix_mode='cpu', uncached=True, reload=reload, full_vocabulary_replay=pathproof,
                backend=p['backend'], lease_id=lease['id'], confirmed=False))
    except BaseException as error:
        failure(out, error); raise


def judge(root, arm, label):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU judgment requires CUDA hidden')
    require_arm(arm); require_label(label)
    p = verify(root); row = candidate(root, p, arm, label)
    generation = Path(root)/'arms'/arm/'generation'/label
    actual = receipts._completed(generation, p)
    if actual.get('arm') != arm or actual.get('label') != label or actual['checkpoint'] != row['checkpoint']:
        raise ValueError('Judgment generation binding differs')
    p, out = start(root, arm, 'judgment', label)
    try:
        with Path(p['judge_lock']).open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            native = read(Path(p['response_cache_root'])/'native/generation.json'); contexts = read(generation/'contexts.json')
            executable = base.task_metrics.grade_generated_tasks(native['corpus'], actual['corpus'], contexts,
                maximum_drop_pp=5., minimum_student_rate=0.)
            write(out/'executable.json', executable)
            semantic = base.judge.judge_pairs(native['corpus'], actual['corpus'], out/'semantic',
                config=base.judge.JudgeConfig(**p['judge']), contexts={k: dict(prompt=v['prompt_text'],
                    source=v['context'], reference=v['response'], task=v['task']) for k, v in contexts.items()})
            quality = parent.prior.task_probe.quality_summary(native, actual, semantic, executable)
            valid = [j for j in semantic['judgments'] if j.get('status') == 'completed']
            available = quality['evidence_available']
            return finish(out, p, dict(status='completed', fingerprint=p['fingerprint'], arm=arm, label=label,
                checkpoint=row['checkpoint'], evidence_available=available,
                qualified=bool(row['mc_eligibility']['passed'] and quality['diagnostic_generation_retention_pass']) if available else None,
                useful_completed_tasks=sum(j['student']['task_success'] and j['student']['generation_success'] for j in valid) if available else None,
                reliable_generations=sum(j['student']['generation_success'] for j in valid) if available else None,
                quality=quality, mc_guardrail=row['mc_eligibility'], nll=row['nll'], update=row['update'], confirmed=False))
    except Exception as error:
        failure(out, error)
        return finish(out, p, dict(status='completed', fingerprint=p['fingerprint'], arm=arm, label=label,
            checkpoint=row['checkpoint'], evidence_available=False, qualified=None, useful_completed_tasks=None,
            reliable_generations=None, error_type=type(error).__name__, error=str(error),
            nll=row['nll'], update=row['update'], confirmed=False))
    except BaseException as error:
        failure(out, error); raise


def paired_mc(control, intervention):
    """Exact matched-ID descriptive wins/losses; no population inference."""
    def index(rows):
        values = {data.digest(row['identity']): row for row in rows}
        if len(values) != len(rows):
            raise ValueError('Duplicate paired MC identities')
        return values
    first, second = index(control), index(intervention)
    if not first or first.keys() != second.keys():
        raise ValueError('Paired MC cases must have exactly matched IDs')
    wins = losses = both_correct = both_wrong = 0
    for key, left in first.items():
        right = second[key]
        if left['answer'] != right['answer'] or left['family'] != right['family']:
            raise ValueError('Paired MC labels differ')
        a, b = left['prediction'] == left['answer'], right['prediction'] == right['answer']
        wins += int(b and not a); losses += int(a and not b)
        both_correct += int(a and b); both_wrong += int(not a and not b)
    delta = 100*(wins-losses)/len(first)
    return dict(count=len(first), exact_matched_ids=[row['identity'] for row in control],
        quadratic_wins=wins, quadratic_losses=losses, both_correct=both_correct, both_wrong=both_wrong,
        accuracy_delta_pp=delta, maximum_drop_pp=5., retention_pass=delta >= -5.,
        independent_population_claim=False)


def check(root):
    root = Path(root).resolve(); p = verify(root, live=False); arms = {}; pending = []; failures = []
    for arm in ARMS:
        ar = root/'arms'/arm; state = {}; judgments = []
        for phase in ('pilot', 'run'):
            path = ar/phase
            if (path/'manifest.json').exists():
                state[phase] = receipts._completed(path, p)
            elif path.exists():
                failures.append(str(path))
        for label in LABELS:
            for phase in ('generation', 'judgment'):
                path = ar/phase/label
                if (path/'manifest.json').exists():
                    row = receipts._completed(path, p)
                    saved = candidate(root, p, arm, label)
                    if row.get('arm') != arm or row.get('label') != label or row.get('checkpoint') != saved['checkpoint']:
                        raise ValueError('Paired phase evidence binding differs')
                    if phase == 'judgment':
                        judgments.append(row)
                        if row.get('evidence_available') is not True:
                            pending.append(f'{arm}/{phase}/{label}/evidence_unavailable')
                else:
                    pending.append(f'{arm}/{phase}/{label}')
                    if path.exists():
                        failures.append(str(path))
        state['task_selection'] = parent.select_quality(judgments)
        state['judgments'] = judgments; arms[arm] = state
    comparison = dict(final_mc=None, final_semantic=None, confirmed=False,
                      scope='Exposed DEV descriptive paired diagnostic; no independent confirmation')
    if all('run' in arms[arm] for arm in ARMS):
        mc = [read(root/'arms'/arm/'run/candidates/update_0071/multiple_choice.json')['records'] for arm in ARMS]
        comparison['final_mc'] = paired_mc(*mc)
    finals = {arm: next((r for r in arms[arm]['judgments'] if r['label'] == 'update_0071'), None) for arm in ARMS}
    if all(row is not None and row.get('evidence_available') is True for row in finals.values()):
        comparison['final_semantic'] = dict(
            softmax_useful_completed_tasks=finals['softmax']['useful_completed_tasks'],
            quadratic_useful_completed_tasks=finals['quadratic']['useful_completed_tasks'],
            useful_completed_delta=finals['quadratic']['useful_completed_tasks']-finals['softmax']['useful_completed_tasks'],
            softmax_reliable_generations=finals['softmax']['reliable_generations'],
            quadratic_reliable_generations=finals['quadratic']['reliable_generations'])
    complete = not pending and not failures and all('run' in arms[a] for a in ARMS)
    return dict(status='completed' if complete else 'blocked' if failures else 'pending', fingerprint=p['fingerprint'],
        arms=arms, comparison=comparison, generation_or_judgment_pending=pending, partial_or_failed_phases=failures,
        selection_final=complete, parent_selection_status=p['parent_selection']['status'],
        parent_qualified=p['parent_selection']['qualified'], confirmed=False, automatic_expansion=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('phase', choices=('prepare', 'pilot', 'run', 'generate', 'judge', 'check'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent-root', type=Path)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--label', choices=LABELS)
    args = parser.parse_args()
    if args.phase == 'prepare':
        if args.parent_root is None:
            parser.error('prepare requires --parent-root')
        result = prepare(args.root, args.parent_root)
    elif args.phase == 'check':
        result = check(args.root)
    else:
        if args.arm is None or args.phase in ('generate', 'judge') and args.label is None:
            parser.error('arm and generation/judgment label required')
        result = globals()[args.phase](args.root, args.arm, *([args.label] if args.phase in ('generate', 'judge') else []))
    print(json.dumps({k: result[k] for k in ('status', 'fingerprint', 'passed', 'arm', 'label') if k in result}, allow_nan=False))


if __name__ == '__main__':
    main()
