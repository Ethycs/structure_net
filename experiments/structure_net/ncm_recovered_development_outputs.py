"""Generate two fixed DEV readouts from one audited recovered-data predictor.

This consumer does not grade summaries, encode references, open Final320, or
choose a checkpoint. All preceding semantic branches must already be closed.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import torch

from experiments.structure_net import ncm_recovered_training_verified_cache as verified
from experiments.structure_net import ncm_source_hierarchy_development as hierarchy
from experiments.structure_net import ncm_recovered_paired_readout as paired
from experiments.structure_net import ncm_recovered_paired_readout_pilot as pilot

study, data, previous = verified.study, verified.study.data, hierarchy.previous
PLAN = study.PLAN
TEST = Path('tests/structure_net/test_ncm_recovered_development_outputs.py')
CPU = study.inputs.BASE/'20261028_recovered_DEV_outputs_CPU_v1.json'
ROOTS = {s: study.inputs.BASE/f'20261028_recovered_DEV_{s}_v1'
         for s in ('preflight', 'inference_pilot', 'outputs')}
DECODER_SHA = '7bc8beea1357c15f344903720f1207086591f0223b2e9ec81690c1af468f789b'
INSTALLED = False


def check_training(value, audit):
    expected = {'status': 'recovered_paraphrase_training_completed', 'stage': 'train',
        'updates': 6416, 'recovered_pairs': 43894, 'source_groups': 38484, 'reviewed_anchors': 2384,
        'config': asdict(study.CONFIG), 'trainable_parameters': 105323136,
        'strict_final_reload_passed': True, 'protected_tensors_unchanged': True,
        'fresh_optimizer_masters_match_initial_weights': True,
        'systems_lifecycle_only_not_quality_evidence': False, 'final_test_opened': False}
    if any(value.get(k) != v for k,v in expected.items()):
        raise ValueError('complete fixed recovered-data training required')
    if (audit.get('status') != 'recovered_full_training_replayed' or audit.get('stage') != 'train'
            or audit.get('updates') != 6416 or not audit.get('actual_saved_protected_tensors_verified')
            or audit.get('source_checkpoint_sha256') != value['checkpoint_sha256']):
        raise ValueError('independent audit of this exact full checkpoint required')
    replay = audit['replay']
    exact = {'updates': 6416, 'presentations': 205248,
        'counts': {'recovered': 153936, 'anchor': 51312, 'noisy': 102624},
        'distinct_recovered_sources': 38484, 'distinct_recovered_targets': 43894,
        'distinct_reviewed_anchors': 2384, 'source_visit_histogram': {'4': 38484},
        'all_sample_and_corruption_counts_replayed': True}
    if any(replay.get(k) != v for k,v in exact.items()):
        raise ValueError('complete four-epoch source, target and noise exposure required')


def closed_history():
    training, evaluation, environment, ancestor, cache, paths = hierarchy.current_history('dense')
    # hierarchy.current_history includes existing, expanded and tree. The last
    # dense branch must also be closed before this new candidate can be read out.
    stages = hierarchy.ROOTS['dense']
    audited = previous.completed(stages['audit'])
    cache = previous.history.parent.original.append_grade(cache, stages['grade'], evaluation)
    if audited['canonical_first_judgments'] != len(cache):
        raise ValueError('complete dense semantic branch must precede recovered evaluation')
    ancestor.audit_cache(cache)
    paths += [stages[s]/n for s in ('grade', 'audit')
              for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]
    paths += [stages['grade']/n for n in ('judgment_cache.json', 'exact_packets.json')]
    return training, evaluation, environment, cache, paths


def context():
    global INSTALLED
    if not INSTALLED:
        verified.install(); INSTALLED = True
    proof = verified.ProcessProofs()
    qualification = proof.read(CPU); proof.verify(qualification['artifacts'])
    if qualification['actual_tool_exit_code'] != 0 or qualification['tests_passed'] != 3:
        raise ValueError('consumer admission and identity contracts must pass first')
    trained, audited = study.require(proof, 'train'), study.require(proof, 'audit')
    check_training(trained, audited)
    if proof.bind(study.ROOTS['train']/'model.pt') != trained['checkpoint_sha256']:
        raise ValueError('fixed final checkpoint bytes differ')
    readout = proof.completed(pilot.ROOT)
    if (readout['status'] != 'paired_source_predicted_readouts_qualified'
            or not readout['original_sequential_outputs_equal'] or not readout['exact_parent_restore']
            or not readout['all_code_pairs_identical'] or readout['decoder_checkpoint_sha256'] != DECODER_SHA):
        raise ValueError('actual paired-readout model qualification required')
    decoder = pilot.decoder.study.require(pilot.decoder.study.ROOTS['train'])
    for stage in ('audit', 'replay'):
        pilot.decoder.study.require(pilot.decoder.evaluation.ROOTS[stage])
    if proof.bind(decoder['checkpoint']) != DECODER_SHA: raise ValueError('fixed step6212 decoder required')
    for root in (pilot.decoder.study.ROOTS['train'], *[pilot.decoder.evaluation.ROOTS[s] for s in ('audit', 'replay')]):
        proof.completed(root)
    training, evaluation, environment, cache, paths = closed_history()
    for path in paths: proof.bind(path)
    catalog = study.inputs.qualified.CATALOG
    proof.completed(catalog)
    recovered = study.inputs.lines(catalog/'sources.jsonl'); proof.bind(catalog/'sources.jsonl')
    binding = proof.read(study.inputs.BINDINGS)
    anchor_root = Path(binding['targets']['full']); proof.completed(anchor_root)
    anchors = proof.read(anchor_root/'cases.json')
    if len(recovered) != 40890 or len(anchors) != 2384: raise ValueError('complete TRAIN union required for split audit')
    cohorts, manifest, split = previous.sources(evaluation, recovered+anchors)
    for path in (evaluation.old.SOURCE/'cases.json', evaluation.old.FEATURES/'manifest.json'):
        proof.bind(path)
    return proof, training, evaluation, environment, cache, trained, decoder, cohorts, manifest, split


def freeze(root, stage, proofs, training, environment, cache, trained):
    if root.exists(): raise ValueError('fresh append-only source-only consumer required')
    files = {Path(__file__).resolve(), TEST.resolve(), PLAN.resolve()}
    files.update(p.resolve() for p in training.native.vendor.VENDOR.rglob('*.py'))
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            p = Path(filename).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()): files.add(p)
    for path in (Path('pixi.lock'), Path('pyproject.toml')): proofs.bind(path)
    value = {'schema': 'ncm-recovered-source-only-DEV-outputs-v1', 'hypothesis_id': study.schedule.HYPOTHESIS,
        'stage': stage, 'environment': environment,
        'config': {'source_only': True, 'arms': list(paired.ARMS), 'checkpoint_sha256': trained['checkpoint_sha256'],
            'decoder_checkpoint_sha256': DECODER_SHA, 'physical_GPU_for_generation': 0,
            'inference_precision': 'FP32', 'TF32': False, 'batch': 1,
            'peak_limit_bytes': study.CONFIG.peak_limit_bytes, 'inherited_first_judgments': len(cache),
            'one_prediction_per_source_per_pass': True, 'pilot_repetition_only_for_determinism': True,
            'references_or_target_codes_used': False, 'final_test_opened': False, 'speed_tested': False},
        'source_hashes': {str(p): proofs.sha(p) for p in sorted(files)}, 'input_hashes': dict(proofs.inputs)}
    return previous.write_proof(root, value, files)


def finish(root, proof, proofs, value):
    proofs.verify(proof['source_hashes']); proofs.verify(proof['input_hashes'])
    result = {'schema': proof['schema'], 'hypothesis_id': study.schedule.HYPOTHESIS, **value,
        'task_accuracy': None, 'new_judge_requests': 0, 'final_test_opened': False, 'speed_tested': False,
        'producer_fingerprint': proof['fingerprint'],
        'artifacts': {str(p): proofs.sha(p) for p in root.rglob('*.json') if 'producer_source' not in p.parts}}
    data.write(root/'result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'artifacts'}), flush=True)


def merge_readout(parent, new, tokenizer, render):
    if set(new) != {'codes', 'raw_word_ids'} or parent['codes'] != new['codes']:
        raise ValueError('decoder may only render unchanged source-predicted codes')
    identity = {k: parent[k] for k in ('id', 'source_sha256', 'split', 'feature_sha256', 'source_positions')}
    return {**identity, **new, **render(new['raw_word_ids'], tokenizer)}


def run(stage):
    root = ROOTS[stage]
    if root.exists(): raise ValueError('fresh append-only output stage required')
    proofs, training, evaluation, environment, cache, trained, decoder, cohorts, manifest, split = context()
    if stage == 'preflight':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized(): raise ValueError('CPU-only preflight required')
        proof = freeze(root, stage, proofs, training, environment, cache, trained)
        data.write(root/'judgment_cache.json', cache); data.write(root/'split_proof.json', split)
        finish(root, proof, proofs, {'status': 'recovered_source_only_DEV_preflight_passed',
            'cases': 128, 'canonical_first_judgments': len(cache), 'checkpoint_sha256': trained['checkpoint_sha256']})
        return
    admitted = proofs.completed(ROOTS['preflight'])
    if (admitted['status'] != 'recovered_source_only_DEV_preflight_passed'
            or admitted['checkpoint_sha256'] != trained['checkpoint_sha256']
            or proofs.read(ROOTS['preflight']/'judgment_cache.json') != cache):
        raise ValueError('same fixed model and complete unchanged semantic ancestry required')
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3060'
            or torch.cuda.mem_get_info()[0] < study.CONFIG.peak_limit_bytes):
        raise ValueError('physical GPU0 RTX3060 with full readout headroom required')
    study.pipeline.determinism.configure_runtime()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    chosen = evaluation.old.inference_probes(cohorts, manifest) if stage == 'inference_pilot' else [
        r for c in previous.COHORTS for r in cohorts[c]]
    before = None
    if stage == 'outputs':
        completed = proofs.completed(ROOTS['inference_pilot'])
        if not completed['all_code_pairs_identical'] or not completed['exact_parent_restore']:
            raise ValueError('complete paired inference pilot required')
        before = proofs.read(ROOTS['inference_pilot']/'outputs.json')
    for row in chosen: proofs.bind(manifest[row['id']]['path'], manifest[row['id']]['sha256'])
    proof = freeze(root, stage, proofs, training, environment, cache, trained)
    data.write(root/'split_proof.json', split)
    torch.cuda.reset_peak_memory_stats()
    model = study.task.load_initial(training)
    saved = torch.load(study.ROOTS['train']/'model.pt', map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(saved['model'], strict=True); del saved
    model.eval().requires_grad_(False)
    outputs = {a: [] for a in paired.ARMS}
    try:
        if (training.native.vendor.module_hash(model) != trained['final_model_sha256']
                or training.prior.protected(model) != trained['frozen_hashes']):
            raise ValueError('exact trained predictor and original frozen readout required')
        tokenizer = training.native.tokenizer()
        for row in chosen:
            output = evaluation.predict(model, row, manifest, tokenizer)
            if stage == 'inference_pilot' and evaluation.predict(model, row, manifest, tokenizer) != output:
                raise ValueError('source-only inference does not repeat exactly')
            outputs['parent_with_source'].append(output); data.write(root/'outputs.json', outputs)
            if torch.cuda.max_memory_allocated() > study.CONFIG.peak_limit_bytes: raise ValueError('inference memory ceiling exceeded')
            if len(outputs['parent_with_source']) % 16 == 0: print(json.dumps({'source_predictions': len(outputs['parent_with_source'])}), flush=True)
        fitted = torch.load(decoder['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
        if fitted['frozen_state_sha256'] != pilot.decoder.study.old.frozen_hash(model.codec):
            raise ValueError('new decoder uses a different frozen encoder or VQ')
        protected = pilot.decoder.parent.decoder.protected(model)
        with paired.standalone_decoder(model, fitted['decoder']):
            if (training.native.vendor.module_hash(model.codec) != decoder['final_codec_sha256']
                    or pilot.decoder.parent.decoder.protected(model) != protected):
                raise ValueError('decoder swap changed protected tensors')
            for parent in outputs['parent_with_source']:
                new = paired.decode_standalone(model, parent['codes'])
                if stage == 'inference_pilot' and new != paired.decode_standalone(model, parent['codes']):
                    raise ValueError('standalone readout does not repeat exactly')
                outputs['new_codec_only'].append(merge_readout(parent, new, tokenizer, evaluation.old.decode))
        count = paired.paired_identity(outputs['parent_with_source'], outputs['new_codec_only'])
        if count != len(chosen) or training.native.vendor.module_hash(model) != trained['final_model_sha256']:
            raise ValueError('complete paired outputs and exact parent restoration required')
        if before is not None:
            for arm in paired.ARMS:
                current = {r['id']: r for r in outputs[arm]}
                if any(current[r['id']] != r for r in before[arm]): raise ValueError('full output differs from fixed pilot')
        if torch.cuda.max_memory_allocated() > study.CONFIG.peak_limit_bytes: raise ValueError('readout memory ceiling exceeded')
        data.write(root/'outputs.json', outputs)
        finish(root, proof, proofs, {'status': 'complete_paired_source_only_'+stage, 'cases': count,
            'all_code_pairs_identical': True, 'exact_parent_restore': True,
            'shared_predictor_calls': count, 'determinism_repetition_calls': count if stage == 'inference_pilot' else 0,
            'outputs_sha256': proofs.sha(root/'outputs.json'), 'model_sha256': trained['final_model_sha256'],
            'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(), 'memory_excludes_live_Llama_encoder': True})
    finally: model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('stage', choices=tuple(ROOTS)); args = parser.parse_args()
    root = ROOTS[args.stage]; existed = root.exists()
    try: run(args.stage)
    except Exception as error:
        if root.exists() and not existed:
            data.write(root/'failure.json', {'type': type(error).__name__, 'message': str(error), 'task_accuracy': None})
        raise
