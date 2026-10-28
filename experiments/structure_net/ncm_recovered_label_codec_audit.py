"""Replay saved codec outputs and expose edits without another model call."""
import difflib
import json
import os
from pathlib import Path
import re

from experiments.structure_net import ncm_recovered_label_codec as study

data = study.data
ROOT = data.ROOT/'20261027_recovered_label_codec_audit_v1'


def run():
    if ROOT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only readback required')
    receipt = data.read(study.ROOT/'actual_tool_receipt.json'); data.verify(receipt['artifacts'])
    if receipt['actual_tool_exit_code'] != 0:
        raise ValueError('successful actual reconstruction terminal required')
    result = data.read(study.ROOT/'result.json'); proof = data.read(study.ROOT/'producer.json')
    data.verify(proof['source_hashes']); data.verify(proof['input_hashes'])
    training, _ = study.codec_source.prior.runtime(); tokenizer = training.native.tokenizer()
    original = data.read(study.sample.ROOT/'cases.json'); cases = data.read(study.ROOT/'cases.json')
    outputs = data.read(study.ROOT/'outputs.json')
    if len(original) != 128 or len(cases) != 128 or len(outputs) != 128:
        raise ValueError('entire fixed sample required')
    edits = []
    for selected, case, output in zip(original, cases, outputs):
        lexical = tokenizer.encode(selected['target_text'], add_special_tokens=False, truncation=False)
        frame = study.targets.catalog.native_frame(lexical, tokenizer.all_special_ids)
        native = tokenizer.decode(lexical, skip_special_tokens=True)
        if case != {**selected, 'native_word_ids': frame, 'native_reference': native}:
            raise ValueError('source, saved target or complete packing differs')
        codes, raw = output['codes'], output['raw_word_ids']
        if len(codes) != 641 or any(type(v) is not int or not 0 <= v < 8192 for v in codes):
            raise ValueError('complete native code hierarchy required')
        if len(raw) != 128 or any(type(v) is not int or not 0 <= v < 30528 for v in raw):
            raise ValueError('complete native raw output required')
        text = training.preparation.rendering.decode(raw, tokenizer)['text']
        expected = {'id': case['id'], 'stratum': case['stratum'], 'target_text': case['target_text'],
            'native_reference': native, 'codes': codes, 'raw_word_ids': raw, 'text': text,
            'invalid_outputs': int(any(v >= len(tokenizer) for v in raw)),
            'tokenizer_digit_sequence_preserved': re.findall(r'\d+', case['target_text']) == re.findall(r'\d+', native),
            **study.codec_source.prior.metrics(frame, raw, native, text)}
        if output != expected:
            raise ValueError('raw text or per-case metrics do not replay')
        if text != native:
            a, b = native.split(), text.split()
            changes = [{'operation': tag, 'reference': ' '.join(a[i:j]), 'decoded': ' '.join(b[k:l]),
                        'reference_word_positions': [i, j], 'decoded_word_positions': [k, l]}
                       for tag, i, j, k, l in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if tag != 'equal']
            edits.append({'id': case['id'], 'stratum': case['stratum'], 'reference': native, 'decoded': text,
                'source': case['source'], 'changes': changes, 'digit_sequence_preserved': output['digit_sequence_preserved']})
    if study.aggregate(outputs) != result['metrics']:
        raise ValueError('aggregate reconstruction metrics differ')
    ROOT.mkdir(); data.write(ROOT/'edit_diagnostics.json', edits)
    value = {'status': 'all128_raw_outputs_and_metrics_replayed', 'cases': 128, 'changed_cases': len(edits),
        'digit_sequence_changed_cases': sum(not r['digit_sequence_preserved'] for r in edits),
        'metrics': result['metrics'], 'model_calls': 0, 'semantic_accuracy': None,
        'artifacts': {str(p): data.sha(p) for p in (Path(__file__), study.ROOT/'producer.json',
            study.ROOT/'result.json', study.ROOT/'actual_tool_receipt.json', study.ROOT/'cases.json',
            study.ROOT/'outputs.json', ROOT/'edit_diagnostics.json')}}
    data.write(ROOT/'result.json', value)
    print(json.dumps({k: v for k, v in value.items() if k not in ('artifacts', 'metrics')}), flush=True)
    print(json.dumps({'edits': [{'id': r['id'], 'changes': r['changes'], 'digits_preserved': r['digit_sequence_preserved']} for r in edits]}), flush=True)


if __name__ == '__main__':
    run()
