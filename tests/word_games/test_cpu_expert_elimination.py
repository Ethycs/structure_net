import hashlib
import json
from pathlib import Path
import re
import struct

import pytest

from experiments.word_games import cpu_expert_elimination as run
from experiments.word_games.prepare_expert_compression import tensor_table, hashes


def test_all_gpu_arms_retain_every_expert_and_the_requested_thread_counts():
    for name, arm in run.ARMS.items():
        assert sorted(i for ids in arm.groups.values() for i in ids) == list(range(26))
        cmd = run.command(arm, Path('/tmp/derived.gguf'), 18088, 'test')
        assert cmd[cmd.index('--threads') + 1] == str(arm.threads)
        assert cmd[cmd.index('--threads-batch') + 1] == str(arm.threads)
        rules = cmd[cmd.index('--override-tensor') + 1].split(',')
        for i in range(26):
            found = [dev for rule in rules for pattern, dev in [rule.split('=')]
                     if re.search(pattern, f'blk.{i}.ffn_down_exps.weight')]
            assert found == [next(d for d, ids in arm.groups.items() if i in ids)]
        assert ('CPU' in [r.split('=')[1] for r in rules]) == (name == 'packed38')
        assert ('/tmp/derived.gguf:/model.gguf:ro' in cmd) == arm.compressed
        assert cmd[cmd.index('--ctx-size') + 1] == '8192'
        assert cmd[cmd.index('--cache-type-k') + 1] == 'q8_0'


def test_verifier_rejects_cpu_fallback_and_missing_projection():
    arm = run.ARMS['gpu40']
    lines = [f'layer {i} assigned to device CUDA0, detail' for i in range(42)]
    lines += [f'tensor blk.{i}.ffn_{part}_exps.weight (144 MiB q4_K) buffer type overridden to {dev}'
              for dev, ids in arm.groups.items() for i in ids for part in ('up', 'gate', 'down')]
    log = '\n'.join(lines)
    assert run.verify(log, arm)['cpu_banks'] == 0
    with pytest.raises(ValueError):
        run.verify(log.replace('CUDA2', 'CUDA_Host'), arm)
    with pytest.raises(ValueError):
        run.verify('\n'.join(lines[:-1]), arm)


def test_cell_restores_hooks_even_when_native_runner_raises(monkeypatch, tmp_path):
    original = run.base.command, run.base.verify, run.base.NativeQwen38, run.base.HYPOTHESIS_ID

    def fail(out, extra, phase, producer, metadata, port):
        # Calling through the installed hook must not recurse.
        cmd = run.base.command(extra, port, 'test')
        assert cmd[cmd.index('--threads') + 1] == '1'
        raise RuntimeError('simulated startup failure')

    monkeypatch.setattr(run.base, 'run_cell', fail)
    with pytest.raises(RuntimeError, match='simulated'):
        run.cell(tmp_path / 'cell', run.ARMS['gpu40_cpu1'], 'pilot', {}, {}, Path('/tmp/q.gguf'), 18088)
    assert (run.base.command, run.base.verify, run.base.NativeQwen38, run.base.HYPOTHESIS_ID) == original


def tiny_gguf(path, payload=b'abcdefgh'):
    def string(s):
        b = s.encode()
        return struct.pack('<Q', len(b)) + b
    data = b'GGUF' + struct.pack('<IQQ', 3, 1, 1)
    data += string('general.alignment') + struct.pack('<II', 4, 32)
    data += string('weight') + struct.pack('<IQIQ', 1, 2, 0, 0)
    data += b'\0' * ((-len(data)) % 32) + payload
    path.write_bytes(data)


def test_streaming_directory_sizes_and_hashes_include_header_and_exact_payload(tmp_path):
    p = tmp_path / 'test.gguf'
    tiny_gguf(p)
    table = tensor_table(p)
    assert len(table) == 1 and table[0]['bytes'] == 8 and table[0]['shape'] == [2]
    whole, tensors = hashes(p, table)
    assert whole == hashlib.sha256(p.read_bytes()).hexdigest()
    assert tensors == {'weight': hashlib.sha256(b'abcdefgh').hexdigest()}
    p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError, match='truncated'):
        tensor_table(p)


def test_receiver_grading_uses_packet_truth_not_baseline_words(monkeypatch, tmp_path):
    packet = {'decision': 'applied', 'entities': [
        {'entity_id': 'entity:a', 'placement': {'slot': 'A1', 'return_by': '2027-01-10'}}]}
    inputs = tmp_path / 'inputs.json'
    inputs.write_text(json.dumps([{'id': 'cases/example/receiver',
        'prompt': 'AUTHORIZED PACKET\n' + json.dumps(packet)}]))
    monkeypatch.setattr(run, 'REQUESTS', inputs)
    monkeypatch.setattr(run, 'Metaphor', lambda **kwargs: kwargs)
    selector = tmp_path / 'runs/metaphor/response.json'
    selector.parent.mkdir(parents=True)
    selector.write_text(json.dumps({'parsed': {}}))
    response = tmp_path / 'runs/cases/example/receiver/response.json'
    response.parent.mkdir(parents=True)
    answer = {'decision': 'applied', 'occupied_slots': [
        {'entity_id': 'entity:a', 'slot': 'A1', 'return_by': '2027-01-10'}], 'summary': 'different wording'}
    response.write_text(json.dumps({'parsed': answer}))
    assert run.receiver_accuracy(tmp_path)['receiver_correct'] == 1
    answer['occupied_slots'][0]['return_by'] = '2027-01-11'
    response.write_text(json.dumps({'parsed': answer}))
    assert run.receiver_accuracy(tmp_path)['receiver_correct'] == 0
