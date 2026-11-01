from pathlib import Path
import pytest

from experiments.word_games import target_70tps as run


def test_whole_layer_commands_preserve_precision_and_pipeline_control():
    for c in run.CONFIGS.values():
        cmd = run.command(c, None, 18088, 'test')
        assert cmd[cmd.index('--ubatch-size') + 1] == str(c.microbatch)
        assert cmd[cmd.index('--cache-type-k') + 1] == 'q8_0'
        assert cmd[cmd.index('--ctx-size') + 1] == '8192'
        assert cmd[cmd.index('--n-cpu-moe') + 1] == '0'
        assert cmd[cmd.index('--threads') + 1] == '1'
        if c.whole:
            assert cmd[cmd.index('--device') + 1] == 'CUDA2,CUDA1,CUDA0'
            assert cmd[cmd.index('--tensor-split') + 1] == '9.5,9,23.5'
            assert ('--override-tensor' not in cmd) == c.pipeline


def test_whole_layer_verifier_rejects_wrong_device_and_pipeline():
    c = run.CONFIGS['whole256']
    log = '\n'.join(f'layer {i} assigned to device ' +
                    ('CUDA2' if i < 10 else 'CUDA1' if i < 19 else 'CUDA0') for i in range(42))
    log += '\n' + '\n'.join(f'load_tensors: {k} model buffer size = {v} MiB' for k, v in run.PARENT_BUFFERS.items())
    log += '\noffloaded 42/42 layers to GPU\ngraph: nodes = 3939, splits = 4'
    assert run.verify(log, c)['graphs'][0]['splits'] == 4
    with pytest.raises(ValueError, match='assignment'):
        run.verify(log.replace('layer 9 assigned to device CUDA2', 'layer 9 assigned to device CUDA1'), c)
    with pytest.raises(ValueError, match='pipeline'):
        run.verify(log + '\npipeline parallelism enabled', c)


def test_failed_cell_restores_measurement_hooks(monkeypatch, tmp_path):
    original = run.cpu.command, run.cpu.verify, run.cpu.HYPOTHESIS
    def fail(*args):
        assert run.cpu.HYPOTHESIS == run.HYPOTHESIS
        raise RuntimeError('simulated failure')
    monkeypatch.setattr(run.cpu, 'cell', fail)
    with pytest.raises(RuntimeError, match='simulated'):
        run.cell(tmp_path, run.CONFIGS['control'], 'pilot', {}, {}, 18088)
    assert (run.cpu.command, run.cpu.verify, run.cpu.HYPOTHESIS) == original


def test_followup_flags_do_not_change_weights_context_or_experts():
    for c in run.FOLLOWUPS.values():
        cmd = run.command(c, None, 18088, 'test')
        assert cmd[cmd.index('--ctx-size') + 1] == '8192'
        assert cmd[cmd.index('--n-cpu-moe') + 1] == '0'
        assert ('--ctx-checkpoints' in cmd) == c.no_checkpoints
        assert ('GGML_CUDA_GRAPH_OPT=1' in cmd) == c.graph_opt
        assert cmd[cmd.index('--ubatch-size') + 1] == str(c.microbatch)
        if c.no_checkpoints:
            assert cmd[cmd.index('--ctx-checkpoints') + 1] == '0'
        if c.graph_opt:
            assert cmd.index('GGML_CUDA_GRAPH_OPT=1') < cmd.index(run.cpu.base.IMAGE)


def test_final_stage_batches_and_mtp_only_put_draft_experts_on_cpu():
    for c in run.BATCH_MTP.values():
        cmd = run.command(c, None, 18088, 'test')
        assert cmd[cmd.index('--batch-size') + 1] == str(c.batch)
        assert cmd[cmd.index('--ctx-size') + 1] == '8192'
        if c.mtp:
            assert cmd[cmd.index('--spec-type') + 1] == 'draft-mtp'
            assert cmd[cmd.index('--spec-draft-n-max') + 1] == '1'
            assert cmd[cmd.index('--threads') + 1] == '12'
            rules = cmd[cmd.index('--override-tensor') + 1].split(',')
            assert [r for r in rules if r.endswith('=CPU')] == ['^blk\\.40\\.ffn_.*_exps\\.weight$=CPU']
