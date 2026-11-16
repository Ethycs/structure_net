from copy import deepcopy
from contextlib import contextmanager

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack
from experiments.structure_net.jspace_3b_dense_adapter import DenseCompactQwen, HFTeacherTargets


def make_models(rank=8, checkpoint=True, full_structure=False, dtype=torch.float32):
    torch.manual_seed(913)
    config = Qwen2Config(hidden_size=8, intermediate_size=12, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=19,
        max_position_embeddings=32, attention_dropout=0., tie_word_embeddings=True)
    config._attn_implementation = "eager"
    teacher = Qwen2ForCausalLM(config).to(dtype).eval()
    projection = (dict(rope_pairs=torch.tensor([[0]]), value_basis=torch.eye(4)[None, :, :2])
                  if full_structure else None)
    blocks = [AffineQwenBlock(layer, torch.eye(8)[:, :rank], torch.randn(12, 8)/10,
                             head_projection=projection) for layer in teacher.model.layers]
    if full_structure:
        for block in blocks:
            block.compile_polynomial(dict(anchors=[0, 1], mean=torch.zeros(12), scale=torch.ones(12),
                coefficients={'2': torch.randn(12, 6)/100}), [10, 11])
    stack = CompactQwenStack(blocks, checkpoint_blocks=checkpoint,
        geometry_mode='bf16' if dtype == torch.bfloat16 else 'fp32')
    student = DenseCompactQwen(stack, deepcopy(teacher.model.norm),
        teacher.model.embed_tokens.weight.detach().clone(), deepcopy(teacher.model.rotary_emb),
        max_length=12, expected_layers=2)
    return teacher, student


def batch():
    return dict(input_ids=torch.tensor([[2, 3, 4, 5, 6], [3, 6, 8, 0, 0]]),
                attention_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]))


def test_full_rank_all_vocabulary_native_parity_and_causal_padding():
    teacher, student = make_models()
    b = batch()
    with torch.no_grad():
        actual = student.readout(student.hidden_forward(b).reshape(-1, 8)).reshape(2, 5, 19)
        expected = teacher(**b).logits
        torch.testing.assert_close(actual[b['attention_mask'].bool()],
                                   expected[b['attention_mask'].bool()], atol=2e-6, rtol=2e-5)
        for row, length in ((0, 5), (1, 3)):
            single = b['input_ids'][row:row+1, :length]
            torch.testing.assert_close(student.next_logits(single), actual[row, length-1],
                                       atol=2e-6, rtol=2e-5)
        modified = {k: v.clone() for k, v in b.items()}
        modified['input_ids'][0, 4] = 12
        after = student.hidden_forward(modified)
        torch.testing.assert_close(after[0, :4], student.hidden_forward(b)[0, :4], rtol=0, atol=0)


def test_compact_backprop_all_layers_fixed_buffers_and_strict_reload(tmp_path):
    teacher, student = make_models(rank=4)
    before = {k: t.clone() for k, t in student.named_buffers()}
    teacher_before = {k: t.clone() for k, t in teacher.state_dict().items()}
    original = deepcopy(student.state_dict())
    optimizer = torch.optim.AdamW(student.parameters(), lr=.005)
    b = batch()
    hidden = student.hidden_forward(b)
    logits = student.readout(hidden[:, 1:3].reshape(-1, 8))
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([4, 5, 8, 1]))
    loss.backward()
    assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters())
               for block in student.stack.blocks)
    assert student.embedding.grad is None
    optimizer.step()
    assert any(not torch.equal(original[name], p) for name, p in student.named_parameters())
    assert all(torch.equal(before[name], t) for name, t in student.named_buffers())
    assert all(torch.equal(teacher_before[name], t) for name, t in teacher.state_dict().items())
    path = tmp_path / 'checkpoint.pt'
    torch.save(student.state_dict(), path)
    _, reloaded = make_models(rank=4)
    reloaded.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.equal(student.next_logits(b['input_ids'][:1]), reloaded.next_logits(b['input_ids'][:1]))


def test_teacher_chunks_match_full_logits_cache_prefix_binding_and_context():
    teacher, _ = make_models()
    events = []
    @contextmanager
    def resident():
        events.append('enter')
        yield
        events.append('exit')
    provider = HFTeacherTargets(teacher, max_length=12, teacher_id='frozen-fixture', teacher_context=resident)
    b = batch()
    positions = torch.tensor([[0, 2], [0, 3], [1, 1]])
    with pytest.raises(ValueError, match='prepared'):
        provider(b, positions)
    provider.prepare(b)
    provider.prepare(b)
    assert provider.capture_count == 1 and events == ['enter', 'exit']
    expected = teacher(**b).logits.detach()[positions[:, 0], positions[:, 1]]
    torch.testing.assert_close(provider(b, positions), expected, rtol=2e-6, atol=2e-7)
    assert not provider(b, positions).requires_grad
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    changed = {k: v.clone() for k, v in b.items()}
    changed['input_ids'][0, 0] = 18
    with pytest.raises(ValueError, match='exact input prefixes'):
        provider(changed, positions)
    with pytest.raises(ValueError, match='padding'):
        provider(b, torch.tensor([[1, 4]]))


def test_context_and_structure_contracts_are_not_silently_relaxed():
    _, student = make_models(rank=4)
    with pytest.raises(ValueError, match='no truncation'):
        student.next_logits(torch.ones(1, 13, dtype=torch.long))
    with pytest.raises(ValueError, match='right-padded'):
        student.hidden_forward(dict(input_ids=torch.ones(1, 3, dtype=torch.long),
                                    attention_mask=torch.tensor([[1, 0, 1]])))
    with pytest.raises(ValueError, match='Packed-document'):
        student.hidden_forward({**batch(), 'segment_ids': torch.zeros(2, 5)})
    wrong = deepcopy(student.state_dict())
    wrong['_extra_state']['max_length'] = 13
    with pytest.raises(ValueError, match='same model and context'):
        student.load_state_dict(wrong)
    with pytest.raises(ValueError, match='every layer'):
        DenseCompactQwen(student.stack, student.norm, student.embedding,
                         student.rotary, max_length=12, expected_layers=3)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_shared_dense_trainer_updates_real_compact_qwen_and_reloads(tmp_path, dtype):
    from experiments.neural_architecture_lab.distillation_recovery import module
    training, data = module('distillation_training'), module('distillation_data')
    teacher, student = make_models(rank=4, full_structure=True, dtype=dtype)
    assert all(b.has_polynomial and b.head_dim == 2 and b.rank == 4 for b in student.stack.blocks)
    teacher.eval().requires_grad_(False)
    teacher_id = training.module_fingerprint(teacher)
    records = [data.ResponseRecord(example_id=f'qwen-{i}', group_id=f'qwen-group-{i}',
        split='train', task='systems_fixture', prompt_ids=(2, 3+i), response_ids=(5+i, 8+i, 18),
        finish_reason='eos', eos_token_id=18, provenance=dict(source_uri='synthetic://compact-qwen',
        source_revision='v1', tokenizer_sha256='a'*64, chat_template_sha256='b'*64,
        prompt_provenance='synthetic token fixture', response_provenance='synthetic token fixture',
        assistant_boundary_verified=True)) for i in range(2)]
    b = data.response_batch(records, pad_token_id=0, max_length=12)
    identities = dict(teacher=teacher_id, student_initializer=training.module_fingerprint(student),
        tokenizer='a'*64, chat_template='b'*64, data=data.dataset_fingerprint(records))
    provider = HFTeacherTargets(teacher, max_length=12, teacher_id=teacher_id,
                                shared_readout=student.embedding)
    config = training.DistillationConfig(token_chunk_size=2, max_supervised_tokens=6, max_updates=1)
    events = []
    trainer = training.DistillationTrainer(student, torch.optim.AdamW(student.parameters(), lr=.005),
        student.hidden_forward, student.readout, provider, config=config, identities=identities)
    result = trainer.fit([b], tmp_path / 'fit', progress=events.append)
    assert result['updates'] == 1 and result['supervised_tokens'] == 6
    assert result['fixed_state_unchanged'] and result['in_process_reload_exact']
    assert all(any(n.startswith(f'stack.blocks.{i}.') for n in result['changed_parameters']) for i in range(2))
    assert training.module_fingerprint(teacher) == teacher_id
    _, fresh = make_models(rank=4, full_structure=True, dtype=dtype)
    training.load_checkpoint(result['final_checkpoint']['path'], fresh,
        expected_sha256=result['final_checkpoint']['sha256'], identities=identities)
    assert torch.equal(student.next_logits(b['input_ids'][:1]), fresh.next_logits(b['input_ids'][:1]))
    assert events and [e for e in events if e['phase'] == 'training'][-1]['cumulative_supervised_tokens'] == 6


def test_shared_readout_requires_exact_frozen_original_weights():
    teacher, student = make_models(rank=4)
    provider = HFTeacherTargets(teacher, max_length=12, teacher_id='fixture',
                                shared_readout=student.embedding)
    b = batch()
    provider.prepare(b)
    assert provider._readout.data_ptr() == student.embedding.data_ptr()
    with pytest.raises(ValueError, match='exactly match'):
        HFTeacherTargets(teacher, max_length=12, teacher_id='fixture',
                         shared_readout=student.embedding + .1)
