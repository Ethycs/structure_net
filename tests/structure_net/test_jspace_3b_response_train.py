"""Real tiny compact stacks test dense response/EOS recovery and prefix behavior."""
from dataclasses import replace

import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding

from experiments.structure_net import jspace_3b_response_train as study
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack

data_module = study.module("distillation_data")


class Dataset:
    def __init__(self, cut=0):
        self.cut = cut
        self.training, self.development = [0, 1], [2]
        provenance = dict(source_uri="fixture", source_revision="fixture-v1",
            tokenizer_sha256="tokenizer", chat_template_sha256="template",
            prompt_provenance="fixture", response_provenance="reference",
            assistant_boundary_verified=True)
        self.records = [data_module.ResponseRecord(example_id=f"case{i}", group_id=f"group{i}",
            split="train" if i < 2 else "development", task="fixture", prompt_ids=(1, 3+i),
            response_ids=((6, 4) if i % 2 == 0 else (8, 9, 4)),
            provenance=provenance, finish_reason="eos", eos_token_id=4) for i in range(3)]
        self.hidden = [torch.randn(len(r.input_ids), 8).bfloat16() for r in self.records]
        self.boundary = [torch.randn(len(r.input_ids), 8).bfloat16() for r in self.records]

    def batch(self, indices):
        assert len(indices) == 1
        i = indices[0]
        batch = data_module.response_batch([self.records[i]], pad_token_id=0, max_length=6)
        batch.update(teacher_hidden=self.hidden[i][None], teacher_input_ids=batch["input_ids"].clone(),
                     teacher_record_ids=[self.records[i].example_id])
        if self.cut:
            batch["boundary_hidden"] = self.boundary[i][None]
        return batch


def make_trainer(cut=0, **changes):
    torch.manual_seed(26351)
    cfg = Qwen2Config(hidden_size=8, intermediate_size=12, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
        attention_dropout=0.)
    cfg._attn_implementation = "eager"
    blocks = []
    for layer in range(cut, 2):
        native = Qwen2DecoderLayer(cfg, layer).bfloat16().eval()
        basis = torch.linalg.qr(torch.randn(8, 8)).Q[:, :4]
        block = AffineQwenBlock(native, basis, torch.randn(6, 8)/10,
            trainable=True, parameter_dtype=torch.bfloat16)
        block.compile_polynomial(dict(anchors=[0, 1], mean=torch.zeros(12), scale=torch.ones(12),
            coefficients={"2": torch.randn(12, 6)/20}), [10, 11])
        for p in block.parameters(): p.data = p.data.bfloat16()
        blocks.append(block)
    stack = CompactQwenStack(blocks, geometry_mode="bf16", pad_to_multiple=0, checkpoint_blocks=True)
    norm = Qwen2RMSNorm(8).bfloat16()
    readout = torch.randn(16, 8).bfloat16()
    rotary = Qwen2RotaryEmbedding(cfg).eval()
    config = replace(study.ResponseConfig(), learning_rate=.02, max_length=6,
        optimizer_chunk_bytes=64, token_chunk_size=1, **changes)
    return study.ResponseTrainer(stack, norm, readout, rotary, Dataset(cut), config=config,
                                 teacher_id="native-fixture", expected_layers=2, host_reserve_bytes=0)


@pytest.mark.parametrize("cut", [0, 1])
def test_full_response_updates_every_block_and_actual_bf16_weights(cut, monkeypatch):
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU fixture initialized CUDA"))
    trainer = make_trainer(cut)
    try:
        before = trainer.parameter_fingerprints()
        initial = trainer.development()[0]
        receipt = trainer.train_window([0, 1])
        assert receipt["supervised_tokens"] == 5  # Includes BOTH observed EOS tokens.
        assert receipt["cumulative_input_tokens"] == 9
        assert receipt["gradients_present_by_block"] == [True] * (2-cut)
        assert receipt["loss"] == receipt["forward_kl"]
        assert receipt["optimizer"]["updated_scalars"] == sum(p.numel() for p in trainer.parameters)
        assert trainer.change_report(before)["bf16_changed_by_block"] == [True] * (2-cut)
        assert trainer.change_report(before)["norm_bf16_storage_changed"]
        assert all(s["step"] == 1 for s in trainer.optimizer.state.values())
        assert all(s[k].dtype == torch.float32 for s in trainer.optimizer.state.values()
                   for k in ("master", "exp_avg", "exp_avg_sq"))
        assert trainer.optimizer.cpu_gradients == trainer.optimizer.received == {}
        assert study.offload.fixed_state(trainer.stack, trainer.readout, trainer.rotary) == trainer.fixed
        assert initial["supervised_tokens"] == trainer.development()[0]["supervised_tokens"] == 2
    finally:
        trainer.close()


def test_complete_response_budget_and_no_repeated_train_examples():
    trainer = make_trainer(max_supervised_tokens=4)
    try:
        with pytest.raises(ValueError, match="budget"):
            trainer.train_window([0, 1])
        assert trainer.updates == 0
        trainer.train_window([0])
        for indices in ([0], [2], [1, 1]):
            with pytest.raises(ValueError, match="TRAIN-only"):
                trainer.train_window(indices)
        with pytest.raises(ValueError, match="budget"):
            trainer.train_window([1])
        assert trainer.supervised_tokens == 2
    finally:
        trainer.close()


def test_cache_cannot_score_different_tokens_or_record_identity():
    trainer = make_trainer()
    try:
        batch = trainer.dataset.batch([0])
        batch["teacher_input_ids"][0, 0] += 1
        with pytest.raises(ValueError, match="cache token identity"):
            trainer.teacher.prepare(batch)
        batch = trainer.dataset.batch([0])
        batch["teacher_record_ids"] = ["wrong"]
        with pytest.raises(ValueError, match="cache token identity"):
            trainer.teacher.prepare(batch)
    finally:
        trainer.close()


def test_progressive_generation_requires_real_prefix_each_generated_history():
    trainer = make_trainer(cut=1)
    try:
        ids = torch.tensor([[1, 3]])
        with pytest.raises(ValueError, match="actual native prefix"):
            trainer.model.next_logits(ids)
        calls = []
        def prefix(tokens, valid):
            calls.append(tokens.tolist())
            return torch.nn.functional.embedding(tokens, trainer.readout)
        trainer.model.native_prefix = prefix
        first = trainer.model.next_logits(ids)
        second = trainer.model.next_logits(torch.tensor([[1, 3, int(first.argmax())]]))
        assert len(calls) == 2 and len(calls[1][0]) == 3
        assert first.shape == second.shape == (16,)
    finally:
        trainer.close()


def test_saved_weights_reload_exactly_with_fixed_geometry(tmp_path):
    trainer = make_trainer()
    try:
        trainer.train_window([0])
        expected = trainer.development()
        receipt = trainer.save_checkpoint(tmp_path / "candidate.pt", identities={"initializer": "toy"})
        assert receipt["resume_requires_optimizer"]
        state = torch.load(receipt["path"], map_location="cpu", weights_only=True)
        restored = make_trainer()
        try:
            restored.stack.load_state_dict(state["stack"], strict=True)
            restored.norm.load_state_dict(state["norm"], strict=True)
            assert restored.development() == expected
        finally:
            restored.close()
    finally:
        trainer.close()
