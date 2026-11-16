from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import Phi3Config, Phi3ForCausalLM

from experiments.structure_net.jspace_logit_sketch import project_sign_probe_groups
from experiments.structure_net.jspace_phi_capture import PhiGroupedTeacherCapture, capture_compiled_development
from experiments.structure_net.jspace_qwen_parallel_fit import finalize_basis


def fixture():
    torch.manual_seed(82164)
    config = Phi3Config(hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=32,
        max_position_embeddings=16, original_max_position_embeddings=16,
        rope_theta=250000, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        tie_word_embeddings=False, resid_pdrop=0., embd_pdrop=0., attention_dropout=0.)
    config._attn_implementation = 'eager'
    model = Phi3ForCausalLM(config).bfloat16().eval().requires_grad_(False)
    cases = [dict(identity=f'case{i}', split='train' if i < 2 else 'development', family='arc',
                  input_ids=[1, 4 + i, 7, 3, 8][:5-i], loss_positions=[0, 2]) for i in range(3)]
    return model, cases


def execute(model, case):
    return model.model(input_ids=torch.tensor([case['input_ids']]), use_cache=False).last_hidden_state


def manual_graph(model, case):
    """Execute the actual layers explicitly, retaining actual fused output."""
    x = model.model.embed_tokens(torch.tensor([case['input_ids']])).detach().requires_grad_(True)
    positions = torch.arange(x.shape[1])[None]
    rope = model.model.rotary_emb(x, positions)
    mask = torch.full((1, 1, x.shape[1], x.shape[1]), torch.finfo(x.dtype).min, dtype=x.dtype).triu(1)
    captured = {}; handles = []
    for i, block in enumerate(model.model.layers):
        captured[i] = {'input': x}
        handles.append(block.self_attn.qkv_proj.register_forward_hook(
            lambda m, a, o, layer=i: captured[layer].__setitem__('qkv', o)))
        handles.append(block.mlp.down_proj.register_forward_pre_hook(
            lambda m, a, layer=i: captured[layer].__setitem__('mlp', a[0])))
        x = block(x, attention_mask=mask, position_ids=positions, cache_position=positions[0],
                  position_embeddings=rope, use_cache=False)[0]
    for handle in handles: handle.remove()
    return model.model.norm(x), captured


def read_bits(path):
    return torch.from_numpy(np.load(path, allow_pickle=False).copy()).view(torch.bfloat16)


def run(root, model, cases, calibration=(0,), **kwargs):
    with PhiGroupedTeacherCapture(root, model, [0, 1], cases, calibration,
            teacher_fingerprint='pinned-untied-teacher', probe_groups=2, probes_per_group=2,
            activation_positions=3, position_capacity=8, seed=493, **kwargs) as recorder:
        for i, case in enumerate(cases): recorder.capture_case(i, lambda c=case: execute(model, c))
        return recorder.finish()


def test_actual_fused_forward_vjp_statistics_and_untied_readout(tmp_path):
    model, cases = fixture(); initial = deepcopy(model.state_dict())
    expected_logits = [F.linear(execute(model, c)[0, c['loss_positions']], model.lm_head.weight) for c in cases]
    output, captured = manual_graph(model, cases[0])
    assert torch.equal(output, execute(model, cases[0]))
    vectors = project_sign_probe_groups(model.lm_head.weight, 2, 2,
                                        torch.Generator().manual_seed(493), groups=2)
    targets = [captured[i][key] for i in range(2) for key in ('input', 'qkv', 'mlp')]
    grams = {i: torch.zeros(2, 16, 16) for i in range(2)}
    masks = {i: torch.zeros(2, 4, 4) for i in range(2)}
    derivatives = {i: [] for i in range(2)}
    for group in range(2):
        for probe in range(2):
            scalar = (output[0, cases[0]['loss_positions']].float() * vectors[group, probe]).sum()
            grads = torch.autograd.grad(scalar, targets, retain_graph=True)
            for layer in range(2):
                gx, gqkv, gm = grads[layer*3:layer*3+3]
                grams[layer][group].addmm_(gx[0].float().T, gx[0].float(), alpha=.5)
                q, k, _ = captured[layer]['qkv'].detach().float().split((16, 8, 8), -1)
                gq, gk, _ = gqkv.float().split((16, 8, 8), -1)
                qmask = (q * gq)[0].reshape(-1, 2, 2, 4).sum((0, 2))
                kmask = (k * gk)[0].reshape(-1, 2, 4).sum(0)
                pair = (qmask[:, :2] + qmask[:, 2:] + kmask[:, :2] + kmask[:, 2:]).flatten()
                masks[layer][group] += pair[:, None] * pair[None] / 2
                derivatives[layer].append(gm[0])
    calls = [0, 0]; handles = []
    for layer in range(2):
        def count(_m, _a, _o, i=layer): calls[i] += 1
        handles.append(model.model.layers[layer].self_attn.qkv_proj.register_forward_hook(count))
    metadata = run(tmp_path/'capture', model, cases)
    for handle in handles: handle.remove()
    assert calls == [len(cases)] * 2  # One native fused forward, not separate Q/K/V.
    assert torch.equal(read_bits(metadata['accepted_teacher_logits']), torch.stack(expected_logits))
    for layer in range(2):
        paths = metadata['blocks'][str(layer)]
        stats = torch.load(paths['statistics'], weights_only=True)
        assert torch.equal(stats['grams'], grams[layer])
        assert torch.equal(stats['mask_grams'], masks[layer])
        assert torch.equal(read_bits(paths['mlp_jacobian_bf16']), torch.cat(derivatives[layer]))
        assert torch.equal(stats['pair_scores'].reshape(2, -1), torch.diagonal(masks[layer], dim1=1, dim2=2))
        assert paths['shape'] == {'width':16, 'intermediate':24, 'heads':4, 'kv_heads':2, 'head_dim':4}
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in initial.items())
    assert all(p.grad is None for p in model.parameters())
    assert not hasattr(model.model.layers[0].self_attn, 'q_proj')
    assert metadata['architecture'] == 'phi3_fused_qkv_phi4'
    assert metadata['probe_groups'] == 2 and metadata['probes_per_group'] == 2


def test_development_cannot_change_training_statistics_and_generic_fit(tmp_path):
    model, cases = fixture()
    left = run(tmp_path/'a', model, cases)
    changed = deepcopy(cases); changed[2]['input_ids'] = [1, 17, 21]
    right = run(tmp_path/'b', model, changed)
    for layer in range(2):
        a = torch.load(left['blocks'][str(layer)]['statistics'], weights_only=True)
        b = torch.load(right['blocks'][str(layer)]['statistics'], weights_only=True)
        assert all(torch.equal(a[k], b[k]) for k in a)
    outputs = finalize_basis(tmp_path/'a', tmp_path/'fit', layer_ids=[1], rank=8, head_dim=2, device='cpu', progress=lambda _:None)
    fitted = torch.load(outputs['1']['fit'], weights_only=True)
    assert fitted['basis'].shape == (16,16) and fitted['rank'] == 8
    assert fitted['rope_pairs'].shape == (2,1) and fitted['value_basis'].shape == (2,4,2)
    assert json.loads((tmp_path/'a'/'items.json').read_text())['items'][0]['jacobian_row_offset'] == 0
    def compiled_forward(block, x, case):
        positions = torch.arange(x.shape[1])[None]
        return block(x, position_ids=positions, position_embeddings=model.model.rotary_emb(x, positions))
    interface = capture_compiled_development(tmp_path/'fit'/'block_01', tmp_path/'a', 1,
        model.model.layers[1], compiled_forward, device='cpu', progress=lambda _:None)
    assert interface['layer'] == 1 and interface['rank'] == 8 and interface['head_dim'] == 2
    assert read_bits(interface['compiled_development']).shape == (3,24)
    weight = torch.load(interface['output_weight'], weights_only=True)
    assert weight.shape == (8,24) and torch.equal(weight, weight.bfloat16().float())


def test_guard_before_artifacts_and_rejects_dev_vjps_or_wrong_precision(tmp_path):
    model, cases = fixture()
    def denied(): raise PermissionError('lease denied')
    with pytest.raises(PermissionError): run(tmp_path/'denied', model, cases, guard=denied)
    assert not (tmp_path/'denied').exists()
    with pytest.raises(ValueError, match='training cases'): run(tmp_path/'dev', model, cases, calibration=(2,))
    assert not (tmp_path/'dev').exists()
    with pytest.raises(ValueError, match='BF16'): run(tmp_path/'float', model.float(), cases)
    assert not (tmp_path/'float').exists()


def test_hook_cleanup_and_frozen_teacher_failure(tmp_path):
    model, cases = fixture()
    with PhiGroupedTeacherCapture(tmp_path/'failed', model, [1], cases, [0],
            teacher_fingerprint='frozen', probe_groups=2, probes_per_group=1) as recorder:
        with pytest.raises(ValueError, match='detached'):
            recorder.capture_case(0, lambda: execute(model, cases[0]).detach())
        assert recorder.failed
        block = model.model.layers[1]
        assert not block._forward_pre_hooks and not block.self_attn.qkv_proj._forward_hooks and not block.mlp.down_proj._forward_pre_hooks
        with pytest.raises(ValueError, match='complete'): recorder.finish()
    with PhiGroupedTeacherCapture(tmp_path/'mutated', model, [1], cases, [0], teacher_fingerprint='frozen') as recorder:
        with torch.no_grad(): model.model.layers[1].self_attn.qkv_proj.weight.add_(.1)
        with pytest.raises(ValueError, match='changed'): recorder.capture_case(0, lambda: execute(model, cases[0]))
