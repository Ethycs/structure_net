import math
import torch
import pytest

from experiments.structure_net.ncm_attention_code_repair_v3 import AttentionCodeRepair, Config, level_loss


def setup_case():
    torch.manual_seed(37)
    c = Config(source_width=6, code_width=8, width=16, heads=2, blocks=2,
        levels=(1, 2, 4), vocabularies=(16, 16, 16), dropout=0.)
    model = AttentionCodeRepair(c)
    books = [torch.randn(16, 8) for _ in c.levels]
    codes = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    features = torch.randn(1, 5, 6, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False]])
    return model, books, codes, features, mask


def test_initialization_preserves_all_codes_even_with_duplicate_book_vectors():
    model, books, codes, features, mask = setup_case()
    books[0][0] = books[0][1]
    assert torch.equal(model.eval().repair(codes, features, mask, books), codes)


def test_real_error_supervision_reaches_source_attention_without_source_or_book_gradients():
    model, books, codes, features, mask = setup_case()
    oracle = (codes + 1) % 16
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    initial = model.delta.weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = model.train()(codes, features, mask, books)
        loss = level_loss(logits, oracle, model.config.levels)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    assert not torch.equal(model.delta.weight, initial)
    assert model.source_kv.weight.grad.abs().sum() > 0
    assert all(b.cross_q.weight.grad.abs().sum() > 0 for b in model.blocks)
    assert features.grad is None
    assert all(b.grad is None for b in books)


def test_masked_source_values_cannot_affect_output_after_learning():
    model, books, codes, features, mask = setup_case()
    torch.nn.init.normal_(model.delta.weight, std=.1)
    model.eval()
    changed = features.detach().clone()
    changed[:, 3:] = float('nan')
    a = model(codes, features, mask, books)
    b = model(codes, changed, mask, books)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    changed[:, :3] += torch.randn_like(changed[:, :3])
    c = model(codes, changed, mask, books)
    assert any(not torch.equal(x, z) for x, z in zip(a, c))


def test_grad_enabled_books_and_incomplete_hierarchies_are_rejected():
    model, books, codes, features, mask = setup_case()
    books[0].requires_grad_(True)
    with pytest.raises(ValueError):
        model(codes, features, mask, books)
    books[0].requires_grad_(False)
    with pytest.raises(ValueError):
        model(codes[:, :-1], features, mask, books)


def test_root_scale_spread_keeps_identity_and_bounds_loss_and_gradients():
    model, books, codes, features, mask = setup_case()
    # Reproduce the native root geometry: tiny entries and norms above 7,000.
    books[0].zero_()
    books[0][1, 0] = 2085.
    books[0][2, 1] = 7222.
    books[0][3, 2] = 1e-30
    before = [book.clone() for book in books]
    assert torch.equal(model.eval().repair(codes, features, mask, books), codes)
    oracle = torch.zeros_like(codes)
    scores = model.train()(codes, features, mask, books)
    loss = level_loss(scores, oracle, model.config.levels)
    assert loss < 4*model.config.score_scale + 8 + math.log(16)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.delta.weight.grad.abs().sum() > 0
    # Even a large learned residual must retain bounded classifier scores.
    with torch.no_grad():
        model.delta.weight.normal_(std=100.)
    for level in model.eval()(codes, features, mask, books):
        assert torch.isfinite(level).all()
        assert level.min() >= -4*model.config.score_scale-.0001 and level.max() <= 8.0001
    assert all(torch.equal(book, saved) for book, saved in zip(books, before))
    assert features.grad is None
